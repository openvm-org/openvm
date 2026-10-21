// Lean compiler output
// Module: Aesop.Builder.Forward
// Imports: public import Init public meta import Init public import Aesop.Builder.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getBinderInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_binderInfo(lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instOrdPremiseIndex_ord(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_forward;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
lean_object* lean_io_get_num_heartbeats();
lean_object* lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_name(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_ElabRuleTerm_scope(lean_object*);
lean_object* lp_aesop_Aesop_PhaseSpec_toRule(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_expr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleInfo_ofExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t lp_aesop_Aesop_instHashableBuilderName_hash(uint8_t);
uint64_t lp_aesop_Aesop_instHashablePhaseName_hash(uint8_t);
uint64_t lp_aesop_Aesop_instHashableScopeName_hash(uint8_t);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_PhaseSpec_phase(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_IndexingMode_format(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_debug;
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RulePattern_elab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_elabRuleTermForApplyLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_forwardTransparency(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_forwardTransparency___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_forwardIndexTransparency(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_forwardIndexTransparency___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 70, .m_capacity = 70, .m_length = 69, .m_data = "aesop: internal error: immediate arg for forward rule is out of range"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "aesop: forward builder: "};
static const lean_object* lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix;
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_RuleBuilder_getImmediatePremises_spec__1(uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_RuleBuilder_getImmediatePremises_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___closed__0 = (const lean_object*)&lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4_spec__4(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "argument '"};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__1;
static lean_once_cell_t lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__2;
static const lean_string_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 66, .m_capacity = 66, .m_length = 65, .m_data = "' cannot be immediate since it is already determined by a pattern"};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__3_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__4;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "function does not have arguments with these names: '"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__1;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__2;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0;
static const lean_string_object lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__2 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__7(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__7___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__0 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__0_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6(lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = ", forward deps "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__3;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "slot "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__4_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__5;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " (premise "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__6_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__7;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = ", deps "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__8_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__9;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = ", common "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__10_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__11;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cluster "};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__2_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__3_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "conclusion deps: "};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__1;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "slot clusters"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__3;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__4;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rule type:"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__5 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_RuleBuilder_forwardCore_spec__0(lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Aesop.Builder.Forward"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Aesop.RuleBuilder.forwardCore"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__1_value;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__3;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "imode: "};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__4 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__5;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "immediate premises: "};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__6 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__7;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forwardCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "decl type: "};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__8 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forwardCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___closed__9;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_forward___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "forward builder"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forward___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forward___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_RuleBuilder_forward___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleBuilder_forward___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_RuleBuilder_forward___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_forward___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forward___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forward___closed__2;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_forward___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_forward___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_forwardTransparency(lean_object* v_opts_1_){
_start:
{
lean_object* v_transparency_x3f_2_; 
v_transparency_x3f_2_ = lean_ctor_get(v_opts_1_, 4);
if (lean_obj_tag(v_transparency_x3f_2_) == 0)
{
uint8_t v___x_3_; 
v___x_3_ = 2;
return v___x_3_;
}
else
{
lean_object* v_val_4_; uint8_t v___x_5_; 
v_val_4_ = lean_ctor_get(v_transparency_x3f_2_, 0);
v___x_5_ = lean_unbox(v_val_4_);
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_forwardTransparency___boxed(lean_object* v_opts_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_aesop_Aesop_RuleBuilderOptions_forwardTransparency(v_opts_6_);
lean_dec_ref(v_opts_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_forwardIndexTransparency(lean_object* v_opts_9_){
_start:
{
lean_object* v_indexTransparency_x3f_10_; 
v_indexTransparency_x3f_10_ = lean_ctor_get(v_opts_9_, 5);
if (lean_obj_tag(v_indexTransparency_x3f_10_) == 0)
{
uint8_t v___x_11_; 
v___x_11_ = 2;
return v___x_11_;
}
else
{
lean_object* v_val_12_; uint8_t v___x_13_; 
v_val_12_ = lean_ctor_get(v_indexTransparency_x3f_10_, 0);
v___x_13_ = lean_unbox(v_val_12_);
return v___x_13_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_forwardIndexTransparency___boxed(lean_object* v_opts_14_){
_start:
{
uint8_t v_res_15_; lean_object* v_r_16_; 
v_res_15_ = lp_aesop_Aesop_RuleBuilderOptions_forwardIndexTransparency(v_opts_14_);
lean_dec_ref(v_opts_14_);
v_r_16_ = lean_box(v_res_15_);
return v_r_16_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg(lean_object* v_x_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = l_Lean_Meta_saveState___redArg(v___y_19_, v___y_21_);
if (lean_obj_tag(v___x_23_) == 0)
{
lean_object* v_a_24_; lean_object* v_r_25_; 
v_a_24_ = lean_ctor_get(v___x_23_, 0);
lean_inc(v_a_24_);
lean_dec_ref_known(v___x_23_, 1);
lean_inc(v___y_21_);
lean_inc_ref(v___y_20_);
lean_inc(v___y_19_);
lean_inc_ref(v___y_18_);
v_r_25_ = lean_apply_5(v_x_17_, v___y_18_, v___y_19_, v___y_20_, v___y_21_, lean_box(0));
if (lean_obj_tag(v_r_25_) == 0)
{
lean_object* v_a_26_; lean_object* v___x_27_; 
v_a_26_ = lean_ctor_get(v_r_25_, 0);
lean_inc(v_a_26_);
lean_dec_ref_known(v_r_25_, 1);
v___x_27_ = l_Lean_Meta_SavedState_restore___redArg(v_a_24_, v___y_19_, v___y_21_);
lean_dec(v_a_24_);
if (lean_obj_tag(v___x_27_) == 0)
{
lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_34_; 
v_isSharedCheck_34_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_34_ == 0)
{
lean_object* v_unused_35_; 
v_unused_35_ = lean_ctor_get(v___x_27_, 0);
lean_dec(v_unused_35_);
v___x_29_ = v___x_27_;
v_isShared_30_ = v_isSharedCheck_34_;
goto v_resetjp_28_;
}
else
{
lean_dec(v___x_27_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_34_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v___x_32_; 
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 0, v_a_26_);
v___x_32_ = v___x_29_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_33_; 
v_reuseFailAlloc_33_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_33_, 0, v_a_26_);
v___x_32_ = v_reuseFailAlloc_33_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
return v___x_32_;
}
}
}
else
{
lean_object* v_a_36_; lean_object* v___x_38_; uint8_t v_isShared_39_; uint8_t v_isSharedCheck_43_; 
lean_dec(v_a_26_);
v_a_36_ = lean_ctor_get(v___x_27_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_43_ == 0)
{
v___x_38_ = v___x_27_;
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
else
{
lean_inc(v_a_36_);
lean_dec(v___x_27_);
v___x_38_ = lean_box(0);
v_isShared_39_ = v_isSharedCheck_43_;
goto v_resetjp_37_;
}
v_resetjp_37_:
{
lean_object* v___x_41_; 
if (v_isShared_39_ == 0)
{
v___x_41_ = v___x_38_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_a_36_);
v___x_41_ = v_reuseFailAlloc_42_;
goto v_reusejp_40_;
}
v_reusejp_40_:
{
return v___x_41_;
}
}
}
}
else
{
lean_object* v_a_44_; lean_object* v___x_45_; 
v_a_44_ = lean_ctor_get(v_r_25_, 0);
lean_inc(v_a_44_);
lean_dec_ref_known(v_r_25_, 1);
v___x_45_ = l_Lean_Meta_SavedState_restore___redArg(v_a_24_, v___y_19_, v___y_21_);
lean_dec(v_a_24_);
if (lean_obj_tag(v___x_45_) == 0)
{
lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_52_; 
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_45_);
if (v_isSharedCheck_52_ == 0)
{
lean_object* v_unused_53_; 
v_unused_53_ = lean_ctor_get(v___x_45_, 0);
lean_dec(v_unused_53_);
v___x_47_ = v___x_45_;
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
else
{
lean_dec(v___x_45_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
lean_object* v___x_50_; 
if (v_isShared_48_ == 0)
{
lean_ctor_set_tag(v___x_47_, 1);
lean_ctor_set(v___x_47_, 0, v_a_44_);
v___x_50_ = v___x_47_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v_a_44_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
else
{
lean_object* v_a_54_; lean_object* v___x_56_; uint8_t v_isShared_57_; uint8_t v_isSharedCheck_61_; 
lean_dec(v_a_44_);
v_a_54_ = lean_ctor_get(v___x_45_, 0);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_45_);
if (v_isSharedCheck_61_ == 0)
{
v___x_56_ = v___x_45_;
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
else
{
lean_inc(v_a_54_);
lean_dec(v___x_45_);
v___x_56_ = lean_box(0);
v_isShared_57_ = v_isSharedCheck_61_;
goto v_resetjp_55_;
}
v_resetjp_55_:
{
lean_object* v___x_59_; 
if (v_isShared_57_ == 0)
{
v___x_59_ = v___x_56_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v_a_54_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
}
else
{
lean_object* v_a_62_; lean_object* v___x_64_; uint8_t v_isShared_65_; uint8_t v_isSharedCheck_69_; 
lean_dec_ref(v_x_17_);
v_a_62_ = lean_ctor_get(v___x_23_, 0);
v_isSharedCheck_69_ = !lean_is_exclusive(v___x_23_);
if (v_isSharedCheck_69_ == 0)
{
v___x_64_ = v___x_23_;
v_isShared_65_ = v_isSharedCheck_69_;
goto v_resetjp_63_;
}
else
{
lean_inc(v_a_62_);
lean_dec(v___x_23_);
v___x_64_ = lean_box(0);
v_isShared_65_ = v_isSharedCheck_69_;
goto v_resetjp_63_;
}
v_resetjp_63_:
{
lean_object* v___x_67_; 
if (v_isShared_65_ == 0)
{
v___x_67_ = v___x_64_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v_a_62_);
v___x_67_ = v_reuseFailAlloc_68_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
return v___x_67_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg___boxed(lean_object* v_x_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg(v_x_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3(lean_object* v_00_u03b1_77_, lean_object* v_x_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg(v_x_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___boxed(lean_object* v_00_u03b1_85_, lean_object* v_x_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3(v_00_u03b1_85_, v_x_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(lean_object* v_msgData_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v___x_99_; lean_object* v_env_100_; lean_object* v___x_101_; lean_object* v_mctx_102_; lean_object* v_lctx_103_; lean_object* v_options_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_99_ = lean_st_ref_get(v___y_97_);
v_env_100_ = lean_ctor_get(v___x_99_, 0);
lean_inc_ref(v_env_100_);
lean_dec(v___x_99_);
v___x_101_ = lean_st_ref_get(v___y_95_);
v_mctx_102_ = lean_ctor_get(v___x_101_, 0);
lean_inc_ref(v_mctx_102_);
lean_dec(v___x_101_);
v_lctx_103_ = lean_ctor_get(v___y_94_, 2);
v_options_104_ = lean_ctor_get(v___y_96_, 2);
lean_inc_ref(v_options_104_);
lean_inc_ref(v_lctx_103_);
v___x_105_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_105_, 0, v_env_100_);
lean_ctor_set(v___x_105_, 1, v_mctx_102_);
lean_ctor_set(v___x_105_, 2, v_lctx_103_);
lean_ctor_set(v___x_105_, 3, v_options_104_);
v___x_106_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v_msgData_93_);
v___x_107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3___boxed(lean_object* v_msgData_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(v_msgData_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(lean_object* v_msg_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_){
_start:
{
lean_object* v_ref_121_; lean_object* v___x_122_; lean_object* v_a_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_131_; 
v_ref_121_ = lean_ctor_get(v___y_118_, 5);
v___x_122_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(v_msg_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_);
v_a_123_ = lean_ctor_get(v___x_122_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_131_ == 0)
{
v___x_125_ = v___x_122_;
v_isShared_126_ = v_isSharedCheck_131_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_a_123_);
lean_dec(v___x_122_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_131_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___x_127_; lean_object* v___x_129_; 
lean_inc(v_ref_121_);
v___x_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_127_, 0, v_ref_121_);
lean_ctor_set(v___x_127_, 1, v_a_123_);
if (v_isShared_126_ == 0)
{
lean_ctor_set_tag(v___x_125_, 1);
lean_ctor_set(v___x_125_, 0, v___x_127_);
v___x_129_ = v___x_125_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_130_; 
v_reuseFailAlloc_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_130_, 0, v___x_127_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg___boxed(lean_object* v_msg_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(v_msg_132_, v___y_133_, v___y_134_, v___y_135_, v___y_136_);
lean_dec(v___y_136_);
lean_dec_ref(v___y_135_);
lean_dec(v___y_134_);
lean_dec_ref(v___y_133_);
return v_res_138_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__1(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__0));
v___x_141_ = l_Lean_stringToMessageData(v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0(uint8_t v___x_142_, lean_object* v_type_143_, lean_object* v___x_144_, uint8_t v___x_145_, lean_object* v_val_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_keyedConfig_152_; uint8_t v_trackZetaDelta_153_; lean_object* v_zetaDeltaSet_154_; lean_object* v_lctx_155_; lean_object* v_localInstances_156_; lean_object* v_defEqCtx_x3f_157_; lean_object* v_synthPendingDepth_158_; lean_object* v_customCanUnfoldPredicate_x3f_159_; uint8_t v_univApprox_160_; uint8_t v_inTypeClassResolution_161_; uint8_t v_cacheInferType_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v_keyedConfig_152_ = lean_ctor_get(v___y_147_, 0);
v_trackZetaDelta_153_ = lean_ctor_get_uint8(v___y_147_, sizeof(void*)*7);
v_zetaDeltaSet_154_ = lean_ctor_get(v___y_147_, 1);
v_lctx_155_ = lean_ctor_get(v___y_147_, 2);
v_localInstances_156_ = lean_ctor_get(v___y_147_, 3);
v_defEqCtx_x3f_157_ = lean_ctor_get(v___y_147_, 4);
v_synthPendingDepth_158_ = lean_ctor_get(v___y_147_, 5);
v_customCanUnfoldPredicate_x3f_159_ = lean_ctor_get(v___y_147_, 6);
v_univApprox_160_ = lean_ctor_get_uint8(v___y_147_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_161_ = lean_ctor_get_uint8(v___y_147_, sizeof(void*)*7 + 2);
v_cacheInferType_162_ = lean_ctor_get_uint8(v___y_147_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_152_);
v___x_163_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_142_, v_keyedConfig_152_);
lean_inc(v_customCanUnfoldPredicate_x3f_159_);
lean_inc(v_synthPendingDepth_158_);
lean_inc(v_defEqCtx_x3f_157_);
lean_inc_ref(v_localInstances_156_);
lean_inc_ref(v_lctx_155_);
lean_inc(v_zetaDeltaSet_154_);
v___x_164_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v_zetaDeltaSet_154_);
lean_ctor_set(v___x_164_, 2, v_lctx_155_);
lean_ctor_set(v___x_164_, 3, v_localInstances_156_);
lean_ctor_set(v___x_164_, 4, v_defEqCtx_x3f_157_);
lean_ctor_set(v___x_164_, 5, v_synthPendingDepth_158_);
lean_ctor_set(v___x_164_, 6, v_customCanUnfoldPredicate_x3f_159_);
lean_ctor_set_uint8(v___x_164_, sizeof(void*)*7, v_trackZetaDelta_153_);
lean_ctor_set_uint8(v___x_164_, sizeof(void*)*7 + 1, v_univApprox_160_);
lean_ctor_set_uint8(v___x_164_, sizeof(void*)*7 + 2, v_inTypeClassResolution_161_);
lean_ctor_set_uint8(v___x_164_, sizeof(void*)*7 + 3, v_cacheInferType_162_);
v___x_165_ = l_Lean_Meta_forallMetaTelescopeReducing(v_type_143_, v___x_144_, v___x_145_, v___x_164_, v___y_148_, v___y_149_, v___y_150_);
lean_dec_ref_known(v___x_164_, 7);
if (lean_obj_tag(v___x_165_) == 0)
{
lean_object* v_a_166_; lean_object* v_fst_167_; lean_object* v___x_168_; uint8_t v___x_169_; 
v_a_166_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_a_166_);
lean_dec_ref_known(v___x_165_, 1);
v_fst_167_ = lean_ctor_get(v_a_166_, 0);
lean_inc(v_fst_167_);
lean_dec(v_a_166_);
v___x_168_ = lean_array_get_size(v_fst_167_);
v___x_169_ = lean_nat_dec_lt(v_val_146_, v___x_168_);
if (v___x_169_ == 0)
{
lean_object* v___x_170_; lean_object* v___x_171_; 
lean_dec(v_fst_167_);
v___x_170_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__1, &lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__1_once, _init_lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___closed__1);
v___x_171_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(v___x_170_, v___y_147_, v___y_148_, v___y_149_, v___y_150_);
lean_dec_ref(v___y_147_);
return v___x_171_;
}
else
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_172_ = lean_array_fget(v_fst_167_, v_val_146_);
lean_dec(v_fst_167_);
v___x_173_ = l_Lean_Expr_mvarId_x21(v___x_172_);
lean_dec(v___x_172_);
v___x_174_ = l_Lean_MVarId_getDecl(v___x_173_, v___y_147_, v___y_148_, v___y_149_, v___y_150_);
if (lean_obj_tag(v___x_174_) == 0)
{
lean_object* v_a_175_; lean_object* v_type_176_; lean_object* v___x_177_; 
v_a_175_ = lean_ctor_get(v___x_174_, 0);
lean_inc(v_a_175_);
lean_dec_ref_known(v___x_174_, 1);
v_type_176_ = lean_ctor_get(v_a_175_, 2);
lean_inc_ref(v_type_176_);
lean_dec(v_a_175_);
v___x_177_ = lp_aesop_Aesop_mkDiscrTreePath(v_type_176_, v___y_147_, v___y_148_, v___y_149_, v___y_150_);
lean_dec_ref(v___y_147_);
if (lean_obj_tag(v___x_177_) == 0)
{
lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_186_; 
v_a_178_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_186_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_186_ == 0)
{
v___x_180_ = v___x_177_;
v_isShared_181_ = v_isSharedCheck_186_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_dec(v___x_177_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_186_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_182_; lean_object* v___x_184_; 
v___x_182_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_182_, 0, v_a_178_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 0, v___x_182_);
v___x_184_ = v___x_180_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_182_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
else
{
lean_object* v_a_187_; lean_object* v___x_189_; uint8_t v_isShared_190_; uint8_t v_isSharedCheck_194_; 
v_a_187_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_194_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_194_ == 0)
{
v___x_189_ = v___x_177_;
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
else
{
lean_inc(v_a_187_);
lean_dec(v___x_177_);
v___x_189_ = lean_box(0);
v_isShared_190_ = v_isSharedCheck_194_;
goto v_resetjp_188_;
}
v_resetjp_188_:
{
lean_object* v___x_192_; 
if (v_isShared_190_ == 0)
{
v___x_192_ = v___x_189_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_193_; 
v_reuseFailAlloc_193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_193_, 0, v_a_187_);
v___x_192_ = v_reuseFailAlloc_193_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
return v___x_192_;
}
}
}
}
else
{
lean_object* v_a_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_202_; 
lean_dec_ref(v___y_147_);
v_a_195_ = lean_ctor_get(v___x_174_, 0);
v_isSharedCheck_202_ = !lean_is_exclusive(v___x_174_);
if (v_isSharedCheck_202_ == 0)
{
v___x_197_ = v___x_174_;
v_isShared_198_ = v_isSharedCheck_202_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_a_195_);
lean_dec(v___x_174_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_202_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v___x_200_; 
if (v_isShared_198_ == 0)
{
v___x_200_ = v___x_197_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_a_195_);
v___x_200_ = v_reuseFailAlloc_201_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
return v___x_200_;
}
}
}
}
}
else
{
lean_object* v_a_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_210_; 
lean_dec_ref(v___y_147_);
v_a_203_ = lean_ctor_get(v___x_165_, 0);
v_isSharedCheck_210_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_210_ == 0)
{
v___x_205_ = v___x_165_;
v_isShared_206_ = v_isSharedCheck_210_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_a_203_);
lean_dec(v___x_165_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_210_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_208_; 
if (v_isShared_206_ == 0)
{
v___x_208_ = v___x_205_;
goto v_reusejp_207_;
}
else
{
lean_object* v_reuseFailAlloc_209_; 
v_reuseFailAlloc_209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_209_, 0, v_a_203_);
v___x_208_ = v_reuseFailAlloc_209_;
goto v_reusejp_207_;
}
v_reusejp_207_:
{
return v___x_208_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___boxed(lean_object* v___x_211_, lean_object* v_type_212_, lean_object* v___x_213_, lean_object* v___x_214_, lean_object* v_val_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_){
_start:
{
uint8_t v___x_3586__boxed_221_; uint8_t v___x_3588__boxed_222_; lean_object* v_res_223_; 
v___x_3586__boxed_221_ = lean_unbox(v___x_211_);
v___x_3588__boxed_222_ = lean_unbox(v___x_214_);
v_res_223_ = lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0(v___x_3586__boxed_221_, v_type_212_, v___x_213_, v___x_3588__boxed_222_, v_val_215_, v___y_216_, v___y_217_, v___y_218_, v___y_219_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
lean_dec(v___y_217_);
lean_dec(v_val_215_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3(lean_object* v_as_224_, size_t v_i_225_, size_t v_stop_226_, lean_object* v_b_227_){
_start:
{
lean_object* v___y_229_; uint8_t v___x_233_; 
v___x_233_ = lean_usize_dec_eq(v_i_225_, v_stop_226_);
if (v___x_233_ == 0)
{
lean_object* v___x_234_; uint8_t v___x_235_; 
v___x_234_ = lean_array_uget_borrowed(v_as_224_, v_i_225_);
v___x_235_ = lean_nat_dec_le(v_b_227_, v___x_234_);
if (v___x_235_ == 0)
{
v___y_229_ = v_b_227_;
goto v___jp_228_;
}
else
{
v___y_229_ = v___x_234_;
goto v___jp_228_;
}
}
else
{
lean_inc(v_b_227_);
return v_b_227_;
}
v___jp_228_:
{
size_t v___x_230_; size_t v___x_231_; 
v___x_230_ = ((size_t)1ULL);
v___x_231_ = lean_usize_add(v_i_225_, v___x_230_);
v_i_225_ = v___x_231_;
v_b_227_ = v___y_229_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3___boxed(lean_object* v_as_236_, lean_object* v_i_237_, lean_object* v_stop_238_, lean_object* v_b_239_){
_start:
{
size_t v_i_boxed_240_; size_t v_stop_boxed_241_; lean_object* v_res_242_; 
v_i_boxed_240_ = lean_unbox_usize(v_i_237_);
lean_dec(v_i_237_);
v_stop_boxed_241_ = lean_unbox_usize(v_stop_238_);
lean_dec(v_stop_238_);
v_res_242_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3(v_as_236_, v_i_boxed_240_, v_stop_boxed_241_, v_b_239_);
lean_dec(v_b_239_);
lean_dec_ref(v_as_236_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg(lean_object* v_arr_243_){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; uint8_t v___x_248_; 
v___x_244_ = lean_unsigned_to_nat(0u);
v___x_245_ = lean_array_fget_borrowed(v_arr_243_, v___x_244_);
v___x_246_ = lean_unsigned_to_nat(1u);
v___x_247_ = lean_array_get_size(v_arr_243_);
v___x_248_ = lean_nat_dec_lt(v___x_246_, v___x_247_);
if (v___x_248_ == 0)
{
lean_inc(v___x_245_);
return v___x_245_;
}
else
{
uint8_t v___x_249_; 
v___x_249_ = lean_nat_dec_le(v___x_247_, v___x_247_);
if (v___x_249_ == 0)
{
if (v___x_248_ == 0)
{
lean_inc(v___x_245_);
return v___x_245_;
}
else
{
size_t v___x_250_; size_t v___x_251_; lean_object* v___x_252_; 
v___x_250_ = ((size_t)1ULL);
v___x_251_ = lean_usize_of_nat(v___x_247_);
v___x_252_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3(v_arr_243_, v___x_250_, v___x_251_, v___x_245_);
return v___x_252_;
}
}
else
{
size_t v___x_253_; size_t v___x_254_; lean_object* v___x_255_; 
v___x_253_ = ((size_t)1ULL);
v___x_254_ = lean_usize_of_nat(v___x_247_);
v___x_255_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1_spec__3(v_arr_243_, v___x_253_, v___x_254_, v___x_245_);
return v___x_255_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg___boxed(lean_object* v_arr_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg(v_arr_256_);
lean_dec_ref(v_arr_256_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1(lean_object* v_arr_258_){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_259_ = lean_array_get_size(v_arr_258_);
v___x_260_ = lean_unsigned_to_nat(0u);
v___x_261_ = lean_nat_dec_eq(v___x_259_, v___x_260_);
if (v___x_261_ == 0)
{
lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_262_ = lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg(v_arr_258_);
v___x_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
return v___x_263_;
}
else
{
lean_object* v___x_264_; 
v___x_264_ = lean_box(0);
return v___x_264_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1___boxed(lean_object* v_arr_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_aesop_Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1(v_arr_265_);
lean_dec_ref(v_arr_265_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__0(size_t v_sz_267_, size_t v_i_268_, lean_object* v_bs_269_){
_start:
{
uint8_t v___x_270_; 
v___x_270_ = lean_usize_dec_lt(v_i_268_, v_sz_267_);
if (v___x_270_ == 0)
{
return v_bs_269_;
}
else
{
lean_object* v_v_271_; lean_object* v___x_272_; lean_object* v_bs_x27_273_; size_t v___x_274_; size_t v___x_275_; lean_object* v___x_276_; 
v_v_271_ = lean_array_uget(v_bs_269_, v_i_268_);
v___x_272_ = lean_unsigned_to_nat(0u);
v_bs_x27_273_ = lean_array_uset(v_bs_269_, v_i_268_, v___x_272_);
v___x_274_ = ((size_t)1ULL);
v___x_275_ = lean_usize_add(v_i_268_, v___x_274_);
v___x_276_ = lean_array_uset(v_bs_x27_273_, v_i_268_, v_v_271_);
v_i_268_ = v___x_275_;
v_bs_269_ = v___x_276_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__0___boxed(lean_object* v_sz_278_, lean_object* v_i_279_, lean_object* v_bs_280_){
_start:
{
size_t v_sz_boxed_281_; size_t v_i_boxed_282_; lean_object* v_res_283_; 
v_sz_boxed_281_ = lean_unbox_usize(v_sz_278_);
lean_dec(v_sz_278_);
v_i_boxed_282_ = lean_unbox_usize(v_i_279_);
lean_dec(v_i_279_);
v_res_283_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__0(v_sz_boxed_281_, v_i_boxed_282_, v_bs_280_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode(lean_object* v_type_284_, lean_object* v_immediate_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_){
_start:
{
size_t v_sz_291_; size_t v___x_292_; lean_object* v_immediate_293_; lean_object* v___x_294_; 
v_sz_291_ = lean_array_size(v_immediate_285_);
v___x_292_ = ((size_t)0ULL);
v_immediate_293_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__0(v_sz_291_, v___x_292_, v_immediate_285_);
v___x_294_ = lp_aesop_Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1(v_immediate_293_);
lean_dec_ref(v_immediate_293_);
if (lean_obj_tag(v___x_294_) == 0)
{
lean_object* v___x_295_; lean_object* v___x_296_; 
lean_dec_ref(v_type_284_);
v___x_295_ = lean_box(0);
v___x_296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_296_, 0, v___x_295_);
return v___x_296_;
}
else
{
lean_object* v_val_297_; lean_object* v___x_298_; uint8_t v___x_299_; uint8_t v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___f_303_; lean_object* v___x_304_; 
v_val_297_ = lean_ctor_get(v___x_294_, 0);
lean_inc(v_val_297_);
lean_dec_ref_known(v___x_294_, 1);
v___x_298_ = lean_box(0);
v___x_299_ = 0;
v___x_300_ = 2;
v___x_301_ = lean_box(v___x_300_);
v___x_302_ = lean_box(v___x_299_);
v___f_303_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___lam__0___boxed), 10, 5);
lean_closure_set(v___f_303_, 0, v___x_301_);
lean_closure_set(v___f_303_, 1, v_type_284_);
lean_closure_set(v___f_303_, 2, v___x_298_);
lean_closure_set(v___f_303_, 3, v___x_302_);
lean_closure_set(v___f_303_, 4, v_val_297_);
v___x_304_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__3___redArg(v___f_303_, v_a_286_, v_a_287_, v_a_288_, v_a_289_);
return v___x_304_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode___boxed(lean_object* v_type_305_, lean_object* v_immediate_306_, lean_object* v_a_307_, lean_object* v_a_308_, lean_object* v_a_309_, lean_object* v_a_310_, lean_object* v_a_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode(v_type_305_, v_immediate_306_, v_a_307_, v_a_308_, v_a_309_, v_a_310_);
lean_dec(v_a_310_);
lean_dec_ref(v_a_309_);
lean_dec(v_a_308_);
lean_dec_ref(v_a_307_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2(lean_object* v_00_u03b1_313_, lean_object* v_msg_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(v_msg_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___boxed(lean_object* v_00_u03b1_321_, lean_object* v_msg_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2(v_00_u03b1_321_, v_msg_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1(lean_object* v_arr_329_, lean_object* v_h_330_){
_start:
{
lean_object* v___x_331_; 
v___x_331_ = lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___redArg(v_arr_329_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1___boxed(lean_object* v_arr_332_, lean_object* v_h_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_aesop_Array_max___at___00Array_max_x3f___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__1_spec__1(v_arr_332_, v_h_333_);
lean_dec_ref(v_arr_332_);
return v_res_334_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated(lean_object* v_pat_x3f_335_, lean_object* v_i_336_){
_start:
{
if (lean_obj_tag(v_pat_x3f_335_) == 0)
{
uint8_t v___x_337_; 
v___x_337_ = 0;
return v___x_337_;
}
else
{
lean_object* v_val_338_; lean_object* v_argMap_339_; lean_object* v___x_340_; uint8_t v___x_341_; 
v_val_338_ = lean_ctor_get(v_pat_x3f_335_, 0);
v_argMap_339_ = lean_ctor_get(v_val_338_, 1);
v___x_340_ = lean_array_get_size(v_argMap_339_);
v___x_341_ = lean_nat_dec_lt(v_i_336_, v___x_340_);
if (v___x_341_ == 0)
{
return v___x_341_;
}
else
{
lean_object* v___x_342_; 
v___x_342_ = lean_array_fget_borrowed(v_argMap_339_, v_i_336_);
if (lean_obj_tag(v___x_342_) == 0)
{
uint8_t v___x_343_; 
v___x_343_ = 0;
return v___x_343_;
}
else
{
return v___x_341_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated___boxed(lean_object* v_pat_x3f_344_, lean_object* v_i_345_){
_start:
{
uint8_t v_res_346_; lean_object* v_r_347_; 
v_res_346_ = lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated(v_pat_x3f_344_, v_i_345_);
lean_dec(v_i_345_);
lean_dec(v_pat_x3f_344_);
v_r_347_ = lean_box(v_res_346_);
return v_r_347_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__1(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_349_ = ((lean_object*)(lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__0));
v___x_350_ = l_Lean_stringToMessageData(v___x_349_);
return v___x_350_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix(void){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lean_obj_once(&lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__1, &lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__1_once, _init_lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix___closed__1);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg(lean_object* v_e_352_, lean_object* v___y_353_){
_start:
{
uint8_t v___x_355_; 
v___x_355_ = l_Lean_Expr_hasMVar(v_e_352_);
if (v___x_355_ == 0)
{
lean_object* v___x_356_; 
v___x_356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_356_, 0, v_e_352_);
return v___x_356_;
}
else
{
lean_object* v___x_357_; lean_object* v_mctx_358_; lean_object* v___x_359_; lean_object* v_fst_360_; lean_object* v_snd_361_; lean_object* v___x_362_; lean_object* v_cache_363_; lean_object* v_zetaDeltaFVarIds_364_; lean_object* v_postponed_365_; lean_object* v_diag_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_375_; 
v___x_357_ = lean_st_ref_get(v___y_353_);
v_mctx_358_ = lean_ctor_get(v___x_357_, 0);
lean_inc_ref(v_mctx_358_);
lean_dec(v___x_357_);
v___x_359_ = l_Lean_instantiateMVarsCore(v_mctx_358_, v_e_352_);
v_fst_360_ = lean_ctor_get(v___x_359_, 0);
lean_inc(v_fst_360_);
v_snd_361_ = lean_ctor_get(v___x_359_, 1);
lean_inc(v_snd_361_);
lean_dec_ref(v___x_359_);
v___x_362_ = lean_st_ref_take(v___y_353_);
v_cache_363_ = lean_ctor_get(v___x_362_, 1);
v_zetaDeltaFVarIds_364_ = lean_ctor_get(v___x_362_, 2);
v_postponed_365_ = lean_ctor_get(v___x_362_, 3);
v_diag_366_ = lean_ctor_get(v___x_362_, 4);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_375_ == 0)
{
lean_object* v_unused_376_; 
v_unused_376_ = lean_ctor_get(v___x_362_, 0);
lean_dec(v_unused_376_);
v___x_368_ = v___x_362_;
v_isShared_369_ = v_isSharedCheck_375_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_diag_366_);
lean_inc(v_postponed_365_);
lean_inc(v_zetaDeltaFVarIds_364_);
lean_inc(v_cache_363_);
lean_dec(v___x_362_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_375_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_371_; 
if (v_isShared_369_ == 0)
{
lean_ctor_set(v___x_368_, 0, v_snd_361_);
v___x_371_ = v___x_368_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_snd_361_);
lean_ctor_set(v_reuseFailAlloc_374_, 1, v_cache_363_);
lean_ctor_set(v_reuseFailAlloc_374_, 2, v_zetaDeltaFVarIds_364_);
lean_ctor_set(v_reuseFailAlloc_374_, 3, v_postponed_365_);
lean_ctor_set(v_reuseFailAlloc_374_, 4, v_diag_366_);
v___x_371_ = v_reuseFailAlloc_374_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_372_ = lean_st_ref_set(v___y_353_, v___x_371_);
v___x_373_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_373_, 0, v_fst_360_);
return v___x_373_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg___boxed(lean_object* v_e_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg(v_e_377_, v___y_378_);
lean_dec(v___y_378_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0(lean_object* v_e_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg(v_e_381_, v___y_383_);
return v___x_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___boxed(lean_object* v_e_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0(v_e_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___lam__0(lean_object* v_k_395_, lean_object* v_b_396_, lean_object* v_c_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_){
_start:
{
lean_object* v___x_403_; 
lean_inc(v___y_401_);
lean_inc_ref(v___y_400_);
lean_inc(v___y_399_);
lean_inc_ref(v___y_398_);
v___x_403_ = lean_apply_7(v_k_395_, v_b_396_, v_c_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_, lean_box(0));
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___lam__0___boxed(lean_object* v_k_404_, lean_object* v_b_405_, lean_object* v_c_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_){
_start:
{
lean_object* v_res_412_; 
v_res_412_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___lam__0(v_k_404_, v_b_405_, v_c_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
lean_dec(v___y_410_);
lean_dec_ref(v___y_409_);
lean_dec(v___y_408_);
lean_dec_ref(v___y_407_);
return v_res_412_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg(lean_object* v_type_413_, lean_object* v_k_414_, uint8_t v_cleanupAnnotations_415_, uint8_t v_whnfType_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_){
_start:
{
lean_object* v___f_422_; lean_object* v___x_423_; 
v___f_422_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_422_, 0, v_k_414_);
v___x_423_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_413_, v___f_422_, v_cleanupAnnotations_415_, v_whnfType_416_, v___y_417_, v___y_418_, v___y_419_, v___y_420_);
if (lean_obj_tag(v___x_423_) == 0)
{
lean_object* v_a_424_; lean_object* v___x_426_; uint8_t v_isShared_427_; uint8_t v_isSharedCheck_431_; 
v_a_424_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_431_ == 0)
{
v___x_426_ = v___x_423_;
v_isShared_427_ = v_isSharedCheck_431_;
goto v_resetjp_425_;
}
else
{
lean_inc(v_a_424_);
lean_dec(v___x_423_);
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
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
v_a_432_ = lean_ctor_get(v___x_423_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_423_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v___x_423_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_423_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_a_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg___boxed(lean_object* v_type_440_, lean_object* v_k_441_, lean_object* v_cleanupAnnotations_442_, lean_object* v_whnfType_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_449_; uint8_t v_whnfType_boxed_450_; lean_object* v_res_451_; 
v_cleanupAnnotations_boxed_449_ = lean_unbox(v_cleanupAnnotations_442_);
v_whnfType_boxed_450_ = lean_unbox(v_whnfType_443_);
v_res_451_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg(v_type_440_, v_k_441_, v_cleanupAnnotations_boxed_449_, v_whnfType_boxed_450_, v___y_444_, v___y_445_, v___y_446_, v___y_447_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
lean_dec(v___y_445_);
lean_dec_ref(v___y_444_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3(lean_object* v_00_u03b1_452_, lean_object* v_type_453_, lean_object* v_k_454_, uint8_t v_cleanupAnnotations_455_, uint8_t v_whnfType_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg(v_type_453_, v_k_454_, v_cleanupAnnotations_455_, v_whnfType_456_, v___y_457_, v___y_458_, v___y_459_, v___y_460_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___boxed(lean_object* v_00_u03b1_463_, lean_object* v_type_464_, lean_object* v_k_465_, lean_object* v_cleanupAnnotations_466_, lean_object* v_whnfType_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_473_; uint8_t v_whnfType_boxed_474_; lean_object* v_res_475_; 
v_cleanupAnnotations_boxed_473_ = lean_unbox(v_cleanupAnnotations_466_);
v_whnfType_boxed_474_ = lean_unbox(v_whnfType_467_);
v_res_475_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3(v_00_u03b1_463_, v_type_464_, v_k_465_, v_cleanupAnnotations_boxed_473_, v_whnfType_boxed_474_, v___y_468_, v___y_469_, v___y_470_, v___y_471_);
lean_dec(v___y_471_);
lean_dec_ref(v___y_470_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_RuleBuilder_getImmediatePremises_spec__1(uint8_t v___x_476_, lean_object* v_fvarId_477_, lean_object* v_as_478_, size_t v_i_479_, size_t v_stop_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
uint8_t v___x_486_; 
v___x_486_ = lean_usize_dec_eq(v_i_479_, v_stop_480_);
if (v___x_486_ == 0)
{
lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_487_ = lean_array_uget_borrowed(v_as_478_, v_i_479_);
lean_inc(v___y_484_);
lean_inc_ref(v___y_483_);
lean_inc(v___y_482_);
lean_inc_ref(v___y_481_);
lean_inc(v___x_487_);
v___x_488_ = lean_infer_type(v___x_487_, v___y_481_, v___y_482_, v___y_483_, v___y_484_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; lean_object* v___x_490_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
lean_inc(v_a_489_);
lean_dec_ref_known(v___x_488_, 1);
v___x_490_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_RuleBuilder_getImmediatePremises_spec__0___redArg(v_a_489_, v___y_482_);
if (lean_obj_tag(v___x_490_) == 0)
{
lean_object* v_a_491_; lean_object* v___x_493_; uint8_t v_isShared_494_; uint8_t v_isSharedCheck_526_; 
v_a_491_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_526_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_526_ == 0)
{
v___x_493_ = v___x_490_;
v_isShared_494_ = v_isSharedCheck_526_;
goto v_resetjp_492_;
}
else
{
lean_inc(v_a_491_);
lean_dec(v___x_490_);
v___x_493_ = lean_box(0);
v_isShared_494_ = v_isSharedCheck_526_;
goto v_resetjp_492_;
}
v_resetjp_492_:
{
lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_495_ = l_Lean_Expr_fvarId_x21(v___x_487_);
v___x_496_ = l_Lean_FVarId_getBinderInfo___redArg(v___x_495_, v___y_481_, v___y_483_, v___y_484_);
if (lean_obj_tag(v___x_496_) == 0)
{
lean_object* v_a_497_; lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_517_; 
v_a_497_ = lean_ctor_get(v___x_496_, 0);
v_isSharedCheck_517_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_517_ == 0)
{
v___x_499_ = v___x_496_;
v_isShared_500_ = v_isSharedCheck_517_;
goto v_resetjp_498_;
}
else
{
lean_inc(v_a_497_);
lean_dec(v___x_496_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_517_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
uint8_t v___x_501_; uint8_t v_a_503_; uint8_t v___x_511_; 
v___x_501_ = 1;
v___x_511_ = lean_unbox(v_a_497_);
lean_dec(v_a_497_);
if (v___x_511_ == 3)
{
lean_del_object(v___x_493_);
lean_dec(v_a_491_);
v_a_503_ = v___x_476_;
goto v___jp_502_;
}
else
{
if (v___x_476_ == 0)
{
uint8_t v___x_512_; 
lean_del_object(v___x_493_);
v___x_512_ = l_Lean_Expr_containsFVar(v_a_491_, v_fvarId_477_);
lean_dec(v_a_491_);
v_a_503_ = v___x_512_;
goto v___jp_502_;
}
else
{
lean_object* v___x_513_; lean_object* v___x_515_; 
lean_del_object(v___x_499_);
lean_dec(v_a_491_);
v___x_513_ = lean_box(v___x_501_);
if (v_isShared_494_ == 0)
{
lean_ctor_set(v___x_493_, 0, v___x_513_);
v___x_515_ = v___x_493_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v___x_513_);
v___x_515_ = v_reuseFailAlloc_516_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
return v___x_515_;
}
}
}
v___jp_502_:
{
if (v_a_503_ == 0)
{
size_t v___x_504_; size_t v___x_505_; 
lean_del_object(v___x_499_);
v___x_504_ = ((size_t)1ULL);
v___x_505_ = lean_usize_add(v_i_479_, v___x_504_);
v_i_479_ = v___x_505_;
goto _start;
}
else
{
lean_object* v___x_507_; lean_object* v___x_509_; 
v___x_507_ = lean_box(v___x_501_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 0, v___x_507_);
v___x_509_ = v___x_499_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v___x_507_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
}
}
}
}
}
else
{
lean_object* v_a_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_525_; 
lean_del_object(v___x_493_);
lean_dec(v_a_491_);
v_a_518_ = lean_ctor_get(v___x_496_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_525_ == 0)
{
v___x_520_ = v___x_496_;
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_a_518_);
lean_dec(v___x_496_);
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
lean_object* v_a_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_534_; 
v_a_527_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_534_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_534_ == 0)
{
v___x_529_ = v___x_490_;
v_isShared_530_ = v_isSharedCheck_534_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_a_527_);
lean_dec(v___x_490_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_534_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
lean_object* v___x_532_; 
if (v_isShared_530_ == 0)
{
v___x_532_ = v___x_529_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v_a_527_);
v___x_532_ = v_reuseFailAlloc_533_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
return v___x_532_;
}
}
}
}
else
{
lean_object* v_a_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_542_; 
v_a_535_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_542_ == 0)
{
v___x_537_ = v___x_488_;
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_a_535_);
lean_dec(v___x_488_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_542_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_540_; 
if (v_isShared_538_ == 0)
{
v___x_540_ = v___x_537_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v_a_535_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
}
}
else
{
uint8_t v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_543_ = 0;
v___x_544_ = lean_box(v___x_543_);
v___x_545_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_545_, 0, v___x_544_);
return v___x_545_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_RuleBuilder_getImmediatePremises_spec__1___boxed(lean_object* v___x_546_, lean_object* v_fvarId_547_, lean_object* v_as_548_, lean_object* v_i_549_, lean_object* v_stop_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
uint8_t v___x_7308__boxed_556_; size_t v_i_boxed_557_; size_t v_stop_boxed_558_; lean_object* v_res_559_; 
v___x_7308__boxed_556_ = lean_unbox(v___x_546_);
v_i_boxed_557_ = lean_unbox_usize(v_i_549_);
lean_dec(v_i_549_);
v_stop_boxed_558_ = lean_unbox_usize(v_stop_550_);
lean_dec(v_stop_550_);
v_res_559_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_RuleBuilder_getImmediatePremises_spec__1(v___x_7308__boxed_556_, v_fvarId_547_, v_as_548_, v_i_boxed_557_, v_stop_boxed_558_, v___y_551_, v___y_552_, v___y_553_, v___y_554_);
lean_dec(v___y_554_);
lean_dec_ref(v___y_553_);
lean_dec(v___y_552_);
lean_dec_ref(v___y_551_);
lean_dec_ref(v_as_548_);
lean_dec(v_fvarId_547_);
return v_res_559_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg(lean_object* v_pat_x3f_560_, lean_object* v_args_561_, lean_object* v___x_562_, lean_object* v_range_563_, lean_object* v_b_564_, lean_object* v_i_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_stop_571_; lean_object* v_step_572_; lean_object* v_a_574_; uint8_t v___x_579_; 
v_stop_571_ = lean_ctor_get(v_range_563_, 1);
v_step_572_ = lean_ctor_get(v_range_563_, 2);
v___x_579_ = lean_nat_dec_lt(v_i_565_, v_stop_571_);
if (v___x_579_ == 0)
{
lean_object* v___x_580_; 
lean_dec(v_i_565_);
lean_dec(v___x_562_);
v___x_580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_580_, 0, v_b_564_);
return v___x_580_;
}
else
{
uint8_t v___x_581_; uint8_t v_a_583_; 
v___x_581_ = lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated(v_pat_x3f_560_, v_i_565_);
if (v___x_581_ == 0)
{
lean_object* v___x_584_; lean_object* v_fvarId_585_; lean_object* v___x_586_; 
v___x_584_ = lean_array_fget_borrowed(v_args_561_, v_i_565_);
v_fvarId_585_ = l_Lean_Expr_fvarId_x21(v___x_584_);
lean_inc(v_fvarId_585_);
v___x_586_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_585_, v___y_566_, v___y_568_, v___y_569_);
if (lean_obj_tag(v___x_586_) == 0)
{
lean_object* v_a_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___y_591_; uint8_t v___x_610_; uint8_t v___x_611_; 
v_a_587_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_a_587_);
lean_dec_ref_known(v___x_586_, 1);
v___x_588_ = lean_unsigned_to_nat(1u);
v___x_589_ = lean_nat_add(v_i_565_, v___x_588_);
v___x_610_ = l_Lean_LocalDecl_binderInfo(v_a_587_);
lean_dec(v_a_587_);
v___x_611_ = l_Lean_BinderInfo_isInstImplicit(v___x_610_);
if (v___x_611_ == 0)
{
goto v___jp_606_;
}
else
{
if (v___x_581_ == 0)
{
lean_dec(v___x_589_);
lean_dec(v_fvarId_585_);
v_a_574_ = v_b_564_;
goto v___jp_573_;
}
else
{
goto v___jp_606_;
}
}
v___jp_590_:
{
uint8_t v___x_592_; 
v___x_592_ = lean_nat_dec_lt(v___x_589_, v___y_591_);
if (v___x_592_ == 0)
{
lean_dec(v___y_591_);
lean_dec(v___x_589_);
lean_dec(v_fvarId_585_);
v_a_583_ = v___x_581_;
goto v___jp_582_;
}
else
{
size_t v___x_593_; size_t v___x_594_; lean_object* v___x_595_; 
v___x_593_ = lean_usize_of_nat(v___x_589_);
lean_dec(v___x_589_);
v___x_594_ = lean_usize_of_nat(v___y_591_);
lean_dec(v___y_591_);
v___x_595_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_RuleBuilder_getImmediatePremises_spec__1(v___x_581_, v_fvarId_585_, v_args_561_, v___x_593_, v___x_594_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
lean_dec(v_fvarId_585_);
if (lean_obj_tag(v___x_595_) == 0)
{
lean_object* v_a_596_; uint8_t v___x_597_; 
v_a_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc(v_a_596_);
lean_dec_ref_known(v___x_595_, 1);
v___x_597_ = lean_unbox(v_a_596_);
lean_dec(v_a_596_);
v_a_583_ = v___x_597_;
goto v___jp_582_;
}
else
{
lean_object* v_a_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_605_; 
lean_dec(v_i_565_);
lean_dec_ref(v_b_564_);
lean_dec(v___x_562_);
v_a_598_ = lean_ctor_get(v___x_595_, 0);
v_isSharedCheck_605_ = !lean_is_exclusive(v___x_595_);
if (v_isSharedCheck_605_ == 0)
{
v___x_600_ = v___x_595_;
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_a_598_);
lean_dec(v___x_595_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_605_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___x_603_; 
if (v_isShared_601_ == 0)
{
v___x_603_ = v___x_600_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v_a_598_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
}
}
v___jp_606_:
{
uint8_t v___x_607_; 
v___x_607_ = lean_nat_dec_lt(v___x_589_, v___x_562_);
if (v___x_607_ == 0)
{
lean_dec(v___x_589_);
lean_dec(v_fvarId_585_);
v_a_583_ = v___x_581_;
goto v___jp_582_;
}
else
{
lean_object* v___x_608_; uint8_t v___x_609_; 
v___x_608_ = lean_array_get_size(v_args_561_);
v___x_609_ = lean_nat_dec_le(v___x_562_, v___x_608_);
if (v___x_609_ == 0)
{
v___y_591_ = v___x_608_;
goto v___jp_590_;
}
else
{
lean_inc(v___x_562_);
v___y_591_ = v___x_562_;
goto v___jp_590_;
}
}
}
}
else
{
lean_object* v_a_612_; lean_object* v___x_614_; uint8_t v_isShared_615_; uint8_t v_isSharedCheck_619_; 
lean_dec(v_fvarId_585_);
lean_dec(v_i_565_);
lean_dec_ref(v_b_564_);
lean_dec(v___x_562_);
v_a_612_ = lean_ctor_get(v___x_586_, 0);
v_isSharedCheck_619_ = !lean_is_exclusive(v___x_586_);
if (v_isSharedCheck_619_ == 0)
{
v___x_614_ = v___x_586_;
v_isShared_615_ = v_isSharedCheck_619_;
goto v_resetjp_613_;
}
else
{
lean_inc(v_a_612_);
lean_dec(v___x_586_);
v___x_614_ = lean_box(0);
v_isShared_615_ = v_isSharedCheck_619_;
goto v_resetjp_613_;
}
v_resetjp_613_:
{
lean_object* v___x_617_; 
if (v_isShared_615_ == 0)
{
v___x_617_ = v___x_614_;
goto v_reusejp_616_;
}
else
{
lean_object* v_reuseFailAlloc_618_; 
v_reuseFailAlloc_618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_618_, 0, v_a_612_);
v___x_617_ = v_reuseFailAlloc_618_;
goto v_reusejp_616_;
}
v_reusejp_616_:
{
return v___x_617_;
}
}
}
}
else
{
v_a_574_ = v_b_564_;
goto v___jp_573_;
}
v___jp_582_:
{
if (v_a_583_ == 0)
{
goto v___jp_577_;
}
else
{
if (v___x_581_ == 0)
{
v_a_574_ = v_b_564_;
goto v___jp_573_;
}
else
{
goto v___jp_577_;
}
}
}
}
v___jp_573_:
{
lean_object* v___x_575_; 
v___x_575_ = lean_nat_add(v_i_565_, v_step_572_);
lean_dec(v_i_565_);
v_b_564_ = v_a_574_;
v_i_565_ = v___x_575_;
goto _start;
}
v___jp_577_:
{
lean_object* v___x_578_; 
lean_inc(v_i_565_);
v___x_578_ = lean_array_push(v_b_564_, v_i_565_);
v_a_574_ = v___x_578_;
goto v___jp_573_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg___boxed(lean_object* v_pat_x3f_620_, lean_object* v_args_621_, lean_object* v___x_622_, lean_object* v_range_623_, lean_object* v_b_624_, lean_object* v_i_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg(v_pat_x3f_620_, v_args_621_, v___x_622_, v_range_623_, v_b_624_, v_i_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_);
lean_dec(v___y_629_);
lean_dec_ref(v___y_628_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
lean_dec_ref(v_range_623_);
lean_dec_ref(v_args_621_);
lean_dec(v_pat_x3f_620_);
return v_res_631_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0(lean_object* v_pat_x3f_634_, lean_object* v_args_635_, lean_object* v_x_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
lean_object* v___x_642_; lean_object* v_result_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_642_ = lean_unsigned_to_nat(0u);
v_result_643_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___closed__0));
v___x_644_ = lean_array_get_size(v_args_635_);
v___x_645_ = lean_unsigned_to_nat(1u);
v___x_646_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_646_, 0, v___x_642_);
lean_ctor_set(v___x_646_, 1, v___x_644_);
lean_ctor_set(v___x_646_, 2, v___x_645_);
v___x_647_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg(v_pat_x3f_634_, v_args_635_, v___x_644_, v___x_646_, v_result_643_, v___x_642_, v___y_637_, v___y_638_, v___y_639_, v___y_640_);
lean_dec_ref_known(v___x_646_, 3);
if (lean_obj_tag(v___x_647_) == 0)
{
lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
v_a_648_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_647_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_647_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_a_648_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
else
{
lean_object* v_a_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_663_; 
v_a_656_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_663_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_663_ == 0)
{
v___x_658_ = v___x_647_;
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_a_656_);
lean_dec(v___x_647_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_661_; 
if (v_isShared_659_ == 0)
{
v___x_661_ = v___x_658_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v_a_656_);
v___x_661_ = v_reuseFailAlloc_662_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
return v___x_661_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___boxed(lean_object* v_pat_x3f_664_, lean_object* v_args_665_, lean_object* v_x_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0(v_pat_x3f_664_, v_args_665_, v_x_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec_ref(v_x_666_);
lean_dec_ref(v_args_665_);
lean_dec(v_pat_x3f_664_);
return v_res_672_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0(uint8_t v___x_673_, lean_object* v_x1_674_, lean_object* v_x2_675_){
_start:
{
uint8_t v___x_676_; 
v___x_676_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_x1_674_, v_x2_675_);
if (v___x_676_ == 0)
{
return v___x_673_;
}
else
{
uint8_t v___x_677_; 
v___x_677_ = 0;
return v___x_677_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0___boxed(lean_object* v___x_678_, lean_object* v_x1_679_, lean_object* v_x2_680_){
_start:
{
uint8_t v___x_7619__boxed_681_; uint8_t v_res_682_; lean_object* v_r_683_; 
v___x_7619__boxed_681_ = lean_unbox(v___x_678_);
v_res_682_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0(v___x_7619__boxed_681_, v_x1_679_, v_x2_680_);
lean_dec(v_x2_680_);
lean_dec(v_x1_679_);
v_r_683_ = lean_box(v_res_682_);
return v_r_683_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg(lean_object* v_hi_684_, lean_object* v_pivot_685_, lean_object* v_as_686_, lean_object* v_i_687_, lean_object* v_k_688_){
_start:
{
uint8_t v___x_689_; 
v___x_689_ = lean_nat_dec_lt(v_k_688_, v_hi_684_);
if (v___x_689_ == 0)
{
lean_object* v___x_690_; lean_object* v___x_691_; 
lean_dec(v_k_688_);
v___x_690_ = lean_array_fswap(v_as_686_, v_i_687_, v_hi_684_);
v___x_691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_691_, 0, v_i_687_);
lean_ctor_set(v___x_691_, 1, v___x_690_);
return v___x_691_;
}
else
{
lean_object* v___x_692_; uint8_t v___x_693_; 
v___x_692_ = lean_array_fget_borrowed(v_as_686_, v_k_688_);
v___x_693_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v___x_692_, v_pivot_685_);
if (v___x_693_ == 0)
{
lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
v___x_694_ = lean_array_fswap(v_as_686_, v_i_687_, v_k_688_);
v___x_695_ = lean_unsigned_to_nat(1u);
v___x_696_ = lean_nat_add(v_i_687_, v___x_695_);
lean_dec(v_i_687_);
v___x_697_ = lean_nat_add(v_k_688_, v___x_695_);
lean_dec(v_k_688_);
v_as_686_ = v___x_694_;
v_i_687_ = v___x_696_;
v_k_688_ = v___x_697_;
goto _start;
}
else
{
lean_object* v___x_699_; lean_object* v___x_700_; 
v___x_699_ = lean_unsigned_to_nat(1u);
v___x_700_ = lean_nat_add(v_k_688_, v___x_699_);
lean_dec(v_k_688_);
v_k_688_ = v___x_700_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg___boxed(lean_object* v_hi_702_, lean_object* v_pivot_703_, lean_object* v_as_704_, lean_object* v_i_705_, lean_object* v_k_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg(v_hi_702_, v_pivot_703_, v_as_704_, v_i_705_, v_k_706_);
lean_dec(v_pivot_703_);
lean_dec(v_hi_702_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg(lean_object* v_n_708_, lean_object* v_as_709_, lean_object* v_lo_710_, lean_object* v_hi_711_){
_start:
{
lean_object* v___y_713_; uint8_t v___x_723_; 
v___x_723_ = lean_nat_dec_lt(v_lo_710_, v_hi_711_);
if (v___x_723_ == 0)
{
lean_dec(v_lo_710_);
return v_as_709_;
}
else
{
lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v_mid_726_; lean_object* v___y_728_; lean_object* v___y_734_; lean_object* v___x_739_; lean_object* v___x_740_; uint8_t v___x_741_; 
v___x_724_ = lean_nat_add(v_lo_710_, v_hi_711_);
v___x_725_ = lean_unsigned_to_nat(1u);
v_mid_726_ = lean_nat_shiftr(v___x_724_, v___x_725_);
lean_dec(v___x_724_);
v___x_739_ = lean_array_fget_borrowed(v_as_709_, v_mid_726_);
v___x_740_ = lean_array_fget_borrowed(v_as_709_, v_lo_710_);
v___x_741_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0(v___x_723_, v___x_739_, v___x_740_);
if (v___x_741_ == 0)
{
v___y_734_ = v_as_709_;
goto v___jp_733_;
}
else
{
lean_object* v___x_742_; 
v___x_742_ = lean_array_fswap(v_as_709_, v_lo_710_, v_mid_726_);
v___y_734_ = v___x_742_;
goto v___jp_733_;
}
v___jp_727_:
{
lean_object* v___x_729_; lean_object* v___x_730_; uint8_t v___x_731_; 
v___x_729_ = lean_array_fget_borrowed(v___y_728_, v_mid_726_);
v___x_730_ = lean_array_fget_borrowed(v___y_728_, v_hi_711_);
v___x_731_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0(v___x_723_, v___x_729_, v___x_730_);
if (v___x_731_ == 0)
{
lean_dec(v_mid_726_);
v___y_713_ = v___y_728_;
goto v___jp_712_;
}
else
{
lean_object* v___x_732_; 
v___x_732_ = lean_array_fswap(v___y_728_, v_mid_726_, v_hi_711_);
lean_dec(v_mid_726_);
v___y_713_ = v___x_732_;
goto v___jp_712_;
}
}
v___jp_733_:
{
lean_object* v___x_735_; lean_object* v___x_736_; uint8_t v___x_737_; 
v___x_735_ = lean_array_fget_borrowed(v___y_734_, v_hi_711_);
v___x_736_ = lean_array_fget_borrowed(v___y_734_, v_lo_710_);
v___x_737_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___lam__0(v___x_723_, v___x_735_, v___x_736_);
if (v___x_737_ == 0)
{
v___y_728_ = v___y_734_;
goto v___jp_727_;
}
else
{
lean_object* v___x_738_; 
v___x_738_ = lean_array_fswap(v___y_734_, v_lo_710_, v_hi_711_);
v___y_728_ = v___x_738_;
goto v___jp_727_;
}
}
}
v___jp_712_:
{
lean_object* v_pivot_714_; lean_object* v___x_715_; lean_object* v_fst_716_; lean_object* v_snd_717_; uint8_t v___x_718_; 
v_pivot_714_ = lean_array_fget(v___y_713_, v_hi_711_);
lean_inc_n(v_lo_710_, 2);
v___x_715_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg(v_hi_711_, v_pivot_714_, v___y_713_, v_lo_710_, v_lo_710_);
lean_dec(v_pivot_714_);
v_fst_716_ = lean_ctor_get(v___x_715_, 0);
lean_inc(v_fst_716_);
v_snd_717_ = lean_ctor_get(v___x_715_, 1);
lean_inc(v_snd_717_);
lean_dec_ref(v___x_715_);
v___x_718_ = lean_nat_dec_le(v_hi_711_, v_fst_716_);
if (v___x_718_ == 0)
{
lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; 
v___x_719_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg(v_n_708_, v_snd_717_, v_lo_710_, v_fst_716_);
v___x_720_ = lean_unsigned_to_nat(1u);
v___x_721_ = lean_nat_add(v_fst_716_, v___x_720_);
lean_dec(v_fst_716_);
v_as_709_ = v___x_719_;
v_lo_710_ = v___x_721_;
goto _start;
}
else
{
lean_dec(v_fst_716_);
lean_dec(v_lo_710_);
return v_snd_717_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg___boxed(lean_object* v_n_743_, lean_object* v_as_744_, lean_object* v_lo_745_, lean_object* v_hi_746_){
_start:
{
lean_object* v_res_747_; 
v_res_747_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg(v_n_743_, v_as_744_, v_lo_745_, v_hi_746_);
lean_dec(v_hi_746_);
lean_dec(v_n_743_);
return v_res_747_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12_spec__15(lean_object* v_f_748_, lean_object* v_xs_749_, lean_object* v_acc_750_, lean_object* v_i_751_, lean_object* v_hd_752_){
_start:
{
lean_object* v___x_753_; uint8_t v___x_754_; 
v___x_753_ = lean_array_get_size(v_xs_749_);
v___x_754_ = lean_nat_dec_lt(v_i_751_, v___x_753_);
if (v___x_754_ == 0)
{
lean_object* v___x_755_; 
lean_dec(v_i_751_);
lean_dec_ref(v_f_748_);
v___x_755_ = lean_array_push(v_acc_750_, v_hd_752_);
return v___x_755_;
}
else
{
lean_object* v_x_756_; uint8_t v___x_757_; 
v_x_756_ = lean_array_fget_borrowed(v_xs_749_, v_i_751_);
v___x_757_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_x_756_, v_hd_752_);
if (v___x_757_ == 1)
{
lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v___x_758_ = lean_unsigned_to_nat(1u);
v___x_759_ = lean_nat_add(v_i_751_, v___x_758_);
lean_dec(v_i_751_);
lean_inc_ref(v_f_748_);
lean_inc(v_x_756_);
v___x_760_ = lean_apply_2(v_f_748_, v_hd_752_, v_x_756_);
v_i_751_ = v___x_759_;
v_hd_752_ = v___x_760_;
goto _start;
}
else
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_762_ = lean_array_push(v_acc_750_, v_hd_752_);
v___x_763_ = lean_unsigned_to_nat(1u);
v___x_764_ = lean_nat_add(v_i_751_, v___x_763_);
lean_dec(v_i_751_);
lean_inc(v_x_756_);
v_acc_750_ = v___x_762_;
v_i_751_ = v___x_764_;
v_hd_752_ = v_x_756_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12_spec__15___boxed(lean_object* v_f_766_, lean_object* v_xs_767_, lean_object* v_acc_768_, lean_object* v_i_769_, lean_object* v_hd_770_){
_start:
{
lean_object* v_res_771_; 
v_res_771_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12_spec__15(v_f_766_, v_xs_767_, v_acc_768_, v_i_769_, v_hd_770_);
lean_dec_ref(v_xs_767_);
return v_res_771_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12(lean_object* v_f_772_, lean_object* v_xs_773_){
_start:
{
lean_object* v___x_774_; lean_object* v___x_775_; uint8_t v___x_776_; 
v___x_774_ = lean_unsigned_to_nat(0u);
v___x_775_ = lean_array_get_size(v_xs_773_);
v___x_776_ = lean_nat_dec_lt(v___x_774_, v___x_775_);
if (v___x_776_ == 0)
{
lean_dec_ref(v_f_772_);
lean_inc_ref(v_xs_773_);
return v_xs_773_;
}
else
{
lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; 
v___x_777_ = lean_mk_empty_array_with_capacity(v___x_775_);
v___x_778_ = lean_unsigned_to_nat(1u);
v___x_779_ = lean_array_fget_borrowed(v_xs_773_, v___x_774_);
lean_inc(v___x_779_);
v___x_780_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12_spec__15(v_f_772_, v_xs_773_, v___x_777_, v___x_778_, v___x_779_);
return v___x_780_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12___boxed(lean_object* v_f_781_, lean_object* v_xs_782_){
_start:
{
lean_object* v_res_783_; 
v_res_783_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12(v_f_781_, v_xs_782_);
lean_dec_ref(v_xs_782_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___lam__0(lean_object* v_x_784_, lean_object* v_x_785_){
_start:
{
lean_inc(v_x_784_);
return v_x_784_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___lam__0___boxed(lean_object* v_x_786_, lean_object* v_x_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___lam__0(v_x_786_, v_x_787_);
lean_dec(v_x_787_);
lean_dec(v_x_786_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9(lean_object* v_xs_790_){
_start:
{
lean_object* v___f_791_; lean_object* v___x_792_; 
v___f_791_ = ((lean_object*)(lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___closed__0));
v___x_792_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9_spec__12(v___f_791_, v_xs_790_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9___boxed(lean_object* v_xs_793_){
_start:
{
lean_object* v_res_794_; 
v_res_794_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9(v_xs_793_);
lean_dec_ref(v_xs_793_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6(lean_object* v_xs_795_){
_start:
{
lean_object* v___x_796_; lean_object* v___y_798_; lean_object* v___y_799_; lean_object* v___x_802_; uint8_t v___x_803_; 
v___x_796_ = lean_array_get_size(v_xs_795_);
v___x_802_ = lean_unsigned_to_nat(0u);
v___x_803_ = lean_nat_dec_eq(v___x_796_, v___x_802_);
if (v___x_803_ == 0)
{
lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___y_807_; uint8_t v___x_809_; 
v___x_804_ = lean_unsigned_to_nat(1u);
v___x_805_ = lean_nat_sub(v___x_796_, v___x_804_);
v___x_809_ = lean_nat_dec_le(v___x_802_, v___x_805_);
if (v___x_809_ == 0)
{
lean_inc(v___x_805_);
v___y_807_ = v___x_805_;
goto v___jp_806_;
}
else
{
v___y_807_ = v___x_802_;
goto v___jp_806_;
}
v___jp_806_:
{
uint8_t v___x_808_; 
v___x_808_ = lean_nat_dec_le(v___y_807_, v___x_805_);
if (v___x_808_ == 0)
{
lean_dec(v___x_805_);
lean_inc(v___y_807_);
v___y_798_ = v___y_807_;
v___y_799_ = v___y_807_;
goto v___jp_797_;
}
else
{
v___y_798_ = v___y_807_;
v___y_799_ = v___x_805_;
goto v___jp_797_;
}
}
}
else
{
lean_object* v___x_810_; 
v___x_810_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9(v_xs_795_);
lean_dec_ref(v_xs_795_);
return v___x_810_;
}
v___jp_797_:
{
lean_object* v___x_800_; lean_object* v___x_801_; 
v___x_800_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg(v___x_796_, v_xs_795_, v___y_798_, v___y_799_);
lean_dec(v___y_799_);
v___x_801_ = lp_aesop_Array_dedupSorted___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__9(v___x_800_);
lean_dec_ref(v___x_800_);
return v___x_801_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__8(lean_object* v_a_811_, lean_object* v_a_812_){
_start:
{
if (lean_obj_tag(v_a_811_) == 0)
{
lean_object* v___x_813_; 
v___x_813_ = l_List_reverse___redArg(v_a_812_);
return v___x_813_;
}
else
{
lean_object* v_head_814_; lean_object* v_tail_815_; lean_object* v___x_817_; uint8_t v_isShared_818_; uint8_t v_isSharedCheck_824_; 
v_head_814_ = lean_ctor_get(v_a_811_, 0);
v_tail_815_ = lean_ctor_get(v_a_811_, 1);
v_isSharedCheck_824_ = !lean_is_exclusive(v_a_811_);
if (v_isSharedCheck_824_ == 0)
{
v___x_817_ = v_a_811_;
v_isShared_818_ = v_isSharedCheck_824_;
goto v_resetjp_816_;
}
else
{
lean_inc(v_tail_815_);
lean_inc(v_head_814_);
lean_dec(v_a_811_);
v___x_817_ = lean_box(0);
v_isShared_818_ = v_isSharedCheck_824_;
goto v_resetjp_816_;
}
v_resetjp_816_:
{
lean_object* v___x_819_; lean_object* v___x_821_; 
v___x_819_ = l_Lean_MessageData_ofName(v_head_814_);
if (v_isShared_818_ == 0)
{
lean_ctor_set(v___x_817_, 1, v_a_812_);
lean_ctor_set(v___x_817_, 0, v___x_819_);
v___x_821_ = v___x_817_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_819_);
lean_ctor_set(v_reuseFailAlloc_823_, 1, v_a_812_);
v___x_821_ = v_reuseFailAlloc_823_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
v_a_811_ = v_tail_815_;
v_a_812_ = v___x_821_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6_spec__7(lean_object* v_xs_825_, lean_object* v_v_826_, lean_object* v_i_827_){
_start:
{
lean_object* v___x_828_; uint8_t v___x_829_; 
v___x_828_ = lean_array_get_size(v_xs_825_);
v___x_829_ = lean_nat_dec_lt(v_i_827_, v___x_828_);
if (v___x_829_ == 0)
{
lean_object* v___x_830_; 
lean_dec(v_i_827_);
v___x_830_ = lean_box(0);
return v___x_830_;
}
else
{
lean_object* v___x_831_; uint8_t v___x_832_; 
v___x_831_ = lean_array_fget_borrowed(v_xs_825_, v_i_827_);
v___x_832_ = lean_name_eq(v___x_831_, v_v_826_);
if (v___x_832_ == 0)
{
lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_833_ = lean_unsigned_to_nat(1u);
v___x_834_ = lean_nat_add(v_i_827_, v___x_833_);
lean_dec(v_i_827_);
v_i_827_ = v___x_834_;
goto _start;
}
else
{
lean_object* v___x_836_; 
v___x_836_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_836_, 0, v_i_827_);
return v___x_836_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6_spec__7___boxed(lean_object* v_xs_837_, lean_object* v_v_838_, lean_object* v_i_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6_spec__7(v_xs_837_, v_v_838_, v_i_839_);
lean_dec(v_v_838_);
lean_dec_ref(v_xs_837_);
return v_res_840_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6(lean_object* v_xs_841_, lean_object* v_v_842_){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_843_ = lean_unsigned_to_nat(0u);
v___x_844_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6_spec__7(v_xs_841_, v_v_842_, v___x_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6___boxed(lean_object* v_xs_845_, lean_object* v_v_846_){
_start:
{
lean_object* v_res_847_; 
v_res_847_ = lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6(v_xs_845_, v_v_846_);
lean_dec(v_v_846_);
lean_dec_ref(v_xs_845_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5(lean_object* v_as_848_, lean_object* v_a_849_){
_start:
{
lean_object* v___x_850_; 
v___x_850_ = lp_aesop_Array_finIdxOf_x3f___at___00Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5_spec__6(v_as_848_, v_a_849_);
if (lean_obj_tag(v___x_850_) == 0)
{
return v_as_848_;
}
else
{
lean_object* v_val_851_; lean_object* v___x_852_; 
v_val_851_ = lean_ctor_get(v___x_850_, 0);
lean_inc(v_val_851_);
lean_dec_ref_known(v___x_850_, 1);
v___x_852_ = l_Array_eraseIdx___redArg(v_as_848_, v_val_851_);
return v___x_852_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5___boxed(lean_object* v_as_853_, lean_object* v_a_854_){
_start:
{
lean_object* v_res_855_; 
v_res_855_ = lp_aesop_Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5(v_as_853_, v_a_854_);
lean_dec(v_a_854_);
return v_res_855_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4_spec__4(lean_object* v_a_856_, lean_object* v_as_857_, size_t v_i_858_, size_t v_stop_859_){
_start:
{
uint8_t v___x_860_; 
v___x_860_ = lean_usize_dec_eq(v_i_858_, v_stop_859_);
if (v___x_860_ == 0)
{
lean_object* v___x_861_; uint8_t v___x_862_; 
v___x_861_ = lean_array_uget_borrowed(v_as_857_, v_i_858_);
v___x_862_ = lean_name_eq(v_a_856_, v___x_861_);
if (v___x_862_ == 0)
{
size_t v___x_863_; size_t v___x_864_; 
v___x_863_ = ((size_t)1ULL);
v___x_864_ = lean_usize_add(v_i_858_, v___x_863_);
v_i_858_ = v___x_864_;
goto _start;
}
else
{
return v___x_862_;
}
}
else
{
uint8_t v___x_866_; 
v___x_866_ = 0;
return v___x_866_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4_spec__4___boxed(lean_object* v_a_867_, lean_object* v_as_868_, lean_object* v_i_869_, lean_object* v_stop_870_){
_start:
{
size_t v_i_boxed_871_; size_t v_stop_boxed_872_; uint8_t v_res_873_; lean_object* v_r_874_; 
v_i_boxed_871_ = lean_unbox_usize(v_i_869_);
lean_dec(v_i_869_);
v_stop_boxed_872_ = lean_unbox_usize(v_stop_870_);
lean_dec(v_stop_870_);
v_res_873_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4_spec__4(v_a_867_, v_as_868_, v_i_boxed_871_, v_stop_boxed_872_);
lean_dec_ref(v_as_868_);
lean_dec(v_a_867_);
v_r_874_ = lean_box(v_res_873_);
return v_r_874_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4(lean_object* v_as_875_, lean_object* v_a_876_){
_start:
{
lean_object* v___x_877_; lean_object* v___x_878_; uint8_t v___x_879_; 
v___x_877_ = lean_unsigned_to_nat(0u);
v___x_878_ = lean_array_get_size(v_as_875_);
v___x_879_ = lean_nat_dec_lt(v___x_877_, v___x_878_);
if (v___x_879_ == 0)
{
return v___x_879_;
}
else
{
if (v___x_879_ == 0)
{
return v___x_879_;
}
else
{
size_t v___x_880_; size_t v___x_881_; uint8_t v___x_882_; 
v___x_880_ = ((size_t)0ULL);
v___x_881_ = lean_usize_of_nat(v___x_878_);
v___x_882_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4_spec__4(v_a_876_, v_as_875_, v___x_880_, v___x_881_);
return v___x_882_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4___boxed(lean_object* v_as_883_, lean_object* v_a_884_){
_start:
{
uint8_t v_res_885_; lean_object* v_r_886_; 
v_res_885_ = lp_aesop_Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4(v_as_883_, v_a_884_);
lean_dec(v_a_884_);
lean_dec_ref(v_as_883_);
v_r_886_ = lean_box(v_res_885_);
return v_r_886_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__1(void){
_start:
{
lean_object* v___x_888_; lean_object* v___x_889_; 
v___x_888_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__0));
v___x_889_ = l_Lean_stringToMessageData(v___x_888_);
return v___x_889_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__2(void){
_start:
{
lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_890_ = lean_obj_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__1, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__1_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__1);
v___x_891_ = lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix;
v___x_892_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_892_, 0, v___x_891_);
lean_ctor_set(v___x_892_, 1, v___x_890_);
return v___x_892_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__4(void){
_start:
{
lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_894_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__3));
v___x_895_ = l_Lean_stringToMessageData(v___x_894_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg(lean_object* v_args_896_, lean_object* v_val_897_, lean_object* v_pat_x3f_898_, lean_object* v_range_899_, lean_object* v_b_900_, lean_object* v_i_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_){
_start:
{
lean_object* v_stop_907_; lean_object* v_step_908_; lean_object* v_a_910_; uint8_t v___x_913_; 
v_stop_907_ = lean_ctor_get(v_range_899_, 1);
v_step_908_ = lean_ctor_get(v_range_899_, 2);
v___x_913_ = lean_nat_dec_lt(v_i_901_, v_stop_907_);
if (v___x_913_ == 0)
{
lean_object* v___x_914_; 
lean_dec(v_i_901_);
v___x_914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_914_, 0, v_b_900_);
return v___x_914_;
}
else
{
lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; 
v___x_915_ = lean_array_fget_borrowed(v_args_896_, v_i_901_);
v___x_916_ = l_Lean_Expr_fvarId_x21(v___x_915_);
v___x_917_ = l_Lean_FVarId_getDecl___redArg(v___x_916_, v___y_902_, v___y_904_, v___y_905_);
if (lean_obj_tag(v___x_917_) == 0)
{
lean_object* v_a_918_; lean_object* v_fst_919_; lean_object* v_snd_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_949_; 
v_a_918_ = lean_ctor_get(v___x_917_, 0);
lean_inc(v_a_918_);
lean_dec_ref_known(v___x_917_, 1);
v_fst_919_ = lean_ctor_get(v_b_900_, 0);
v_snd_920_ = lean_ctor_get(v_b_900_, 1);
v_isSharedCheck_949_ = !lean_is_exclusive(v_b_900_);
if (v_isSharedCheck_949_ == 0)
{
v___x_922_ = v_b_900_;
v_isShared_923_ = v_isSharedCheck_949_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_snd_920_);
lean_inc(v_fst_919_);
lean_dec(v_b_900_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_949_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_924_; uint8_t v___x_925_; 
v___x_924_ = l_Lean_LocalDecl_userName(v_a_918_);
lean_dec(v_a_918_);
v___x_925_ = lp_aesop_Array_contains___at___00Aesop_RuleBuilder_getImmediatePremises_spec__4(v_val_897_, v___x_924_);
if (v___x_925_ == 0)
{
lean_object* v___x_927_; 
lean_dec(v___x_924_);
if (v_isShared_923_ == 0)
{
v___x_927_ = v___x_922_;
goto v_reusejp_926_;
}
else
{
lean_object* v_reuseFailAlloc_928_; 
v_reuseFailAlloc_928_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_928_, 0, v_fst_919_);
lean_ctor_set(v_reuseFailAlloc_928_, 1, v_snd_920_);
v___x_927_ = v_reuseFailAlloc_928_;
goto v_reusejp_926_;
}
v_reusejp_926_:
{
v_a_910_ = v___x_927_;
goto v___jp_909_;
}
}
else
{
uint8_t v___x_929_; 
v___x_929_ = lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_isPatternInstantiated(v_pat_x3f_898_, v_i_901_);
if (v___x_929_ == 0)
{
lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_933_; 
lean_inc(v_i_901_);
v___x_930_ = lean_array_push(v_snd_920_, v_i_901_);
v___x_931_ = lp_aesop_Array_erase___at___00Aesop_RuleBuilder_getImmediatePremises_spec__5(v_fst_919_, v___x_924_);
lean_dec(v___x_924_);
if (v_isShared_923_ == 0)
{
lean_ctor_set(v___x_922_, 1, v___x_930_);
lean_ctor_set(v___x_922_, 0, v___x_931_);
v___x_933_ = v___x_922_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v___x_931_);
lean_ctor_set(v_reuseFailAlloc_934_, 1, v___x_930_);
v___x_933_ = v_reuseFailAlloc_934_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
v_a_910_ = v___x_933_;
goto v___jp_909_;
}
}
else
{
lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_948_; 
lean_del_object(v___x_922_);
lean_dec(v_snd_920_);
lean_dec(v_fst_919_);
lean_dec(v_i_901_);
v___x_935_ = lean_obj_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__2, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__2_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__2);
v___x_936_ = l_Lean_MessageData_ofName(v___x_924_);
v___x_937_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_937_, 0, v___x_935_);
lean_ctor_set(v___x_937_, 1, v___x_936_);
v___x_938_ = lean_obj_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__4, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__4_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___closed__4);
v___x_939_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_939_, 0, v___x_937_);
lean_ctor_set(v___x_939_, 1, v___x_938_);
v___x_940_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(v___x_939_, v___y_902_, v___y_903_, v___y_904_, v___y_905_);
v_a_941_ = lean_ctor_get(v___x_940_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___x_940_);
if (v_isSharedCheck_948_ == 0)
{
v___x_943_ = v___x_940_;
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_940_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_946_; 
if (v_isShared_944_ == 0)
{
v___x_946_ = v___x_943_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v_a_941_);
v___x_946_ = v_reuseFailAlloc_947_;
goto v_reusejp_945_;
}
v_reusejp_945_:
{
return v___x_946_;
}
}
}
}
}
}
else
{
lean_object* v_a_950_; lean_object* v___x_952_; uint8_t v_isShared_953_; uint8_t v_isSharedCheck_957_; 
lean_dec(v_i_901_);
lean_dec_ref(v_b_900_);
v_a_950_ = lean_ctor_get(v___x_917_, 0);
v_isSharedCheck_957_ = !lean_is_exclusive(v___x_917_);
if (v_isSharedCheck_957_ == 0)
{
v___x_952_ = v___x_917_;
v_isShared_953_ = v_isSharedCheck_957_;
goto v_resetjp_951_;
}
else
{
lean_inc(v_a_950_);
lean_dec(v___x_917_);
v___x_952_ = lean_box(0);
v_isShared_953_ = v_isSharedCheck_957_;
goto v_resetjp_951_;
}
v_resetjp_951_:
{
lean_object* v___x_955_; 
if (v_isShared_953_ == 0)
{
v___x_955_ = v___x_952_;
goto v_reusejp_954_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v_a_950_);
v___x_955_ = v_reuseFailAlloc_956_;
goto v_reusejp_954_;
}
v_reusejp_954_:
{
return v___x_955_;
}
}
}
}
v___jp_909_:
{
lean_object* v___x_911_; 
v___x_911_ = lean_nat_add(v_i_901_, v_step_908_);
lean_dec(v_i_901_);
v_b_900_ = v_a_910_;
v_i_901_ = v___x_911_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg___boxed(lean_object* v_args_958_, lean_object* v_val_959_, lean_object* v_pat_x3f_960_, lean_object* v_range_961_, lean_object* v_b_962_, lean_object* v_i_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_){
_start:
{
lean_object* v_res_969_; 
v_res_969_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg(v_args_958_, v_val_959_, v_pat_x3f_960_, v_range_961_, v_b_962_, v_i_963_, v___y_964_, v___y_965_, v___y_966_, v___y_967_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
lean_dec(v___y_965_);
lean_dec_ref(v___y_964_);
lean_dec_ref(v_range_961_);
lean_dec(v_pat_x3f_960_);
lean_dec_ref(v_val_959_);
lean_dec_ref(v_args_958_);
return v_res_969_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__1(void){
_start:
{
lean_object* v___x_971_; lean_object* v___x_972_; 
v___x_971_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__0));
v___x_972_ = l_Lean_stringToMessageData(v___x_971_);
return v___x_972_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__2(void){
_start:
{
lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; 
v___x_973_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__1, &lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__1_once, _init_lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__1);
v___x_974_ = lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix;
v___x_975_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_975_, 0, v___x_974_);
lean_ctor_set(v___x_975_, 1, v___x_973_);
return v___x_975_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__4(void){
_start:
{
lean_object* v___x_977_; lean_object* v___x_978_; 
v___x_977_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__3));
v___x_978_ = l_Lean_stringToMessageData(v___x_977_);
return v___x_978_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1(lean_object* v_val_979_, lean_object* v_pat_x3f_980_, lean_object* v_args_981_, lean_object* v_x_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_){
_start:
{
lean_object* v_unseen_988_; lean_object* v___x_989_; lean_object* v_result_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; 
lean_inc_ref(v_val_979_);
v_unseen_988_ = lp_aesop_Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6(v_val_979_);
v___x_989_ = lean_unsigned_to_nat(0u);
v_result_990_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___closed__0));
v___x_991_ = lean_array_get_size(v_args_981_);
v___x_992_ = lean_unsigned_to_nat(1u);
v___x_993_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_993_, 0, v___x_989_);
lean_ctor_set(v___x_993_, 1, v___x_991_);
lean_ctor_set(v___x_993_, 2, v___x_992_);
v___x_994_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_994_, 0, v_unseen_988_);
lean_ctor_set(v___x_994_, 1, v_result_990_);
v___x_995_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg(v_args_981_, v_val_979_, v_pat_x3f_980_, v___x_993_, v___x_994_, v___x_989_, v___y_983_, v___y_984_, v___y_985_, v___y_986_);
lean_dec_ref_known(v___x_993_, 3);
lean_dec_ref(v_val_979_);
if (lean_obj_tag(v___x_995_) == 0)
{
lean_object* v_a_996_; lean_object* v___x_998_; uint8_t v_isShared_999_; uint8_t v_isSharedCheck_1030_; 
v_a_996_ = lean_ctor_get(v___x_995_, 0);
v_isSharedCheck_1030_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1030_ == 0)
{
v___x_998_ = v___x_995_;
v_isShared_999_ = v_isSharedCheck_1030_;
goto v_resetjp_997_;
}
else
{
lean_inc(v_a_996_);
lean_dec(v___x_995_);
v___x_998_ = lean_box(0);
v_isShared_999_ = v_isSharedCheck_1030_;
goto v_resetjp_997_;
}
v_resetjp_997_:
{
lean_object* v_fst_1000_; lean_object* v_snd_1001_; lean_object* v___x_1003_; uint8_t v_isShared_1004_; uint8_t v_isSharedCheck_1029_; 
v_fst_1000_ = lean_ctor_get(v_a_996_, 0);
v_snd_1001_ = lean_ctor_get(v_a_996_, 1);
v_isSharedCheck_1029_ = !lean_is_exclusive(v_a_996_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1003_ = v_a_996_;
v_isShared_1004_ = v_isSharedCheck_1029_;
goto v_resetjp_1002_;
}
else
{
lean_inc(v_snd_1001_);
lean_inc(v_fst_1000_);
lean_dec(v_a_996_);
v___x_1003_ = lean_box(0);
v_isShared_1004_ = v_isSharedCheck_1029_;
goto v_resetjp_1002_;
}
v_resetjp_1002_:
{
lean_object* v___x_1005_; uint8_t v___x_1006_; 
v___x_1005_ = lean_array_get_size(v_fst_1000_);
v___x_1006_ = lean_nat_dec_eq(v___x_1005_, v___x_989_);
if (v___x_1006_ == 0)
{
lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1013_; 
lean_dec(v_snd_1001_);
lean_del_object(v___x_998_);
v___x_1007_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__2, &lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__2_once, _init_lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__2);
v___x_1008_ = lean_array_to_list(v_fst_1000_);
v___x_1009_ = lean_box(0);
v___x_1010_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__8(v___x_1008_, v___x_1009_);
v___x_1011_ = l_Lean_MessageData_ofList(v___x_1010_);
if (v_isShared_1004_ == 0)
{
lean_ctor_set_tag(v___x_1003_, 7);
lean_ctor_set(v___x_1003_, 1, v___x_1011_);
lean_ctor_set(v___x_1003_, 0, v___x_1007_);
v___x_1013_ = v___x_1003_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1025_; 
v_reuseFailAlloc_1025_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1025_, 0, v___x_1007_);
lean_ctor_set(v_reuseFailAlloc_1025_, 1, v___x_1011_);
v___x_1013_ = v_reuseFailAlloc_1025_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v_a_1017_; lean_object* v___x_1019_; uint8_t v_isShared_1020_; uint8_t v_isSharedCheck_1024_; 
v___x_1014_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__4, &lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__4_once, _init_lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___closed__4);
v___x_1015_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1013_);
lean_ctor_set(v___x_1015_, 1, v___x_1014_);
v___x_1016_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2___redArg(v___x_1015_, v___y_983_, v___y_984_, v___y_985_, v___y_986_);
v_a_1017_ = lean_ctor_get(v___x_1016_, 0);
v_isSharedCheck_1024_ = !lean_is_exclusive(v___x_1016_);
if (v_isSharedCheck_1024_ == 0)
{
v___x_1019_ = v___x_1016_;
v_isShared_1020_ = v_isSharedCheck_1024_;
goto v_resetjp_1018_;
}
else
{
lean_inc(v_a_1017_);
lean_dec(v___x_1016_);
v___x_1019_ = lean_box(0);
v_isShared_1020_ = v_isSharedCheck_1024_;
goto v_resetjp_1018_;
}
v_resetjp_1018_:
{
lean_object* v___x_1022_; 
if (v_isShared_1020_ == 0)
{
v___x_1022_ = v___x_1019_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v_a_1017_);
v___x_1022_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
return v___x_1022_;
}
}
}
}
else
{
lean_object* v___x_1027_; 
lean_del_object(v___x_1003_);
lean_dec(v_fst_1000_);
if (v_isShared_999_ == 0)
{
lean_ctor_set(v___x_998_, 0, v_snd_1001_);
v___x_1027_ = v___x_998_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v_snd_1001_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
}
}
}
else
{
lean_object* v_a_1031_; lean_object* v___x_1033_; uint8_t v_isShared_1034_; uint8_t v_isSharedCheck_1038_; 
v_a_1031_ = lean_ctor_get(v___x_995_, 0);
v_isSharedCheck_1038_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1038_ == 0)
{
v___x_1033_ = v___x_995_;
v_isShared_1034_ = v_isSharedCheck_1038_;
goto v_resetjp_1032_;
}
else
{
lean_inc(v_a_1031_);
lean_dec(v___x_995_);
v___x_1033_ = lean_box(0);
v_isShared_1034_ = v_isSharedCheck_1038_;
goto v_resetjp_1032_;
}
v_resetjp_1032_:
{
lean_object* v___x_1036_; 
if (v_isShared_1034_ == 0)
{
v___x_1036_ = v___x_1033_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1037_; 
v_reuseFailAlloc_1037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1037_, 0, v_a_1031_);
v___x_1036_ = v_reuseFailAlloc_1037_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
return v___x_1036_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___boxed(lean_object* v_val_1039_, lean_object* v_pat_x3f_1040_, lean_object* v_args_1041_, lean_object* v_x_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_){
_start:
{
lean_object* v_res_1048_; 
v_res_1048_ = lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1(v_val_1039_, v_pat_x3f_1040_, v_args_1041_, v_x_1042_, v___y_1043_, v___y_1044_, v___y_1045_, v___y_1046_);
lean_dec(v___y_1046_);
lean_dec_ref(v___y_1045_);
lean_dec(v___y_1044_);
lean_dec_ref(v___y_1043_);
lean_dec_ref(v_x_1042_);
lean_dec_ref(v_args_1041_);
lean_dec(v_pat_x3f_1040_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises(lean_object* v_type_1049_, lean_object* v_pat_x3f_1050_, lean_object* v_x_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_, lean_object* v_a_1055_){
_start:
{
if (lean_obj_tag(v_x_1051_) == 0)
{
lean_object* v_keyedConfig_1057_; uint8_t v_trackZetaDelta_1058_; lean_object* v_zetaDeltaSet_1059_; lean_object* v_lctx_1060_; lean_object* v_localInstances_1061_; lean_object* v_defEqCtx_x3f_1062_; lean_object* v_synthPendingDepth_1063_; lean_object* v_customCanUnfoldPredicate_x3f_1064_; uint8_t v_univApprox_1065_; uint8_t v_inTypeClassResolution_1066_; uint8_t v_cacheInferType_1067_; lean_object* v___f_1068_; uint8_t v___x_1069_; uint8_t v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; 
v_keyedConfig_1057_ = lean_ctor_get(v_a_1052_, 0);
v_trackZetaDelta_1058_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7);
v_zetaDeltaSet_1059_ = lean_ctor_get(v_a_1052_, 1);
v_lctx_1060_ = lean_ctor_get(v_a_1052_, 2);
v_localInstances_1061_ = lean_ctor_get(v_a_1052_, 3);
v_defEqCtx_x3f_1062_ = lean_ctor_get(v_a_1052_, 4);
v_synthPendingDepth_1063_ = lean_ctor_get(v_a_1052_, 5);
v_customCanUnfoldPredicate_x3f_1064_ = lean_ctor_get(v_a_1052_, 6);
v_univApprox_1065_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1066_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7 + 2);
v_cacheInferType_1067_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7 + 3);
v___f_1068_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1068_, 0, v_pat_x3f_1050_);
v___x_1069_ = 0;
v___x_1070_ = 2;
lean_inc_ref(v_keyedConfig_1057_);
v___x_1071_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1070_, v_keyedConfig_1057_);
lean_inc(v_customCanUnfoldPredicate_x3f_1064_);
lean_inc(v_synthPendingDepth_1063_);
lean_inc(v_defEqCtx_x3f_1062_);
lean_inc_ref(v_localInstances_1061_);
lean_inc_ref(v_lctx_1060_);
lean_inc(v_zetaDeltaSet_1059_);
v___x_1072_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1072_, 0, v___x_1071_);
lean_ctor_set(v___x_1072_, 1, v_zetaDeltaSet_1059_);
lean_ctor_set(v___x_1072_, 2, v_lctx_1060_);
lean_ctor_set(v___x_1072_, 3, v_localInstances_1061_);
lean_ctor_set(v___x_1072_, 4, v_defEqCtx_x3f_1062_);
lean_ctor_set(v___x_1072_, 5, v_synthPendingDepth_1063_);
lean_ctor_set(v___x_1072_, 6, v_customCanUnfoldPredicate_x3f_1064_);
lean_ctor_set_uint8(v___x_1072_, sizeof(void*)*7, v_trackZetaDelta_1058_);
lean_ctor_set_uint8(v___x_1072_, sizeof(void*)*7 + 1, v_univApprox_1065_);
lean_ctor_set_uint8(v___x_1072_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1066_);
lean_ctor_set_uint8(v___x_1072_, sizeof(void*)*7 + 3, v_cacheInferType_1067_);
v___x_1073_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg(v_type_1049_, v___f_1068_, v___x_1069_, v___x_1069_, v___x_1072_, v_a_1053_, v_a_1054_, v_a_1055_);
lean_dec_ref_known(v___x_1072_, 7);
return v___x_1073_;
}
else
{
lean_object* v_val_1074_; lean_object* v_keyedConfig_1075_; uint8_t v_trackZetaDelta_1076_; lean_object* v_zetaDeltaSet_1077_; lean_object* v_lctx_1078_; lean_object* v_localInstances_1079_; lean_object* v_defEqCtx_x3f_1080_; lean_object* v_synthPendingDepth_1081_; lean_object* v_customCanUnfoldPredicate_x3f_1082_; uint8_t v_univApprox_1083_; uint8_t v_inTypeClassResolution_1084_; uint8_t v_cacheInferType_1085_; lean_object* v___f_1086_; uint8_t v___x_1087_; uint8_t v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; 
v_val_1074_ = lean_ctor_get(v_x_1051_, 0);
lean_inc(v_val_1074_);
lean_dec_ref_known(v_x_1051_, 1);
v_keyedConfig_1075_ = lean_ctor_get(v_a_1052_, 0);
v_trackZetaDelta_1076_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7);
v_zetaDeltaSet_1077_ = lean_ctor_get(v_a_1052_, 1);
v_lctx_1078_ = lean_ctor_get(v_a_1052_, 2);
v_localInstances_1079_ = lean_ctor_get(v_a_1052_, 3);
v_defEqCtx_x3f_1080_ = lean_ctor_get(v_a_1052_, 4);
v_synthPendingDepth_1081_ = lean_ctor_get(v_a_1052_, 5);
v_customCanUnfoldPredicate_x3f_1082_ = lean_ctor_get(v_a_1052_, 6);
v_univApprox_1083_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1084_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7 + 2);
v_cacheInferType_1085_ = lean_ctor_get_uint8(v_a_1052_, sizeof(void*)*7 + 3);
v___f_1086_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleBuilder_getImmediatePremises___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1086_, 0, v_val_1074_);
lean_closure_set(v___f_1086_, 1, v_pat_x3f_1050_);
v___x_1087_ = 0;
v___x_1088_ = 2;
lean_inc_ref(v_keyedConfig_1075_);
v___x_1089_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1088_, v_keyedConfig_1075_);
lean_inc(v_customCanUnfoldPredicate_x3f_1082_);
lean_inc(v_synthPendingDepth_1081_);
lean_inc(v_defEqCtx_x3f_1080_);
lean_inc_ref(v_localInstances_1079_);
lean_inc_ref(v_lctx_1078_);
lean_inc(v_zetaDeltaSet_1077_);
v___x_1090_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1090_, 0, v___x_1089_);
lean_ctor_set(v___x_1090_, 1, v_zetaDeltaSet_1077_);
lean_ctor_set(v___x_1090_, 2, v_lctx_1078_);
lean_ctor_set(v___x_1090_, 3, v_localInstances_1079_);
lean_ctor_set(v___x_1090_, 4, v_defEqCtx_x3f_1080_);
lean_ctor_set(v___x_1090_, 5, v_synthPendingDepth_1081_);
lean_ctor_set(v___x_1090_, 6, v_customCanUnfoldPredicate_x3f_1082_);
lean_ctor_set_uint8(v___x_1090_, sizeof(void*)*7, v_trackZetaDelta_1076_);
lean_ctor_set_uint8(v___x_1090_, sizeof(void*)*7 + 1, v_univApprox_1083_);
lean_ctor_set_uint8(v___x_1090_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1084_);
lean_ctor_set_uint8(v___x_1090_, sizeof(void*)*7 + 3, v_cacheInferType_1085_);
v___x_1091_ = lp_aesop_Lean_Meta_forallTelescopeReducing___at___00Aesop_RuleBuilder_getImmediatePremises_spec__3___redArg(v_type_1049_, v___f_1086_, v___x_1087_, v___x_1087_, v___x_1090_, v_a_1053_, v_a_1054_, v_a_1055_);
lean_dec_ref_known(v___x_1090_, 7);
return v___x_1091_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getImmediatePremises___boxed(lean_object* v_type_1092_, lean_object* v_pat_x3f_1093_, lean_object* v_x_1094_, lean_object* v_a_1095_, lean_object* v_a_1096_, lean_object* v_a_1097_, lean_object* v_a_1098_, lean_object* v_a_1099_){
_start:
{
lean_object* v_res_1100_; 
v_res_1100_ = lp_aesop_Aesop_RuleBuilder_getImmediatePremises(v_type_1092_, v_pat_x3f_1093_, v_x_1094_, v_a_1095_, v_a_1096_, v_a_1097_, v_a_1098_);
lean_dec(v_a_1098_);
lean_dec_ref(v_a_1097_);
lean_dec(v_a_1096_);
lean_dec_ref(v_a_1095_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2(lean_object* v_pat_x3f_1101_, lean_object* v_args_1102_, lean_object* v___x_1103_, lean_object* v_range_1104_, lean_object* v_b_1105_, lean_object* v_i_1106_, lean_object* v_hs_1107_, lean_object* v_hl_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_){
_start:
{
lean_object* v___x_1114_; 
v___x_1114_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___redArg(v_pat_x3f_1101_, v_args_1102_, v___x_1103_, v_range_1104_, v_b_1105_, v_i_1106_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_);
return v___x_1114_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2___boxed(lean_object* v_pat_x3f_1115_, lean_object* v_args_1116_, lean_object* v___x_1117_, lean_object* v_range_1118_, lean_object* v_b_1119_, lean_object* v_i_1120_, lean_object* v_hs_1121_, lean_object* v_hl_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_){
_start:
{
lean_object* v_res_1128_; 
v_res_1128_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__2(v_pat_x3f_1115_, v_args_1116_, v___x_1117_, v_range_1118_, v_b_1119_, v_i_1120_, v_hs_1121_, v_hl_1122_, v___y_1123_, v___y_1124_, v___y_1125_, v___y_1126_);
lean_dec(v___y_1126_);
lean_dec_ref(v___y_1125_);
lean_dec(v___y_1124_);
lean_dec_ref(v___y_1123_);
lean_dec_ref(v_range_1118_);
lean_dec_ref(v_args_1116_);
lean_dec(v_pat_x3f_1115_);
return v_res_1128_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7(lean_object* v_args_1129_, lean_object* v_val_1130_, lean_object* v_pat_x3f_1131_, lean_object* v_range_1132_, lean_object* v_b_1133_, lean_object* v_i_1134_, lean_object* v_hs_1135_, lean_object* v_hl_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_){
_start:
{
lean_object* v___x_1142_; 
v___x_1142_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___redArg(v_args_1129_, v_val_1130_, v_pat_x3f_1131_, v_range_1132_, v_b_1133_, v_i_1134_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_);
return v___x_1142_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7___boxed(lean_object* v_args_1143_, lean_object* v_val_1144_, lean_object* v_pat_x3f_1145_, lean_object* v_range_1146_, lean_object* v_b_1147_, lean_object* v_i_1148_, lean_object* v_hs_1149_, lean_object* v_hl_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_){
_start:
{
lean_object* v_res_1156_; 
v_res_1156_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_getImmediatePremises_spec__7(v_args_1143_, v_val_1144_, v_pat_x3f_1145_, v_range_1146_, v_b_1147_, v_i_1148_, v_hs_1149_, v_hl_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_);
lean_dec(v___y_1154_);
lean_dec_ref(v___y_1153_);
lean_dec(v___y_1152_);
lean_dec_ref(v___y_1151_);
lean_dec_ref(v_range_1146_);
lean_dec(v_pat_x3f_1145_);
lean_dec_ref(v_val_1144_);
lean_dec_ref(v_args_1143_);
return v_res_1156_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8(lean_object* v_n_1157_, lean_object* v_as_1158_, lean_object* v_lo_1159_, lean_object* v_hi_1160_, lean_object* v_w_1161_, lean_object* v_hlo_1162_, lean_object* v_hhi_1163_){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___redArg(v_n_1157_, v_as_1158_, v_lo_1159_, v_hi_1160_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8___boxed(lean_object* v_n_1165_, lean_object* v_as_1166_, lean_object* v_lo_1167_, lean_object* v_hi_1168_, lean_object* v_w_1169_, lean_object* v_hlo_1170_, lean_object* v_hhi_1171_){
_start:
{
lean_object* v_res_1172_; 
v_res_1172_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8(v_n_1165_, v_as_1166_, v_lo_1167_, v_hi_1168_, v_w_1169_, v_hlo_1170_, v_hhi_1171_);
lean_dec(v_hi_1168_);
lean_dec(v_n_1165_);
return v_res_1172_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10(lean_object* v_n_1173_, lean_object* v_lo_1174_, lean_object* v_hi_1175_, lean_object* v_hhi_1176_, lean_object* v_pivot_1177_, lean_object* v_as_1178_, lean_object* v_i_1179_, lean_object* v_k_1180_, lean_object* v_ilo_1181_, lean_object* v_ik_1182_, lean_object* v_w_1183_){
_start:
{
lean_object* v___x_1184_; 
v___x_1184_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___redArg(v_hi_1175_, v_pivot_1177_, v_as_1178_, v_i_1179_, v_k_1180_);
return v___x_1184_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10___boxed(lean_object* v_n_1185_, lean_object* v_lo_1186_, lean_object* v_hi_1187_, lean_object* v_hhi_1188_, lean_object* v_pivot_1189_, lean_object* v_as_1190_, lean_object* v_i_1191_, lean_object* v_k_1192_, lean_object* v_ilo_1193_, lean_object* v_ik_1194_, lean_object* v_w_1195_){
_start:
{
lean_object* v_res_1196_; 
v_res_1196_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_sortDedup___at___00Aesop_RuleBuilder_getImmediatePremises_spec__6_spec__8_spec__10(v_n_1185_, v_lo_1186_, v_hi_1187_, v_hhi_1188_, v_pivot_1189_, v_as_1190_, v_i_1191_, v_k_1192_, v_ilo_1193_, v_ik_1194_, v_w_1195_);
lean_dec(v_pivot_1189_);
lean_dec(v_hi_1187_);
lean_dec(v_lo_1186_);
lean_dec(v_n_1185_);
return v_res_1196_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v___x_1197_ = lean_unsigned_to_nat(32u);
v___x_1198_ = lean_mk_empty_array_with_capacity(v___x_1197_);
v___x_1199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1198_);
return v___x_1199_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1(void){
_start:
{
size_t v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1200_ = ((size_t)5ULL);
v___x_1201_ = lean_unsigned_to_nat(0u);
v___x_1202_ = lean_unsigned_to_nat(32u);
v___x_1203_ = lean_mk_empty_array_with_capacity(v___x_1202_);
v___x_1204_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__0);
v___x_1205_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1205_, 0, v___x_1204_);
lean_ctor_set(v___x_1205_, 1, v___x_1203_);
lean_ctor_set(v___x_1205_, 2, v___x_1201_);
lean_ctor_set(v___x_1205_, 3, v___x_1201_);
lean_ctor_set_usize(v___x_1205_, 4, v___x_1200_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg(lean_object* v___y_1206_){
_start:
{
lean_object* v___x_1208_; lean_object* v_traceState_1209_; lean_object* v_traces_1210_; lean_object* v___x_1211_; lean_object* v_traceState_1212_; lean_object* v_env_1213_; lean_object* v_nextMacroScope_1214_; lean_object* v_ngen_1215_; lean_object* v_auxDeclNGen_1216_; lean_object* v_cache_1217_; lean_object* v_messages_1218_; lean_object* v_infoState_1219_; lean_object* v_snapshotTasks_1220_; lean_object* v___x_1222_; uint8_t v_isShared_1223_; uint8_t v_isSharedCheck_1239_; 
v___x_1208_ = lean_st_ref_get(v___y_1206_);
v_traceState_1209_ = lean_ctor_get(v___x_1208_, 4);
lean_inc_ref(v_traceState_1209_);
lean_dec(v___x_1208_);
v_traces_1210_ = lean_ctor_get(v_traceState_1209_, 0);
lean_inc_ref(v_traces_1210_);
lean_dec_ref(v_traceState_1209_);
v___x_1211_ = lean_st_ref_take(v___y_1206_);
v_traceState_1212_ = lean_ctor_get(v___x_1211_, 4);
v_env_1213_ = lean_ctor_get(v___x_1211_, 0);
v_nextMacroScope_1214_ = lean_ctor_get(v___x_1211_, 1);
v_ngen_1215_ = lean_ctor_get(v___x_1211_, 2);
v_auxDeclNGen_1216_ = lean_ctor_get(v___x_1211_, 3);
v_cache_1217_ = lean_ctor_get(v___x_1211_, 5);
v_messages_1218_ = lean_ctor_get(v___x_1211_, 6);
v_infoState_1219_ = lean_ctor_get(v___x_1211_, 7);
v_snapshotTasks_1220_ = lean_ctor_get(v___x_1211_, 8);
v_isSharedCheck_1239_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1239_ == 0)
{
v___x_1222_ = v___x_1211_;
v_isShared_1223_ = v_isSharedCheck_1239_;
goto v_resetjp_1221_;
}
else
{
lean_inc(v_snapshotTasks_1220_);
lean_inc(v_infoState_1219_);
lean_inc(v_messages_1218_);
lean_inc(v_cache_1217_);
lean_inc(v_traceState_1212_);
lean_inc(v_auxDeclNGen_1216_);
lean_inc(v_ngen_1215_);
lean_inc(v_nextMacroScope_1214_);
lean_inc(v_env_1213_);
lean_dec(v___x_1211_);
v___x_1222_ = lean_box(0);
v_isShared_1223_ = v_isSharedCheck_1239_;
goto v_resetjp_1221_;
}
v_resetjp_1221_:
{
uint64_t v_tid_1224_; lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1237_; 
v_tid_1224_ = lean_ctor_get_uint64(v_traceState_1212_, sizeof(void*)*1);
v_isSharedCheck_1237_ = !lean_is_exclusive(v_traceState_1212_);
if (v_isSharedCheck_1237_ == 0)
{
lean_object* v_unused_1238_; 
v_unused_1238_ = lean_ctor_get(v_traceState_1212_, 0);
lean_dec(v_unused_1238_);
v___x_1226_ = v_traceState_1212_;
v_isShared_1227_ = v_isSharedCheck_1237_;
goto v_resetjp_1225_;
}
else
{
lean_dec(v_traceState_1212_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1237_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v___x_1228_; lean_object* v___x_1230_; 
v___x_1228_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1);
if (v_isShared_1227_ == 0)
{
lean_ctor_set(v___x_1226_, 0, v___x_1228_);
v___x_1230_ = v___x_1226_;
goto v_reusejp_1229_;
}
else
{
lean_object* v_reuseFailAlloc_1236_; 
v_reuseFailAlloc_1236_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1236_, 0, v___x_1228_);
lean_ctor_set_uint64(v_reuseFailAlloc_1236_, sizeof(void*)*1, v_tid_1224_);
v___x_1230_ = v_reuseFailAlloc_1236_;
goto v_reusejp_1229_;
}
v_reusejp_1229_:
{
lean_object* v___x_1232_; 
if (v_isShared_1223_ == 0)
{
lean_ctor_set(v___x_1222_, 4, v___x_1230_);
v___x_1232_ = v___x_1222_;
goto v_reusejp_1231_;
}
else
{
lean_object* v_reuseFailAlloc_1235_; 
v_reuseFailAlloc_1235_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1235_, 0, v_env_1213_);
lean_ctor_set(v_reuseFailAlloc_1235_, 1, v_nextMacroScope_1214_);
lean_ctor_set(v_reuseFailAlloc_1235_, 2, v_ngen_1215_);
lean_ctor_set(v_reuseFailAlloc_1235_, 3, v_auxDeclNGen_1216_);
lean_ctor_set(v_reuseFailAlloc_1235_, 4, v___x_1230_);
lean_ctor_set(v_reuseFailAlloc_1235_, 5, v_cache_1217_);
lean_ctor_set(v_reuseFailAlloc_1235_, 6, v_messages_1218_);
lean_ctor_set(v_reuseFailAlloc_1235_, 7, v_infoState_1219_);
lean_ctor_set(v_reuseFailAlloc_1235_, 8, v_snapshotTasks_1220_);
v___x_1232_ = v_reuseFailAlloc_1235_;
goto v_reusejp_1231_;
}
v_reusejp_1231_:
{
lean_object* v___x_1233_; lean_object* v___x_1234_; 
v___x_1233_ = lean_st_ref_set(v___y_1206_, v___x_1232_);
v___x_1234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1234_, 0, v_traces_1210_);
return v___x_1234_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___boxed(lean_object* v___y_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg(v___y_1240_);
lean_dec(v___y_1240_);
return v_res_1242_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3(lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg(v___y_1246_);
return v___x_1248_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___boxed(lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_){
_start:
{
lean_object* v_res_1254_; 
v_res_1254_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3(v___y_1249_, v___y_1250_, v___y_1251_, v___y_1252_);
lean_dec(v___y_1252_);
lean_dec_ref(v___y_1251_);
lean_dec(v___y_1250_);
lean_dec_ref(v___y_1249_);
return v_res_1254_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(lean_object* v_opts_1255_, lean_object* v_opt_1256_){
_start:
{
lean_object* v_name_1257_; lean_object* v_defValue_1258_; lean_object* v_map_1259_; lean_object* v___x_1260_; 
v_name_1257_ = lean_ctor_get(v_opt_1256_, 0);
v_defValue_1258_ = lean_ctor_get(v_opt_1256_, 1);
v_map_1259_ = lean_ctor_get(v_opts_1255_, 0);
v___x_1260_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1259_, v_name_1257_);
if (lean_obj_tag(v___x_1260_) == 0)
{
uint8_t v___x_1261_; 
v___x_1261_ = lean_unbox(v_defValue_1258_);
return v___x_1261_;
}
else
{
lean_object* v_val_1262_; 
v_val_1262_ = lean_ctor_get(v___x_1260_, 0);
lean_inc(v_val_1262_);
lean_dec_ref_known(v___x_1260_, 1);
if (lean_obj_tag(v_val_1262_) == 1)
{
uint8_t v_v_1263_; 
v_v_1263_ = lean_ctor_get_uint8(v_val_1262_, 0);
lean_dec_ref_known(v_val_1262_, 0);
return v_v_1263_;
}
else
{
uint8_t v___x_1264_; 
lean_dec(v_val_1262_);
v___x_1264_ = lean_unbox(v_defValue_1258_);
return v___x_1264_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4___boxed(lean_object* v_opts_1265_, lean_object* v_opt_1266_){
_start:
{
uint8_t v_res_1267_; lean_object* v_r_1268_; 
v_res_1267_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_opts_1265_, v_opt_1266_);
lean_dec_ref(v_opt_1266_);
lean_dec_ref(v_opts_1265_);
v_r_1268_ = lean_box(v_res_1267_);
return v_r_1268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0(lean_object* v___x_1269_, lean_object* v_x_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_){
_start:
{
lean_object* v___x_1276_; 
v___x_1276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1269_);
return v___x_1276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0___boxed(lean_object* v___x_1277_, lean_object* v_x_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0(v___x_1277_, v_x_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_);
lean_dec(v___y_1282_);
lean_dec_ref(v___y_1281_);
lean_dec(v___y_1280_);
lean_dec_ref(v___y_1279_);
lean_dec_ref(v_x_1278_);
return v_res_1284_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1285_; double v___x_1286_; 
v___x_1285_ = lean_unsigned_to_nat(0u);
v___x_1286_ = lean_float_of_nat(v___x_1285_);
return v___x_1286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(lean_object* v_cls_1290_, lean_object* v_msg_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_){
_start:
{
lean_object* v_ref_1297_; lean_object* v___x_1298_; lean_object* v_a_1299_; lean_object* v___x_1301_; uint8_t v_isShared_1302_; uint8_t v_isSharedCheck_1343_; 
v_ref_1297_ = lean_ctor_get(v___y_1294_, 5);
v___x_1298_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(v_msg_1291_, v___y_1292_, v___y_1293_, v___y_1294_, v___y_1295_);
v_a_1299_ = lean_ctor_get(v___x_1298_, 0);
v_isSharedCheck_1343_ = !lean_is_exclusive(v___x_1298_);
if (v_isSharedCheck_1343_ == 0)
{
v___x_1301_ = v___x_1298_;
v_isShared_1302_ = v_isSharedCheck_1343_;
goto v_resetjp_1300_;
}
else
{
lean_inc(v_a_1299_);
lean_dec(v___x_1298_);
v___x_1301_ = lean_box(0);
v_isShared_1302_ = v_isSharedCheck_1343_;
goto v_resetjp_1300_;
}
v_resetjp_1300_:
{
lean_object* v___x_1303_; lean_object* v_traceState_1304_; lean_object* v_env_1305_; lean_object* v_nextMacroScope_1306_; lean_object* v_ngen_1307_; lean_object* v_auxDeclNGen_1308_; lean_object* v_cache_1309_; lean_object* v_messages_1310_; lean_object* v_infoState_1311_; lean_object* v_snapshotTasks_1312_; lean_object* v___x_1314_; uint8_t v_isShared_1315_; uint8_t v_isSharedCheck_1342_; 
v___x_1303_ = lean_st_ref_take(v___y_1295_);
v_traceState_1304_ = lean_ctor_get(v___x_1303_, 4);
v_env_1305_ = lean_ctor_get(v___x_1303_, 0);
v_nextMacroScope_1306_ = lean_ctor_get(v___x_1303_, 1);
v_ngen_1307_ = lean_ctor_get(v___x_1303_, 2);
v_auxDeclNGen_1308_ = lean_ctor_get(v___x_1303_, 3);
v_cache_1309_ = lean_ctor_get(v___x_1303_, 5);
v_messages_1310_ = lean_ctor_get(v___x_1303_, 6);
v_infoState_1311_ = lean_ctor_get(v___x_1303_, 7);
v_snapshotTasks_1312_ = lean_ctor_get(v___x_1303_, 8);
v_isSharedCheck_1342_ = !lean_is_exclusive(v___x_1303_);
if (v_isSharedCheck_1342_ == 0)
{
v___x_1314_ = v___x_1303_;
v_isShared_1315_ = v_isSharedCheck_1342_;
goto v_resetjp_1313_;
}
else
{
lean_inc(v_snapshotTasks_1312_);
lean_inc(v_infoState_1311_);
lean_inc(v_messages_1310_);
lean_inc(v_cache_1309_);
lean_inc(v_traceState_1304_);
lean_inc(v_auxDeclNGen_1308_);
lean_inc(v_ngen_1307_);
lean_inc(v_nextMacroScope_1306_);
lean_inc(v_env_1305_);
lean_dec(v___x_1303_);
v___x_1314_ = lean_box(0);
v_isShared_1315_ = v_isSharedCheck_1342_;
goto v_resetjp_1313_;
}
v_resetjp_1313_:
{
uint64_t v_tid_1316_; lean_object* v_traces_1317_; lean_object* v___x_1319_; uint8_t v_isShared_1320_; uint8_t v_isSharedCheck_1341_; 
v_tid_1316_ = lean_ctor_get_uint64(v_traceState_1304_, sizeof(void*)*1);
v_traces_1317_ = lean_ctor_get(v_traceState_1304_, 0);
v_isSharedCheck_1341_ = !lean_is_exclusive(v_traceState_1304_);
if (v_isSharedCheck_1341_ == 0)
{
v___x_1319_ = v_traceState_1304_;
v_isShared_1320_ = v_isSharedCheck_1341_;
goto v_resetjp_1318_;
}
else
{
lean_inc(v_traces_1317_);
lean_dec(v_traceState_1304_);
v___x_1319_ = lean_box(0);
v_isShared_1320_ = v_isSharedCheck_1341_;
goto v_resetjp_1318_;
}
v_resetjp_1318_:
{
lean_object* v___x_1321_; double v___x_1322_; uint8_t v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1331_; 
v___x_1321_ = lean_box(0);
v___x_1322_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0);
v___x_1323_ = 0;
v___x_1324_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1));
v___x_1325_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1325_, 0, v_cls_1290_);
lean_ctor_set(v___x_1325_, 1, v___x_1321_);
lean_ctor_set(v___x_1325_, 2, v___x_1324_);
lean_ctor_set_float(v___x_1325_, sizeof(void*)*3, v___x_1322_);
lean_ctor_set_float(v___x_1325_, sizeof(void*)*3 + 8, v___x_1322_);
lean_ctor_set_uint8(v___x_1325_, sizeof(void*)*3 + 16, v___x_1323_);
v___x_1326_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__2));
v___x_1327_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1325_);
lean_ctor_set(v___x_1327_, 1, v_a_1299_);
lean_ctor_set(v___x_1327_, 2, v___x_1326_);
lean_inc(v_ref_1297_);
v___x_1328_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1328_, 0, v_ref_1297_);
lean_ctor_set(v___x_1328_, 1, v___x_1327_);
v___x_1329_ = l_Lean_PersistentArray_push___redArg(v_traces_1317_, v___x_1328_);
if (v_isShared_1320_ == 0)
{
lean_ctor_set(v___x_1319_, 0, v___x_1329_);
v___x_1331_ = v___x_1319_;
goto v_reusejp_1330_;
}
else
{
lean_object* v_reuseFailAlloc_1340_; 
v_reuseFailAlloc_1340_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1340_, 0, v___x_1329_);
lean_ctor_set_uint64(v_reuseFailAlloc_1340_, sizeof(void*)*1, v_tid_1316_);
v___x_1331_ = v_reuseFailAlloc_1340_;
goto v_reusejp_1330_;
}
v_reusejp_1330_:
{
lean_object* v___x_1333_; 
if (v_isShared_1315_ == 0)
{
lean_ctor_set(v___x_1314_, 4, v___x_1331_);
v___x_1333_ = v___x_1314_;
goto v_reusejp_1332_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v_env_1305_);
lean_ctor_set(v_reuseFailAlloc_1339_, 1, v_nextMacroScope_1306_);
lean_ctor_set(v_reuseFailAlloc_1339_, 2, v_ngen_1307_);
lean_ctor_set(v_reuseFailAlloc_1339_, 3, v_auxDeclNGen_1308_);
lean_ctor_set(v_reuseFailAlloc_1339_, 4, v___x_1331_);
lean_ctor_set(v_reuseFailAlloc_1339_, 5, v_cache_1309_);
lean_ctor_set(v_reuseFailAlloc_1339_, 6, v_messages_1310_);
lean_ctor_set(v_reuseFailAlloc_1339_, 7, v_infoState_1311_);
lean_ctor_set(v_reuseFailAlloc_1339_, 8, v_snapshotTasks_1312_);
v___x_1333_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1332_;
}
v_reusejp_1332_:
{
lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1337_; 
v___x_1334_ = lean_st_ref_set(v___y_1295_, v___x_1333_);
v___x_1335_ = lean_box(0);
if (v_isShared_1302_ == 0)
{
lean_ctor_set(v___x_1301_, 0, v___x_1335_);
v___x_1337_ = v___x_1301_;
goto v_reusejp_1336_;
}
else
{
lean_object* v_reuseFailAlloc_1338_; 
v_reuseFailAlloc_1338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1338_, 0, v___x_1335_);
v___x_1337_ = v_reuseFailAlloc_1338_;
goto v_reusejp_1336_;
}
v_reusejp_1336_:
{
return v___x_1337_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___boxed(lean_object* v_cls_1344_, lean_object* v_msg_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_){
_start:
{
lean_object* v_res_1351_; 
v_res_1351_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v_cls_1344_, v_msg_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_);
lean_dec(v___y_1349_);
lean_dec_ref(v___y_1348_);
lean_dec(v___y_1347_);
lean_dec_ref(v___y_1346_);
return v_res_1351_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(lean_object* v_opt_1352_, lean_object* v___y_1353_){
_start:
{
lean_object* v_options_1355_; lean_object* v_option_1356_; uint8_t v___x_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; 
v_options_1355_ = lean_ctor_get(v___y_1353_, 2);
v_option_1356_ = lean_ctor_get(v_opt_1352_, 1);
v___x_1357_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_options_1355_, v_option_1356_);
v___x_1358_ = lean_box(v___x_1357_);
v___x_1359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1359_, 0, v___x_1358_);
return v___x_1359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg___boxed(lean_object* v_opt_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_){
_start:
{
lean_object* v_res_1363_; 
v_res_1363_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v_opt_1360_, v___y_1361_);
lean_dec_ref(v___y_1361_);
lean_dec_ref(v_opt_1360_);
return v_res_1363_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(lean_object* v_a_1364_, lean_object* v_a_1365_){
_start:
{
if (lean_obj_tag(v_a_1364_) == 0)
{
lean_object* v___x_1366_; 
v___x_1366_ = l_List_reverse___redArg(v_a_1365_);
return v___x_1366_;
}
else
{
lean_object* v_head_1367_; lean_object* v_tail_1368_; lean_object* v___x_1370_; uint8_t v_isShared_1371_; uint8_t v_isSharedCheck_1379_; 
v_head_1367_ = lean_ctor_get(v_a_1364_, 0);
v_tail_1368_ = lean_ctor_get(v_a_1364_, 1);
v_isSharedCheck_1379_ = !lean_is_exclusive(v_a_1364_);
if (v_isSharedCheck_1379_ == 0)
{
v___x_1370_ = v_a_1364_;
v_isShared_1371_ = v_isSharedCheck_1379_;
goto v_resetjp_1369_;
}
else
{
lean_inc(v_tail_1368_);
lean_inc(v_head_1367_);
lean_dec(v_a_1364_);
v___x_1370_ = lean_box(0);
v_isShared_1371_ = v_isSharedCheck_1379_;
goto v_resetjp_1369_;
}
v_resetjp_1369_:
{
lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1376_; 
v___x_1372_ = l_Nat_reprFast(v_head_1367_);
v___x_1373_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1372_);
v___x_1374_ = l_Lean_MessageData_ofFormat(v___x_1373_);
if (v_isShared_1371_ == 0)
{
lean_ctor_set(v___x_1370_, 1, v_a_1365_);
lean_ctor_set(v___x_1370_, 0, v___x_1374_);
v___x_1376_ = v___x_1370_;
goto v_reusejp_1375_;
}
else
{
lean_object* v_reuseFailAlloc_1378_; 
v_reuseFailAlloc_1378_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1378_, 0, v___x_1374_);
lean_ctor_set(v_reuseFailAlloc_1378_, 1, v_a_1365_);
v___x_1376_ = v_reuseFailAlloc_1378_;
goto v_reusejp_1375_;
}
v_reusejp_1375_:
{
v_a_1364_ = v_tail_1368_;
v_a_1365_ = v___x_1376_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6(size_t v_sz_1380_, size_t v_i_1381_, lean_object* v_bs_1382_){
_start:
{
uint8_t v___x_1383_; 
v___x_1383_ = lean_usize_dec_lt(v_i_1381_, v_sz_1380_);
if (v___x_1383_ == 0)
{
return v_bs_1382_;
}
else
{
lean_object* v_v_1384_; lean_object* v_msg_1385_; lean_object* v___x_1386_; lean_object* v_bs_x27_1387_; size_t v___x_1388_; size_t v___x_1389_; lean_object* v___x_1390_; 
v_v_1384_ = lean_array_uget_borrowed(v_bs_1382_, v_i_1381_);
v_msg_1385_ = lean_ctor_get(v_v_1384_, 1);
lean_inc_ref(v_msg_1385_);
v___x_1386_ = lean_unsigned_to_nat(0u);
v_bs_x27_1387_ = lean_array_uset(v_bs_1382_, v_i_1381_, v___x_1386_);
v___x_1388_ = ((size_t)1ULL);
v___x_1389_ = lean_usize_add(v_i_1381_, v___x_1388_);
v___x_1390_ = lean_array_uset(v_bs_x27_1387_, v_i_1381_, v_msg_1385_);
v_i_1381_ = v___x_1389_;
v_bs_1382_ = v___x_1390_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6___boxed(lean_object* v_sz_1392_, lean_object* v_i_1393_, lean_object* v_bs_1394_){
_start:
{
size_t v_sz_boxed_1395_; size_t v_i_boxed_1396_; lean_object* v_res_1397_; 
v_sz_boxed_1395_ = lean_unbox_usize(v_sz_1392_);
lean_dec(v_sz_1392_);
v_i_boxed_1396_ = lean_unbox_usize(v_i_1393_);
lean_dec(v_i_1393_);
v_res_1397_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6(v_sz_boxed_1395_, v_i_boxed_1396_, v_bs_1394_);
return v_res_1397_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5(lean_object* v_oldTraces_1398_, lean_object* v_data_1399_, lean_object* v_ref_1400_, lean_object* v_msg_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_){
_start:
{
lean_object* v_fileName_1407_; lean_object* v_fileMap_1408_; lean_object* v_options_1409_; lean_object* v_currRecDepth_1410_; lean_object* v_maxRecDepth_1411_; lean_object* v_ref_1412_; lean_object* v_currNamespace_1413_; lean_object* v_openDecls_1414_; lean_object* v_initHeartbeats_1415_; lean_object* v_maxHeartbeats_1416_; lean_object* v_quotContext_1417_; lean_object* v_currMacroScope_1418_; uint8_t v_diag_1419_; lean_object* v_cancelTk_x3f_1420_; uint8_t v_suppressElabErrors_1421_; lean_object* v_inheritedTraceOptions_1422_; lean_object* v___x_1423_; lean_object* v_traceState_1424_; lean_object* v_traces_1425_; lean_object* v_ref_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; size_t v_sz_1429_; size_t v___x_1430_; lean_object* v___x_1431_; lean_object* v_msg_1432_; lean_object* v___x_1433_; lean_object* v_a_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1471_; 
v_fileName_1407_ = lean_ctor_get(v___y_1404_, 0);
v_fileMap_1408_ = lean_ctor_get(v___y_1404_, 1);
v_options_1409_ = lean_ctor_get(v___y_1404_, 2);
v_currRecDepth_1410_ = lean_ctor_get(v___y_1404_, 3);
v_maxRecDepth_1411_ = lean_ctor_get(v___y_1404_, 4);
v_ref_1412_ = lean_ctor_get(v___y_1404_, 5);
v_currNamespace_1413_ = lean_ctor_get(v___y_1404_, 6);
v_openDecls_1414_ = lean_ctor_get(v___y_1404_, 7);
v_initHeartbeats_1415_ = lean_ctor_get(v___y_1404_, 8);
v_maxHeartbeats_1416_ = lean_ctor_get(v___y_1404_, 9);
v_quotContext_1417_ = lean_ctor_get(v___y_1404_, 10);
v_currMacroScope_1418_ = lean_ctor_get(v___y_1404_, 11);
v_diag_1419_ = lean_ctor_get_uint8(v___y_1404_, sizeof(void*)*14);
v_cancelTk_x3f_1420_ = lean_ctor_get(v___y_1404_, 12);
v_suppressElabErrors_1421_ = lean_ctor_get_uint8(v___y_1404_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1422_ = lean_ctor_get(v___y_1404_, 13);
v___x_1423_ = lean_st_ref_get(v___y_1405_);
v_traceState_1424_ = lean_ctor_get(v___x_1423_, 4);
lean_inc_ref(v_traceState_1424_);
lean_dec(v___x_1423_);
v_traces_1425_ = lean_ctor_get(v_traceState_1424_, 0);
lean_inc_ref(v_traces_1425_);
lean_dec_ref(v_traceState_1424_);
v_ref_1426_ = l_Lean_replaceRef(v_ref_1400_, v_ref_1412_);
lean_inc_ref(v_inheritedTraceOptions_1422_);
lean_inc(v_cancelTk_x3f_1420_);
lean_inc(v_currMacroScope_1418_);
lean_inc(v_quotContext_1417_);
lean_inc(v_maxHeartbeats_1416_);
lean_inc(v_initHeartbeats_1415_);
lean_inc(v_openDecls_1414_);
lean_inc(v_currNamespace_1413_);
lean_inc(v_maxRecDepth_1411_);
lean_inc(v_currRecDepth_1410_);
lean_inc_ref(v_options_1409_);
lean_inc_ref(v_fileMap_1408_);
lean_inc_ref(v_fileName_1407_);
v___x_1427_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1427_, 0, v_fileName_1407_);
lean_ctor_set(v___x_1427_, 1, v_fileMap_1408_);
lean_ctor_set(v___x_1427_, 2, v_options_1409_);
lean_ctor_set(v___x_1427_, 3, v_currRecDepth_1410_);
lean_ctor_set(v___x_1427_, 4, v_maxRecDepth_1411_);
lean_ctor_set(v___x_1427_, 5, v_ref_1426_);
lean_ctor_set(v___x_1427_, 6, v_currNamespace_1413_);
lean_ctor_set(v___x_1427_, 7, v_openDecls_1414_);
lean_ctor_set(v___x_1427_, 8, v_initHeartbeats_1415_);
lean_ctor_set(v___x_1427_, 9, v_maxHeartbeats_1416_);
lean_ctor_set(v___x_1427_, 10, v_quotContext_1417_);
lean_ctor_set(v___x_1427_, 11, v_currMacroScope_1418_);
lean_ctor_set(v___x_1427_, 12, v_cancelTk_x3f_1420_);
lean_ctor_set(v___x_1427_, 13, v_inheritedTraceOptions_1422_);
lean_ctor_set_uint8(v___x_1427_, sizeof(void*)*14, v_diag_1419_);
lean_ctor_set_uint8(v___x_1427_, sizeof(void*)*14 + 1, v_suppressElabErrors_1421_);
v___x_1428_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1425_);
lean_dec_ref(v_traces_1425_);
v_sz_1429_ = lean_array_size(v___x_1428_);
v___x_1430_ = ((size_t)0ULL);
v___x_1431_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6(v_sz_1429_, v___x_1430_, v___x_1428_);
v_msg_1432_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1432_, 0, v_data_1399_);
lean_ctor_set(v_msg_1432_, 1, v_msg_1401_);
lean_ctor_set(v_msg_1432_, 2, v___x_1431_);
v___x_1433_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(v_msg_1432_, v___y_1402_, v___y_1403_, v___x_1427_, v___y_1405_);
lean_dec_ref_known(v___x_1427_, 14);
v_a_1434_ = lean_ctor_get(v___x_1433_, 0);
v_isSharedCheck_1471_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1436_ = v___x_1433_;
v_isShared_1437_ = v_isSharedCheck_1471_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_a_1434_);
lean_dec(v___x_1433_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1471_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v___x_1438_; lean_object* v_traceState_1439_; lean_object* v_env_1440_; lean_object* v_nextMacroScope_1441_; lean_object* v_ngen_1442_; lean_object* v_auxDeclNGen_1443_; lean_object* v_cache_1444_; lean_object* v_messages_1445_; lean_object* v_infoState_1446_; lean_object* v_snapshotTasks_1447_; lean_object* v___x_1449_; uint8_t v_isShared_1450_; uint8_t v_isSharedCheck_1470_; 
v___x_1438_ = lean_st_ref_take(v___y_1405_);
v_traceState_1439_ = lean_ctor_get(v___x_1438_, 4);
v_env_1440_ = lean_ctor_get(v___x_1438_, 0);
v_nextMacroScope_1441_ = lean_ctor_get(v___x_1438_, 1);
v_ngen_1442_ = lean_ctor_get(v___x_1438_, 2);
v_auxDeclNGen_1443_ = lean_ctor_get(v___x_1438_, 3);
v_cache_1444_ = lean_ctor_get(v___x_1438_, 5);
v_messages_1445_ = lean_ctor_get(v___x_1438_, 6);
v_infoState_1446_ = lean_ctor_get(v___x_1438_, 7);
v_snapshotTasks_1447_ = lean_ctor_get(v___x_1438_, 8);
v_isSharedCheck_1470_ = !lean_is_exclusive(v___x_1438_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1449_ = v___x_1438_;
v_isShared_1450_ = v_isSharedCheck_1470_;
goto v_resetjp_1448_;
}
else
{
lean_inc(v_snapshotTasks_1447_);
lean_inc(v_infoState_1446_);
lean_inc(v_messages_1445_);
lean_inc(v_cache_1444_);
lean_inc(v_traceState_1439_);
lean_inc(v_auxDeclNGen_1443_);
lean_inc(v_ngen_1442_);
lean_inc(v_nextMacroScope_1441_);
lean_inc(v_env_1440_);
lean_dec(v___x_1438_);
v___x_1449_ = lean_box(0);
v_isShared_1450_ = v_isSharedCheck_1470_;
goto v_resetjp_1448_;
}
v_resetjp_1448_:
{
uint64_t v_tid_1451_; lean_object* v___x_1453_; uint8_t v_isShared_1454_; uint8_t v_isSharedCheck_1468_; 
v_tid_1451_ = lean_ctor_get_uint64(v_traceState_1439_, sizeof(void*)*1);
v_isSharedCheck_1468_ = !lean_is_exclusive(v_traceState_1439_);
if (v_isSharedCheck_1468_ == 0)
{
lean_object* v_unused_1469_; 
v_unused_1469_ = lean_ctor_get(v_traceState_1439_, 0);
lean_dec(v_unused_1469_);
v___x_1453_ = v_traceState_1439_;
v_isShared_1454_ = v_isSharedCheck_1468_;
goto v_resetjp_1452_;
}
else
{
lean_dec(v_traceState_1439_);
v___x_1453_ = lean_box(0);
v_isShared_1454_ = v_isSharedCheck_1468_;
goto v_resetjp_1452_;
}
v_resetjp_1452_:
{
lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1458_; 
v___x_1455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1455_, 0, v_ref_1400_);
lean_ctor_set(v___x_1455_, 1, v_a_1434_);
v___x_1456_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1398_, v___x_1455_);
if (v_isShared_1454_ == 0)
{
lean_ctor_set(v___x_1453_, 0, v___x_1456_);
v___x_1458_ = v___x_1453_;
goto v_reusejp_1457_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v___x_1456_);
lean_ctor_set_uint64(v_reuseFailAlloc_1467_, sizeof(void*)*1, v_tid_1451_);
v___x_1458_ = v_reuseFailAlloc_1467_;
goto v_reusejp_1457_;
}
v_reusejp_1457_:
{
lean_object* v___x_1460_; 
if (v_isShared_1450_ == 0)
{
lean_ctor_set(v___x_1449_, 4, v___x_1458_);
v___x_1460_ = v___x_1449_;
goto v_reusejp_1459_;
}
else
{
lean_object* v_reuseFailAlloc_1466_; 
v_reuseFailAlloc_1466_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1466_, 0, v_env_1440_);
lean_ctor_set(v_reuseFailAlloc_1466_, 1, v_nextMacroScope_1441_);
lean_ctor_set(v_reuseFailAlloc_1466_, 2, v_ngen_1442_);
lean_ctor_set(v_reuseFailAlloc_1466_, 3, v_auxDeclNGen_1443_);
lean_ctor_set(v_reuseFailAlloc_1466_, 4, v___x_1458_);
lean_ctor_set(v_reuseFailAlloc_1466_, 5, v_cache_1444_);
lean_ctor_set(v_reuseFailAlloc_1466_, 6, v_messages_1445_);
lean_ctor_set(v_reuseFailAlloc_1466_, 7, v_infoState_1446_);
lean_ctor_set(v_reuseFailAlloc_1466_, 8, v_snapshotTasks_1447_);
v___x_1460_ = v_reuseFailAlloc_1466_;
goto v_reusejp_1459_;
}
v_reusejp_1459_:
{
lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1464_; 
v___x_1461_ = lean_st_ref_set(v___y_1405_, v___x_1460_);
v___x_1462_ = lean_box(0);
if (v_isShared_1437_ == 0)
{
lean_ctor_set(v___x_1436_, 0, v___x_1462_);
v___x_1464_ = v___x_1436_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1465_; 
v_reuseFailAlloc_1465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1465_, 0, v___x_1462_);
v___x_1464_ = v_reuseFailAlloc_1465_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
return v___x_1464_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5___boxed(lean_object* v_oldTraces_1472_, lean_object* v_data_1473_, lean_object* v_ref_1474_, lean_object* v_msg_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_){
_start:
{
lean_object* v_res_1481_; 
v_res_1481_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5(v_oldTraces_1472_, v_data_1473_, v_ref_1474_, v_msg_1475_, v___y_1476_, v___y_1477_, v___y_1478_, v___y_1479_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v___y_1477_);
lean_dec_ref(v___y_1476_);
return v_res_1481_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(lean_object* v_opts_1482_, lean_object* v_opt_1483_){
_start:
{
lean_object* v_name_1484_; lean_object* v_defValue_1485_; lean_object* v_map_1486_; lean_object* v___x_1487_; 
v_name_1484_ = lean_ctor_get(v_opt_1483_, 0);
v_defValue_1485_ = lean_ctor_get(v_opt_1483_, 1);
v_map_1486_ = lean_ctor_get(v_opts_1482_, 0);
v___x_1487_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1486_, v_name_1484_);
if (lean_obj_tag(v___x_1487_) == 0)
{
lean_inc(v_defValue_1485_);
return v_defValue_1485_;
}
else
{
lean_object* v_val_1488_; 
v_val_1488_ = lean_ctor_get(v___x_1487_, 0);
lean_inc(v_val_1488_);
lean_dec_ref_known(v___x_1487_, 1);
if (lean_obj_tag(v_val_1488_) == 3)
{
lean_object* v_v_1489_; 
v_v_1489_ = lean_ctor_get(v_val_1488_, 0);
lean_inc(v_v_1489_);
lean_dec_ref_known(v_val_1488_, 1);
return v_v_1489_;
}
else
{
lean_dec(v_val_1488_);
lean_inc(v_defValue_1485_);
return v_defValue_1485_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8___boxed(lean_object* v_opts_1490_, lean_object* v_opt_1491_){
_start:
{
lean_object* v_res_1492_; 
v_res_1492_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(v_opts_1490_, v_opt_1491_);
lean_dec_ref(v_opt_1491_);
lean_dec_ref(v_opts_1490_);
return v_res_1492_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__7(lean_object* v_e_1493_){
_start:
{
if (lean_obj_tag(v_e_1493_) == 0)
{
uint8_t v___x_1494_; 
v___x_1494_ = 2;
return v___x_1494_;
}
else
{
uint8_t v___x_1495_; 
v___x_1495_ = 0;
return v___x_1495_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__7___boxed(lean_object* v_e_1496_){
_start:
{
uint8_t v_res_1497_; lean_object* v_r_1498_; 
v_res_1497_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__7(v_e_1496_);
lean_dec_ref(v_e_1496_);
v_r_1498_ = lean_box(v_res_1497_);
return v_r_1498_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg(lean_object* v_x_1499_){
_start:
{
if (lean_obj_tag(v_x_1499_) == 0)
{
lean_object* v_a_1501_; lean_object* v___x_1503_; uint8_t v_isShared_1504_; uint8_t v_isSharedCheck_1508_; 
v_a_1501_ = lean_ctor_get(v_x_1499_, 0);
v_isSharedCheck_1508_ = !lean_is_exclusive(v_x_1499_);
if (v_isSharedCheck_1508_ == 0)
{
v___x_1503_ = v_x_1499_;
v_isShared_1504_ = v_isSharedCheck_1508_;
goto v_resetjp_1502_;
}
else
{
lean_inc(v_a_1501_);
lean_dec(v_x_1499_);
v___x_1503_ = lean_box(0);
v_isShared_1504_ = v_isSharedCheck_1508_;
goto v_resetjp_1502_;
}
v_resetjp_1502_:
{
lean_object* v___x_1506_; 
if (v_isShared_1504_ == 0)
{
lean_ctor_set_tag(v___x_1503_, 1);
v___x_1506_ = v___x_1503_;
goto v_reusejp_1505_;
}
else
{
lean_object* v_reuseFailAlloc_1507_; 
v_reuseFailAlloc_1507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1507_, 0, v_a_1501_);
v___x_1506_ = v_reuseFailAlloc_1507_;
goto v_reusejp_1505_;
}
v_reusejp_1505_:
{
return v___x_1506_;
}
}
}
else
{
lean_object* v_a_1509_; lean_object* v___x_1511_; uint8_t v_isShared_1512_; uint8_t v_isSharedCheck_1516_; 
v_a_1509_ = lean_ctor_get(v_x_1499_, 0);
v_isSharedCheck_1516_ = !lean_is_exclusive(v_x_1499_);
if (v_isSharedCheck_1516_ == 0)
{
v___x_1511_ = v_x_1499_;
v_isShared_1512_ = v_isSharedCheck_1516_;
goto v_resetjp_1510_;
}
else
{
lean_inc(v_a_1509_);
lean_dec(v_x_1499_);
v___x_1511_ = lean_box(0);
v_isShared_1512_ = v_isSharedCheck_1516_;
goto v_resetjp_1510_;
}
v_resetjp_1510_:
{
lean_object* v___x_1514_; 
if (v_isShared_1512_ == 0)
{
lean_ctor_set_tag(v___x_1511_, 0);
v___x_1514_ = v___x_1511_;
goto v_reusejp_1513_;
}
else
{
lean_object* v_reuseFailAlloc_1515_; 
v_reuseFailAlloc_1515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1515_, 0, v_a_1509_);
v___x_1514_ = v_reuseFailAlloc_1515_;
goto v_reusejp_1513_;
}
v_reusejp_1513_:
{
return v___x_1514_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg___boxed(lean_object* v_x_1517_, lean_object* v___y_1518_){
_start:
{
lean_object* v_res_1519_; 
v_res_1519_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg(v_x_1517_);
return v_res_1519_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1(void){
_start:
{
lean_object* v___x_1521_; lean_object* v___x_1522_; 
v___x_1521_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__0));
v___x_1522_ = l_Lean_stringToMessageData(v___x_1521_);
return v___x_1522_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2(void){
_start:
{
lean_object* v___x_1523_; double v___x_1524_; 
v___x_1523_ = lean_unsigned_to_nat(1000u);
v___x_1524_ = lean_float_of_nat(v___x_1523_);
return v___x_1524_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(lean_object* v_cls_1525_, uint8_t v_collapsed_1526_, lean_object* v_tag_1527_, lean_object* v_opts_1528_, uint8_t v_clsEnabled_1529_, lean_object* v_oldTraces_1530_, lean_object* v_msg_1531_, lean_object* v_resStartStop_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_){
_start:
{
lean_object* v_fst_1538_; lean_object* v_snd_1539_; lean_object* v___y_1541_; lean_object* v___y_1542_; lean_object* v_data_1543_; lean_object* v_fst_1546_; lean_object* v_snd_1547_; lean_object* v___x_1548_; uint8_t v___x_1549_; lean_object* v___y_1551_; lean_object* v_a_1552_; uint8_t v___y_1567_; double v___y_1598_; 
v_fst_1538_ = lean_ctor_get(v_resStartStop_1532_, 0);
lean_inc(v_fst_1538_);
v_snd_1539_ = lean_ctor_get(v_resStartStop_1532_, 1);
lean_inc(v_snd_1539_);
lean_dec_ref(v_resStartStop_1532_);
v_fst_1546_ = lean_ctor_get(v_snd_1539_, 0);
lean_inc(v_fst_1546_);
v_snd_1547_ = lean_ctor_get(v_snd_1539_, 1);
lean_inc(v_snd_1547_);
lean_dec(v_snd_1539_);
v___x_1548_ = l_Lean_trace_profiler;
v___x_1549_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_opts_1528_, v___x_1548_);
if (v___x_1549_ == 0)
{
v___y_1567_ = v___x_1549_;
goto v___jp_1566_;
}
else
{
lean_object* v___x_1603_; uint8_t v___x_1604_; 
v___x_1603_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1604_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_opts_1528_, v___x_1603_);
if (v___x_1604_ == 0)
{
lean_object* v___x_1605_; lean_object* v___x_1606_; double v___x_1607_; double v___x_1608_; double v___x_1609_; 
v___x_1605_ = l_Lean_trace_profiler_threshold;
v___x_1606_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(v_opts_1528_, v___x_1605_);
v___x_1607_ = lean_float_of_nat(v___x_1606_);
v___x_1608_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2);
v___x_1609_ = lean_float_div(v___x_1607_, v___x_1608_);
v___y_1598_ = v___x_1609_;
goto v___jp_1597_;
}
else
{
lean_object* v___x_1610_; lean_object* v___x_1611_; double v___x_1612_; 
v___x_1610_ = l_Lean_trace_profiler_threshold;
v___x_1611_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(v_opts_1528_, v___x_1610_);
v___x_1612_ = lean_float_of_nat(v___x_1611_);
v___y_1598_ = v___x_1612_;
goto v___jp_1597_;
}
}
v___jp_1540_:
{
lean_object* v___x_1544_; 
lean_inc(v___y_1541_);
v___x_1544_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5(v_oldTraces_1530_, v_data_1543_, v___y_1541_, v___y_1542_, v___y_1533_, v___y_1534_, v___y_1535_, v___y_1536_);
if (lean_obj_tag(v___x_1544_) == 0)
{
lean_object* v___x_1545_; 
lean_dec_ref_known(v___x_1544_, 1);
v___x_1545_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg(v_fst_1538_);
return v___x_1545_;
}
else
{
lean_dec(v_fst_1538_);
return v___x_1544_;
}
}
v___jp_1550_:
{
uint8_t v_result_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; double v___x_1556_; lean_object* v_data_1557_; 
v_result_1553_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__7(v_fst_1538_);
v___x_1554_ = lean_box(v_result_1553_);
v___x_1555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1555_, 0, v___x_1554_);
v___x_1556_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0);
lean_inc_ref(v_tag_1527_);
lean_inc_ref(v___x_1555_);
lean_inc(v_cls_1525_);
v_data_1557_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1557_, 0, v_cls_1525_);
lean_ctor_set(v_data_1557_, 1, v___x_1555_);
lean_ctor_set(v_data_1557_, 2, v_tag_1527_);
lean_ctor_set_float(v_data_1557_, sizeof(void*)*3, v___x_1556_);
lean_ctor_set_float(v_data_1557_, sizeof(void*)*3 + 8, v___x_1556_);
lean_ctor_set_uint8(v_data_1557_, sizeof(void*)*3 + 16, v_collapsed_1526_);
if (v___x_1549_ == 0)
{
lean_dec_ref_known(v___x_1555_, 1);
lean_dec(v_snd_1547_);
lean_dec(v_fst_1546_);
lean_dec_ref(v_tag_1527_);
lean_dec(v_cls_1525_);
v___y_1541_ = v___y_1551_;
v___y_1542_ = v_a_1552_;
v_data_1543_ = v_data_1557_;
goto v___jp_1540_;
}
else
{
lean_object* v_data_1558_; double v___x_1559_; double v___x_1560_; 
lean_dec_ref_known(v_data_1557_, 3);
v_data_1558_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1558_, 0, v_cls_1525_);
lean_ctor_set(v_data_1558_, 1, v___x_1555_);
lean_ctor_set(v_data_1558_, 2, v_tag_1527_);
v___x_1559_ = lean_unbox_float(v_fst_1546_);
lean_dec(v_fst_1546_);
lean_ctor_set_float(v_data_1558_, sizeof(void*)*3, v___x_1559_);
v___x_1560_ = lean_unbox_float(v_snd_1547_);
lean_dec(v_snd_1547_);
lean_ctor_set_float(v_data_1558_, sizeof(void*)*3 + 8, v___x_1560_);
lean_ctor_set_uint8(v_data_1558_, sizeof(void*)*3 + 16, v_collapsed_1526_);
v___y_1541_ = v___y_1551_;
v___y_1542_ = v_a_1552_;
v_data_1543_ = v_data_1558_;
goto v___jp_1540_;
}
}
v___jp_1561_:
{
lean_object* v_ref_1562_; lean_object* v___x_1563_; 
v_ref_1562_ = lean_ctor_get(v___y_1535_, 5);
lean_inc(v___y_1536_);
lean_inc_ref(v___y_1535_);
lean_inc(v___y_1534_);
lean_inc_ref(v___y_1533_);
lean_inc(v_fst_1538_);
v___x_1563_ = lean_apply_6(v_msg_1531_, v_fst_1538_, v___y_1533_, v___y_1534_, v___y_1535_, v___y_1536_, lean_box(0));
if (lean_obj_tag(v___x_1563_) == 0)
{
lean_object* v_a_1564_; 
v_a_1564_ = lean_ctor_get(v___x_1563_, 0);
lean_inc(v_a_1564_);
lean_dec_ref_known(v___x_1563_, 1);
v___y_1551_ = v_ref_1562_;
v_a_1552_ = v_a_1564_;
goto v___jp_1550_;
}
else
{
lean_object* v___x_1565_; 
lean_dec_ref_known(v___x_1563_, 1);
v___x_1565_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1);
v___y_1551_ = v_ref_1562_;
v_a_1552_ = v___x_1565_;
goto v___jp_1550_;
}
}
v___jp_1566_:
{
if (v_clsEnabled_1529_ == 0)
{
if (v___y_1567_ == 0)
{
lean_object* v___x_1568_; lean_object* v_traceState_1569_; lean_object* v_env_1570_; lean_object* v_nextMacroScope_1571_; lean_object* v_ngen_1572_; lean_object* v_auxDeclNGen_1573_; lean_object* v_cache_1574_; lean_object* v_messages_1575_; lean_object* v_infoState_1576_; lean_object* v_snapshotTasks_1577_; lean_object* v___x_1579_; uint8_t v_isShared_1580_; uint8_t v_isSharedCheck_1596_; 
lean_dec(v_snd_1547_);
lean_dec(v_fst_1546_);
lean_dec_ref(v_msg_1531_);
lean_dec_ref(v_tag_1527_);
lean_dec(v_cls_1525_);
v___x_1568_ = lean_st_ref_take(v___y_1536_);
v_traceState_1569_ = lean_ctor_get(v___x_1568_, 4);
v_env_1570_ = lean_ctor_get(v___x_1568_, 0);
v_nextMacroScope_1571_ = lean_ctor_get(v___x_1568_, 1);
v_ngen_1572_ = lean_ctor_get(v___x_1568_, 2);
v_auxDeclNGen_1573_ = lean_ctor_get(v___x_1568_, 3);
v_cache_1574_ = lean_ctor_get(v___x_1568_, 5);
v_messages_1575_ = lean_ctor_get(v___x_1568_, 6);
v_infoState_1576_ = lean_ctor_get(v___x_1568_, 7);
v_snapshotTasks_1577_ = lean_ctor_get(v___x_1568_, 8);
v_isSharedCheck_1596_ = !lean_is_exclusive(v___x_1568_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1579_ = v___x_1568_;
v_isShared_1580_ = v_isSharedCheck_1596_;
goto v_resetjp_1578_;
}
else
{
lean_inc(v_snapshotTasks_1577_);
lean_inc(v_infoState_1576_);
lean_inc(v_messages_1575_);
lean_inc(v_cache_1574_);
lean_inc(v_traceState_1569_);
lean_inc(v_auxDeclNGen_1573_);
lean_inc(v_ngen_1572_);
lean_inc(v_nextMacroScope_1571_);
lean_inc(v_env_1570_);
lean_dec(v___x_1568_);
v___x_1579_ = lean_box(0);
v_isShared_1580_ = v_isSharedCheck_1596_;
goto v_resetjp_1578_;
}
v_resetjp_1578_:
{
uint64_t v_tid_1581_; lean_object* v_traces_1582_; lean_object* v___x_1584_; uint8_t v_isShared_1585_; uint8_t v_isSharedCheck_1595_; 
v_tid_1581_ = lean_ctor_get_uint64(v_traceState_1569_, sizeof(void*)*1);
v_traces_1582_ = lean_ctor_get(v_traceState_1569_, 0);
v_isSharedCheck_1595_ = !lean_is_exclusive(v_traceState_1569_);
if (v_isSharedCheck_1595_ == 0)
{
v___x_1584_ = v_traceState_1569_;
v_isShared_1585_ = v_isSharedCheck_1595_;
goto v_resetjp_1583_;
}
else
{
lean_inc(v_traces_1582_);
lean_dec(v_traceState_1569_);
v___x_1584_ = lean_box(0);
v_isShared_1585_ = v_isSharedCheck_1595_;
goto v_resetjp_1583_;
}
v_resetjp_1583_:
{
lean_object* v___x_1586_; lean_object* v___x_1588_; 
v___x_1586_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1530_, v_traces_1582_);
lean_dec_ref(v_traces_1582_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 0, v___x_1586_);
v___x_1588_ = v___x_1584_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v___x_1586_);
lean_ctor_set_uint64(v_reuseFailAlloc_1594_, sizeof(void*)*1, v_tid_1581_);
v___x_1588_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
lean_object* v___x_1590_; 
if (v_isShared_1580_ == 0)
{
lean_ctor_set(v___x_1579_, 4, v___x_1588_);
v___x_1590_ = v___x_1579_;
goto v_reusejp_1589_;
}
else
{
lean_object* v_reuseFailAlloc_1593_; 
v_reuseFailAlloc_1593_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1593_, 0, v_env_1570_);
lean_ctor_set(v_reuseFailAlloc_1593_, 1, v_nextMacroScope_1571_);
lean_ctor_set(v_reuseFailAlloc_1593_, 2, v_ngen_1572_);
lean_ctor_set(v_reuseFailAlloc_1593_, 3, v_auxDeclNGen_1573_);
lean_ctor_set(v_reuseFailAlloc_1593_, 4, v___x_1588_);
lean_ctor_set(v_reuseFailAlloc_1593_, 5, v_cache_1574_);
lean_ctor_set(v_reuseFailAlloc_1593_, 6, v_messages_1575_);
lean_ctor_set(v_reuseFailAlloc_1593_, 7, v_infoState_1576_);
lean_ctor_set(v_reuseFailAlloc_1593_, 8, v_snapshotTasks_1577_);
v___x_1590_ = v_reuseFailAlloc_1593_;
goto v_reusejp_1589_;
}
v_reusejp_1589_:
{
lean_object* v___x_1591_; lean_object* v___x_1592_; 
v___x_1591_ = lean_st_ref_set(v___y_1536_, v___x_1590_);
v___x_1592_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg(v_fst_1538_);
return v___x_1592_;
}
}
}
}
}
else
{
goto v___jp_1561_;
}
}
else
{
goto v___jp_1561_;
}
}
v___jp_1597_:
{
double v___x_1599_; double v___x_1600_; double v___x_1601_; uint8_t v___x_1602_; 
v___x_1599_ = lean_unbox_float(v_snd_1547_);
v___x_1600_ = lean_unbox_float(v_fst_1546_);
v___x_1601_ = lean_float_sub(v___x_1599_, v___x_1600_);
v___x_1602_ = lean_float_decLt(v___y_1598_, v___x_1601_);
v___y_1567_ = v___x_1602_;
goto v___jp_1566_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___boxed(lean_object* v_cls_1613_, lean_object* v_collapsed_1614_, lean_object* v_tag_1615_, lean_object* v_opts_1616_, lean_object* v_clsEnabled_1617_, lean_object* v_oldTraces_1618_, lean_object* v_msg_1619_, lean_object* v_resStartStop_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
uint8_t v_collapsed_boxed_1626_; uint8_t v_clsEnabled_boxed_1627_; lean_object* v_res_1628_; 
v_collapsed_boxed_1626_ = lean_unbox(v_collapsed_1614_);
v_clsEnabled_boxed_1627_ = lean_unbox(v_clsEnabled_1617_);
v_res_1628_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(v_cls_1613_, v_collapsed_boxed_1626_, v_tag_1615_, v_opts_1616_, v_clsEnabled_boxed_1627_, v_oldTraces_1618_, v_msg_1619_, v_resStartStop_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_);
lean_dec(v___y_1624_);
lean_dec_ref(v___y_1623_);
lean_dec(v___y_1622_);
lean_dec_ref(v___y_1621_);
lean_dec_ref(v_opts_1616_);
return v_res_1628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__7(lean_object* v_x_1629_, lean_object* v_x_1630_){
_start:
{
if (lean_obj_tag(v_x_1630_) == 0)
{
return v_x_1629_;
}
else
{
lean_object* v_key_1631_; lean_object* v_tail_1632_; lean_object* v___x_1633_; 
v_key_1631_ = lean_ctor_get(v_x_1630_, 0);
lean_inc(v_key_1631_);
v_tail_1632_ = lean_ctor_get(v_x_1630_, 2);
lean_inc(v_tail_1632_);
lean_dec_ref_known(v_x_1630_, 3);
v___x_1633_ = lean_array_push(v_x_1629_, v_key_1631_);
v_x_1629_ = v___x_1633_;
v_x_1630_ = v_tail_1632_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(lean_object* v_as_1635_, size_t v_i_1636_, size_t v_stop_1637_, lean_object* v_b_1638_){
_start:
{
uint8_t v___x_1639_; 
v___x_1639_ = lean_usize_dec_eq(v_i_1636_, v_stop_1637_);
if (v___x_1639_ == 0)
{
lean_object* v___x_1640_; lean_object* v___x_1641_; size_t v___x_1642_; size_t v___x_1643_; 
v___x_1640_ = lean_array_uget_borrowed(v_as_1635_, v_i_1636_);
lean_inc(v___x_1640_);
v___x_1641_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__7(v_b_1638_, v___x_1640_);
v___x_1642_ = ((size_t)1ULL);
v___x_1643_ = lean_usize_add(v_i_1636_, v___x_1642_);
v_i_1636_ = v___x_1643_;
v_b_1638_ = v___x_1641_;
goto _start;
}
else
{
return v_b_1638_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8___boxed(lean_object* v_as_1645_, lean_object* v_i_1646_, lean_object* v_stop_1647_, lean_object* v_b_1648_){
_start:
{
size_t v_i_boxed_1649_; size_t v_stop_boxed_1650_; lean_object* v_res_1651_; 
v_i_boxed_1649_ = lean_unbox_usize(v_i_1646_);
lean_dec(v_i_1646_);
v_stop_boxed_1650_ = lean_unbox_usize(v_stop_1647_);
lean_dec(v_stop_1647_);
v_res_1651_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(v_as_1645_, v_i_boxed_1649_, v_stop_boxed_1650_, v_b_1648_);
lean_dec_ref(v_as_1645_);
return v_res_1651_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg(lean_object* v_hi_1652_, lean_object* v_pivot_1653_, lean_object* v_as_1654_, lean_object* v_i_1655_, lean_object* v_k_1656_){
_start:
{
uint8_t v___x_1657_; 
v___x_1657_ = lean_nat_dec_lt(v_k_1656_, v_hi_1652_);
if (v___x_1657_ == 0)
{
lean_object* v___x_1658_; lean_object* v___x_1659_; 
lean_dec(v_k_1656_);
v___x_1658_ = lean_array_fswap(v_as_1654_, v_i_1655_, v_hi_1652_);
v___x_1659_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1659_, 0, v_i_1655_);
lean_ctor_set(v___x_1659_, 1, v___x_1658_);
return v___x_1659_;
}
else
{
lean_object* v___x_1660_; uint8_t v___x_1661_; 
v___x_1660_ = lean_array_fget_borrowed(v_as_1654_, v_k_1656_);
v___x_1661_ = lp_aesop_Aesop_instOrdPremiseIndex_ord(v___x_1660_, v_pivot_1653_);
if (v___x_1661_ == 0)
{
lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1662_ = lean_array_fswap(v_as_1654_, v_i_1655_, v_k_1656_);
v___x_1663_ = lean_unsigned_to_nat(1u);
v___x_1664_ = lean_nat_add(v_i_1655_, v___x_1663_);
lean_dec(v_i_1655_);
v___x_1665_ = lean_nat_add(v_k_1656_, v___x_1663_);
lean_dec(v_k_1656_);
v_as_1654_ = v___x_1662_;
v_i_1655_ = v___x_1664_;
v_k_1656_ = v___x_1665_;
goto _start;
}
else
{
lean_object* v___x_1667_; lean_object* v___x_1668_; 
v___x_1667_ = lean_unsigned_to_nat(1u);
v___x_1668_ = lean_nat_add(v_k_1656_, v___x_1667_);
lean_dec(v_k_1656_);
v_k_1656_ = v___x_1668_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg___boxed(lean_object* v_hi_1670_, lean_object* v_pivot_1671_, lean_object* v_as_1672_, lean_object* v_i_1673_, lean_object* v_k_1674_){
_start:
{
lean_object* v_res_1675_; 
v_res_1675_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg(v_hi_1670_, v_pivot_1671_, v_as_1672_, v_i_1673_, v_k_1674_);
lean_dec(v_pivot_1671_);
lean_dec(v_hi_1670_);
return v_res_1675_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0(uint8_t v___x_1676_, lean_object* v_x_1677_, lean_object* v_y_1678_){
_start:
{
uint8_t v___x_1679_; 
v___x_1679_ = lp_aesop_Aesop_instOrdPremiseIndex_ord(v_x_1677_, v_y_1678_);
if (v___x_1679_ == 0)
{
return v___x_1676_;
}
else
{
uint8_t v___x_1680_; 
v___x_1680_ = 0;
return v___x_1680_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0___boxed(lean_object* v___x_1681_, lean_object* v_x_1682_, lean_object* v_y_1683_){
_start:
{
uint8_t v___x_58689__boxed_1684_; uint8_t v_res_1685_; lean_object* v_r_1686_; 
v___x_58689__boxed_1684_ = lean_unbox(v___x_1681_);
v_res_1685_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0(v___x_58689__boxed_1684_, v_x_1682_, v_y_1683_);
lean_dec(v_y_1683_);
lean_dec(v_x_1682_);
v_r_1686_ = lean_box(v_res_1685_);
return v_r_1686_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(lean_object* v_n_1687_, lean_object* v_as_1688_, lean_object* v_lo_1689_, lean_object* v_hi_1690_){
_start:
{
lean_object* v___y_1692_; uint8_t v___x_1702_; 
v___x_1702_ = lean_nat_dec_lt(v_lo_1689_, v_hi_1690_);
if (v___x_1702_ == 0)
{
lean_dec(v_lo_1689_);
return v_as_1688_;
}
else
{
lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v_mid_1705_; lean_object* v___y_1707_; lean_object* v___y_1713_; lean_object* v___x_1718_; lean_object* v___x_1719_; uint8_t v___x_1720_; 
v___x_1703_ = lean_nat_add(v_lo_1689_, v_hi_1690_);
v___x_1704_ = lean_unsigned_to_nat(1u);
v_mid_1705_ = lean_nat_shiftr(v___x_1703_, v___x_1704_);
lean_dec(v___x_1703_);
v___x_1718_ = lean_array_fget_borrowed(v_as_1688_, v_mid_1705_);
v___x_1719_ = lean_array_fget_borrowed(v_as_1688_, v_lo_1689_);
v___x_1720_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0(v___x_1702_, v___x_1718_, v___x_1719_);
if (v___x_1720_ == 0)
{
v___y_1713_ = v_as_1688_;
goto v___jp_1712_;
}
else
{
lean_object* v___x_1721_; 
v___x_1721_ = lean_array_fswap(v_as_1688_, v_lo_1689_, v_mid_1705_);
v___y_1713_ = v___x_1721_;
goto v___jp_1712_;
}
v___jp_1706_:
{
lean_object* v___x_1708_; lean_object* v___x_1709_; uint8_t v___x_1710_; 
v___x_1708_ = lean_array_fget_borrowed(v___y_1707_, v_mid_1705_);
v___x_1709_ = lean_array_fget_borrowed(v___y_1707_, v_hi_1690_);
v___x_1710_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0(v___x_1702_, v___x_1708_, v___x_1709_);
if (v___x_1710_ == 0)
{
lean_dec(v_mid_1705_);
v___y_1692_ = v___y_1707_;
goto v___jp_1691_;
}
else
{
lean_object* v___x_1711_; 
v___x_1711_ = lean_array_fswap(v___y_1707_, v_mid_1705_, v_hi_1690_);
lean_dec(v_mid_1705_);
v___y_1692_ = v___x_1711_;
goto v___jp_1691_;
}
}
v___jp_1712_:
{
lean_object* v___x_1714_; lean_object* v___x_1715_; uint8_t v___x_1716_; 
v___x_1714_ = lean_array_fget_borrowed(v___y_1713_, v_hi_1690_);
v___x_1715_ = lean_array_fget_borrowed(v___y_1713_, v_lo_1689_);
v___x_1716_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___lam__0(v___x_1702_, v___x_1714_, v___x_1715_);
if (v___x_1716_ == 0)
{
v___y_1707_ = v___y_1713_;
goto v___jp_1706_;
}
else
{
lean_object* v___x_1717_; 
v___x_1717_ = lean_array_fswap(v___y_1713_, v_lo_1689_, v_hi_1690_);
v___y_1707_ = v___x_1717_;
goto v___jp_1706_;
}
}
}
v___jp_1691_:
{
lean_object* v_pivot_1693_; lean_object* v___x_1694_; lean_object* v_fst_1695_; lean_object* v_snd_1696_; uint8_t v___x_1697_; 
v_pivot_1693_ = lean_array_fget(v___y_1692_, v_hi_1690_);
lean_inc_n(v_lo_1689_, 2);
v___x_1694_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg(v_hi_1690_, v_pivot_1693_, v___y_1692_, v_lo_1689_, v_lo_1689_);
lean_dec(v_pivot_1693_);
v_fst_1695_ = lean_ctor_get(v___x_1694_, 0);
lean_inc(v_fst_1695_);
v_snd_1696_ = lean_ctor_get(v___x_1694_, 1);
lean_inc(v_snd_1696_);
lean_dec_ref(v___x_1694_);
v___x_1697_ = lean_nat_dec_le(v_hi_1690_, v_fst_1695_);
if (v___x_1697_ == 0)
{
lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; 
v___x_1698_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(v_n_1687_, v_snd_1696_, v_lo_1689_, v_fst_1695_);
v___x_1699_ = lean_unsigned_to_nat(1u);
v___x_1700_ = lean_nat_add(v_fst_1695_, v___x_1699_);
lean_dec(v_fst_1695_);
v_as_1688_ = v___x_1698_;
v_lo_1689_ = v___x_1700_;
goto _start;
}
else
{
lean_dec(v_fst_1695_);
lean_dec(v_lo_1689_);
return v_snd_1696_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg___boxed(lean_object* v_n_1722_, lean_object* v_as_1723_, lean_object* v_lo_1724_, lean_object* v_hi_1725_){
_start:
{
lean_object* v_res_1726_; 
v_res_1726_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(v_n_1722_, v_as_1723_, v_lo_1724_, v_hi_1725_);
lean_dec(v_hi_1725_);
lean_dec(v_n_1722_);
return v_res_1726_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6(lean_object* v_xs_1727_){
_start:
{
lean_object* v___x_1728_; lean_object* v___x_1729_; uint8_t v___x_1730_; 
v___x_1728_ = lean_array_get_size(v_xs_1727_);
v___x_1729_ = lean_unsigned_to_nat(0u);
v___x_1730_ = lean_nat_dec_eq(v___x_1728_, v___x_1729_);
if (v___x_1730_ == 0)
{
lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___y_1734_; uint8_t v___x_1738_; 
v___x_1731_ = lean_unsigned_to_nat(1u);
v___x_1732_ = lean_nat_sub(v___x_1728_, v___x_1731_);
v___x_1738_ = lean_nat_dec_le(v___x_1729_, v___x_1732_);
if (v___x_1738_ == 0)
{
lean_inc(v___x_1732_);
v___y_1734_ = v___x_1732_;
goto v___jp_1733_;
}
else
{
v___y_1734_ = v___x_1729_;
goto v___jp_1733_;
}
v___jp_1733_:
{
uint8_t v___x_1735_; 
v___x_1735_ = lean_nat_dec_le(v___y_1734_, v___x_1732_);
if (v___x_1735_ == 0)
{
lean_object* v___x_1736_; 
lean_dec(v___x_1732_);
lean_inc(v___y_1734_);
v___x_1736_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(v___x_1728_, v_xs_1727_, v___y_1734_, v___y_1734_);
lean_dec(v___y_1734_);
return v___x_1736_;
}
else
{
lean_object* v___x_1737_; 
v___x_1737_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(v___x_1728_, v_xs_1727_, v___y_1734_, v___x_1732_);
lean_dec(v___x_1732_);
return v___x_1737_;
}
}
}
else
{
return v_xs_1727_;
}
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__1(void){
_start:
{
lean_object* v___x_1740_; lean_object* v___x_1741_; 
v___x_1740_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__0));
v___x_1741_ = l_Lean_stringToMessageData(v___x_1740_);
return v___x_1741_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__3(void){
_start:
{
lean_object* v___x_1743_; lean_object* v___x_1744_; 
v___x_1743_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__2));
v___x_1744_ = l_Lean_stringToMessageData(v___x_1743_);
return v___x_1744_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__5(void){
_start:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; 
v___x_1746_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__4));
v___x_1747_ = l_Lean_stringToMessageData(v___x_1746_);
return v___x_1747_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__7(void){
_start:
{
lean_object* v___x_1749_; lean_object* v___x_1750_; 
v___x_1749_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__6));
v___x_1750_ = l_Lean_stringToMessageData(v___x_1749_);
return v___x_1750_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__9(void){
_start:
{
lean_object* v___x_1752_; lean_object* v___x_1753_; 
v___x_1752_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__8));
v___x_1753_ = l_Lean_stringToMessageData(v___x_1752_);
return v___x_1753_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__11(void){
_start:
{
lean_object* v___x_1755_; lean_object* v___x_1756_; 
v___x_1755_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__10));
v___x_1756_ = l_Lean_stringToMessageData(v___x_1755_);
return v___x_1756_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(lean_object* v_as_1757_, size_t v_sz_1758_, size_t v_i_1759_, lean_object* v_b_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_){
_start:
{
uint8_t v___x_1766_; 
v___x_1766_ = lean_usize_dec_lt(v_i_1759_, v_sz_1758_);
if (v___x_1766_ == 0)
{
lean_object* v___x_1767_; 
v___x_1767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1767_, 0, v_b_1760_);
return v___x_1767_;
}
else
{
lean_object* v___x_1768_; lean_object* v_traceClass_1769_; lean_object* v_a_1770_; lean_object* v_deps_1771_; lean_object* v_index_1772_; lean_object* v_premiseIndex_1773_; lean_object* v_common_1774_; lean_object* v_forwardDeps_1775_; lean_object* v_size_1776_; lean_object* v_buckets_1777_; lean_object* v___x_1779_; uint8_t v_isShared_1780_; uint8_t v_isSharedCheck_1859_; 
v___x_1768_ = lp_aesop_Aesop_TraceOption_forward;
v_traceClass_1769_ = lean_ctor_get(v___x_1768_, 0);
v_a_1770_ = lean_array_uget_borrowed(v_as_1757_, v_i_1759_);
v_deps_1771_ = lean_ctor_get(v_a_1770_, 3);
lean_inc_ref(v_deps_1771_);
v_index_1772_ = lean_ctor_get(v_a_1770_, 1);
v_premiseIndex_1773_ = lean_ctor_get(v_a_1770_, 2);
v_common_1774_ = lean_ctor_get(v_a_1770_, 4);
lean_inc_ref(v_common_1774_);
v_forwardDeps_1775_ = lean_ctor_get(v_a_1770_, 5);
v_size_1776_ = lean_ctor_get(v_deps_1771_, 0);
v_buckets_1777_ = lean_ctor_get(v_deps_1771_, 1);
v_isSharedCheck_1859_ = !lean_is_exclusive(v_deps_1771_);
if (v_isSharedCheck_1859_ == 0)
{
v___x_1779_ = v_deps_1771_;
v_isShared_1780_ = v_isSharedCheck_1859_;
goto v_resetjp_1778_;
}
else
{
lean_inc(v_buckets_1777_);
lean_inc(v_size_1776_);
lean_dec(v_deps_1771_);
v___x_1779_ = lean_box(0);
v_isShared_1780_ = v_isSharedCheck_1859_;
goto v_resetjp_1778_;
}
v_resetjp_1778_:
{
lean_object* v___x_1781_; lean_object* v___y_1783_; lean_object* v___y_1784_; lean_object* v___y_1785_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___y_1820_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; uint8_t v___x_1851_; 
v___x_1781_ = lean_box(0);
v___x_1806_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__5, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__5_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__5);
lean_inc(v_index_1772_);
v___x_1807_ = l_Nat_reprFast(v_index_1772_);
v___x_1808_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1808_, 0, v___x_1807_);
v___x_1809_ = l_Lean_MessageData_ofFormat(v___x_1808_);
v___x_1810_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1810_, 0, v___x_1806_);
lean_ctor_set(v___x_1810_, 1, v___x_1809_);
v___x_1811_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__7, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__7_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__7);
v___x_1812_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1812_, 0, v___x_1810_);
lean_ctor_set(v___x_1812_, 1, v___x_1811_);
lean_inc(v_premiseIndex_1773_);
v___x_1813_ = l_Nat_reprFast(v_premiseIndex_1773_);
v___x_1814_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1814_, 0, v___x_1813_);
v___x_1815_ = l_Lean_MessageData_ofFormat(v___x_1814_);
v___x_1816_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1816_, 0, v___x_1812_);
lean_ctor_set(v___x_1816_, 1, v___x_1815_);
v___x_1817_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__9, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__9_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__9);
v___x_1818_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1818_, 0, v___x_1816_);
lean_ctor_set(v___x_1818_, 1, v___x_1817_);
v___x_1848_ = lean_mk_empty_array_with_capacity(v_size_1776_);
lean_dec(v_size_1776_);
v___x_1849_ = lean_unsigned_to_nat(0u);
v___x_1850_ = lean_array_get_size(v_buckets_1777_);
v___x_1851_ = lean_nat_dec_lt(v___x_1849_, v___x_1850_);
if (v___x_1851_ == 0)
{
lean_dec_ref(v_buckets_1777_);
v___y_1820_ = v___x_1848_;
goto v___jp_1819_;
}
else
{
uint8_t v___x_1852_; 
v___x_1852_ = lean_nat_dec_le(v___x_1850_, v___x_1850_);
if (v___x_1852_ == 0)
{
if (v___x_1851_ == 0)
{
lean_dec_ref(v_buckets_1777_);
v___y_1820_ = v___x_1848_;
goto v___jp_1819_;
}
else
{
size_t v___x_1853_; size_t v___x_1854_; lean_object* v___x_1855_; 
v___x_1853_ = ((size_t)0ULL);
v___x_1854_ = lean_usize_of_nat(v___x_1850_);
v___x_1855_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(v_buckets_1777_, v___x_1853_, v___x_1854_, v___x_1848_);
lean_dec_ref(v_buckets_1777_);
v___y_1820_ = v___x_1855_;
goto v___jp_1819_;
}
}
else
{
size_t v___x_1856_; size_t v___x_1857_; lean_object* v___x_1858_; 
v___x_1856_ = ((size_t)0ULL);
v___x_1857_ = lean_usize_of_nat(v___x_1850_);
v___x_1858_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(v_buckets_1777_, v___x_1856_, v___x_1857_, v___x_1848_);
lean_dec_ref(v_buckets_1777_);
v___y_1820_ = v___x_1858_;
goto v___jp_1819_;
}
}
v___jp_1782_:
{
lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1791_; 
v___x_1786_ = lp_aesop_Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6(v___y_1785_);
v___x_1787_ = lean_array_to_list(v___x_1786_);
lean_inc(v___y_1783_);
v___x_1788_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(v___x_1787_, v___y_1783_);
v___x_1789_ = l_Lean_MessageData_ofList(v___x_1788_);
if (v_isShared_1780_ == 0)
{
lean_ctor_set_tag(v___x_1779_, 7);
lean_ctor_set(v___x_1779_, 1, v___x_1789_);
lean_ctor_set(v___x_1779_, 0, v___y_1784_);
v___x_1791_ = v___x_1779_;
goto v_reusejp_1790_;
}
else
{
lean_object* v_reuseFailAlloc_1805_; 
v_reuseFailAlloc_1805_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1805_, 0, v___y_1784_);
lean_ctor_set(v_reuseFailAlloc_1805_, 1, v___x_1789_);
v___x_1791_ = v_reuseFailAlloc_1805_;
goto v_reusejp_1790_;
}
v_reusejp_1790_:
{
lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; 
v___x_1792_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__1);
v___x_1793_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1793_, 0, v___x_1791_);
lean_ctor_set(v___x_1793_, 1, v___x_1792_);
lean_inc_ref(v_forwardDeps_1775_);
v___x_1794_ = lp_aesop_Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6(v_forwardDeps_1775_);
v___x_1795_ = lean_array_to_list(v___x_1794_);
v___x_1796_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(v___x_1795_, v___y_1783_);
v___x_1797_ = l_Lean_MessageData_ofList(v___x_1796_);
v___x_1798_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1798_, 0, v___x_1793_);
lean_ctor_set(v___x_1798_, 1, v___x_1797_);
v___x_1799_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__3);
v___x_1800_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1800_, 0, v___x_1798_);
lean_ctor_set(v___x_1800_, 1, v___x_1799_);
lean_inc(v_traceClass_1769_);
v___x_1801_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v_traceClass_1769_, v___x_1800_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_);
if (lean_obj_tag(v___x_1801_) == 0)
{
size_t v___x_1802_; size_t v___x_1803_; 
lean_dec_ref_known(v___x_1801_, 1);
v___x_1802_ = ((size_t)1ULL);
v___x_1803_ = lean_usize_add(v_i_1759_, v___x_1802_);
v_i_1759_ = v___x_1803_;
v_b_1760_ = v___x_1781_;
goto _start;
}
else
{
return v___x_1801_;
}
}
}
v___jp_1819_:
{
lean_object* v_size_1821_; lean_object* v_buckets_1822_; lean_object* v___x_1824_; uint8_t v_isShared_1825_; uint8_t v_isSharedCheck_1847_; 
v_size_1821_ = lean_ctor_get(v_common_1774_, 0);
v_buckets_1822_ = lean_ctor_get(v_common_1774_, 1);
v_isSharedCheck_1847_ = !lean_is_exclusive(v_common_1774_);
if (v_isSharedCheck_1847_ == 0)
{
v___x_1824_ = v_common_1774_;
v_isShared_1825_ = v_isSharedCheck_1847_;
goto v_resetjp_1823_;
}
else
{
lean_inc(v_buckets_1822_);
lean_inc(v_size_1821_);
lean_dec(v_common_1774_);
v___x_1824_ = lean_box(0);
v_isShared_1825_ = v_isSharedCheck_1847_;
goto v_resetjp_1823_;
}
v_resetjp_1823_:
{
lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1832_; 
v___x_1826_ = lp_aesop_Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6(v___y_1820_);
v___x_1827_ = lean_array_to_list(v___x_1826_);
v___x_1828_ = lean_box(0);
v___x_1829_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(v___x_1827_, v___x_1828_);
v___x_1830_ = l_Lean_MessageData_ofList(v___x_1829_);
if (v_isShared_1825_ == 0)
{
lean_ctor_set_tag(v___x_1824_, 7);
lean_ctor_set(v___x_1824_, 1, v___x_1830_);
lean_ctor_set(v___x_1824_, 0, v___x_1818_);
v___x_1832_ = v___x_1824_;
goto v_reusejp_1831_;
}
else
{
lean_object* v_reuseFailAlloc_1846_; 
v_reuseFailAlloc_1846_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1846_, 0, v___x_1818_);
lean_ctor_set(v_reuseFailAlloc_1846_, 1, v___x_1830_);
v___x_1832_ = v_reuseFailAlloc_1846_;
goto v_reusejp_1831_;
}
v_reusejp_1831_:
{
lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; uint8_t v___x_1838_; 
v___x_1833_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__11, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__11_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___closed__11);
v___x_1834_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1832_);
lean_ctor_set(v___x_1834_, 1, v___x_1833_);
v___x_1835_ = lean_mk_empty_array_with_capacity(v_size_1821_);
lean_dec(v_size_1821_);
v___x_1836_ = lean_unsigned_to_nat(0u);
v___x_1837_ = lean_array_get_size(v_buckets_1822_);
v___x_1838_ = lean_nat_dec_lt(v___x_1836_, v___x_1837_);
if (v___x_1838_ == 0)
{
lean_dec_ref(v_buckets_1822_);
v___y_1783_ = v___x_1828_;
v___y_1784_ = v___x_1834_;
v___y_1785_ = v___x_1835_;
goto v___jp_1782_;
}
else
{
uint8_t v___x_1839_; 
v___x_1839_ = lean_nat_dec_le(v___x_1837_, v___x_1837_);
if (v___x_1839_ == 0)
{
if (v___x_1838_ == 0)
{
lean_dec_ref(v_buckets_1822_);
v___y_1783_ = v___x_1828_;
v___y_1784_ = v___x_1834_;
v___y_1785_ = v___x_1835_;
goto v___jp_1782_;
}
else
{
size_t v___x_1840_; size_t v___x_1841_; lean_object* v___x_1842_; 
v___x_1840_ = ((size_t)0ULL);
v___x_1841_ = lean_usize_of_nat(v___x_1837_);
v___x_1842_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(v_buckets_1822_, v___x_1840_, v___x_1841_, v___x_1835_);
lean_dec_ref(v_buckets_1822_);
v___y_1783_ = v___x_1828_;
v___y_1784_ = v___x_1834_;
v___y_1785_ = v___x_1842_;
goto v___jp_1782_;
}
}
else
{
size_t v___x_1843_; size_t v___x_1844_; lean_object* v___x_1845_; 
v___x_1843_ = ((size_t)0ULL);
v___x_1844_ = lean_usize_of_nat(v___x_1837_);
v___x_1845_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__8(v_buckets_1822_, v___x_1843_, v___x_1844_, v___x_1835_);
lean_dec_ref(v_buckets_1822_);
v___y_1783_ = v___x_1828_;
v___y_1784_ = v___x_1834_;
v___y_1785_ = v___x_1845_;
goto v___jp_1782_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9___boxed(lean_object* v_as_1860_, lean_object* v_sz_1861_, lean_object* v_i_1862_, lean_object* v_b_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_){
_start:
{
size_t v_sz_boxed_1869_; size_t v_i_boxed_1870_; lean_object* v_res_1871_; 
v_sz_boxed_1869_ = lean_unbox_usize(v_sz_1861_);
lean_dec(v_sz_1861_);
v_i_boxed_1870_ = lean_unbox_usize(v_i_1862_);
lean_dec(v_i_1862_);
v_res_1871_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(v_as_1860_, v_sz_boxed_1869_, v_i_boxed_1870_, v_b_1863_, v___y_1864_, v___y_1865_, v___y_1866_, v___y_1867_);
lean_dec(v___y_1867_);
lean_dec_ref(v___y_1866_);
lean_dec(v___y_1865_);
lean_dec_ref(v___y_1864_);
lean_dec_ref(v_as_1860_);
return v_res_1871_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__1(void){
_start:
{
lean_object* v___x_1873_; lean_object* v___x_1874_; 
v___x_1873_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__0));
v___x_1874_ = l_Lean_stringToMessageData(v___x_1873_);
return v___x_1874_;
}
}
static double _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4(void){
_start:
{
lean_object* v___x_1878_; double v___x_1879_; 
v___x_1878_ = lean_unsigned_to_nat(1000000000u);
v___x_1879_ = lean_float_of_nat(v___x_1878_);
return v___x_1879_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(lean_object* v___x_1880_, uint8_t v___x_1881_, lean_object* v_range_1882_, lean_object* v_b_1883_, lean_object* v_i_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_){
_start:
{
lean_object* v_stop_1890_; lean_object* v_step_1891_; uint8_t v___x_1892_; 
v_stop_1890_ = lean_ctor_get(v_range_1882_, 1);
v_step_1891_ = lean_ctor_get(v_range_1882_, 2);
v___x_1892_ = lean_nat_dec_lt(v_i_1884_, v_stop_1890_);
if (v___x_1892_ == 0)
{
lean_object* v___x_1893_; 
lean_dec(v_i_1884_);
v___x_1893_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1893_, 0, v_b_1883_);
return v___x_1893_;
}
else
{
lean_object* v_options_1894_; lean_object* v_inheritedTraceOptions_1895_; uint8_t v_hasTrace_1896_; lean_object* v___x_1897_; lean_object* v___y_1902_; lean_object* v___x_1903_; size_t v_sz_1904_; size_t v___x_1905_; 
v_options_1894_ = lean_ctor_get(v___y_1887_, 2);
v_inheritedTraceOptions_1895_ = lean_ctor_get(v___y_1887_, 13);
v_hasTrace_1896_ = lean_ctor_get_uint8(v_options_1894_, sizeof(void*)*1);
v___x_1897_ = lean_box(0);
v___x_1903_ = lean_array_fget_borrowed(v___x_1880_, v_i_1884_);
v_sz_1904_ = lean_array_size(v___x_1903_);
v___x_1905_ = ((size_t)0ULL);
if (v_hasTrace_1896_ == 0)
{
lean_object* v___x_1906_; 
v___x_1906_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(v___x_1903_, v_sz_1904_, v___x_1905_, v___x_1897_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_);
if (lean_obj_tag(v___x_1906_) == 0)
{
lean_dec_ref_known(v___x_1906_, 1);
goto v___jp_1898_;
}
else
{
v___y_1902_ = v___x_1906_;
goto v___jp_1901_;
}
}
else
{
lean_object* v___x_1907_; lean_object* v_traceClass_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___f_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; uint8_t v___x_1918_; lean_object* v___y_1920_; lean_object* v___y_1921_; lean_object* v_a_1922_; lean_object* v___y_1935_; lean_object* v___y_1936_; lean_object* v_a_1937_; lean_object* v___y_1940_; lean_object* v___y_1941_; lean_object* v_a_1942_; lean_object* v___y_1952_; lean_object* v___y_1953_; lean_object* v_a_1954_; 
v___x_1907_ = lp_aesop_Aesop_TraceOption_forward;
v_traceClass_1908_ = lean_ctor_get(v___x_1907_, 0);
v___x_1909_ = lean_obj_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__1, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__1_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__1);
lean_inc(v_i_1884_);
v___x_1910_ = l_Nat_reprFast(v_i_1884_);
v___x_1911_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1911_, 0, v___x_1910_);
v___x_1912_ = l_Lean_MessageData_ofFormat(v___x_1911_);
v___x_1913_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1913_, 0, v___x_1909_);
lean_ctor_set(v___x_1913_, 1, v___x_1912_);
v___f_1914_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0___boxed), 7, 1);
lean_closure_set(v___f_1914_, 0, v___x_1913_);
v___x_1915_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1));
v___x_1916_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__3));
lean_inc(v_traceClass_1908_);
v___x_1917_ = l_Lean_Name_append(v___x_1916_, v_traceClass_1908_);
v___x_1918_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1895_, v_options_1894_, v___x_1917_);
lean_dec(v___x_1917_);
if (v___x_1918_ == 0)
{
lean_object* v___x_1983_; uint8_t v___x_1984_; 
v___x_1983_ = l_Lean_trace_profiler;
v___x_1984_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_options_1894_, v___x_1983_);
if (v___x_1984_ == 0)
{
lean_object* v___x_1985_; 
lean_dec_ref(v___f_1914_);
v___x_1985_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(v___x_1903_, v_sz_1904_, v___x_1905_, v___x_1897_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_);
if (lean_obj_tag(v___x_1985_) == 0)
{
lean_dec_ref_known(v___x_1985_, 1);
goto v___jp_1898_;
}
else
{
v___y_1902_ = v___x_1985_;
goto v___jp_1901_;
}
}
else
{
goto v___jp_1956_;
}
}
else
{
goto v___jp_1956_;
}
v___jp_1919_:
{
lean_object* v___x_1923_; double v___x_1924_; double v___x_1925_; double v___x_1926_; double v___x_1927_; double v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; 
v___x_1923_ = lean_io_mono_nanos_now();
v___x_1924_ = lean_float_of_nat(v___y_1920_);
v___x_1925_ = lean_float_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4);
v___x_1926_ = lean_float_div(v___x_1924_, v___x_1925_);
v___x_1927_ = lean_float_of_nat(v___x_1923_);
v___x_1928_ = lean_float_div(v___x_1927_, v___x_1925_);
v___x_1929_ = lean_box_float(v___x_1926_);
v___x_1930_ = lean_box_float(v___x_1928_);
v___x_1931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1931_, 0, v___x_1929_);
lean_ctor_set(v___x_1931_, 1, v___x_1930_);
v___x_1932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1932_, 0, v_a_1922_);
lean_ctor_set(v___x_1932_, 1, v___x_1931_);
lean_inc(v_traceClass_1908_);
v___x_1933_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(v_traceClass_1908_, v___x_1881_, v___x_1915_, v_options_1894_, v___x_1918_, v___y_1921_, v___f_1914_, v___x_1932_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_);
v___y_1902_ = v___x_1933_;
goto v___jp_1901_;
}
v___jp_1934_:
{
lean_object* v___x_1938_; 
v___x_1938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1938_, 0, v_a_1937_);
v___y_1920_ = v___y_1935_;
v___y_1921_ = v___y_1936_;
v_a_1922_ = v___x_1938_;
goto v___jp_1919_;
}
v___jp_1939_:
{
lean_object* v___x_1943_; double v___x_1944_; double v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; 
v___x_1943_ = lean_io_get_num_heartbeats();
v___x_1944_ = lean_float_of_nat(v___y_1940_);
v___x_1945_ = lean_float_of_nat(v___x_1943_);
v___x_1946_ = lean_box_float(v___x_1944_);
v___x_1947_ = lean_box_float(v___x_1945_);
v___x_1948_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1948_, 0, v___x_1946_);
lean_ctor_set(v___x_1948_, 1, v___x_1947_);
v___x_1949_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1949_, 0, v_a_1942_);
lean_ctor_set(v___x_1949_, 1, v___x_1948_);
lean_inc(v_traceClass_1908_);
v___x_1950_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(v_traceClass_1908_, v___x_1881_, v___x_1915_, v_options_1894_, v___x_1918_, v___y_1941_, v___f_1914_, v___x_1949_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_);
v___y_1902_ = v___x_1950_;
goto v___jp_1901_;
}
v___jp_1951_:
{
lean_object* v___x_1955_; 
v___x_1955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1955_, 0, v_a_1954_);
v___y_1940_ = v___y_1952_;
v___y_1941_ = v___y_1953_;
v_a_1942_ = v___x_1955_;
goto v___jp_1939_;
}
v___jp_1956_:
{
lean_object* v___x_1957_; lean_object* v_a_1958_; lean_object* v___x_1959_; uint8_t v___x_1960_; 
v___x_1957_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg(v___y_1888_);
v_a_1958_ = lean_ctor_get(v___x_1957_, 0);
lean_inc(v_a_1958_);
lean_dec_ref(v___x_1957_);
v___x_1959_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1960_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_options_1894_, v___x_1959_);
if (v___x_1960_ == 0)
{
lean_object* v___x_1961_; lean_object* v___x_1962_; 
v___x_1961_ = lean_io_mono_nanos_now();
v___x_1962_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(v___x_1903_, v_sz_1904_, v___x_1905_, v___x_1897_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_);
if (lean_obj_tag(v___x_1962_) == 0)
{
lean_dec_ref_known(v___x_1962_, 1);
v___y_1935_ = v___x_1961_;
v___y_1936_ = v_a_1958_;
v_a_1937_ = v___x_1897_;
goto v___jp_1934_;
}
else
{
if (lean_obj_tag(v___x_1962_) == 0)
{
lean_object* v_a_1963_; 
v_a_1963_ = lean_ctor_get(v___x_1962_, 0);
lean_inc(v_a_1963_);
lean_dec_ref_known(v___x_1962_, 1);
v___y_1935_ = v___x_1961_;
v___y_1936_ = v_a_1958_;
v_a_1937_ = v_a_1963_;
goto v___jp_1934_;
}
else
{
lean_object* v_a_1964_; lean_object* v___x_1966_; uint8_t v_isShared_1967_; uint8_t v_isSharedCheck_1971_; 
v_a_1964_ = lean_ctor_get(v___x_1962_, 0);
v_isSharedCheck_1971_ = !lean_is_exclusive(v___x_1962_);
if (v_isSharedCheck_1971_ == 0)
{
v___x_1966_ = v___x_1962_;
v_isShared_1967_ = v_isSharedCheck_1971_;
goto v_resetjp_1965_;
}
else
{
lean_inc(v_a_1964_);
lean_dec(v___x_1962_);
v___x_1966_ = lean_box(0);
v_isShared_1967_ = v_isSharedCheck_1971_;
goto v_resetjp_1965_;
}
v_resetjp_1965_:
{
lean_object* v___x_1969_; 
if (v_isShared_1967_ == 0)
{
lean_ctor_set_tag(v___x_1966_, 0);
v___x_1969_ = v___x_1966_;
goto v_reusejp_1968_;
}
else
{
lean_object* v_reuseFailAlloc_1970_; 
v_reuseFailAlloc_1970_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1970_, 0, v_a_1964_);
v___x_1969_ = v_reuseFailAlloc_1970_;
goto v_reusejp_1968_;
}
v_reusejp_1968_:
{
v___y_1920_ = v___x_1961_;
v___y_1921_ = v_a_1958_;
v_a_1922_ = v___x_1969_;
goto v___jp_1919_;
}
}
}
}
}
else
{
lean_object* v___x_1972_; lean_object* v___x_1973_; 
v___x_1972_ = lean_io_get_num_heartbeats();
v___x_1973_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__9(v___x_1903_, v_sz_1904_, v___x_1905_, v___x_1897_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_);
if (lean_obj_tag(v___x_1973_) == 0)
{
lean_dec_ref_known(v___x_1973_, 1);
v___y_1952_ = v___x_1972_;
v___y_1953_ = v_a_1958_;
v_a_1954_ = v___x_1897_;
goto v___jp_1951_;
}
else
{
if (lean_obj_tag(v___x_1973_) == 0)
{
lean_object* v_a_1974_; 
v_a_1974_ = lean_ctor_get(v___x_1973_, 0);
lean_inc(v_a_1974_);
lean_dec_ref_known(v___x_1973_, 1);
v___y_1952_ = v___x_1972_;
v___y_1953_ = v_a_1958_;
v_a_1954_ = v_a_1974_;
goto v___jp_1951_;
}
else
{
lean_object* v_a_1975_; lean_object* v___x_1977_; uint8_t v_isShared_1978_; uint8_t v_isSharedCheck_1982_; 
v_a_1975_ = lean_ctor_get(v___x_1973_, 0);
v_isSharedCheck_1982_ = !lean_is_exclusive(v___x_1973_);
if (v_isSharedCheck_1982_ == 0)
{
v___x_1977_ = v___x_1973_;
v_isShared_1978_ = v_isSharedCheck_1982_;
goto v_resetjp_1976_;
}
else
{
lean_inc(v_a_1975_);
lean_dec(v___x_1973_);
v___x_1977_ = lean_box(0);
v_isShared_1978_ = v_isSharedCheck_1982_;
goto v_resetjp_1976_;
}
v_resetjp_1976_:
{
lean_object* v___x_1980_; 
if (v_isShared_1978_ == 0)
{
lean_ctor_set_tag(v___x_1977_, 0);
v___x_1980_ = v___x_1977_;
goto v_reusejp_1979_;
}
else
{
lean_object* v_reuseFailAlloc_1981_; 
v_reuseFailAlloc_1981_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1981_, 0, v_a_1975_);
v___x_1980_ = v_reuseFailAlloc_1981_;
goto v_reusejp_1979_;
}
v_reusejp_1979_:
{
v___y_1940_ = v___x_1972_;
v___y_1941_ = v_a_1958_;
v_a_1942_ = v___x_1980_;
goto v___jp_1939_;
}
}
}
}
}
}
}
v___jp_1898_:
{
lean_object* v___x_1899_; 
v___x_1899_ = lean_nat_add(v_i_1884_, v_step_1891_);
lean_dec(v_i_1884_);
v_b_1883_ = v___x_1897_;
v_i_1884_ = v___x_1899_;
goto _start;
}
v___jp_1901_:
{
if (lean_obj_tag(v___y_1902_) == 0)
{
lean_dec_ref_known(v___y_1902_, 1);
goto v___jp_1898_;
}
else
{
lean_dec(v_i_1884_);
return v___y_1902_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___boxed(lean_object* v___x_1986_, lean_object* v___x_1987_, lean_object* v_range_1988_, lean_object* v_b_1989_, lean_object* v_i_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_){
_start:
{
uint8_t v___x_59031__boxed_1996_; lean_object* v_res_1997_; 
v___x_59031__boxed_1996_ = lean_unbox(v___x_1987_);
v_res_1997_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(v___x_1986_, v___x_59031__boxed_1996_, v_range_1988_, v_b_1989_, v_i_1990_, v___y_1991_, v___y_1992_, v___y_1993_, v___y_1994_);
lean_dec(v___y_1994_);
lean_dec_ref(v___y_1993_);
lean_dec(v___y_1992_);
lean_dec_ref(v___y_1991_);
lean_dec_ref(v_range_1988_);
lean_dec_ref(v___x_1986_);
return v_res_1997_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__1(void){
_start:
{
lean_object* v___x_1999_; lean_object* v___x_2000_; 
v___x_1999_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__0));
v___x_2000_ = l_Lean_stringToMessageData(v___x_1999_);
return v___x_2000_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__3(void){
_start:
{
lean_object* v___x_2002_; lean_object* v___x_2003_; 
v___x_2002_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__2));
v___x_2003_ = l_Lean_stringToMessageData(v___x_2002_);
return v___x_2003_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__4(void){
_start:
{
lean_object* v___x_2004_; lean_object* v___f_2005_; 
v___x_2004_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__3, &lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__3_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__3);
v___f_2005_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___lam__0___boxed), 7, 1);
lean_closure_set(v___f_2005_, 0, v___x_2004_);
return v___f_2005_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__6(void){
_start:
{
lean_object* v___x_2007_; lean_object* v___x_2008_; 
v___x_2007_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__5));
v___x_2008_ = l_Lean_stringToMessageData(v___x_2007_);
return v___x_2008_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082(lean_object* v_t_2009_, lean_object* v_immediate_x3f_2010_, lean_object* v_pat_x3f_2011_, lean_object* v_phase_2012_, uint8_t v_isDestruct_2013_, lean_object* v_a_2014_, lean_object* v_a_2015_, lean_object* v_a_2016_, lean_object* v_a_2017_){
_start:
{
lean_object* v___x_2019_; 
lean_inc_ref(v_t_2009_);
v___x_2019_ = lp_aesop_Aesop_ElabRuleTerm_expr(v_t_2009_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2019_) == 0)
{
lean_object* v_a_2020_; lean_object* v___x_2021_; 
v_a_2020_ = lean_ctor_get(v___x_2019_, 0);
lean_inc(v_a_2020_);
lean_dec_ref_known(v___x_2019_, 1);
lean_inc_ref(v_t_2009_);
v___x_2021_ = lp_aesop_Aesop_ElabRuleTerm_name(v_t_2009_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2021_) == 0)
{
lean_object* v_a_2022_; lean_object* v___x_2023_; 
v_a_2022_ = lean_ctor_get(v___x_2021_, 0);
lean_inc(v_a_2022_);
lean_dec_ref_known(v___x_2021_, 1);
lean_inc(v_a_2017_);
lean_inc_ref(v_a_2016_);
lean_inc(v_a_2015_);
lean_inc_ref(v_a_2014_);
lean_inc(v_a_2020_);
v___x_2023_ = lean_infer_type(v_a_2020_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2023_) == 0)
{
lean_object* v_a_2024_; lean_object* v___x_2025_; 
v_a_2024_ = lean_ctor_get(v___x_2023_, 0);
lean_inc(v_a_2024_);
lean_dec_ref_known(v___x_2023_, 1);
lean_inc(v_pat_x3f_2011_);
v___x_2025_ = lp_aesop_Aesop_RuleBuilder_getImmediatePremises(v_a_2024_, v_pat_x3f_2011_, v_immediate_x3f_2010_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2025_) == 0)
{
lean_object* v_a_2026_; lean_object* v___x_2027_; 
v_a_2026_ = lean_ctor_get(v___x_2025_, 0);
lean_inc(v_a_2026_);
lean_dec_ref_known(v___x_2025_, 1);
lean_inc(v_a_2020_);
v___x_2027_ = lp_aesop_Aesop_ForwardRuleInfo_ofExpr(v_a_2020_, v_pat_x3f_2011_, v_a_2026_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2027_) == 0)
{
lean_object* v_a_2028_; lean_object* v___x_2030_; uint8_t v_isShared_2031_; uint8_t v_isSharedCheck_2325_; 
v_a_2028_ = lean_ctor_get(v___x_2027_, 0);
v_isSharedCheck_2325_ = !lean_is_exclusive(v___x_2027_);
if (v_isSharedCheck_2325_ == 0)
{
v___x_2030_ = v___x_2027_;
v_isShared_2031_ = v_isSharedCheck_2325_;
goto v_resetjp_2029_;
}
else
{
lean_inc(v_a_2028_);
lean_dec(v___x_2027_);
v___x_2030_ = lean_box(0);
v_isShared_2031_ = v_isSharedCheck_2325_;
goto v_resetjp_2029_;
}
v_resetjp_2029_:
{
uint8_t v___y_2033_; lean_object* v___y_2034_; uint8_t v___y_2035_; uint8_t v___y_2036_; uint64_t v___y_2037_; lean_object* v___y_2051_; uint8_t v___y_2052_; lean_object* v___y_2058_; lean_object* v___x_2069_; lean_object* v___y_2071_; lean_object* v___y_2072_; lean_object* v___y_2073_; lean_object* v___y_2074_; lean_object* v___y_2075_; lean_object* v___y_2096_; lean_object* v___y_2097_; lean_object* v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2111_; lean_object* v___y_2112_; lean_object* v___y_2113_; lean_object* v___y_2114_; lean_object* v___y_2115_; lean_object* v___y_2116_; lean_object* v___y_2117_; uint8_t v___y_2118_; lean_object* v___y_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; uint8_t v___y_2122_; lean_object* v_a_2123_; lean_object* v___y_2133_; lean_object* v___y_2134_; lean_object* v___y_2135_; lean_object* v___y_2136_; lean_object* v___y_2137_; lean_object* v___y_2138_; uint8_t v___y_2139_; lean_object* v___y_2140_; lean_object* v___y_2141_; lean_object* v___y_2142_; uint8_t v___y_2143_; lean_object* v___y_2144_; lean_object* v_a_2145_; lean_object* v___y_2148_; lean_object* v___y_2149_; lean_object* v___y_2150_; lean_object* v___y_2151_; lean_object* v___y_2152_; lean_object* v___y_2153_; uint8_t v___y_2154_; lean_object* v___y_2155_; lean_object* v___y_2156_; lean_object* v___y_2157_; uint8_t v___y_2158_; lean_object* v___y_2159_; lean_object* v_a_2160_; lean_object* v___y_2163_; lean_object* v___y_2164_; lean_object* v___y_2165_; lean_object* v___y_2166_; lean_object* v___y_2167_; lean_object* v___y_2168_; lean_object* v___y_2169_; uint8_t v___y_2170_; lean_object* v___y_2171_; lean_object* v___y_2172_; lean_object* v___y_2173_; uint8_t v___y_2174_; lean_object* v_a_2175_; lean_object* v___y_2188_; lean_object* v___y_2189_; lean_object* v___y_2190_; lean_object* v___y_2191_; lean_object* v___y_2192_; lean_object* v___y_2193_; uint8_t v___y_2194_; lean_object* v___y_2195_; lean_object* v___y_2196_; lean_object* v___y_2197_; uint8_t v___y_2198_; lean_object* v___y_2199_; lean_object* v_a_2200_; lean_object* v___y_2203_; lean_object* v___y_2204_; lean_object* v___y_2205_; lean_object* v___y_2206_; lean_object* v___y_2207_; lean_object* v___y_2208_; uint8_t v___y_2209_; lean_object* v___y_2210_; lean_object* v___y_2211_; lean_object* v___y_2212_; uint8_t v___y_2213_; lean_object* v___y_2214_; lean_object* v_a_2215_; lean_object* v___y_2218_; lean_object* v___y_2219_; lean_object* v___y_2220_; lean_object* v___y_2221_; lean_object* v___y_2222_; uint8_t v___y_2223_; lean_object* v___y_2224_; uint8_t v___y_2225_; lean_object* v___y_2226_; uint8_t v___y_2227_; lean_object* v___y_2228_; lean_object* v___y_2262_; lean_object* v___y_2263_; lean_object* v___y_2264_; lean_object* v___y_2265_; lean_object* v___x_2299_; lean_object* v_a_2300_; uint8_t v___x_2301_; 
v___x_2069_ = lp_aesop_Aesop_TraceOption_forward;
v___x_2299_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2069_, v_a_2016_);
v_a_2300_ = lean_ctor_get(v___x_2299_, 0);
lean_inc(v_a_2300_);
lean_dec_ref(v___x_2299_);
v___x_2301_ = lean_unbox(v_a_2300_);
lean_dec(v_a_2300_);
if (v___x_2301_ == 0)
{
lean_dec(v_a_2020_);
v___y_2262_ = v_a_2014_;
v___y_2263_ = v_a_2015_;
v___y_2264_ = v_a_2016_;
v___y_2265_ = v_a_2017_;
goto v___jp_2261_;
}
else
{
lean_object* v___x_2302_; 
lean_inc(v_a_2017_);
lean_inc_ref(v_a_2016_);
lean_inc(v_a_2015_);
lean_inc_ref(v_a_2014_);
v___x_2302_ = lean_infer_type(v_a_2020_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2302_) == 0)
{
lean_object* v_a_2303_; lean_object* v_traceClass_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; 
v_a_2303_ = lean_ctor_get(v___x_2302_, 0);
lean_inc(v_a_2303_);
lean_dec_ref_known(v___x_2302_, 1);
v_traceClass_2304_ = lean_ctor_get(v___x_2069_, 0);
v___x_2305_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__6, &lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__6_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__6);
v___x_2306_ = l_Lean_indentExpr(v_a_2303_);
v___x_2307_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2307_, 0, v___x_2305_);
lean_ctor_set(v___x_2307_, 1, v___x_2306_);
lean_inc(v_traceClass_2304_);
v___x_2308_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v_traceClass_2304_, v___x_2307_, v_a_2014_, v_a_2015_, v_a_2016_, v_a_2017_);
if (lean_obj_tag(v___x_2308_) == 0)
{
lean_dec_ref_known(v___x_2308_, 1);
v___y_2262_ = v_a_2014_;
v___y_2263_ = v_a_2015_;
v___y_2264_ = v_a_2016_;
v___y_2265_ = v_a_2017_;
goto v___jp_2261_;
}
else
{
lean_object* v_a_2309_; lean_object* v___x_2311_; uint8_t v_isShared_2312_; uint8_t v_isSharedCheck_2316_; 
lean_del_object(v___x_2030_);
lean_dec(v_a_2028_);
lean_dec(v_a_2022_);
lean_dec_ref(v_t_2009_);
v_a_2309_ = lean_ctor_get(v___x_2308_, 0);
v_isSharedCheck_2316_ = !lean_is_exclusive(v___x_2308_);
if (v_isSharedCheck_2316_ == 0)
{
v___x_2311_ = v___x_2308_;
v_isShared_2312_ = v_isSharedCheck_2316_;
goto v_resetjp_2310_;
}
else
{
lean_inc(v_a_2309_);
lean_dec(v___x_2308_);
v___x_2311_ = lean_box(0);
v_isShared_2312_ = v_isSharedCheck_2316_;
goto v_resetjp_2310_;
}
v_resetjp_2310_:
{
lean_object* v___x_2314_; 
if (v_isShared_2312_ == 0)
{
v___x_2314_ = v___x_2311_;
goto v_reusejp_2313_;
}
else
{
lean_object* v_reuseFailAlloc_2315_; 
v_reuseFailAlloc_2315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2315_, 0, v_a_2309_);
v___x_2314_ = v_reuseFailAlloc_2315_;
goto v_reusejp_2313_;
}
v_reusejp_2313_:
{
return v___x_2314_;
}
}
}
}
else
{
lean_object* v_a_2317_; lean_object* v___x_2319_; uint8_t v_isShared_2320_; uint8_t v_isSharedCheck_2324_; 
lean_del_object(v___x_2030_);
lean_dec(v_a_2028_);
lean_dec(v_a_2022_);
lean_dec_ref(v_t_2009_);
v_a_2317_ = lean_ctor_get(v___x_2302_, 0);
v_isSharedCheck_2324_ = !lean_is_exclusive(v___x_2302_);
if (v_isSharedCheck_2324_ == 0)
{
v___x_2319_ = v___x_2302_;
v_isShared_2320_ = v_isSharedCheck_2324_;
goto v_resetjp_2318_;
}
else
{
lean_inc(v_a_2317_);
lean_dec(v___x_2302_);
v___x_2319_ = lean_box(0);
v_isShared_2320_ = v_isSharedCheck_2324_;
goto v_resetjp_2318_;
}
v_resetjp_2318_:
{
lean_object* v___x_2322_; 
if (v_isShared_2320_ == 0)
{
v___x_2322_ = v___x_2319_;
goto v_reusejp_2321_;
}
else
{
lean_object* v_reuseFailAlloc_2323_; 
v_reuseFailAlloc_2323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2323_, 0, v_a_2317_);
v___x_2322_ = v_reuseFailAlloc_2323_;
goto v_reusejp_2321_;
}
v_reusejp_2321_:
{
return v___x_2322_;
}
}
}
}
v___jp_2032_:
{
uint64_t v___x_2038_; uint64_t v___x_2039_; uint64_t v___x_2040_; uint64_t v___x_2041_; uint64_t v___x_2042_; uint64_t v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2048_; 
v___x_2038_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___y_2033_);
v___x_2039_ = lp_aesop_Aesop_instHashablePhaseName_hash(v___y_2035_);
v___x_2040_ = lp_aesop_Aesop_instHashableScopeName_hash(v___y_2036_);
v___x_2041_ = lean_uint64_mix_hash(v___x_2039_, v___x_2040_);
v___x_2042_ = lean_uint64_mix_hash(v___x_2038_, v___x_2041_);
v___x_2043_ = lean_uint64_mix_hash(v___y_2037_, v___x_2042_);
v___x_2044_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_2044_, 0, v_a_2022_);
lean_ctor_set_uint8(v___x_2044_, sizeof(void*)*1 + 8, v___y_2033_);
lean_ctor_set_uint8(v___x_2044_, sizeof(void*)*1 + 9, v___y_2035_);
lean_ctor_set_uint8(v___x_2044_, sizeof(void*)*1 + 10, v___y_2036_);
lean_ctor_set_uint64(v___x_2044_, sizeof(void*)*1, v___x_2043_);
v___x_2045_ = lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(v_t_2009_);
v___x_2046_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2046_, 0, v_a_2028_);
lean_ctor_set(v___x_2046_, 1, v___x_2044_);
lean_ctor_set(v___x_2046_, 2, v___x_2045_);
lean_ctor_set(v___x_2046_, 3, v___y_2034_);
if (v_isShared_2031_ == 0)
{
lean_ctor_set(v___x_2030_, 0, v___x_2046_);
v___x_2048_ = v___x_2030_;
goto v_reusejp_2047_;
}
else
{
lean_object* v_reuseFailAlloc_2049_; 
v_reuseFailAlloc_2049_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2049_, 0, v___x_2046_);
v___x_2048_ = v_reuseFailAlloc_2049_;
goto v_reusejp_2047_;
}
v_reusejp_2047_:
{
return v___x_2048_;
}
}
v___jp_2050_:
{
uint8_t v___x_2053_; uint8_t v___x_2054_; 
v___x_2053_ = lp_aesop_Aesop_PhaseSpec_phase(v_phase_2012_);
v___x_2054_ = lp_aesop_Aesop_ElabRuleTerm_scope(v_t_2009_);
if (lean_obj_tag(v_a_2022_) == 0)
{
uint64_t v___x_2055_; 
v___x_2055_ = 1723ULL;
v___y_2033_ = v___y_2052_;
v___y_2034_ = v___y_2051_;
v___y_2035_ = v___x_2053_;
v___y_2036_ = v___x_2054_;
v___y_2037_ = v___x_2055_;
goto v___jp_2032_;
}
else
{
uint64_t v_hash_2056_; 
v_hash_2056_ = lean_ctor_get_uint64(v_a_2022_, sizeof(void*)*2);
v___y_2033_ = v___y_2052_;
v___y_2034_ = v___y_2051_;
v___y_2035_ = v___x_2053_;
v___y_2036_ = v___x_2054_;
v___y_2037_ = v_hash_2056_;
goto v___jp_2032_;
}
}
v___jp_2057_:
{
if (v_isDestruct_2013_ == 0)
{
uint8_t v___x_2059_; 
v___x_2059_ = 4;
v___y_2051_ = v___y_2058_;
v___y_2052_ = v___x_2059_;
goto v___jp_2050_;
}
else
{
uint8_t v___x_2060_; 
v___x_2060_ = 3;
v___y_2051_ = v___y_2058_;
v___y_2052_ = v___x_2060_;
goto v___jp_2050_;
}
}
v___jp_2061_:
{
switch(lean_obj_tag(v_phase_2012_))
{
case 0:
{
lean_object* v_info_2062_; lean_object* v_penalty_2063_; lean_object* v___x_2064_; 
v_info_2062_ = lean_ctor_get(v_phase_2012_, 0);
v_penalty_2063_ = lean_ctor_get(v_info_2062_, 0);
lean_inc(v_penalty_2063_);
v___x_2064_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2064_, 0, v_penalty_2063_);
v___y_2058_ = v___x_2064_;
goto v___jp_2057_;
}
case 1:
{
lean_object* v_info_2065_; lean_object* v___x_2066_; 
v_info_2065_ = lean_ctor_get(v_phase_2012_, 0);
lean_inc(v_info_2065_);
v___x_2066_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2066_, 0, v_info_2065_);
v___y_2058_ = v___x_2066_;
goto v___jp_2057_;
}
default: 
{
double v_info_2067_; lean_object* v___x_2068_; 
v_info_2067_ = lean_ctor_get_float(v_phase_2012_, 0);
v___x_2068_ = lean_alloc_ctor(1, 0, 8);
lean_ctor_set_float(v___x_2068_, 0, v_info_2067_);
v___y_2058_ = v___x_2068_;
goto v___jp_2057_;
}
}
}
v___jp_2070_:
{
lean_object* v___x_2076_; lean_object* v_a_2077_; uint8_t v___x_2078_; 
v___x_2076_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2069_, v___y_2075_);
v_a_2077_ = lean_ctor_get(v___x_2076_, 0);
lean_inc(v_a_2077_);
lean_dec_ref(v___x_2076_);
v___x_2078_ = lean_unbox(v_a_2077_);
lean_dec(v_a_2077_);
if (v___x_2078_ == 0)
{
lean_dec(v___y_2071_);
goto v___jp_2061_;
}
else
{
lean_object* v_conclusionDeps_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; 
v_conclusionDeps_2079_ = lean_ctor_get(v_a_2028_, 3);
v___x_2080_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__1, &lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__1_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__1);
lean_inc_ref(v_conclusionDeps_2079_);
v___x_2081_ = lean_array_to_list(v_conclusionDeps_2079_);
v___x_2082_ = lean_box(0);
v___x_2083_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(v___x_2081_, v___x_2082_);
v___x_2084_ = l_Lean_MessageData_ofList(v___x_2083_);
v___x_2085_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2085_, 0, v___x_2080_);
lean_ctor_set(v___x_2085_, 1, v___x_2084_);
v___x_2086_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v___y_2071_, v___x_2085_, v___y_2074_, v___y_2073_, v___y_2075_, v___y_2072_);
if (lean_obj_tag(v___x_2086_) == 0)
{
lean_dec_ref_known(v___x_2086_, 1);
goto v___jp_2061_;
}
else
{
lean_object* v_a_2087_; lean_object* v___x_2089_; uint8_t v_isShared_2090_; uint8_t v_isSharedCheck_2094_; 
lean_del_object(v___x_2030_);
lean_dec(v_a_2028_);
lean_dec(v_a_2022_);
lean_dec_ref(v_t_2009_);
v_a_2087_ = lean_ctor_get(v___x_2086_, 0);
v_isSharedCheck_2094_ = !lean_is_exclusive(v___x_2086_);
if (v_isSharedCheck_2094_ == 0)
{
v___x_2089_ = v___x_2086_;
v_isShared_2090_ = v_isSharedCheck_2094_;
goto v_resetjp_2088_;
}
else
{
lean_inc(v_a_2087_);
lean_dec(v___x_2086_);
v___x_2089_ = lean_box(0);
v_isShared_2090_ = v_isSharedCheck_2094_;
goto v_resetjp_2088_;
}
v_resetjp_2088_:
{
lean_object* v___x_2092_; 
if (v_isShared_2090_ == 0)
{
v___x_2092_ = v___x_2089_;
goto v_reusejp_2091_;
}
else
{
lean_object* v_reuseFailAlloc_2093_; 
v_reuseFailAlloc_2093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2093_, 0, v_a_2087_);
v___x_2092_ = v_reuseFailAlloc_2093_;
goto v_reusejp_2091_;
}
v_reusejp_2091_:
{
return v___x_2092_;
}
}
}
}
}
v___jp_2095_:
{
if (lean_obj_tag(v___y_2101_) == 0)
{
lean_dec_ref_known(v___y_2101_, 1);
v___y_2071_ = v___y_2096_;
v___y_2072_ = v___y_2097_;
v___y_2073_ = v___y_2098_;
v___y_2074_ = v___y_2099_;
v___y_2075_ = v___y_2100_;
goto v___jp_2070_;
}
else
{
lean_object* v_a_2102_; lean_object* v___x_2104_; uint8_t v_isShared_2105_; uint8_t v_isSharedCheck_2109_; 
lean_dec(v___y_2096_);
lean_del_object(v___x_2030_);
lean_dec(v_a_2028_);
lean_dec(v_a_2022_);
lean_dec_ref(v_t_2009_);
v_a_2102_ = lean_ctor_get(v___y_2101_, 0);
v_isSharedCheck_2109_ = !lean_is_exclusive(v___y_2101_);
if (v_isSharedCheck_2109_ == 0)
{
v___x_2104_ = v___y_2101_;
v_isShared_2105_ = v_isSharedCheck_2109_;
goto v_resetjp_2103_;
}
else
{
lean_inc(v_a_2102_);
lean_dec(v___y_2101_);
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
v___jp_2110_:
{
lean_object* v___x_2124_; double v___x_2125_; double v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; 
v___x_2124_ = lean_io_get_num_heartbeats();
v___x_2125_ = lean_float_of_nat(v___y_2112_);
v___x_2126_ = lean_float_of_nat(v___x_2124_);
v___x_2127_ = lean_box_float(v___x_2125_);
v___x_2128_ = lean_box_float(v___x_2126_);
v___x_2129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2129_, 0, v___x_2127_);
lean_ctor_set(v___x_2129_, 1, v___x_2128_);
v___x_2130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2130_, 0, v_a_2123_);
lean_ctor_set(v___x_2130_, 1, v___x_2129_);
lean_inc_ref(v___y_2119_);
lean_inc_ref(v___y_2115_);
lean_inc(v___y_2111_);
v___x_2131_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(v___y_2111_, v___y_2122_, v___y_2115_, v___y_2116_, v___y_2118_, v___y_2120_, v___y_2119_, v___x_2130_, v___y_2117_, v___y_2114_, v___y_2121_, v___y_2113_);
v___y_2096_ = v___y_2111_;
v___y_2097_ = v___y_2113_;
v___y_2098_ = v___y_2114_;
v___y_2099_ = v___y_2117_;
v___y_2100_ = v___y_2121_;
v___y_2101_ = v___x_2131_;
goto v___jp_2095_;
}
v___jp_2132_:
{
lean_object* v___x_2146_; 
v___x_2146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2146_, 0, v_a_2145_);
v___y_2111_ = v___y_2133_;
v___y_2112_ = v___y_2134_;
v___y_2113_ = v___y_2135_;
v___y_2114_ = v___y_2136_;
v___y_2115_ = v___y_2137_;
v___y_2116_ = v___y_2138_;
v___y_2117_ = v___y_2140_;
v___y_2118_ = v___y_2139_;
v___y_2119_ = v___y_2142_;
v___y_2120_ = v___y_2141_;
v___y_2121_ = v___y_2144_;
v___y_2122_ = v___y_2143_;
v_a_2123_ = v___x_2146_;
goto v___jp_2110_;
}
v___jp_2147_:
{
lean_object* v___x_2161_; 
v___x_2161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2161_, 0, v_a_2160_);
v___y_2111_ = v___y_2148_;
v___y_2112_ = v___y_2149_;
v___y_2113_ = v___y_2150_;
v___y_2114_ = v___y_2151_;
v___y_2115_ = v___y_2152_;
v___y_2116_ = v___y_2153_;
v___y_2117_ = v___y_2155_;
v___y_2118_ = v___y_2154_;
v___y_2119_ = v___y_2157_;
v___y_2120_ = v___y_2156_;
v___y_2121_ = v___y_2159_;
v___y_2122_ = v___y_2158_;
v_a_2123_ = v___x_2161_;
goto v___jp_2110_;
}
v___jp_2162_:
{
lean_object* v___x_2176_; double v___x_2177_; double v___x_2178_; double v___x_2179_; double v___x_2180_; double v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; 
v___x_2176_ = lean_io_mono_nanos_now();
v___x_2177_ = lean_float_of_nat(v___y_2165_);
v___x_2178_ = lean_float_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4);
v___x_2179_ = lean_float_div(v___x_2177_, v___x_2178_);
v___x_2180_ = lean_float_of_nat(v___x_2176_);
v___x_2181_ = lean_float_div(v___x_2180_, v___x_2178_);
v___x_2182_ = lean_box_float(v___x_2179_);
v___x_2183_ = lean_box_float(v___x_2181_);
v___x_2184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2184_, 0, v___x_2182_);
lean_ctor_set(v___x_2184_, 1, v___x_2183_);
v___x_2185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2185_, 0, v_a_2175_);
lean_ctor_set(v___x_2185_, 1, v___x_2184_);
lean_inc_ref(v___y_2171_);
lean_inc_ref(v___y_2167_);
lean_inc(v___y_2163_);
v___x_2186_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5(v___y_2163_, v___y_2174_, v___y_2167_, v___y_2168_, v___y_2170_, v___y_2172_, v___y_2171_, v___x_2185_, v___y_2169_, v___y_2166_, v___y_2173_, v___y_2164_);
v___y_2096_ = v___y_2163_;
v___y_2097_ = v___y_2164_;
v___y_2098_ = v___y_2166_;
v___y_2099_ = v___y_2169_;
v___y_2100_ = v___y_2173_;
v___y_2101_ = v___x_2186_;
goto v___jp_2095_;
}
v___jp_2187_:
{
lean_object* v___x_2201_; 
v___x_2201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2201_, 0, v_a_2200_);
v___y_2163_ = v___y_2188_;
v___y_2164_ = v___y_2189_;
v___y_2165_ = v___y_2190_;
v___y_2166_ = v___y_2191_;
v___y_2167_ = v___y_2192_;
v___y_2168_ = v___y_2193_;
v___y_2169_ = v___y_2195_;
v___y_2170_ = v___y_2194_;
v___y_2171_ = v___y_2197_;
v___y_2172_ = v___y_2196_;
v___y_2173_ = v___y_2199_;
v___y_2174_ = v___y_2198_;
v_a_2175_ = v___x_2201_;
goto v___jp_2162_;
}
v___jp_2202_:
{
lean_object* v___x_2216_; 
v___x_2216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2216_, 0, v_a_2215_);
v___y_2163_ = v___y_2203_;
v___y_2164_ = v___y_2204_;
v___y_2165_ = v___y_2205_;
v___y_2166_ = v___y_2206_;
v___y_2167_ = v___y_2207_;
v___y_2168_ = v___y_2208_;
v___y_2169_ = v___y_2210_;
v___y_2170_ = v___y_2209_;
v___y_2171_ = v___y_2212_;
v___y_2172_ = v___y_2211_;
v___y_2173_ = v___y_2214_;
v___y_2174_ = v___y_2213_;
v_a_2175_ = v___x_2216_;
goto v___jp_2162_;
}
v___jp_2217_:
{
lean_object* v___x_2229_; lean_object* v_a_2230_; lean_object* v___x_2231_; uint8_t v___x_2232_; 
v___x_2229_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg(v___y_2219_);
v_a_2230_ = lean_ctor_get(v___x_2229_, 0);
lean_inc(v_a_2230_);
lean_dec_ref(v___x_2229_);
v___x_2231_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2232_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v___y_2222_, v___x_2231_);
if (v___x_2232_ == 0)
{
lean_object* v___x_2233_; lean_object* v___x_2234_; lean_object* v_a_2235_; uint8_t v___x_2236_; 
v___x_2233_ = lean_io_mono_nanos_now();
v___x_2234_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2069_, v___y_2228_);
v_a_2235_ = lean_ctor_get(v___x_2234_, 0);
lean_inc(v_a_2235_);
lean_dec_ref(v___x_2234_);
v___x_2236_ = lean_unbox(v_a_2235_);
lean_dec(v_a_2235_);
if (v___x_2236_ == 0)
{
lean_object* v___x_2237_; 
v___x_2237_ = lean_box(0);
v___y_2188_ = v___y_2218_;
v___y_2189_ = v___y_2219_;
v___y_2190_ = v___x_2233_;
v___y_2191_ = v___y_2220_;
v___y_2192_ = v___y_2221_;
v___y_2193_ = v___y_2222_;
v___y_2194_ = v___y_2223_;
v___y_2195_ = v___y_2224_;
v___y_2196_ = v_a_2230_;
v___y_2197_ = v___y_2226_;
v___y_2198_ = v___y_2227_;
v___y_2199_ = v___y_2228_;
v_a_2200_ = v___x_2237_;
goto v___jp_2187_;
}
else
{
lean_object* v_slotClusters_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; 
v_slotClusters_2238_ = lean_ctor_get(v_a_2028_, 2);
v___x_2239_ = lean_unsigned_to_nat(0u);
v___x_2240_ = lean_array_get_size(v_slotClusters_2238_);
v___x_2241_ = lean_unsigned_to_nat(1u);
v___x_2242_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2242_, 0, v___x_2239_);
lean_ctor_set(v___x_2242_, 1, v___x_2240_);
lean_ctor_set(v___x_2242_, 2, v___x_2241_);
v___x_2243_ = lean_box(0);
v___x_2244_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(v_slotClusters_2238_, v___y_2225_, v___x_2242_, v___x_2243_, v___x_2239_, v___y_2224_, v___y_2220_, v___y_2228_, v___y_2219_);
lean_dec_ref_known(v___x_2242_, 3);
if (lean_obj_tag(v___x_2244_) == 0)
{
lean_dec_ref_known(v___x_2244_, 1);
v___y_2188_ = v___y_2218_;
v___y_2189_ = v___y_2219_;
v___y_2190_ = v___x_2233_;
v___y_2191_ = v___y_2220_;
v___y_2192_ = v___y_2221_;
v___y_2193_ = v___y_2222_;
v___y_2194_ = v___y_2223_;
v___y_2195_ = v___y_2224_;
v___y_2196_ = v_a_2230_;
v___y_2197_ = v___y_2226_;
v___y_2198_ = v___y_2227_;
v___y_2199_ = v___y_2228_;
v_a_2200_ = v___x_2243_;
goto v___jp_2187_;
}
else
{
if (lean_obj_tag(v___x_2244_) == 0)
{
lean_object* v_a_2245_; 
v_a_2245_ = lean_ctor_get(v___x_2244_, 0);
lean_inc(v_a_2245_);
lean_dec_ref_known(v___x_2244_, 1);
v___y_2188_ = v___y_2218_;
v___y_2189_ = v___y_2219_;
v___y_2190_ = v___x_2233_;
v___y_2191_ = v___y_2220_;
v___y_2192_ = v___y_2221_;
v___y_2193_ = v___y_2222_;
v___y_2194_ = v___y_2223_;
v___y_2195_ = v___y_2224_;
v___y_2196_ = v_a_2230_;
v___y_2197_ = v___y_2226_;
v___y_2198_ = v___y_2227_;
v___y_2199_ = v___y_2228_;
v_a_2200_ = v_a_2245_;
goto v___jp_2187_;
}
else
{
lean_object* v_a_2246_; 
v_a_2246_ = lean_ctor_get(v___x_2244_, 0);
lean_inc(v_a_2246_);
lean_dec_ref_known(v___x_2244_, 1);
v___y_2203_ = v___y_2218_;
v___y_2204_ = v___y_2219_;
v___y_2205_ = v___x_2233_;
v___y_2206_ = v___y_2220_;
v___y_2207_ = v___y_2221_;
v___y_2208_ = v___y_2222_;
v___y_2209_ = v___y_2223_;
v___y_2210_ = v___y_2224_;
v___y_2211_ = v_a_2230_;
v___y_2212_ = v___y_2226_;
v___y_2213_ = v___y_2227_;
v___y_2214_ = v___y_2228_;
v_a_2215_ = v_a_2246_;
goto v___jp_2202_;
}
}
}
}
else
{
lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v_a_2249_; uint8_t v___x_2250_; 
v___x_2247_ = lean_io_get_num_heartbeats();
v___x_2248_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2069_, v___y_2228_);
v_a_2249_ = lean_ctor_get(v___x_2248_, 0);
lean_inc(v_a_2249_);
lean_dec_ref(v___x_2248_);
v___x_2250_ = lean_unbox(v_a_2249_);
lean_dec(v_a_2249_);
if (v___x_2250_ == 0)
{
lean_object* v___x_2251_; 
v___x_2251_ = lean_box(0);
v___y_2133_ = v___y_2218_;
v___y_2134_ = v___x_2247_;
v___y_2135_ = v___y_2219_;
v___y_2136_ = v___y_2220_;
v___y_2137_ = v___y_2221_;
v___y_2138_ = v___y_2222_;
v___y_2139_ = v___y_2223_;
v___y_2140_ = v___y_2224_;
v___y_2141_ = v_a_2230_;
v___y_2142_ = v___y_2226_;
v___y_2143_ = v___y_2227_;
v___y_2144_ = v___y_2228_;
v_a_2145_ = v___x_2251_;
goto v___jp_2132_;
}
else
{
lean_object* v_slotClusters_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; 
v_slotClusters_2252_ = lean_ctor_get(v_a_2028_, 2);
v___x_2253_ = lean_unsigned_to_nat(0u);
v___x_2254_ = lean_array_get_size(v_slotClusters_2252_);
v___x_2255_ = lean_unsigned_to_nat(1u);
v___x_2256_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2256_, 0, v___x_2253_);
lean_ctor_set(v___x_2256_, 1, v___x_2254_);
lean_ctor_set(v___x_2256_, 2, v___x_2255_);
v___x_2257_ = lean_box(0);
v___x_2258_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(v_slotClusters_2252_, v___x_2232_, v___x_2256_, v___x_2257_, v___x_2253_, v___y_2224_, v___y_2220_, v___y_2228_, v___y_2219_);
lean_dec_ref_known(v___x_2256_, 3);
if (lean_obj_tag(v___x_2258_) == 0)
{
lean_dec_ref_known(v___x_2258_, 1);
v___y_2133_ = v___y_2218_;
v___y_2134_ = v___x_2247_;
v___y_2135_ = v___y_2219_;
v___y_2136_ = v___y_2220_;
v___y_2137_ = v___y_2221_;
v___y_2138_ = v___y_2222_;
v___y_2139_ = v___y_2223_;
v___y_2140_ = v___y_2224_;
v___y_2141_ = v_a_2230_;
v___y_2142_ = v___y_2226_;
v___y_2143_ = v___y_2227_;
v___y_2144_ = v___y_2228_;
v_a_2145_ = v___x_2257_;
goto v___jp_2132_;
}
else
{
if (lean_obj_tag(v___x_2258_) == 0)
{
lean_object* v_a_2259_; 
v_a_2259_ = lean_ctor_get(v___x_2258_, 0);
lean_inc(v_a_2259_);
lean_dec_ref_known(v___x_2258_, 1);
v___y_2133_ = v___y_2218_;
v___y_2134_ = v___x_2247_;
v___y_2135_ = v___y_2219_;
v___y_2136_ = v___y_2220_;
v___y_2137_ = v___y_2221_;
v___y_2138_ = v___y_2222_;
v___y_2139_ = v___y_2223_;
v___y_2140_ = v___y_2224_;
v___y_2141_ = v_a_2230_;
v___y_2142_ = v___y_2226_;
v___y_2143_ = v___y_2227_;
v___y_2144_ = v___y_2228_;
v_a_2145_ = v_a_2259_;
goto v___jp_2132_;
}
else
{
lean_object* v_a_2260_; 
v_a_2260_ = lean_ctor_get(v___x_2258_, 0);
lean_inc(v_a_2260_);
lean_dec_ref_known(v___x_2258_, 1);
v___y_2148_ = v___y_2218_;
v___y_2149_ = v___x_2247_;
v___y_2150_ = v___y_2219_;
v___y_2151_ = v___y_2220_;
v___y_2152_ = v___y_2221_;
v___y_2153_ = v___y_2222_;
v___y_2154_ = v___y_2223_;
v___y_2155_ = v___y_2224_;
v___y_2156_ = v_a_2230_;
v___y_2157_ = v___y_2226_;
v___y_2158_ = v___y_2227_;
v___y_2159_ = v___y_2228_;
v_a_2160_ = v_a_2260_;
goto v___jp_2147_;
}
}
}
}
}
v___jp_2261_:
{
lean_object* v_options_2266_; uint8_t v_hasTrace_2267_; 
v_options_2266_ = lean_ctor_get(v___y_2264_, 2);
v_hasTrace_2267_ = lean_ctor_get_uint8(v_options_2266_, sizeof(void*)*1);
if (v_hasTrace_2267_ == 0)
{
lean_object* v_traceClass_2268_; lean_object* v___x_2269_; lean_object* v_a_2270_; uint8_t v___x_2271_; 
v_traceClass_2268_ = lean_ctor_get(v___x_2069_, 0);
v___x_2269_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2069_, v___y_2264_);
v_a_2270_ = lean_ctor_get(v___x_2269_, 0);
lean_inc(v_a_2270_);
lean_dec_ref(v___x_2269_);
v___x_2271_ = lean_unbox(v_a_2270_);
if (v___x_2271_ == 0)
{
lean_dec(v_a_2270_);
lean_inc(v_traceClass_2268_);
v___y_2071_ = v_traceClass_2268_;
v___y_2072_ = v___y_2265_;
v___y_2073_ = v___y_2263_;
v___y_2074_ = v___y_2262_;
v___y_2075_ = v___y_2264_;
goto v___jp_2070_;
}
else
{
lean_object* v_slotClusters_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; uint8_t v___x_2278_; lean_object* v___x_2279_; 
v_slotClusters_2272_ = lean_ctor_get(v_a_2028_, 2);
v___x_2273_ = lean_unsigned_to_nat(0u);
v___x_2274_ = lean_array_get_size(v_slotClusters_2272_);
v___x_2275_ = lean_unsigned_to_nat(1u);
v___x_2276_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2276_, 0, v___x_2273_);
lean_ctor_set(v___x_2276_, 1, v___x_2274_);
lean_ctor_set(v___x_2276_, 2, v___x_2275_);
v___x_2277_ = lean_box(0);
v___x_2278_ = lean_unbox(v_a_2270_);
lean_dec(v_a_2270_);
v___x_2279_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(v_slotClusters_2272_, v___x_2278_, v___x_2276_, v___x_2277_, v___x_2273_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_);
lean_dec_ref_known(v___x_2276_, 3);
if (lean_obj_tag(v___x_2279_) == 0)
{
lean_dec_ref_known(v___x_2279_, 1);
lean_inc(v_traceClass_2268_);
v___y_2071_ = v_traceClass_2268_;
v___y_2072_ = v___y_2265_;
v___y_2073_ = v___y_2263_;
v___y_2074_ = v___y_2262_;
v___y_2075_ = v___y_2264_;
goto v___jp_2070_;
}
else
{
lean_inc(v_traceClass_2268_);
v___y_2096_ = v_traceClass_2268_;
v___y_2097_ = v___y_2265_;
v___y_2098_ = v___y_2263_;
v___y_2099_ = v___y_2262_;
v___y_2100_ = v___y_2264_;
v___y_2101_ = v___x_2279_;
goto v___jp_2095_;
}
}
}
else
{
lean_object* v_traceClass_2280_; lean_object* v_inheritedTraceOptions_2281_; lean_object* v___f_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; uint8_t v___x_2286_; 
v_traceClass_2280_ = lean_ctor_get(v___x_2069_, 0);
v_inheritedTraceOptions_2281_ = lean_ctor_get(v___y_2264_, 13);
v___f_2282_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__4, &lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__4_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___closed__4);
v___x_2283_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1));
v___x_2284_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__3));
lean_inc(v_traceClass_2280_);
v___x_2285_ = l_Lean_Name_append(v___x_2284_, v_traceClass_2280_);
v___x_2286_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2281_, v_options_2266_, v___x_2285_);
lean_dec(v___x_2285_);
if (v___x_2286_ == 0)
{
lean_object* v___x_2287_; uint8_t v___x_2288_; 
v___x_2287_ = l_Lean_trace_profiler;
v___x_2288_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_options_2266_, v___x_2287_);
if (v___x_2288_ == 0)
{
lean_object* v___x_2289_; lean_object* v_a_2290_; uint8_t v___x_2291_; 
v___x_2289_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2069_, v___y_2264_);
v_a_2290_ = lean_ctor_get(v___x_2289_, 0);
lean_inc(v_a_2290_);
lean_dec_ref(v___x_2289_);
v___x_2291_ = lean_unbox(v_a_2290_);
lean_dec(v_a_2290_);
if (v___x_2291_ == 0)
{
lean_inc(v_traceClass_2280_);
v___y_2071_ = v_traceClass_2280_;
v___y_2072_ = v___y_2265_;
v___y_2073_ = v___y_2263_;
v___y_2074_ = v___y_2262_;
v___y_2075_ = v___y_2264_;
goto v___jp_2070_;
}
else
{
lean_object* v_slotClusters_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; 
v_slotClusters_2292_ = lean_ctor_get(v_a_2028_, 2);
v___x_2293_ = lean_unsigned_to_nat(0u);
v___x_2294_ = lean_array_get_size(v_slotClusters_2292_);
v___x_2295_ = lean_unsigned_to_nat(1u);
v___x_2296_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2296_, 0, v___x_2293_);
lean_ctor_set(v___x_2296_, 1, v___x_2294_);
lean_ctor_set(v___x_2296_, 2, v___x_2295_);
v___x_2297_ = lean_box(0);
v___x_2298_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(v_slotClusters_2292_, v_hasTrace_2267_, v___x_2296_, v___x_2297_, v___x_2293_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_);
lean_dec_ref_known(v___x_2296_, 3);
if (lean_obj_tag(v___x_2298_) == 0)
{
lean_dec_ref_known(v___x_2298_, 1);
lean_inc(v_traceClass_2280_);
v___y_2071_ = v_traceClass_2280_;
v___y_2072_ = v___y_2265_;
v___y_2073_ = v___y_2263_;
v___y_2074_ = v___y_2262_;
v___y_2075_ = v___y_2264_;
goto v___jp_2070_;
}
else
{
lean_inc(v_traceClass_2280_);
v___y_2096_ = v_traceClass_2280_;
v___y_2097_ = v___y_2265_;
v___y_2098_ = v___y_2263_;
v___y_2099_ = v___y_2262_;
v___y_2100_ = v___y_2264_;
v___y_2101_ = v___x_2298_;
goto v___jp_2095_;
}
}
}
else
{
lean_inc(v_traceClass_2280_);
v___y_2218_ = v_traceClass_2280_;
v___y_2219_ = v___y_2265_;
v___y_2220_ = v___y_2263_;
v___y_2221_ = v___x_2283_;
v___y_2222_ = v_options_2266_;
v___y_2223_ = v___x_2286_;
v___y_2224_ = v___y_2262_;
v___y_2225_ = v_hasTrace_2267_;
v___y_2226_ = v___f_2282_;
v___y_2227_ = v_hasTrace_2267_;
v___y_2228_ = v___y_2264_;
goto v___jp_2217_;
}
}
else
{
lean_inc(v_traceClass_2280_);
v___y_2218_ = v_traceClass_2280_;
v___y_2219_ = v___y_2265_;
v___y_2220_ = v___y_2263_;
v___y_2221_ = v___x_2283_;
v___y_2222_ = v_options_2266_;
v___y_2223_ = v___x_2286_;
v___y_2224_ = v___y_2262_;
v___y_2225_ = v_hasTrace_2267_;
v___y_2226_ = v___f_2282_;
v___y_2227_ = v_hasTrace_2267_;
v___y_2228_ = v___y_2264_;
goto v___jp_2217_;
}
}
}
}
}
else
{
lean_object* v_a_2326_; lean_object* v___x_2328_; uint8_t v_isShared_2329_; uint8_t v_isSharedCheck_2333_; 
lean_dec(v_a_2022_);
lean_dec(v_a_2020_);
lean_dec_ref(v_t_2009_);
v_a_2326_ = lean_ctor_get(v___x_2027_, 0);
v_isSharedCheck_2333_ = !lean_is_exclusive(v___x_2027_);
if (v_isSharedCheck_2333_ == 0)
{
v___x_2328_ = v___x_2027_;
v_isShared_2329_ = v_isSharedCheck_2333_;
goto v_resetjp_2327_;
}
else
{
lean_inc(v_a_2326_);
lean_dec(v___x_2027_);
v___x_2328_ = lean_box(0);
v_isShared_2329_ = v_isSharedCheck_2333_;
goto v_resetjp_2327_;
}
v_resetjp_2327_:
{
lean_object* v___x_2331_; 
if (v_isShared_2329_ == 0)
{
v___x_2331_ = v___x_2328_;
goto v_reusejp_2330_;
}
else
{
lean_object* v_reuseFailAlloc_2332_; 
v_reuseFailAlloc_2332_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2332_, 0, v_a_2326_);
v___x_2331_ = v_reuseFailAlloc_2332_;
goto v_reusejp_2330_;
}
v_reusejp_2330_:
{
return v___x_2331_;
}
}
}
}
else
{
lean_object* v_a_2334_; lean_object* v___x_2336_; uint8_t v_isShared_2337_; uint8_t v_isSharedCheck_2341_; 
lean_dec(v_a_2022_);
lean_dec(v_a_2020_);
lean_dec(v_pat_x3f_2011_);
lean_dec_ref(v_t_2009_);
v_a_2334_ = lean_ctor_get(v___x_2025_, 0);
v_isSharedCheck_2341_ = !lean_is_exclusive(v___x_2025_);
if (v_isSharedCheck_2341_ == 0)
{
v___x_2336_ = v___x_2025_;
v_isShared_2337_ = v_isSharedCheck_2341_;
goto v_resetjp_2335_;
}
else
{
lean_inc(v_a_2334_);
lean_dec(v___x_2025_);
v___x_2336_ = lean_box(0);
v_isShared_2337_ = v_isSharedCheck_2341_;
goto v_resetjp_2335_;
}
v_resetjp_2335_:
{
lean_object* v___x_2339_; 
if (v_isShared_2337_ == 0)
{
v___x_2339_ = v___x_2336_;
goto v_reusejp_2338_;
}
else
{
lean_object* v_reuseFailAlloc_2340_; 
v_reuseFailAlloc_2340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2340_, 0, v_a_2334_);
v___x_2339_ = v_reuseFailAlloc_2340_;
goto v_reusejp_2338_;
}
v_reusejp_2338_:
{
return v___x_2339_;
}
}
}
}
else
{
lean_object* v_a_2342_; lean_object* v___x_2344_; uint8_t v_isShared_2345_; uint8_t v_isSharedCheck_2349_; 
lean_dec(v_a_2022_);
lean_dec(v_a_2020_);
lean_dec(v_pat_x3f_2011_);
lean_dec(v_immediate_x3f_2010_);
lean_dec_ref(v_t_2009_);
v_a_2342_ = lean_ctor_get(v___x_2023_, 0);
v_isSharedCheck_2349_ = !lean_is_exclusive(v___x_2023_);
if (v_isSharedCheck_2349_ == 0)
{
v___x_2344_ = v___x_2023_;
v_isShared_2345_ = v_isSharedCheck_2349_;
goto v_resetjp_2343_;
}
else
{
lean_inc(v_a_2342_);
lean_dec(v___x_2023_);
v___x_2344_ = lean_box(0);
v_isShared_2345_ = v_isSharedCheck_2349_;
goto v_resetjp_2343_;
}
v_resetjp_2343_:
{
lean_object* v___x_2347_; 
if (v_isShared_2345_ == 0)
{
v___x_2347_ = v___x_2344_;
goto v_reusejp_2346_;
}
else
{
lean_object* v_reuseFailAlloc_2348_; 
v_reuseFailAlloc_2348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2348_, 0, v_a_2342_);
v___x_2347_ = v_reuseFailAlloc_2348_;
goto v_reusejp_2346_;
}
v_reusejp_2346_:
{
return v___x_2347_;
}
}
}
}
else
{
lean_object* v_a_2350_; lean_object* v___x_2352_; uint8_t v_isShared_2353_; uint8_t v_isSharedCheck_2357_; 
lean_dec(v_a_2020_);
lean_dec(v_pat_x3f_2011_);
lean_dec(v_immediate_x3f_2010_);
lean_dec_ref(v_t_2009_);
v_a_2350_ = lean_ctor_get(v___x_2021_, 0);
v_isSharedCheck_2357_ = !lean_is_exclusive(v___x_2021_);
if (v_isSharedCheck_2357_ == 0)
{
v___x_2352_ = v___x_2021_;
v_isShared_2353_ = v_isSharedCheck_2357_;
goto v_resetjp_2351_;
}
else
{
lean_inc(v_a_2350_);
lean_dec(v___x_2021_);
v___x_2352_ = lean_box(0);
v_isShared_2353_ = v_isSharedCheck_2357_;
goto v_resetjp_2351_;
}
v_resetjp_2351_:
{
lean_object* v___x_2355_; 
if (v_isShared_2353_ == 0)
{
v___x_2355_ = v___x_2352_;
goto v_reusejp_2354_;
}
else
{
lean_object* v_reuseFailAlloc_2356_; 
v_reuseFailAlloc_2356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2356_, 0, v_a_2350_);
v___x_2355_ = v_reuseFailAlloc_2356_;
goto v_reusejp_2354_;
}
v_reusejp_2354_:
{
return v___x_2355_;
}
}
}
}
else
{
lean_object* v_a_2358_; lean_object* v___x_2360_; uint8_t v_isShared_2361_; uint8_t v_isSharedCheck_2365_; 
lean_dec(v_pat_x3f_2011_);
lean_dec(v_immediate_x3f_2010_);
lean_dec_ref(v_t_2009_);
v_a_2358_ = lean_ctor_get(v___x_2019_, 0);
v_isSharedCheck_2365_ = !lean_is_exclusive(v___x_2019_);
if (v_isSharedCheck_2365_ == 0)
{
v___x_2360_ = v___x_2019_;
v_isShared_2361_ = v_isSharedCheck_2365_;
goto v_resetjp_2359_;
}
else
{
lean_inc(v_a_2358_);
lean_dec(v___x_2019_);
v___x_2360_ = lean_box(0);
v_isShared_2361_ = v_isSharedCheck_2365_;
goto v_resetjp_2359_;
}
v_resetjp_2359_:
{
lean_object* v___x_2363_; 
if (v_isShared_2361_ == 0)
{
v___x_2363_ = v___x_2360_;
goto v_reusejp_2362_;
}
else
{
lean_object* v_reuseFailAlloc_2364_; 
v_reuseFailAlloc_2364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2364_, 0, v_a_2358_);
v___x_2363_ = v_reuseFailAlloc_2364_;
goto v_reusejp_2362_;
}
v_reusejp_2362_:
{
return v___x_2363_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore_u2082___boxed(lean_object* v_t_2366_, lean_object* v_immediate_x3f_2367_, lean_object* v_pat_x3f_2368_, lean_object* v_phase_2369_, lean_object* v_isDestruct_2370_, lean_object* v_a_2371_, lean_object* v_a_2372_, lean_object* v_a_2373_, lean_object* v_a_2374_, lean_object* v_a_2375_){
_start:
{
uint8_t v_isDestruct_boxed_2376_; lean_object* v_res_2377_; 
v_isDestruct_boxed_2376_ = lean_unbox(v_isDestruct_2370_);
v_res_2377_ = lp_aesop_Aesop_RuleBuilder_forwardCore_u2082(v_t_2366_, v_immediate_x3f_2367_, v_pat_x3f_2368_, v_phase_2369_, v_isDestruct_boxed_2376_, v_a_2371_, v_a_2372_, v_a_2373_, v_a_2374_);
lean_dec(v_a_2374_);
lean_dec_ref(v_a_2373_);
lean_dec(v_a_2372_);
lean_dec_ref(v_a_2371_);
lean_dec_ref(v_phase_2369_);
return v_res_2377_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0(lean_object* v_opt_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_){
_start:
{
lean_object* v___x_2384_; 
v___x_2384_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v_opt_2378_, v___y_2381_);
return v___x_2384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___boxed(lean_object* v_opt_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_){
_start:
{
lean_object* v_res_2391_; 
v_res_2391_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0(v_opt_2385_, v___y_2386_, v___y_2387_, v___y_2388_, v___y_2389_);
lean_dec(v___y_2389_);
lean_dec_ref(v___y_2388_);
lean_dec(v___y_2387_);
lean_dec_ref(v___y_2386_);
lean_dec_ref(v_opt_2385_);
return v_res_2391_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6(lean_object* v_00_u03b1_2392_, lean_object* v_x_2393_, lean_object* v___y_2394_, lean_object* v___y_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_){
_start:
{
lean_object* v___x_2399_; 
v___x_2399_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___redArg(v_x_2393_);
return v___x_2399_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6___boxed(lean_object* v_00_u03b1_2400_, lean_object* v_x_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_, lean_object* v___y_2406_){
_start:
{
lean_object* v_res_2407_; 
v_res_2407_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__6(v_00_u03b1_2400_, v_x_2401_, v___y_2402_, v___y_2403_, v___y_2404_, v___y_2405_);
lean_dec(v___y_2405_);
lean_dec_ref(v___y_2404_);
lean_dec(v___y_2403_);
lean_dec_ref(v___y_2402_);
return v_res_2407_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10(lean_object* v___x_2408_, uint8_t v___x_2409_, lean_object* v_range_2410_, lean_object* v_b_2411_, lean_object* v_i_2412_, lean_object* v_hs_2413_, lean_object* v_hl_2414_, lean_object* v___y_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_, lean_object* v___y_2418_){
_start:
{
lean_object* v___x_2420_; 
v___x_2420_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg(v___x_2408_, v___x_2409_, v_range_2410_, v_b_2411_, v_i_2412_, v___y_2415_, v___y_2416_, v___y_2417_, v___y_2418_);
return v___x_2420_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___boxed(lean_object* v___x_2421_, lean_object* v___x_2422_, lean_object* v_range_2423_, lean_object* v_b_2424_, lean_object* v_i_2425_, lean_object* v_hs_2426_, lean_object* v_hl_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_, lean_object* v___y_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_){
_start:
{
uint8_t v___x_60010__boxed_2433_; lean_object* v_res_2434_; 
v___x_60010__boxed_2433_ = lean_unbox(v___x_2422_);
v_res_2434_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10(v___x_2421_, v___x_60010__boxed_2433_, v_range_2423_, v_b_2424_, v_i_2425_, v_hs_2426_, v_hl_2427_, v___y_2428_, v___y_2429_, v___y_2430_, v___y_2431_);
lean_dec(v___y_2431_);
lean_dec_ref(v___y_2430_);
lean_dec(v___y_2429_);
lean_dec_ref(v___y_2428_);
lean_dec_ref(v_range_2423_);
lean_dec_ref(v___x_2421_);
return v_res_2434_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10(lean_object* v_n_2435_, lean_object* v_as_2436_, lean_object* v_lo_2437_, lean_object* v_hi_2438_, lean_object* v_w_2439_, lean_object* v_hlo_2440_, lean_object* v_hhi_2441_){
_start:
{
lean_object* v___x_2442_; 
v___x_2442_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___redArg(v_n_2435_, v_as_2436_, v_lo_2437_, v_hi_2438_);
return v___x_2442_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10___boxed(lean_object* v_n_2443_, lean_object* v_as_2444_, lean_object* v_lo_2445_, lean_object* v_hi_2446_, lean_object* v_w_2447_, lean_object* v_hlo_2448_, lean_object* v_hhi_2449_){
_start:
{
lean_object* v_res_2450_; 
v_res_2450_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10(v_n_2443_, v_as_2444_, v_lo_2445_, v_hi_2446_, v_w_2447_, v_hlo_2448_, v_hhi_2449_);
lean_dec(v_hi_2446_);
lean_dec(v_n_2443_);
return v_res_2450_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12(lean_object* v_n_2451_, lean_object* v_lo_2452_, lean_object* v_hi_2453_, lean_object* v_hhi_2454_, lean_object* v_pivot_2455_, lean_object* v_as_2456_, lean_object* v_i_2457_, lean_object* v_k_2458_, lean_object* v_ilo_2459_, lean_object* v_ik_2460_, lean_object* v_w_2461_){
_start:
{
lean_object* v___x_2462_; 
v___x_2462_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___redArg(v_hi_2453_, v_pivot_2455_, v_as_2456_, v_i_2457_, v_k_2458_);
return v___x_2462_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12___boxed(lean_object* v_n_2463_, lean_object* v_lo_2464_, lean_object* v_hi_2465_, lean_object* v_hhi_2466_, lean_object* v_pivot_2467_, lean_object* v_as_2468_, lean_object* v_i_2469_, lean_object* v_k_2470_, lean_object* v_ilo_2471_, lean_object* v_ik_2472_, lean_object* v_w_2473_){
_start:
{
lean_object* v_res_2474_; 
v_res_2474_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__6_spec__10_spec__12(v_n_2463_, v_lo_2464_, v_hi_2465_, v_hhi_2466_, v_pivot_2467_, v_as_2468_, v_i_2469_, v_k_2470_, v_ilo_2471_, v_ik_2472_, v_w_2473_);
lean_dec(v_pivot_2467_);
lean_dec(v_hi_2465_);
lean_dec(v_lo_2464_);
lean_dec(v_n_2463_);
return v_res_2474_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_RuleBuilder_forwardCore_spec__0(lean_object* v_msg_2475_){
_start:
{
lean_object* v___x_2476_; lean_object* v___x_2477_; 
v___x_2476_ = lp_aesop_Aesop_instInhabitedBaseRuleSetMember_default;
v___x_2477_ = lean_panic_fn_borrowed(v___x_2476_, v_msg_2475_);
return v___x_2477_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__3(void){
_start:
{
lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; 
v___x_2481_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore___closed__2));
v___x_2482_ = lean_unsigned_to_nat(11u);
v___x_2483_ = lean_unsigned_to_nat(145u);
v___x_2484_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore___closed__1));
v___x_2485_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore___closed__0));
v___x_2486_ = l_mkPanicMessageWithDecl(v___x_2485_, v___x_2484_, v___x_2483_, v___x_2482_, v___x_2481_);
return v___x_2486_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__5(void){
_start:
{
lean_object* v___x_2488_; lean_object* v___x_2489_; 
v___x_2488_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore___closed__4));
v___x_2489_ = l_Lean_stringToMessageData(v___x_2488_);
return v___x_2489_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__7(void){
_start:
{
lean_object* v___x_2491_; lean_object* v___x_2492_; 
v___x_2491_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore___closed__6));
v___x_2492_ = l_Lean_stringToMessageData(v___x_2491_);
return v___x_2492_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__9(void){
_start:
{
lean_object* v___x_2494_; lean_object* v___x_2495_; 
v___x_2494_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forwardCore___closed__8));
v___x_2495_ = l_Lean_stringToMessageData(v___x_2494_);
return v___x_2495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore(lean_object* v_t_2496_, lean_object* v_immediate_x3f_2497_, lean_object* v_pat_x3f_2498_, lean_object* v_phase_2499_, uint8_t v_isDestruct_2500_, lean_object* v_a_2501_, lean_object* v_a_2502_, lean_object* v_a_2503_, lean_object* v_a_2504_){
_start:
{
lean_object* v___y_2507_; lean_object* v___y_2512_; lean_object* v___y_2513_; uint8_t v___y_2514_; lean_object* v___y_2515_; lean_object* v___y_2516_; lean_object* v___y_2517_; lean_object* v___y_2518_; lean_object* v___y_2554_; lean_object* v___y_2555_; lean_object* v___y_2556_; uint8_t v___y_2557_; lean_object* v___y_2558_; lean_object* v___y_2559_; lean_object* v___y_2560_; lean_object* v___y_2561_; lean_object* v___y_2590_; lean_object* v___y_2591_; uint8_t v___y_2592_; lean_object* v___y_2593_; lean_object* v___y_2594_; lean_object* v___y_2595_; lean_object* v___y_2596_; uint8_t v___y_2627_; 
if (v_isDestruct_2500_ == 0)
{
uint8_t v___x_2665_; 
v___x_2665_ = 4;
v___y_2627_ = v___x_2665_;
goto v___jp_2626_;
}
else
{
uint8_t v___x_2666_; 
v___x_2666_ = 3;
v___y_2627_ = v___x_2666_;
goto v___jp_2626_;
}
v___jp_2506_:
{
lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; 
v___x_2508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2508_, 0, v___y_2507_);
v___x_2509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2509_, 0, v___x_2508_);
v___x_2510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2510_, 0, v___x_2509_);
return v___x_2510_;
}
v___jp_2511_:
{
lean_object* v___x_2519_; lean_object* v___x_2520_; 
lean_inc_ref_n(v_t_2496_, 2);
v___x_2519_ = lp_aesop_Aesop_ElabRuleTerm_toRuleTerm(v_t_2496_);
v___x_2520_ = lp_aesop_Aesop_ElabRuleTerm_name(v_t_2496_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
if (lean_obj_tag(v___x_2520_) == 0)
{
lean_object* v_a_2521_; lean_object* v___x_2522_; uint8_t v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; 
v_a_2521_ = lean_ctor_get(v___x_2520_, 0);
lean_inc(v_a_2521_);
lean_dec_ref_known(v___x_2520_, 1);
v___x_2522_ = lean_alloc_ctor(2, 2, 1);
lean_ctor_set(v___x_2522_, 0, v___x_2519_);
lean_ctor_set(v___x_2522_, 1, v___y_2512_);
lean_ctor_set_uint8(v___x_2522_, sizeof(void*)*2, v_isDestruct_2500_);
v___x_2523_ = lp_aesop_Aesop_ElabRuleTerm_scope(v_t_2496_);
lean_inc(v_pat_x3f_2498_);
lean_inc_ref(v_phase_2499_);
v___x_2524_ = lp_aesop_Aesop_PhaseSpec_toRule(v_phase_2499_, v_a_2521_, v___y_2514_, v___x_2523_, v___x_2522_, v___y_2513_, v_pat_x3f_2498_);
v___x_2525_ = lp_aesop_Aesop_RuleBuilder_forwardCore_u2082(v_t_2496_, v_immediate_x3f_2497_, v_pat_x3f_2498_, v_phase_2499_, v_isDestruct_2500_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
lean_dec_ref(v_phase_2499_);
if (lean_obj_tag(v___x_2525_) == 0)
{
switch(lean_obj_tag(v___x_2524_))
{
case 0:
{
lean_object* v_a_2526_; lean_object* v_r_2527_; lean_object* v___x_2528_; 
v_a_2526_ = lean_ctor_get(v___x_2525_, 0);
lean_inc(v_a_2526_);
lean_dec_ref_known(v___x_2525_, 1);
v_r_2527_ = lean_ctor_get(v___x_2524_, 0);
lean_inc_ref(v_r_2527_);
lean_dec_ref_known(v___x_2524_, 1);
v___x_2528_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2528_, 0, v_a_2526_);
lean_ctor_set(v___x_2528_, 1, v_r_2527_);
v___y_2507_ = v___x_2528_;
goto v___jp_2506_;
}
case 2:
{
lean_object* v_a_2529_; lean_object* v_r_2530_; lean_object* v___x_2531_; 
v_a_2529_ = lean_ctor_get(v___x_2525_, 0);
lean_inc(v_a_2529_);
lean_dec_ref_known(v___x_2525_, 1);
v_r_2530_ = lean_ctor_get(v___x_2524_, 0);
lean_inc_ref(v_r_2530_);
lean_dec_ref_known(v___x_2524_, 1);
v___x_2531_ = lean_alloc_ctor(6, 2, 0);
lean_ctor_set(v___x_2531_, 0, v_a_2529_);
lean_ctor_set(v___x_2531_, 1, v_r_2530_);
v___y_2507_ = v___x_2531_;
goto v___jp_2506_;
}
case 1:
{
lean_object* v_a_2532_; lean_object* v_r_2533_; lean_object* v___x_2534_; 
v_a_2532_ = lean_ctor_get(v___x_2525_, 0);
lean_inc(v_a_2532_);
lean_dec_ref_known(v___x_2525_, 1);
v_r_2533_ = lean_ctor_get(v___x_2524_, 0);
lean_inc_ref(v_r_2533_);
lean_dec_ref_known(v___x_2524_, 1);
v___x_2534_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2534_, 0, v_a_2532_);
lean_ctor_set(v___x_2534_, 1, v_r_2533_);
v___y_2507_ = v___x_2534_;
goto v___jp_2506_;
}
default: 
{
lean_object* v___x_2535_; lean_object* v___x_2536_; 
lean_dec_ref_known(v___x_2525_, 1);
lean_dec_ref(v___x_2524_);
v___x_2535_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__3, &lp_aesop_Aesop_RuleBuilder_forwardCore___closed__3_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__3);
v___x_2536_ = lp_aesop_panic___at___00Aesop_RuleBuilder_forwardCore_spec__0(v___x_2535_);
v___y_2507_ = v___x_2536_;
goto v___jp_2506_;
}
}
}
else
{
lean_object* v_a_2537_; lean_object* v___x_2539_; uint8_t v_isShared_2540_; uint8_t v_isSharedCheck_2544_; 
lean_dec_ref(v___x_2524_);
v_a_2537_ = lean_ctor_get(v___x_2525_, 0);
v_isSharedCheck_2544_ = !lean_is_exclusive(v___x_2525_);
if (v_isSharedCheck_2544_ == 0)
{
v___x_2539_ = v___x_2525_;
v_isShared_2540_ = v_isSharedCheck_2544_;
goto v_resetjp_2538_;
}
else
{
lean_inc(v_a_2537_);
lean_dec(v___x_2525_);
v___x_2539_ = lean_box(0);
v_isShared_2540_ = v_isSharedCheck_2544_;
goto v_resetjp_2538_;
}
v_resetjp_2538_:
{
lean_object* v___x_2542_; 
if (v_isShared_2540_ == 0)
{
v___x_2542_ = v___x_2539_;
goto v_reusejp_2541_;
}
else
{
lean_object* v_reuseFailAlloc_2543_; 
v_reuseFailAlloc_2543_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2543_, 0, v_a_2537_);
v___x_2542_ = v_reuseFailAlloc_2543_;
goto v_reusejp_2541_;
}
v_reusejp_2541_:
{
return v___x_2542_;
}
}
}
}
else
{
lean_object* v_a_2545_; lean_object* v___x_2547_; uint8_t v_isShared_2548_; uint8_t v_isSharedCheck_2552_; 
lean_dec_ref(v___x_2519_);
lean_dec(v___y_2513_);
lean_dec_ref(v___y_2512_);
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2545_ = lean_ctor_get(v___x_2520_, 0);
v_isSharedCheck_2552_ = !lean_is_exclusive(v___x_2520_);
if (v_isSharedCheck_2552_ == 0)
{
v___x_2547_ = v___x_2520_;
v_isShared_2548_ = v_isSharedCheck_2552_;
goto v_resetjp_2546_;
}
else
{
lean_inc(v_a_2545_);
lean_dec(v___x_2520_);
v___x_2547_ = lean_box(0);
v_isShared_2548_ = v_isSharedCheck_2552_;
goto v_resetjp_2546_;
}
v_resetjp_2546_:
{
lean_object* v___x_2550_; 
if (v_isShared_2548_ == 0)
{
v___x_2550_ = v___x_2547_;
goto v_reusejp_2549_;
}
else
{
lean_object* v_reuseFailAlloc_2551_; 
v_reuseFailAlloc_2551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2551_, 0, v_a_2545_);
v___x_2550_ = v_reuseFailAlloc_2551_;
goto v_reusejp_2549_;
}
v_reusejp_2549_:
{
return v___x_2550_;
}
}
}
}
v___jp_2553_:
{
lean_object* v___x_2562_; 
lean_inc_ref(v___y_2554_);
v___x_2562_ = lp_aesop_Aesop_RuleBuilder_getForwardIndexingMode(v___y_2556_, v___y_2554_, v___y_2558_, v___y_2559_, v___y_2560_, v___y_2561_);
if (lean_obj_tag(v___x_2562_) == 0)
{
lean_object* v_a_2563_; lean_object* v___x_2564_; lean_object* v_a_2565_; uint8_t v___x_2566_; 
v_a_2563_ = lean_ctor_get(v___x_2562_, 0);
lean_inc(v_a_2563_);
lean_dec_ref_known(v___x_2562_, 1);
v___x_2564_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___y_2555_, v___y_2560_);
v_a_2565_ = lean_ctor_get(v___x_2564_, 0);
lean_inc(v_a_2565_);
lean_dec_ref(v___x_2564_);
v___x_2566_ = lean_unbox(v_a_2565_);
lean_dec(v_a_2565_);
if (v___x_2566_ == 0)
{
lean_dec_ref(v___y_2555_);
v___y_2512_ = v___y_2554_;
v___y_2513_ = v_a_2563_;
v___y_2514_ = v___y_2557_;
v___y_2515_ = v___y_2558_;
v___y_2516_ = v___y_2559_;
v___y_2517_ = v___y_2560_;
v___y_2518_ = v___y_2561_;
goto v___jp_2511_;
}
else
{
lean_object* v_traceClass_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; 
v_traceClass_2567_ = lean_ctor_get(v___y_2555_, 0);
lean_inc(v_traceClass_2567_);
lean_dec_ref(v___y_2555_);
v___x_2568_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__5, &lp_aesop_Aesop_RuleBuilder_forwardCore___closed__5_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__5);
lean_inc(v_a_2563_);
v___x_2569_ = lp_aesop_Aesop_IndexingMode_format(v_a_2563_);
v___x_2570_ = l_Lean_MessageData_ofFormat(v___x_2569_);
v___x_2571_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2571_, 0, v___x_2568_);
lean_ctor_set(v___x_2571_, 1, v___x_2570_);
v___x_2572_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v_traceClass_2567_, v___x_2571_, v___y_2558_, v___y_2559_, v___y_2560_, v___y_2561_);
if (lean_obj_tag(v___x_2572_) == 0)
{
lean_dec_ref_known(v___x_2572_, 1);
v___y_2512_ = v___y_2554_;
v___y_2513_ = v_a_2563_;
v___y_2514_ = v___y_2557_;
v___y_2515_ = v___y_2558_;
v___y_2516_ = v___y_2559_;
v___y_2517_ = v___y_2560_;
v___y_2518_ = v___y_2561_;
goto v___jp_2511_;
}
else
{
lean_object* v_a_2573_; lean_object* v___x_2575_; uint8_t v_isShared_2576_; uint8_t v_isSharedCheck_2580_; 
lean_dec(v_a_2563_);
lean_dec_ref(v___y_2554_);
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2573_ = lean_ctor_get(v___x_2572_, 0);
v_isSharedCheck_2580_ = !lean_is_exclusive(v___x_2572_);
if (v_isSharedCheck_2580_ == 0)
{
v___x_2575_ = v___x_2572_;
v_isShared_2576_ = v_isSharedCheck_2580_;
goto v_resetjp_2574_;
}
else
{
lean_inc(v_a_2573_);
lean_dec(v___x_2572_);
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
else
{
lean_object* v_a_2581_; lean_object* v___x_2583_; uint8_t v_isShared_2584_; uint8_t v_isSharedCheck_2588_; 
lean_dec_ref(v___y_2555_);
lean_dec_ref(v___y_2554_);
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2581_ = lean_ctor_get(v___x_2562_, 0);
v_isSharedCheck_2588_ = !lean_is_exclusive(v___x_2562_);
if (v_isSharedCheck_2588_ == 0)
{
v___x_2583_ = v___x_2562_;
v_isShared_2584_ = v_isSharedCheck_2588_;
goto v_resetjp_2582_;
}
else
{
lean_inc(v_a_2581_);
lean_dec(v___x_2562_);
v___x_2583_ = lean_box(0);
v_isShared_2584_ = v_isSharedCheck_2588_;
goto v_resetjp_2582_;
}
v_resetjp_2582_:
{
lean_object* v___x_2586_; 
if (v_isShared_2584_ == 0)
{
v___x_2586_ = v___x_2583_;
goto v_reusejp_2585_;
}
else
{
lean_object* v_reuseFailAlloc_2587_; 
v_reuseFailAlloc_2587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2587_, 0, v_a_2581_);
v___x_2586_ = v_reuseFailAlloc_2587_;
goto v_reusejp_2585_;
}
v_reusejp_2585_:
{
return v___x_2586_;
}
}
}
}
v___jp_2589_:
{
lean_object* v___x_2597_; 
lean_inc(v_immediate_x3f_2497_);
lean_inc(v_pat_x3f_2498_);
lean_inc_ref(v___y_2591_);
v___x_2597_ = lp_aesop_Aesop_RuleBuilder_getImmediatePremises(v___y_2591_, v_pat_x3f_2498_, v_immediate_x3f_2497_, v___y_2593_, v___y_2594_, v___y_2595_, v___y_2596_);
if (lean_obj_tag(v___x_2597_) == 0)
{
lean_object* v_a_2598_; lean_object* v___x_2599_; lean_object* v_a_2600_; uint8_t v___x_2601_; 
v_a_2598_ = lean_ctor_get(v___x_2597_, 0);
lean_inc(v_a_2598_);
lean_dec_ref_known(v___x_2597_, 1);
v___x_2599_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___y_2590_, v___y_2595_);
v_a_2600_ = lean_ctor_get(v___x_2599_, 0);
lean_inc(v_a_2600_);
lean_dec_ref(v___x_2599_);
v___x_2601_ = lean_unbox(v_a_2600_);
lean_dec(v_a_2600_);
if (v___x_2601_ == 0)
{
v___y_2554_ = v_a_2598_;
v___y_2555_ = v___y_2590_;
v___y_2556_ = v___y_2591_;
v___y_2557_ = v___y_2592_;
v___y_2558_ = v___y_2593_;
v___y_2559_ = v___y_2594_;
v___y_2560_ = v___y_2595_;
v___y_2561_ = v___y_2596_;
goto v___jp_2553_;
}
else
{
lean_object* v_traceClass_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; 
v_traceClass_2602_ = lean_ctor_get(v___y_2590_, 0);
v___x_2603_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__7, &lp_aesop_Aesop_RuleBuilder_forwardCore___closed__7_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__7);
lean_inc(v_a_2598_);
v___x_2604_ = lean_array_to_list(v_a_2598_);
v___x_2605_ = lean_box(0);
v___x_2606_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__1(v___x_2604_, v___x_2605_);
v___x_2607_ = l_Lean_MessageData_ofList(v___x_2606_);
v___x_2608_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2608_, 0, v___x_2603_);
lean_ctor_set(v___x_2608_, 1, v___x_2607_);
lean_inc(v_traceClass_2602_);
v___x_2609_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v_traceClass_2602_, v___x_2608_, v___y_2593_, v___y_2594_, v___y_2595_, v___y_2596_);
if (lean_obj_tag(v___x_2609_) == 0)
{
lean_dec_ref_known(v___x_2609_, 1);
v___y_2554_ = v_a_2598_;
v___y_2555_ = v___y_2590_;
v___y_2556_ = v___y_2591_;
v___y_2557_ = v___y_2592_;
v___y_2558_ = v___y_2593_;
v___y_2559_ = v___y_2594_;
v___y_2560_ = v___y_2595_;
v___y_2561_ = v___y_2596_;
goto v___jp_2553_;
}
else
{
lean_object* v_a_2610_; lean_object* v___x_2612_; uint8_t v_isShared_2613_; uint8_t v_isSharedCheck_2617_; 
lean_dec(v_a_2598_);
lean_dec_ref(v___y_2591_);
lean_dec_ref(v___y_2590_);
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2610_ = lean_ctor_get(v___x_2609_, 0);
v_isSharedCheck_2617_ = !lean_is_exclusive(v___x_2609_);
if (v_isSharedCheck_2617_ == 0)
{
v___x_2612_ = v___x_2609_;
v_isShared_2613_ = v_isSharedCheck_2617_;
goto v_resetjp_2611_;
}
else
{
lean_inc(v_a_2610_);
lean_dec(v___x_2609_);
v___x_2612_ = lean_box(0);
v_isShared_2613_ = v_isSharedCheck_2617_;
goto v_resetjp_2611_;
}
v_resetjp_2611_:
{
lean_object* v___x_2615_; 
if (v_isShared_2613_ == 0)
{
v___x_2615_ = v___x_2612_;
goto v_reusejp_2614_;
}
else
{
lean_object* v_reuseFailAlloc_2616_; 
v_reuseFailAlloc_2616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2616_, 0, v_a_2610_);
v___x_2615_ = v_reuseFailAlloc_2616_;
goto v_reusejp_2614_;
}
v_reusejp_2614_:
{
return v___x_2615_;
}
}
}
}
}
else
{
lean_object* v_a_2618_; lean_object* v___x_2620_; uint8_t v_isShared_2621_; uint8_t v_isSharedCheck_2625_; 
lean_dec_ref(v___y_2591_);
lean_dec_ref(v___y_2590_);
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2618_ = lean_ctor_get(v___x_2597_, 0);
v_isSharedCheck_2625_ = !lean_is_exclusive(v___x_2597_);
if (v_isSharedCheck_2625_ == 0)
{
v___x_2620_ = v___x_2597_;
v_isShared_2621_ = v_isSharedCheck_2625_;
goto v_resetjp_2619_;
}
else
{
lean_inc(v_a_2618_);
lean_dec(v___x_2597_);
v___x_2620_ = lean_box(0);
v_isShared_2621_ = v_isSharedCheck_2625_;
goto v_resetjp_2619_;
}
v_resetjp_2619_:
{
lean_object* v___x_2623_; 
if (v_isShared_2621_ == 0)
{
v___x_2623_ = v___x_2620_;
goto v_reusejp_2622_;
}
else
{
lean_object* v_reuseFailAlloc_2624_; 
v_reuseFailAlloc_2624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2624_, 0, v_a_2618_);
v___x_2623_ = v_reuseFailAlloc_2624_;
goto v_reusejp_2622_;
}
v_reusejp_2622_:
{
return v___x_2623_;
}
}
}
}
v___jp_2626_:
{
lean_object* v___x_2628_; 
lean_inc_ref(v_t_2496_);
v___x_2628_ = lp_aesop_Aesop_ElabRuleTerm_expr(v_t_2496_, v_a_2501_, v_a_2502_, v_a_2503_, v_a_2504_);
if (lean_obj_tag(v___x_2628_) == 0)
{
lean_object* v_a_2629_; lean_object* v___x_2630_; 
v_a_2629_ = lean_ctor_get(v___x_2628_, 0);
lean_inc(v_a_2629_);
lean_dec_ref_known(v___x_2628_, 1);
lean_inc(v_a_2504_);
lean_inc_ref(v_a_2503_);
lean_inc(v_a_2502_);
lean_inc_ref(v_a_2501_);
v___x_2630_ = lean_infer_type(v_a_2629_, v_a_2501_, v_a_2502_, v_a_2503_, v_a_2504_);
if (lean_obj_tag(v___x_2630_) == 0)
{
lean_object* v_a_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v_a_2634_; uint8_t v___x_2635_; 
v_a_2631_ = lean_ctor_get(v___x_2630_, 0);
lean_inc(v_a_2631_);
lean_dec_ref_known(v___x_2630_, 1);
v___x_2632_ = lp_aesop_Aesop_TraceOption_debug;
v___x_2633_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__0___redArg(v___x_2632_, v_a_2503_);
v_a_2634_ = lean_ctor_get(v___x_2633_, 0);
lean_inc(v_a_2634_);
lean_dec_ref(v___x_2633_);
v___x_2635_ = lean_unbox(v_a_2634_);
lean_dec(v_a_2634_);
if (v___x_2635_ == 0)
{
v___y_2590_ = v___x_2632_;
v___y_2591_ = v_a_2631_;
v___y_2592_ = v___y_2627_;
v___y_2593_ = v_a_2501_;
v___y_2594_ = v_a_2502_;
v___y_2595_ = v_a_2503_;
v___y_2596_ = v_a_2504_;
goto v___jp_2589_;
}
else
{
lean_object* v_traceClass_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; 
v_traceClass_2636_ = lean_ctor_get(v___x_2632_, 0);
v___x_2637_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forwardCore___closed__9, &lp_aesop_Aesop_RuleBuilder_forwardCore___closed__9_once, _init_lp_aesop_Aesop_RuleBuilder_forwardCore___closed__9);
lean_inc(v_a_2631_);
v___x_2638_ = l_Lean_MessageData_ofExpr(v_a_2631_);
v___x_2639_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2639_, 0, v___x_2637_);
lean_ctor_set(v___x_2639_, 1, v___x_2638_);
lean_inc(v_traceClass_2636_);
v___x_2640_ = lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2(v_traceClass_2636_, v___x_2639_, v_a_2501_, v_a_2502_, v_a_2503_, v_a_2504_);
if (lean_obj_tag(v___x_2640_) == 0)
{
lean_dec_ref_known(v___x_2640_, 1);
v___y_2590_ = v___x_2632_;
v___y_2591_ = v_a_2631_;
v___y_2592_ = v___y_2627_;
v___y_2593_ = v_a_2501_;
v___y_2594_ = v_a_2502_;
v___y_2595_ = v_a_2503_;
v___y_2596_ = v_a_2504_;
goto v___jp_2589_;
}
else
{
lean_object* v_a_2641_; lean_object* v___x_2643_; uint8_t v_isShared_2644_; uint8_t v_isSharedCheck_2648_; 
lean_dec(v_a_2631_);
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2641_ = lean_ctor_get(v___x_2640_, 0);
v_isSharedCheck_2648_ = !lean_is_exclusive(v___x_2640_);
if (v_isSharedCheck_2648_ == 0)
{
v___x_2643_ = v___x_2640_;
v_isShared_2644_ = v_isSharedCheck_2648_;
goto v_resetjp_2642_;
}
else
{
lean_inc(v_a_2641_);
lean_dec(v___x_2640_);
v___x_2643_ = lean_box(0);
v_isShared_2644_ = v_isSharedCheck_2648_;
goto v_resetjp_2642_;
}
v_resetjp_2642_:
{
lean_object* v___x_2646_; 
if (v_isShared_2644_ == 0)
{
v___x_2646_ = v___x_2643_;
goto v_reusejp_2645_;
}
else
{
lean_object* v_reuseFailAlloc_2647_; 
v_reuseFailAlloc_2647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2647_, 0, v_a_2641_);
v___x_2646_ = v_reuseFailAlloc_2647_;
goto v_reusejp_2645_;
}
v_reusejp_2645_:
{
return v___x_2646_;
}
}
}
}
}
else
{
lean_object* v_a_2649_; lean_object* v___x_2651_; uint8_t v_isShared_2652_; uint8_t v_isSharedCheck_2656_; 
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2649_ = lean_ctor_get(v___x_2630_, 0);
v_isSharedCheck_2656_ = !lean_is_exclusive(v___x_2630_);
if (v_isSharedCheck_2656_ == 0)
{
v___x_2651_ = v___x_2630_;
v_isShared_2652_ = v_isSharedCheck_2656_;
goto v_resetjp_2650_;
}
else
{
lean_inc(v_a_2649_);
lean_dec(v___x_2630_);
v___x_2651_ = lean_box(0);
v_isShared_2652_ = v_isSharedCheck_2656_;
goto v_resetjp_2650_;
}
v_resetjp_2650_:
{
lean_object* v___x_2654_; 
if (v_isShared_2652_ == 0)
{
v___x_2654_ = v___x_2651_;
goto v_reusejp_2653_;
}
else
{
lean_object* v_reuseFailAlloc_2655_; 
v_reuseFailAlloc_2655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2655_, 0, v_a_2649_);
v___x_2654_ = v_reuseFailAlloc_2655_;
goto v_reusejp_2653_;
}
v_reusejp_2653_:
{
return v___x_2654_;
}
}
}
}
else
{
lean_object* v_a_2657_; lean_object* v___x_2659_; uint8_t v_isShared_2660_; uint8_t v_isSharedCheck_2664_; 
lean_dec_ref(v_phase_2499_);
lean_dec(v_pat_x3f_2498_);
lean_dec(v_immediate_x3f_2497_);
lean_dec_ref(v_t_2496_);
v_a_2657_ = lean_ctor_get(v___x_2628_, 0);
v_isSharedCheck_2664_ = !lean_is_exclusive(v___x_2628_);
if (v_isSharedCheck_2664_ == 0)
{
v___x_2659_ = v___x_2628_;
v_isShared_2660_ = v_isSharedCheck_2664_;
goto v_resetjp_2658_;
}
else
{
lean_inc(v_a_2657_);
lean_dec(v___x_2628_);
v___x_2659_ = lean_box(0);
v_isShared_2660_ = v_isSharedCheck_2664_;
goto v_resetjp_2658_;
}
v_resetjp_2658_:
{
lean_object* v___x_2662_; 
if (v_isShared_2660_ == 0)
{
v___x_2662_ = v___x_2659_;
goto v_reusejp_2661_;
}
else
{
lean_object* v_reuseFailAlloc_2663_; 
v_reuseFailAlloc_2663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2663_, 0, v_a_2657_);
v___x_2662_ = v_reuseFailAlloc_2663_;
goto v_reusejp_2661_;
}
v_reusejp_2661_:
{
return v___x_2662_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forwardCore___boxed(lean_object* v_t_2667_, lean_object* v_immediate_x3f_2668_, lean_object* v_pat_x3f_2669_, lean_object* v_phase_2670_, lean_object* v_isDestruct_2671_, lean_object* v_a_2672_, lean_object* v_a_2673_, lean_object* v_a_2674_, lean_object* v_a_2675_, lean_object* v_a_2676_){
_start:
{
uint8_t v_isDestruct_boxed_2677_; lean_object* v_res_2678_; 
v_isDestruct_boxed_2677_ = lean_unbox(v_isDestruct_2671_);
v_res_2678_ = lp_aesop_Aesop_RuleBuilder_forwardCore(v_t_2667_, v_immediate_x3f_2668_, v_pat_x3f_2669_, v_phase_2670_, v_isDestruct_boxed_2677_, v_a_2672_, v_a_2673_, v_a_2674_, v_a_2675_);
lean_dec(v_a_2675_);
lean_dec_ref(v_a_2674_);
lean_dec(v_a_2673_);
lean_dec_ref(v_a_2672_);
return v_res_2678_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg(lean_object* v___y_2679_){
_start:
{
lean_object* v___x_2681_; lean_object* v_traceState_2682_; lean_object* v_traces_2683_; lean_object* v___x_2684_; lean_object* v_traceState_2685_; lean_object* v_env_2686_; lean_object* v_nextMacroScope_2687_; lean_object* v_ngen_2688_; lean_object* v_auxDeclNGen_2689_; lean_object* v_cache_2690_; lean_object* v_messages_2691_; lean_object* v_infoState_2692_; lean_object* v_snapshotTasks_2693_; lean_object* v___x_2695_; uint8_t v_isShared_2696_; uint8_t v_isSharedCheck_2714_; 
v___x_2681_ = lean_st_ref_get(v___y_2679_);
v_traceState_2682_ = lean_ctor_get(v___x_2681_, 4);
lean_inc_ref(v_traceState_2682_);
lean_dec(v___x_2681_);
v_traces_2683_ = lean_ctor_get(v_traceState_2682_, 0);
lean_inc_ref(v_traces_2683_);
lean_dec_ref(v_traceState_2682_);
v___x_2684_ = lean_st_ref_take(v___y_2679_);
v_traceState_2685_ = lean_ctor_get(v___x_2684_, 4);
v_env_2686_ = lean_ctor_get(v___x_2684_, 0);
v_nextMacroScope_2687_ = lean_ctor_get(v___x_2684_, 1);
v_ngen_2688_ = lean_ctor_get(v___x_2684_, 2);
v_auxDeclNGen_2689_ = lean_ctor_get(v___x_2684_, 3);
v_cache_2690_ = lean_ctor_get(v___x_2684_, 5);
v_messages_2691_ = lean_ctor_get(v___x_2684_, 6);
v_infoState_2692_ = lean_ctor_get(v___x_2684_, 7);
v_snapshotTasks_2693_ = lean_ctor_get(v___x_2684_, 8);
v_isSharedCheck_2714_ = !lean_is_exclusive(v___x_2684_);
if (v_isSharedCheck_2714_ == 0)
{
v___x_2695_ = v___x_2684_;
v_isShared_2696_ = v_isSharedCheck_2714_;
goto v_resetjp_2694_;
}
else
{
lean_inc(v_snapshotTasks_2693_);
lean_inc(v_infoState_2692_);
lean_inc(v_messages_2691_);
lean_inc(v_cache_2690_);
lean_inc(v_traceState_2685_);
lean_inc(v_auxDeclNGen_2689_);
lean_inc(v_ngen_2688_);
lean_inc(v_nextMacroScope_2687_);
lean_inc(v_env_2686_);
lean_dec(v___x_2684_);
v___x_2695_ = lean_box(0);
v_isShared_2696_ = v_isSharedCheck_2714_;
goto v_resetjp_2694_;
}
v_resetjp_2694_:
{
uint64_t v_tid_2697_; lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2712_; 
v_tid_2697_ = lean_ctor_get_uint64(v_traceState_2685_, sizeof(void*)*1);
v_isSharedCheck_2712_ = !lean_is_exclusive(v_traceState_2685_);
if (v_isSharedCheck_2712_ == 0)
{
lean_object* v_unused_2713_; 
v_unused_2713_ = lean_ctor_get(v_traceState_2685_, 0);
lean_dec(v_unused_2713_);
v___x_2699_ = v_traceState_2685_;
v_isShared_2700_ = v_isSharedCheck_2712_;
goto v_resetjp_2698_;
}
else
{
lean_dec(v_traceState_2685_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2712_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2705_; 
v___x_2701_ = lean_unsigned_to_nat(32u);
v___x_2702_ = lean_mk_empty_array_with_capacity(v___x_2701_);
lean_dec_ref(v___x_2702_);
v___x_2703_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__3___redArg___closed__1);
if (v_isShared_2700_ == 0)
{
lean_ctor_set(v___x_2699_, 0, v___x_2703_);
v___x_2705_ = v___x_2699_;
goto v_reusejp_2704_;
}
else
{
lean_object* v_reuseFailAlloc_2711_; 
v_reuseFailAlloc_2711_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2711_, 0, v___x_2703_);
lean_ctor_set_uint64(v_reuseFailAlloc_2711_, sizeof(void*)*1, v_tid_2697_);
v___x_2705_ = v_reuseFailAlloc_2711_;
goto v_reusejp_2704_;
}
v_reusejp_2704_:
{
lean_object* v___x_2707_; 
if (v_isShared_2696_ == 0)
{
lean_ctor_set(v___x_2695_, 4, v___x_2705_);
v___x_2707_ = v___x_2695_;
goto v_reusejp_2706_;
}
else
{
lean_object* v_reuseFailAlloc_2710_; 
v_reuseFailAlloc_2710_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2710_, 0, v_env_2686_);
lean_ctor_set(v_reuseFailAlloc_2710_, 1, v_nextMacroScope_2687_);
lean_ctor_set(v_reuseFailAlloc_2710_, 2, v_ngen_2688_);
lean_ctor_set(v_reuseFailAlloc_2710_, 3, v_auxDeclNGen_2689_);
lean_ctor_set(v_reuseFailAlloc_2710_, 4, v___x_2705_);
lean_ctor_set(v_reuseFailAlloc_2710_, 5, v_cache_2690_);
lean_ctor_set(v_reuseFailAlloc_2710_, 6, v_messages_2691_);
lean_ctor_set(v_reuseFailAlloc_2710_, 7, v_infoState_2692_);
lean_ctor_set(v_reuseFailAlloc_2710_, 8, v_snapshotTasks_2693_);
v___x_2707_ = v_reuseFailAlloc_2710_;
goto v_reusejp_2706_;
}
v_reusejp_2706_:
{
lean_object* v___x_2708_; lean_object* v___x_2709_; 
v___x_2708_ = lean_st_ref_set(v___y_2679_, v___x_2707_);
v___x_2709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2709_, 0, v_traces_2683_);
return v___x_2709_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg___boxed(lean_object* v___y_2715_, lean_object* v___y_2716_){
_start:
{
lean_object* v_res_2717_; 
v_res_2717_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg(v___y_2715_);
lean_dec(v___y_2715_);
return v_res_2717_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0(lean_object* v___y_2718_, lean_object* v___y_2719_, lean_object* v___y_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_){
_start:
{
lean_object* v___x_2726_; 
v___x_2726_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg(v___y_2724_);
return v___x_2726_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___boxed(lean_object* v___y_2727_, lean_object* v___y_2728_, lean_object* v___y_2729_, lean_object* v___y_2730_, lean_object* v___y_2731_, lean_object* v___y_2732_, lean_object* v___y_2733_, lean_object* v___y_2734_){
_start:
{
lean_object* v_res_2735_; 
v_res_2735_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0(v___y_2727_, v___y_2728_, v___y_2729_, v___y_2730_, v___y_2731_, v___y_2732_, v___y_2733_);
lean_dec(v___y_2733_);
lean_dec_ref(v___y_2732_);
lean_dec(v___y_2731_);
lean_dec_ref(v___y_2730_);
lean_dec(v___y_2729_);
lean_dec_ref(v___y_2728_);
lean_dec_ref(v___y_2727_);
return v_res_2735_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward___lam__0(lean_object* v___x_2736_, lean_object* v_x_2737_, lean_object* v___y_2738_, lean_object* v___y_2739_, lean_object* v___y_2740_, lean_object* v___y_2741_, lean_object* v___y_2742_, lean_object* v___y_2743_, lean_object* v___y_2744_){
_start:
{
lean_object* v___x_2746_; 
v___x_2746_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2746_, 0, v___x_2736_);
return v___x_2746_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward___lam__0___boxed(lean_object* v___x_2747_, lean_object* v_x_2748_, lean_object* v___y_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_, lean_object* v___y_2755_, lean_object* v___y_2756_){
_start:
{
lean_object* v_res_2757_; 
v_res_2757_ = lp_aesop_Aesop_RuleBuilder_forward___lam__0(v___x_2747_, v_x_2748_, v___y_2749_, v___y_2750_, v___y_2751_, v___y_2752_, v___y_2753_, v___y_2754_, v___y_2755_);
lean_dec(v___y_2755_);
lean_dec_ref(v___y_2754_);
lean_dec(v___y_2753_);
lean_dec_ref(v___y_2752_);
lean_dec(v___y_2751_);
lean_dec_ref(v___y_2750_);
lean_dec_ref(v___y_2749_);
lean_dec_ref(v_x_2748_);
return v_res_2757_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__3(lean_object* v_e_2758_){
_start:
{
if (lean_obj_tag(v_e_2758_) == 0)
{
uint8_t v___x_2759_; 
v___x_2759_ = 2;
return v___x_2759_;
}
else
{
uint8_t v___x_2760_; 
v___x_2760_ = 0;
return v___x_2760_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__3___boxed(lean_object* v_e_2761_){
_start:
{
uint8_t v_res_2762_; lean_object* v_r_2763_; 
v_res_2762_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__3(v_e_2761_);
lean_dec_ref(v_e_2761_);
v_r_2763_ = lean_box(v_res_2762_);
return v_r_2763_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg(lean_object* v_x_2764_){
_start:
{
if (lean_obj_tag(v_x_2764_) == 0)
{
lean_object* v_a_2766_; lean_object* v___x_2768_; uint8_t v_isShared_2769_; uint8_t v_isSharedCheck_2773_; 
v_a_2766_ = lean_ctor_get(v_x_2764_, 0);
v_isSharedCheck_2773_ = !lean_is_exclusive(v_x_2764_);
if (v_isSharedCheck_2773_ == 0)
{
v___x_2768_ = v_x_2764_;
v_isShared_2769_ = v_isSharedCheck_2773_;
goto v_resetjp_2767_;
}
else
{
lean_inc(v_a_2766_);
lean_dec(v_x_2764_);
v___x_2768_ = lean_box(0);
v_isShared_2769_ = v_isSharedCheck_2773_;
goto v_resetjp_2767_;
}
v_resetjp_2767_:
{
lean_object* v___x_2771_; 
if (v_isShared_2769_ == 0)
{
lean_ctor_set_tag(v___x_2768_, 1);
v___x_2771_ = v___x_2768_;
goto v_reusejp_2770_;
}
else
{
lean_object* v_reuseFailAlloc_2772_; 
v_reuseFailAlloc_2772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2772_, 0, v_a_2766_);
v___x_2771_ = v_reuseFailAlloc_2772_;
goto v_reusejp_2770_;
}
v_reusejp_2770_:
{
return v___x_2771_;
}
}
}
else
{
lean_object* v_a_2774_; lean_object* v___x_2776_; uint8_t v_isShared_2777_; uint8_t v_isSharedCheck_2781_; 
v_a_2774_ = lean_ctor_get(v_x_2764_, 0);
v_isSharedCheck_2781_ = !lean_is_exclusive(v_x_2764_);
if (v_isSharedCheck_2781_ == 0)
{
v___x_2776_ = v_x_2764_;
v_isShared_2777_ = v_isSharedCheck_2781_;
goto v_resetjp_2775_;
}
else
{
lean_inc(v_a_2774_);
lean_dec(v_x_2764_);
v___x_2776_ = lean_box(0);
v_isShared_2777_ = v_isSharedCheck_2781_;
goto v_resetjp_2775_;
}
v_resetjp_2775_:
{
lean_object* v___x_2779_; 
if (v_isShared_2777_ == 0)
{
lean_ctor_set_tag(v___x_2776_, 0);
v___x_2779_ = v___x_2776_;
goto v_reusejp_2778_;
}
else
{
lean_object* v_reuseFailAlloc_2780_; 
v_reuseFailAlloc_2780_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2780_, 0, v_a_2774_);
v___x_2779_ = v_reuseFailAlloc_2780_;
goto v_reusejp_2778_;
}
v_reusejp_2778_:
{
return v___x_2779_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg___boxed(lean_object* v_x_2782_, lean_object* v___y_2783_){
_start:
{
lean_object* v_res_2784_; 
v_res_2784_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg(v_x_2782_);
return v_res_2784_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg(lean_object* v_oldTraces_2785_, lean_object* v_data_2786_, lean_object* v_ref_2787_, lean_object* v_msg_2788_, lean_object* v___y_2789_, lean_object* v___y_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_){
_start:
{
lean_object* v_fileName_2794_; lean_object* v_fileMap_2795_; lean_object* v_options_2796_; lean_object* v_currRecDepth_2797_; lean_object* v_maxRecDepth_2798_; lean_object* v_ref_2799_; lean_object* v_currNamespace_2800_; lean_object* v_openDecls_2801_; lean_object* v_initHeartbeats_2802_; lean_object* v_maxHeartbeats_2803_; lean_object* v_quotContext_2804_; lean_object* v_currMacroScope_2805_; uint8_t v_diag_2806_; lean_object* v_cancelTk_x3f_2807_; uint8_t v_suppressElabErrors_2808_; lean_object* v_inheritedTraceOptions_2809_; lean_object* v___x_2810_; lean_object* v_traceState_2811_; lean_object* v_traces_2812_; lean_object* v_ref_2813_; lean_object* v___x_2814_; lean_object* v___x_2815_; size_t v_sz_2816_; size_t v___x_2817_; lean_object* v___x_2818_; lean_object* v_msg_2819_; lean_object* v___x_2820_; lean_object* v_a_2821_; lean_object* v___x_2823_; uint8_t v_isShared_2824_; uint8_t v_isSharedCheck_2858_; 
v_fileName_2794_ = lean_ctor_get(v___y_2791_, 0);
v_fileMap_2795_ = lean_ctor_get(v___y_2791_, 1);
v_options_2796_ = lean_ctor_get(v___y_2791_, 2);
v_currRecDepth_2797_ = lean_ctor_get(v___y_2791_, 3);
v_maxRecDepth_2798_ = lean_ctor_get(v___y_2791_, 4);
v_ref_2799_ = lean_ctor_get(v___y_2791_, 5);
v_currNamespace_2800_ = lean_ctor_get(v___y_2791_, 6);
v_openDecls_2801_ = lean_ctor_get(v___y_2791_, 7);
v_initHeartbeats_2802_ = lean_ctor_get(v___y_2791_, 8);
v_maxHeartbeats_2803_ = lean_ctor_get(v___y_2791_, 9);
v_quotContext_2804_ = lean_ctor_get(v___y_2791_, 10);
v_currMacroScope_2805_ = lean_ctor_get(v___y_2791_, 11);
v_diag_2806_ = lean_ctor_get_uint8(v___y_2791_, sizeof(void*)*14);
v_cancelTk_x3f_2807_ = lean_ctor_get(v___y_2791_, 12);
v_suppressElabErrors_2808_ = lean_ctor_get_uint8(v___y_2791_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2809_ = lean_ctor_get(v___y_2791_, 13);
v___x_2810_ = lean_st_ref_get(v___y_2792_);
v_traceState_2811_ = lean_ctor_get(v___x_2810_, 4);
lean_inc_ref(v_traceState_2811_);
lean_dec(v___x_2810_);
v_traces_2812_ = lean_ctor_get(v_traceState_2811_, 0);
lean_inc_ref(v_traces_2812_);
lean_dec_ref(v_traceState_2811_);
v_ref_2813_ = l_Lean_replaceRef(v_ref_2787_, v_ref_2799_);
lean_inc_ref(v_inheritedTraceOptions_2809_);
lean_inc(v_cancelTk_x3f_2807_);
lean_inc(v_currMacroScope_2805_);
lean_inc(v_quotContext_2804_);
lean_inc(v_maxHeartbeats_2803_);
lean_inc(v_initHeartbeats_2802_);
lean_inc(v_openDecls_2801_);
lean_inc(v_currNamespace_2800_);
lean_inc(v_maxRecDepth_2798_);
lean_inc(v_currRecDepth_2797_);
lean_inc_ref(v_options_2796_);
lean_inc_ref(v_fileMap_2795_);
lean_inc_ref(v_fileName_2794_);
v___x_2814_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2814_, 0, v_fileName_2794_);
lean_ctor_set(v___x_2814_, 1, v_fileMap_2795_);
lean_ctor_set(v___x_2814_, 2, v_options_2796_);
lean_ctor_set(v___x_2814_, 3, v_currRecDepth_2797_);
lean_ctor_set(v___x_2814_, 4, v_maxRecDepth_2798_);
lean_ctor_set(v___x_2814_, 5, v_ref_2813_);
lean_ctor_set(v___x_2814_, 6, v_currNamespace_2800_);
lean_ctor_set(v___x_2814_, 7, v_openDecls_2801_);
lean_ctor_set(v___x_2814_, 8, v_initHeartbeats_2802_);
lean_ctor_set(v___x_2814_, 9, v_maxHeartbeats_2803_);
lean_ctor_set(v___x_2814_, 10, v_quotContext_2804_);
lean_ctor_set(v___x_2814_, 11, v_currMacroScope_2805_);
lean_ctor_set(v___x_2814_, 12, v_cancelTk_x3f_2807_);
lean_ctor_set(v___x_2814_, 13, v_inheritedTraceOptions_2809_);
lean_ctor_set_uint8(v___x_2814_, sizeof(void*)*14, v_diag_2806_);
lean_ctor_set_uint8(v___x_2814_, sizeof(void*)*14 + 1, v_suppressElabErrors_2808_);
v___x_2815_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2812_);
lean_dec_ref(v_traces_2812_);
v_sz_2816_ = lean_array_size(v___x_2815_);
v___x_2817_ = ((size_t)0ULL);
v___x_2818_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__5_spec__6(v_sz_2816_, v___x_2817_, v___x_2815_);
v_msg_2819_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2819_, 0, v_data_2786_);
lean_ctor_set(v_msg_2819_, 1, v_msg_2788_);
lean_ctor_set(v_msg_2819_, 2, v___x_2818_);
v___x_2820_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_RuleBuilder_getForwardIndexingMode_spec__2_spec__3(v_msg_2819_, v___y_2789_, v___y_2790_, v___x_2814_, v___y_2792_);
lean_dec_ref_known(v___x_2814_, 14);
v_a_2821_ = lean_ctor_get(v___x_2820_, 0);
v_isSharedCheck_2858_ = !lean_is_exclusive(v___x_2820_);
if (v_isSharedCheck_2858_ == 0)
{
v___x_2823_ = v___x_2820_;
v_isShared_2824_ = v_isSharedCheck_2858_;
goto v_resetjp_2822_;
}
else
{
lean_inc(v_a_2821_);
lean_dec(v___x_2820_);
v___x_2823_ = lean_box(0);
v_isShared_2824_ = v_isSharedCheck_2858_;
goto v_resetjp_2822_;
}
v_resetjp_2822_:
{
lean_object* v___x_2825_; lean_object* v_traceState_2826_; lean_object* v_env_2827_; lean_object* v_nextMacroScope_2828_; lean_object* v_ngen_2829_; lean_object* v_auxDeclNGen_2830_; lean_object* v_cache_2831_; lean_object* v_messages_2832_; lean_object* v_infoState_2833_; lean_object* v_snapshotTasks_2834_; lean_object* v___x_2836_; uint8_t v_isShared_2837_; uint8_t v_isSharedCheck_2857_; 
v___x_2825_ = lean_st_ref_take(v___y_2792_);
v_traceState_2826_ = lean_ctor_get(v___x_2825_, 4);
v_env_2827_ = lean_ctor_get(v___x_2825_, 0);
v_nextMacroScope_2828_ = lean_ctor_get(v___x_2825_, 1);
v_ngen_2829_ = lean_ctor_get(v___x_2825_, 2);
v_auxDeclNGen_2830_ = lean_ctor_get(v___x_2825_, 3);
v_cache_2831_ = lean_ctor_get(v___x_2825_, 5);
v_messages_2832_ = lean_ctor_get(v___x_2825_, 6);
v_infoState_2833_ = lean_ctor_get(v___x_2825_, 7);
v_snapshotTasks_2834_ = lean_ctor_get(v___x_2825_, 8);
v_isSharedCheck_2857_ = !lean_is_exclusive(v___x_2825_);
if (v_isSharedCheck_2857_ == 0)
{
v___x_2836_ = v___x_2825_;
v_isShared_2837_ = v_isSharedCheck_2857_;
goto v_resetjp_2835_;
}
else
{
lean_inc(v_snapshotTasks_2834_);
lean_inc(v_infoState_2833_);
lean_inc(v_messages_2832_);
lean_inc(v_cache_2831_);
lean_inc(v_traceState_2826_);
lean_inc(v_auxDeclNGen_2830_);
lean_inc(v_ngen_2829_);
lean_inc(v_nextMacroScope_2828_);
lean_inc(v_env_2827_);
lean_dec(v___x_2825_);
v___x_2836_ = lean_box(0);
v_isShared_2837_ = v_isSharedCheck_2857_;
goto v_resetjp_2835_;
}
v_resetjp_2835_:
{
uint64_t v_tid_2838_; lean_object* v___x_2840_; uint8_t v_isShared_2841_; uint8_t v_isSharedCheck_2855_; 
v_tid_2838_ = lean_ctor_get_uint64(v_traceState_2826_, sizeof(void*)*1);
v_isSharedCheck_2855_ = !lean_is_exclusive(v_traceState_2826_);
if (v_isSharedCheck_2855_ == 0)
{
lean_object* v_unused_2856_; 
v_unused_2856_ = lean_ctor_get(v_traceState_2826_, 0);
lean_dec(v_unused_2856_);
v___x_2840_ = v_traceState_2826_;
v_isShared_2841_ = v_isSharedCheck_2855_;
goto v_resetjp_2839_;
}
else
{
lean_dec(v_traceState_2826_);
v___x_2840_ = lean_box(0);
v_isShared_2841_ = v_isSharedCheck_2855_;
goto v_resetjp_2839_;
}
v_resetjp_2839_:
{
lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2845_; 
v___x_2842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2842_, 0, v_ref_2787_);
lean_ctor_set(v___x_2842_, 1, v_a_2821_);
v___x_2843_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2785_, v___x_2842_);
if (v_isShared_2841_ == 0)
{
lean_ctor_set(v___x_2840_, 0, v___x_2843_);
v___x_2845_ = v___x_2840_;
goto v_reusejp_2844_;
}
else
{
lean_object* v_reuseFailAlloc_2854_; 
v_reuseFailAlloc_2854_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2854_, 0, v___x_2843_);
lean_ctor_set_uint64(v_reuseFailAlloc_2854_, sizeof(void*)*1, v_tid_2838_);
v___x_2845_ = v_reuseFailAlloc_2854_;
goto v_reusejp_2844_;
}
v_reusejp_2844_:
{
lean_object* v___x_2847_; 
if (v_isShared_2837_ == 0)
{
lean_ctor_set(v___x_2836_, 4, v___x_2845_);
v___x_2847_ = v___x_2836_;
goto v_reusejp_2846_;
}
else
{
lean_object* v_reuseFailAlloc_2853_; 
v_reuseFailAlloc_2853_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2853_, 0, v_env_2827_);
lean_ctor_set(v_reuseFailAlloc_2853_, 1, v_nextMacroScope_2828_);
lean_ctor_set(v_reuseFailAlloc_2853_, 2, v_ngen_2829_);
lean_ctor_set(v_reuseFailAlloc_2853_, 3, v_auxDeclNGen_2830_);
lean_ctor_set(v_reuseFailAlloc_2853_, 4, v___x_2845_);
lean_ctor_set(v_reuseFailAlloc_2853_, 5, v_cache_2831_);
lean_ctor_set(v_reuseFailAlloc_2853_, 6, v_messages_2832_);
lean_ctor_set(v_reuseFailAlloc_2853_, 7, v_infoState_2833_);
lean_ctor_set(v_reuseFailAlloc_2853_, 8, v_snapshotTasks_2834_);
v___x_2847_ = v_reuseFailAlloc_2853_;
goto v_reusejp_2846_;
}
v_reusejp_2846_:
{
lean_object* v___x_2848_; lean_object* v___x_2849_; lean_object* v___x_2851_; 
v___x_2848_ = lean_st_ref_set(v___y_2792_, v___x_2847_);
v___x_2849_ = lean_box(0);
if (v_isShared_2824_ == 0)
{
lean_ctor_set(v___x_2823_, 0, v___x_2849_);
v___x_2851_ = v___x_2823_;
goto v_reusejp_2850_;
}
else
{
lean_object* v_reuseFailAlloc_2852_; 
v_reuseFailAlloc_2852_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2852_, 0, v___x_2849_);
v___x_2851_ = v_reuseFailAlloc_2852_;
goto v_reusejp_2850_;
}
v_reusejp_2850_:
{
return v___x_2851_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg___boxed(lean_object* v_oldTraces_2859_, lean_object* v_data_2860_, lean_object* v_ref_2861_, lean_object* v_msg_2862_, lean_object* v___y_2863_, lean_object* v___y_2864_, lean_object* v___y_2865_, lean_object* v___y_2866_, lean_object* v___y_2867_){
_start:
{
lean_object* v_res_2868_; 
v_res_2868_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg(v_oldTraces_2859_, v_data_2860_, v_ref_2861_, v_msg_2862_, v___y_2863_, v___y_2864_, v___y_2865_, v___y_2866_);
lean_dec(v___y_2866_);
lean_dec_ref(v___y_2865_);
lean_dec(v___y_2864_);
lean_dec_ref(v___y_2863_);
return v_res_2868_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1(lean_object* v_cls_2869_, uint8_t v_collapsed_2870_, lean_object* v_tag_2871_, lean_object* v_opts_2872_, uint8_t v_clsEnabled_2873_, lean_object* v_oldTraces_2874_, lean_object* v_msg_2875_, lean_object* v_resStartStop_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_, lean_object* v___y_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_){
_start:
{
lean_object* v_fst_2885_; lean_object* v_snd_2886_; lean_object* v___y_2888_; lean_object* v___y_2889_; lean_object* v_data_2890_; lean_object* v_fst_2901_; lean_object* v_snd_2902_; lean_object* v___x_2903_; uint8_t v___x_2904_; lean_object* v___y_2906_; lean_object* v_a_2907_; uint8_t v___y_2922_; double v___y_2953_; 
v_fst_2885_ = lean_ctor_get(v_resStartStop_2876_, 0);
lean_inc(v_fst_2885_);
v_snd_2886_ = lean_ctor_get(v_resStartStop_2876_, 1);
lean_inc(v_snd_2886_);
lean_dec_ref(v_resStartStop_2876_);
v_fst_2901_ = lean_ctor_get(v_snd_2886_, 0);
lean_inc(v_fst_2901_);
v_snd_2902_ = lean_ctor_get(v_snd_2886_, 1);
lean_inc(v_snd_2902_);
lean_dec(v_snd_2886_);
v___x_2903_ = l_Lean_trace_profiler;
v___x_2904_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_opts_2872_, v___x_2903_);
if (v___x_2904_ == 0)
{
v___y_2922_ = v___x_2904_;
goto v___jp_2921_;
}
else
{
lean_object* v___x_2958_; uint8_t v___x_2959_; 
v___x_2958_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2959_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_opts_2872_, v___x_2958_);
if (v___x_2959_ == 0)
{
lean_object* v___x_2960_; lean_object* v___x_2961_; double v___x_2962_; double v___x_2963_; double v___x_2964_; 
v___x_2960_ = l_Lean_trace_profiler_threshold;
v___x_2961_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(v_opts_2872_, v___x_2960_);
v___x_2962_ = lean_float_of_nat(v___x_2961_);
v___x_2963_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__2);
v___x_2964_ = lean_float_div(v___x_2962_, v___x_2963_);
v___y_2953_ = v___x_2964_;
goto v___jp_2952_;
}
else
{
lean_object* v___x_2965_; lean_object* v___x_2966_; double v___x_2967_; 
v___x_2965_ = l_Lean_trace_profiler_threshold;
v___x_2966_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5_spec__8(v_opts_2872_, v___x_2965_);
v___x_2967_ = lean_float_of_nat(v___x_2966_);
v___y_2953_ = v___x_2967_;
goto v___jp_2952_;
}
}
v___jp_2887_:
{
lean_object* v___x_2891_; 
lean_inc(v___y_2888_);
v___x_2891_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg(v_oldTraces_2874_, v_data_2890_, v___y_2888_, v___y_2889_, v___y_2880_, v___y_2881_, v___y_2882_, v___y_2883_);
if (lean_obj_tag(v___x_2891_) == 0)
{
lean_object* v___x_2892_; 
lean_dec_ref_known(v___x_2891_, 1);
v___x_2892_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg(v_fst_2885_);
return v___x_2892_;
}
else
{
lean_object* v_a_2893_; lean_object* v___x_2895_; uint8_t v_isShared_2896_; uint8_t v_isSharedCheck_2900_; 
lean_dec(v_fst_2885_);
v_a_2893_ = lean_ctor_get(v___x_2891_, 0);
v_isSharedCheck_2900_ = !lean_is_exclusive(v___x_2891_);
if (v_isSharedCheck_2900_ == 0)
{
v___x_2895_ = v___x_2891_;
v_isShared_2896_ = v_isSharedCheck_2900_;
goto v_resetjp_2894_;
}
else
{
lean_inc(v_a_2893_);
lean_dec(v___x_2891_);
v___x_2895_ = lean_box(0);
v_isShared_2896_ = v_isSharedCheck_2900_;
goto v_resetjp_2894_;
}
v_resetjp_2894_:
{
lean_object* v___x_2898_; 
if (v_isShared_2896_ == 0)
{
v___x_2898_ = v___x_2895_;
goto v_reusejp_2897_;
}
else
{
lean_object* v_reuseFailAlloc_2899_; 
v_reuseFailAlloc_2899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2899_, 0, v_a_2893_);
v___x_2898_ = v_reuseFailAlloc_2899_;
goto v_reusejp_2897_;
}
v_reusejp_2897_:
{
return v___x_2898_;
}
}
}
}
v___jp_2905_:
{
uint8_t v_result_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; double v___x_2911_; lean_object* v_data_2912_; 
v_result_2908_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__3(v_fst_2885_);
v___x_2909_ = lean_box(v_result_2908_);
v___x_2910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2910_, 0, v___x_2909_);
v___x_2911_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__0);
lean_inc_ref(v_tag_2871_);
lean_inc_ref(v___x_2910_);
lean_inc(v_cls_2869_);
v_data_2912_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2912_, 0, v_cls_2869_);
lean_ctor_set(v_data_2912_, 1, v___x_2910_);
lean_ctor_set(v_data_2912_, 2, v_tag_2871_);
lean_ctor_set_float(v_data_2912_, sizeof(void*)*3, v___x_2911_);
lean_ctor_set_float(v_data_2912_, sizeof(void*)*3 + 8, v___x_2911_);
lean_ctor_set_uint8(v_data_2912_, sizeof(void*)*3 + 16, v_collapsed_2870_);
if (v___x_2904_ == 0)
{
lean_dec_ref_known(v___x_2910_, 1);
lean_dec(v_snd_2902_);
lean_dec(v_fst_2901_);
lean_dec_ref(v_tag_2871_);
lean_dec(v_cls_2869_);
v___y_2888_ = v___y_2906_;
v___y_2889_ = v_a_2907_;
v_data_2890_ = v_data_2912_;
goto v___jp_2887_;
}
else
{
lean_object* v_data_2913_; double v___x_2914_; double v___x_2915_; 
lean_dec_ref_known(v_data_2912_, 3);
v_data_2913_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2913_, 0, v_cls_2869_);
lean_ctor_set(v_data_2913_, 1, v___x_2910_);
lean_ctor_set(v_data_2913_, 2, v_tag_2871_);
v___x_2914_ = lean_unbox_float(v_fst_2901_);
lean_dec(v_fst_2901_);
lean_ctor_set_float(v_data_2913_, sizeof(void*)*3, v___x_2914_);
v___x_2915_ = lean_unbox_float(v_snd_2902_);
lean_dec(v_snd_2902_);
lean_ctor_set_float(v_data_2913_, sizeof(void*)*3 + 8, v___x_2915_);
lean_ctor_set_uint8(v_data_2913_, sizeof(void*)*3 + 16, v_collapsed_2870_);
v___y_2888_ = v___y_2906_;
v___y_2889_ = v_a_2907_;
v_data_2890_ = v_data_2913_;
goto v___jp_2887_;
}
}
v___jp_2916_:
{
lean_object* v_ref_2917_; lean_object* v___x_2918_; 
v_ref_2917_ = lean_ctor_get(v___y_2882_, 5);
lean_inc(v___y_2883_);
lean_inc_ref(v___y_2882_);
lean_inc(v___y_2881_);
lean_inc_ref(v___y_2880_);
lean_inc(v___y_2879_);
lean_inc_ref(v___y_2878_);
lean_inc_ref(v___y_2877_);
lean_inc(v_fst_2885_);
v___x_2918_ = lean_apply_9(v_msg_2875_, v_fst_2885_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_, v___y_2881_, v___y_2882_, v___y_2883_, lean_box(0));
if (lean_obj_tag(v___x_2918_) == 0)
{
lean_object* v_a_2919_; 
v_a_2919_ = lean_ctor_get(v___x_2918_, 0);
lean_inc(v_a_2919_);
lean_dec_ref_known(v___x_2918_, 1);
v___y_2906_ = v_ref_2917_;
v_a_2907_ = v_a_2919_;
goto v___jp_2905_;
}
else
{
lean_object* v___x_2920_; 
lean_dec_ref_known(v___x_2918_, 1);
v___x_2920_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__5___closed__1);
v___y_2906_ = v_ref_2917_;
v_a_2907_ = v___x_2920_;
goto v___jp_2905_;
}
}
v___jp_2921_:
{
if (v_clsEnabled_2873_ == 0)
{
if (v___y_2922_ == 0)
{
lean_object* v___x_2923_; lean_object* v_traceState_2924_; lean_object* v_env_2925_; lean_object* v_nextMacroScope_2926_; lean_object* v_ngen_2927_; lean_object* v_auxDeclNGen_2928_; lean_object* v_cache_2929_; lean_object* v_messages_2930_; lean_object* v_infoState_2931_; lean_object* v_snapshotTasks_2932_; lean_object* v___x_2934_; uint8_t v_isShared_2935_; uint8_t v_isSharedCheck_2951_; 
lean_dec(v_snd_2902_);
lean_dec(v_fst_2901_);
lean_dec_ref(v_msg_2875_);
lean_dec_ref(v_tag_2871_);
lean_dec(v_cls_2869_);
v___x_2923_ = lean_st_ref_take(v___y_2883_);
v_traceState_2924_ = lean_ctor_get(v___x_2923_, 4);
v_env_2925_ = lean_ctor_get(v___x_2923_, 0);
v_nextMacroScope_2926_ = lean_ctor_get(v___x_2923_, 1);
v_ngen_2927_ = lean_ctor_get(v___x_2923_, 2);
v_auxDeclNGen_2928_ = lean_ctor_get(v___x_2923_, 3);
v_cache_2929_ = lean_ctor_get(v___x_2923_, 5);
v_messages_2930_ = lean_ctor_get(v___x_2923_, 6);
v_infoState_2931_ = lean_ctor_get(v___x_2923_, 7);
v_snapshotTasks_2932_ = lean_ctor_get(v___x_2923_, 8);
v_isSharedCheck_2951_ = !lean_is_exclusive(v___x_2923_);
if (v_isSharedCheck_2951_ == 0)
{
v___x_2934_ = v___x_2923_;
v_isShared_2935_ = v_isSharedCheck_2951_;
goto v_resetjp_2933_;
}
else
{
lean_inc(v_snapshotTasks_2932_);
lean_inc(v_infoState_2931_);
lean_inc(v_messages_2930_);
lean_inc(v_cache_2929_);
lean_inc(v_traceState_2924_);
lean_inc(v_auxDeclNGen_2928_);
lean_inc(v_ngen_2927_);
lean_inc(v_nextMacroScope_2926_);
lean_inc(v_env_2925_);
lean_dec(v___x_2923_);
v___x_2934_ = lean_box(0);
v_isShared_2935_ = v_isSharedCheck_2951_;
goto v_resetjp_2933_;
}
v_resetjp_2933_:
{
uint64_t v_tid_2936_; lean_object* v_traces_2937_; lean_object* v___x_2939_; uint8_t v_isShared_2940_; uint8_t v_isSharedCheck_2950_; 
v_tid_2936_ = lean_ctor_get_uint64(v_traceState_2924_, sizeof(void*)*1);
v_traces_2937_ = lean_ctor_get(v_traceState_2924_, 0);
v_isSharedCheck_2950_ = !lean_is_exclusive(v_traceState_2924_);
if (v_isSharedCheck_2950_ == 0)
{
v___x_2939_ = v_traceState_2924_;
v_isShared_2940_ = v_isSharedCheck_2950_;
goto v_resetjp_2938_;
}
else
{
lean_inc(v_traces_2937_);
lean_dec(v_traceState_2924_);
v___x_2939_ = lean_box(0);
v_isShared_2940_ = v_isSharedCheck_2950_;
goto v_resetjp_2938_;
}
v_resetjp_2938_:
{
lean_object* v___x_2941_; lean_object* v___x_2943_; 
v___x_2941_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2874_, v_traces_2937_);
lean_dec_ref(v_traces_2937_);
if (v_isShared_2940_ == 0)
{
lean_ctor_set(v___x_2939_, 0, v___x_2941_);
v___x_2943_ = v___x_2939_;
goto v_reusejp_2942_;
}
else
{
lean_object* v_reuseFailAlloc_2949_; 
v_reuseFailAlloc_2949_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2949_, 0, v___x_2941_);
lean_ctor_set_uint64(v_reuseFailAlloc_2949_, sizeof(void*)*1, v_tid_2936_);
v___x_2943_ = v_reuseFailAlloc_2949_;
goto v_reusejp_2942_;
}
v_reusejp_2942_:
{
lean_object* v___x_2945_; 
if (v_isShared_2935_ == 0)
{
lean_ctor_set(v___x_2934_, 4, v___x_2943_);
v___x_2945_ = v___x_2934_;
goto v_reusejp_2944_;
}
else
{
lean_object* v_reuseFailAlloc_2948_; 
v_reuseFailAlloc_2948_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2948_, 0, v_env_2925_);
lean_ctor_set(v_reuseFailAlloc_2948_, 1, v_nextMacroScope_2926_);
lean_ctor_set(v_reuseFailAlloc_2948_, 2, v_ngen_2927_);
lean_ctor_set(v_reuseFailAlloc_2948_, 3, v_auxDeclNGen_2928_);
lean_ctor_set(v_reuseFailAlloc_2948_, 4, v___x_2943_);
lean_ctor_set(v_reuseFailAlloc_2948_, 5, v_cache_2929_);
lean_ctor_set(v_reuseFailAlloc_2948_, 6, v_messages_2930_);
lean_ctor_set(v_reuseFailAlloc_2948_, 7, v_infoState_2931_);
lean_ctor_set(v_reuseFailAlloc_2948_, 8, v_snapshotTasks_2932_);
v___x_2945_ = v_reuseFailAlloc_2948_;
goto v_reusejp_2944_;
}
v_reusejp_2944_:
{
lean_object* v___x_2946_; lean_object* v___x_2947_; 
v___x_2946_ = lean_st_ref_set(v___y_2883_, v___x_2945_);
v___x_2947_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg(v_fst_2885_);
return v___x_2947_;
}
}
}
}
}
else
{
goto v___jp_2916_;
}
}
else
{
goto v___jp_2916_;
}
}
v___jp_2952_:
{
double v___x_2954_; double v___x_2955_; double v___x_2956_; uint8_t v___x_2957_; 
v___x_2954_ = lean_unbox_float(v_snd_2902_);
v___x_2955_ = lean_unbox_float(v_fst_2901_);
v___x_2956_ = lean_float_sub(v___x_2954_, v___x_2955_);
v___x_2957_ = lean_float_decLt(v___y_2953_, v___x_2956_);
v___y_2922_ = v___x_2957_;
goto v___jp_2921_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1___boxed(lean_object* v_cls_2968_, lean_object* v_collapsed_2969_, lean_object* v_tag_2970_, lean_object* v_opts_2971_, lean_object* v_clsEnabled_2972_, lean_object* v_oldTraces_2973_, lean_object* v_msg_2974_, lean_object* v_resStartStop_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_){
_start:
{
uint8_t v_collapsed_boxed_2984_; uint8_t v_clsEnabled_boxed_2985_; lean_object* v_res_2986_; 
v_collapsed_boxed_2984_ = lean_unbox(v_collapsed_2969_);
v_clsEnabled_boxed_2985_ = lean_unbox(v_clsEnabled_2972_);
v_res_2986_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1(v_cls_2968_, v_collapsed_boxed_2984_, v_tag_2970_, v_opts_2971_, v_clsEnabled_boxed_2985_, v_oldTraces_2973_, v_msg_2974_, v_resStartStop_2975_, v___y_2976_, v___y_2977_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_);
lean_dec(v___y_2982_);
lean_dec_ref(v___y_2981_);
lean_dec(v___y_2980_);
lean_dec_ref(v___y_2979_);
lean_dec(v___y_2978_);
lean_dec_ref(v___y_2977_);
lean_dec_ref(v___y_2976_);
lean_dec_ref(v_opts_2971_);
return v_res_2986_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forward___closed__2(void){
_start:
{
lean_object* v___x_2990_; lean_object* v___x_2991_; 
v___x_2990_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_forward___closed__1));
v___x_2991_ = l_Lean_MessageData_ofFormat(v___x_2990_);
return v___x_2991_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_forward___closed__3(void){
_start:
{
lean_object* v___x_2992_; lean_object* v___f_2993_; 
v___x_2992_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forward___closed__2, &lp_aesop_Aesop_RuleBuilder_forward___closed__2_once, _init_lp_aesop_Aesop_RuleBuilder_forward___closed__2);
v___f_2993_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleBuilder_forward___lam__0___boxed), 10, 1);
lean_closure_set(v___f_2993_, 0, v___x_2992_);
return v___f_2993_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward(uint8_t v_isDestruct_2994_, lean_object* v_input_2995_, lean_object* v_a_2996_, lean_object* v_a_2997_, lean_object* v_a_2998_, lean_object* v_a_2999_, lean_object* v_a_3000_, lean_object* v_a_3001_, lean_object* v_a_3002_){
_start:
{
lean_object* v_term_3004_; lean_object* v_options_3005_; lean_object* v_phase_3006_; lean_object* v___y_3008_; lean_object* v___y_3009_; lean_object* v___y_3010_; lean_object* v___y_3011_; lean_object* v___y_3012_; lean_object* v_a_3013_; lean_object* v_e_3017_; lean_object* v___y_3018_; lean_object* v___y_3019_; lean_object* v___y_3020_; lean_object* v___y_3021_; lean_object* v___y_3022_; lean_object* v___y_3023_; lean_object* v_options_3045_; uint8_t v_hasTrace_3046_; 
v_term_3004_ = lean_ctor_get(v_input_2995_, 0);
lean_inc(v_term_3004_);
v_options_3005_ = lean_ctor_get(v_input_2995_, 1);
lean_inc_ref(v_options_3005_);
v_phase_3006_ = lean_ctor_get(v_input_2995_, 2);
lean_inc_ref(v_phase_3006_);
lean_dec_ref(v_input_2995_);
v_options_3045_ = lean_ctor_get(v_a_3001_, 2);
v_hasTrace_3046_ = lean_ctor_get_uint8(v_options_3045_, sizeof(void*)*1);
if (v_hasTrace_3046_ == 0)
{
lean_object* v___x_3047_; 
lean_inc(v_term_3004_);
v___x_3047_ = lp_aesop_Aesop_elabRuleTermForApplyLike(v_term_3004_, v_a_2996_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3047_) == 0)
{
lean_object* v_a_3048_; 
v_a_3048_ = lean_ctor_get(v___x_3047_, 0);
lean_inc(v_a_3048_);
lean_dec_ref_known(v___x_3047_, 1);
v_e_3017_ = v_a_3048_;
v___y_3018_ = v_a_2997_;
v___y_3019_ = v_a_2998_;
v___y_3020_ = v_a_2999_;
v___y_3021_ = v_a_3000_;
v___y_3022_ = v_a_3001_;
v___y_3023_ = v_a_3002_;
goto v___jp_3016_;
}
else
{
lean_object* v_a_3049_; lean_object* v___x_3051_; uint8_t v_isShared_3052_; uint8_t v_isSharedCheck_3056_; 
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
lean_dec(v_term_3004_);
v_a_3049_ = lean_ctor_get(v___x_3047_, 0);
v_isSharedCheck_3056_ = !lean_is_exclusive(v___x_3047_);
if (v_isSharedCheck_3056_ == 0)
{
v___x_3051_ = v___x_3047_;
v_isShared_3052_ = v_isSharedCheck_3056_;
goto v_resetjp_3050_;
}
else
{
lean_inc(v_a_3049_);
lean_dec(v___x_3047_);
v___x_3051_ = lean_box(0);
v_isShared_3052_ = v_isSharedCheck_3056_;
goto v_resetjp_3050_;
}
v_resetjp_3050_:
{
lean_object* v___x_3054_; 
if (v_isShared_3052_ == 0)
{
v___x_3054_ = v___x_3051_;
goto v_reusejp_3053_;
}
else
{
lean_object* v_reuseFailAlloc_3055_; 
v_reuseFailAlloc_3055_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3055_, 0, v_a_3049_);
v___x_3054_ = v_reuseFailAlloc_3055_;
goto v_reusejp_3053_;
}
v_reusejp_3053_:
{
return v___x_3054_;
}
}
}
}
else
{
lean_object* v_inheritedTraceOptions_3057_; lean_object* v___x_3058_; lean_object* v_traceClass_3059_; lean_object* v___f_3060_; lean_object* v___x_3061_; lean_object* v___x_3062_; lean_object* v___x_3063_; uint8_t v___x_3064_; lean_object* v___y_3066_; lean_object* v___y_3067_; lean_object* v_a_3068_; lean_object* v___y_3081_; lean_object* v___y_3082_; lean_object* v_a_3083_; lean_object* v___y_3086_; lean_object* v___y_3087_; lean_object* v___y_3088_; lean_object* v_a_3089_; lean_object* v___y_3102_; lean_object* v___y_3103_; lean_object* v_a_3104_; lean_object* v___y_3114_; lean_object* v___y_3115_; lean_object* v_a_3116_; lean_object* v___y_3119_; lean_object* v___y_3120_; lean_object* v___y_3121_; lean_object* v_a_3122_; 
v_inheritedTraceOptions_3057_ = lean_ctor_get(v_a_3001_, 13);
v___x_3058_ = lp_aesop_Aesop_TraceOption_debug;
v_traceClass_3059_ = lean_ctor_get(v___x_3058_, 0);
v___f_3060_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_forward___closed__3, &lp_aesop_Aesop_RuleBuilder_forward___closed__3_once, _init_lp_aesop_Aesop_RuleBuilder_forward___closed__3);
v___x_3061_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__2___closed__1));
v___x_3062_ = ((lean_object*)(lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__3));
lean_inc(v_traceClass_3059_);
v___x_3063_ = l_Lean_Name_append(v___x_3062_, v_traceClass_3059_);
v___x_3064_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3057_, v_options_3045_, v___x_3063_);
lean_dec(v___x_3063_);
if (v___x_3064_ == 0)
{
lean_object* v___x_3175_; uint8_t v___x_3176_; 
v___x_3175_ = l_Lean_trace_profiler;
v___x_3176_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_options_3045_, v___x_3175_);
if (v___x_3176_ == 0)
{
lean_object* v___x_3177_; 
lean_inc(v_term_3004_);
v___x_3177_ = lp_aesop_Aesop_elabRuleTermForApplyLike(v_term_3004_, v_a_2996_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3177_) == 0)
{
lean_object* v_a_3178_; 
v_a_3178_ = lean_ctor_get(v___x_3177_, 0);
lean_inc(v_a_3178_);
lean_dec_ref_known(v___x_3177_, 1);
v_e_3017_ = v_a_3178_;
v___y_3018_ = v_a_2997_;
v___y_3019_ = v_a_2998_;
v___y_3020_ = v_a_2999_;
v___y_3021_ = v_a_3000_;
v___y_3022_ = v_a_3001_;
v___y_3023_ = v_a_3002_;
goto v___jp_3016_;
}
else
{
lean_object* v_a_3179_; lean_object* v___x_3181_; uint8_t v_isShared_3182_; uint8_t v_isSharedCheck_3186_; 
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
lean_dec(v_term_3004_);
v_a_3179_ = lean_ctor_get(v___x_3177_, 0);
v_isSharedCheck_3186_ = !lean_is_exclusive(v___x_3177_);
if (v_isSharedCheck_3186_ == 0)
{
v___x_3181_ = v___x_3177_;
v_isShared_3182_ = v_isSharedCheck_3186_;
goto v_resetjp_3180_;
}
else
{
lean_inc(v_a_3179_);
lean_dec(v___x_3177_);
v___x_3181_ = lean_box(0);
v_isShared_3182_ = v_isSharedCheck_3186_;
goto v_resetjp_3180_;
}
v_resetjp_3180_:
{
lean_object* v___x_3184_; 
if (v_isShared_3182_ == 0)
{
v___x_3184_ = v___x_3181_;
goto v_reusejp_3183_;
}
else
{
lean_object* v_reuseFailAlloc_3185_; 
v_reuseFailAlloc_3185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3185_, 0, v_a_3179_);
v___x_3184_ = v_reuseFailAlloc_3185_;
goto v_reusejp_3183_;
}
v_reusejp_3183_:
{
return v___x_3184_;
}
}
}
}
else
{
goto v___jp_3134_;
}
}
else
{
goto v___jp_3134_;
}
v___jp_3065_:
{
lean_object* v___x_3069_; double v___x_3070_; double v___x_3071_; double v___x_3072_; double v___x_3073_; double v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; lean_object* v___x_3079_; 
v___x_3069_ = lean_io_mono_nanos_now();
v___x_3070_ = lean_float_of_nat(v___y_3066_);
v___x_3071_ = lean_float_once(&lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4, &lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4_once, _init_lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__10___redArg___closed__4);
v___x_3072_ = lean_float_div(v___x_3070_, v___x_3071_);
v___x_3073_ = lean_float_of_nat(v___x_3069_);
v___x_3074_ = lean_float_div(v___x_3073_, v___x_3071_);
v___x_3075_ = lean_box_float(v___x_3072_);
v___x_3076_ = lean_box_float(v___x_3074_);
v___x_3077_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3077_, 0, v___x_3075_);
lean_ctor_set(v___x_3077_, 1, v___x_3076_);
v___x_3078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3078_, 0, v_a_3068_);
lean_ctor_set(v___x_3078_, 1, v___x_3077_);
lean_inc(v_traceClass_3059_);
v___x_3079_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1(v_traceClass_3059_, v_hasTrace_3046_, v___x_3061_, v_options_3045_, v___x_3064_, v___y_3067_, v___f_3060_, v___x_3078_, v_a_2996_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
return v___x_3079_;
}
v___jp_3080_:
{
lean_object* v___x_3084_; 
v___x_3084_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3084_, 0, v_a_3083_);
v___y_3066_ = v___y_3081_;
v___y_3067_ = v___y_3082_;
v_a_3068_ = v___x_3084_;
goto v___jp_3065_;
}
v___jp_3085_:
{
lean_object* v_immediatePremises_x3f_3090_; lean_object* v___x_3091_; 
v_immediatePremises_x3f_3090_ = lean_ctor_get(v_options_3005_, 0);
lean_inc(v_immediatePremises_x3f_3090_);
lean_dec_ref(v_options_3005_);
v___x_3091_ = lp_aesop_Aesop_RuleBuilder_forwardCore(v___y_3088_, v_immediatePremises_x3f_3090_, v_a_3089_, v_phase_3006_, v_isDestruct_2994_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3091_) == 0)
{
lean_object* v_a_3092_; lean_object* v___x_3094_; uint8_t v_isShared_3095_; uint8_t v_isSharedCheck_3099_; 
v_a_3092_ = lean_ctor_get(v___x_3091_, 0);
v_isSharedCheck_3099_ = !lean_is_exclusive(v___x_3091_);
if (v_isSharedCheck_3099_ == 0)
{
v___x_3094_ = v___x_3091_;
v_isShared_3095_ = v_isSharedCheck_3099_;
goto v_resetjp_3093_;
}
else
{
lean_inc(v_a_3092_);
lean_dec(v___x_3091_);
v___x_3094_ = lean_box(0);
v_isShared_3095_ = v_isSharedCheck_3099_;
goto v_resetjp_3093_;
}
v_resetjp_3093_:
{
lean_object* v___x_3097_; 
if (v_isShared_3095_ == 0)
{
lean_ctor_set_tag(v___x_3094_, 1);
v___x_3097_ = v___x_3094_;
goto v_reusejp_3096_;
}
else
{
lean_object* v_reuseFailAlloc_3098_; 
v_reuseFailAlloc_3098_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3098_, 0, v_a_3092_);
v___x_3097_ = v_reuseFailAlloc_3098_;
goto v_reusejp_3096_;
}
v_reusejp_3096_:
{
v___y_3066_ = v___y_3086_;
v___y_3067_ = v___y_3087_;
v_a_3068_ = v___x_3097_;
goto v___jp_3065_;
}
}
}
else
{
lean_object* v_a_3100_; 
v_a_3100_ = lean_ctor_get(v___x_3091_, 0);
lean_inc(v_a_3100_);
lean_dec_ref_known(v___x_3091_, 1);
v___y_3081_ = v___y_3086_;
v___y_3082_ = v___y_3087_;
v_a_3083_ = v_a_3100_;
goto v___jp_3080_;
}
}
v___jp_3101_:
{
lean_object* v___x_3105_; double v___x_3106_; double v___x_3107_; lean_object* v___x_3108_; lean_object* v___x_3109_; lean_object* v___x_3110_; lean_object* v___x_3111_; lean_object* v___x_3112_; 
v___x_3105_ = lean_io_get_num_heartbeats();
v___x_3106_ = lean_float_of_nat(v___y_3103_);
v___x_3107_ = lean_float_of_nat(v___x_3105_);
v___x_3108_ = lean_box_float(v___x_3106_);
v___x_3109_ = lean_box_float(v___x_3107_);
v___x_3110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3110_, 0, v___x_3108_);
lean_ctor_set(v___x_3110_, 1, v___x_3109_);
v___x_3111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3111_, 0, v_a_3104_);
lean_ctor_set(v___x_3111_, 1, v___x_3110_);
lean_inc(v_traceClass_3059_);
v___x_3112_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1(v_traceClass_3059_, v_hasTrace_3046_, v___x_3061_, v_options_3045_, v___x_3064_, v___y_3102_, v___f_3060_, v___x_3111_, v_a_2996_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
return v___x_3112_;
}
v___jp_3113_:
{
lean_object* v___x_3117_; 
v___x_3117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3117_, 0, v_a_3116_);
v___y_3102_ = v___y_3114_;
v___y_3103_ = v___y_3115_;
v_a_3104_ = v___x_3117_;
goto v___jp_3101_;
}
v___jp_3118_:
{
lean_object* v_immediatePremises_x3f_3123_; lean_object* v___x_3124_; 
v_immediatePremises_x3f_3123_ = lean_ctor_get(v_options_3005_, 0);
lean_inc(v_immediatePremises_x3f_3123_);
lean_dec_ref(v_options_3005_);
v___x_3124_ = lp_aesop_Aesop_RuleBuilder_forwardCore(v___y_3119_, v_immediatePremises_x3f_3123_, v_a_3122_, v_phase_3006_, v_isDestruct_2994_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3124_) == 0)
{
lean_object* v_a_3125_; lean_object* v___x_3127_; uint8_t v_isShared_3128_; uint8_t v_isSharedCheck_3132_; 
v_a_3125_ = lean_ctor_get(v___x_3124_, 0);
v_isSharedCheck_3132_ = !lean_is_exclusive(v___x_3124_);
if (v_isSharedCheck_3132_ == 0)
{
v___x_3127_ = v___x_3124_;
v_isShared_3128_ = v_isSharedCheck_3132_;
goto v_resetjp_3126_;
}
else
{
lean_inc(v_a_3125_);
lean_dec(v___x_3124_);
v___x_3127_ = lean_box(0);
v_isShared_3128_ = v_isSharedCheck_3132_;
goto v_resetjp_3126_;
}
v_resetjp_3126_:
{
lean_object* v___x_3130_; 
if (v_isShared_3128_ == 0)
{
lean_ctor_set_tag(v___x_3127_, 1);
v___x_3130_ = v___x_3127_;
goto v_reusejp_3129_;
}
else
{
lean_object* v_reuseFailAlloc_3131_; 
v_reuseFailAlloc_3131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3131_, 0, v_a_3125_);
v___x_3130_ = v_reuseFailAlloc_3131_;
goto v_reusejp_3129_;
}
v_reusejp_3129_:
{
v___y_3102_ = v___y_3120_;
v___y_3103_ = v___y_3121_;
v_a_3104_ = v___x_3130_;
goto v___jp_3101_;
}
}
}
else
{
lean_object* v_a_3133_; 
v_a_3133_ = lean_ctor_get(v___x_3124_, 0);
lean_inc(v_a_3133_);
lean_dec_ref_known(v___x_3124_, 1);
v___y_3114_ = v___y_3120_;
v___y_3115_ = v___y_3121_;
v_a_3116_ = v_a_3133_;
goto v___jp_3113_;
}
}
v___jp_3134_:
{
lean_object* v___x_3135_; lean_object* v_a_3136_; lean_object* v___x_3137_; uint8_t v___x_3138_; 
v___x_3135_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_RuleBuilder_forward_spec__0___redArg(v_a_3002_);
v_a_3136_ = lean_ctor_get(v___x_3135_, 0);
lean_inc(v_a_3136_);
lean_dec_ref(v___x_3135_);
v___x_3137_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3138_ = lp_aesop_Lean_Option_get___at___00Aesop_RuleBuilder_forwardCore_u2082_spec__4(v_options_3045_, v___x_3137_);
if (v___x_3138_ == 0)
{
lean_object* v___x_3139_; lean_object* v___x_3140_; 
v___x_3139_ = lean_io_mono_nanos_now();
lean_inc(v_term_3004_);
v___x_3140_ = lp_aesop_Aesop_elabRuleTermForApplyLike(v_term_3004_, v_a_2996_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3140_) == 0)
{
lean_object* v_a_3141_; lean_object* v_pattern_x3f_3142_; lean_object* v___x_3143_; 
v_a_3141_ = lean_ctor_get(v___x_3140_, 0);
lean_inc_n(v_a_3141_, 2);
lean_dec_ref_known(v___x_3140_, 1);
v_pattern_x3f_3142_ = lean_ctor_get(v_options_3005_, 3);
lean_inc(v_pattern_x3f_3142_);
v___x_3143_ = lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(v_term_3004_, v_a_3141_);
if (lean_obj_tag(v_pattern_x3f_3142_) == 0)
{
lean_object* v___x_3144_; 
lean_dec(v_a_3141_);
v___x_3144_ = lean_box(0);
v___y_3086_ = v___x_3139_;
v___y_3087_ = v_a_3136_;
v___y_3088_ = v___x_3143_;
v_a_3089_ = v___x_3144_;
goto v___jp_3085_;
}
else
{
lean_object* v_val_3145_; lean_object* v___x_3147_; uint8_t v_isShared_3148_; uint8_t v_isSharedCheck_3155_; 
v_val_3145_ = lean_ctor_get(v_pattern_x3f_3142_, 0);
v_isSharedCheck_3155_ = !lean_is_exclusive(v_pattern_x3f_3142_);
if (v_isSharedCheck_3155_ == 0)
{
v___x_3147_ = v_pattern_x3f_3142_;
v_isShared_3148_ = v_isSharedCheck_3155_;
goto v_resetjp_3146_;
}
else
{
lean_inc(v_val_3145_);
lean_dec(v_pattern_x3f_3142_);
v___x_3147_ = lean_box(0);
v_isShared_3148_ = v_isSharedCheck_3155_;
goto v_resetjp_3146_;
}
v_resetjp_3146_:
{
lean_object* v___x_3149_; 
v___x_3149_ = lp_aesop_Aesop_RulePattern_elab(v_val_3145_, v_a_3141_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3149_) == 0)
{
lean_object* v_a_3150_; lean_object* v___x_3152_; 
v_a_3150_ = lean_ctor_get(v___x_3149_, 0);
lean_inc(v_a_3150_);
lean_dec_ref_known(v___x_3149_, 1);
if (v_isShared_3148_ == 0)
{
lean_ctor_set(v___x_3147_, 0, v_a_3150_);
v___x_3152_ = v___x_3147_;
goto v_reusejp_3151_;
}
else
{
lean_object* v_reuseFailAlloc_3153_; 
v_reuseFailAlloc_3153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3153_, 0, v_a_3150_);
v___x_3152_ = v_reuseFailAlloc_3153_;
goto v_reusejp_3151_;
}
v_reusejp_3151_:
{
v___y_3086_ = v___x_3139_;
v___y_3087_ = v_a_3136_;
v___y_3088_ = v___x_3143_;
v_a_3089_ = v___x_3152_;
goto v___jp_3085_;
}
}
else
{
lean_object* v_a_3154_; 
lean_del_object(v___x_3147_);
lean_dec_ref(v___x_3143_);
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
v_a_3154_ = lean_ctor_get(v___x_3149_, 0);
lean_inc(v_a_3154_);
lean_dec_ref_known(v___x_3149_, 1);
v___y_3081_ = v___x_3139_;
v___y_3082_ = v_a_3136_;
v_a_3083_ = v_a_3154_;
goto v___jp_3080_;
}
}
}
}
else
{
lean_object* v_a_3156_; 
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
lean_dec(v_term_3004_);
v_a_3156_ = lean_ctor_get(v___x_3140_, 0);
lean_inc(v_a_3156_);
lean_dec_ref_known(v___x_3140_, 1);
v___y_3081_ = v___x_3139_;
v___y_3082_ = v_a_3136_;
v_a_3083_ = v_a_3156_;
goto v___jp_3080_;
}
}
else
{
lean_object* v___x_3157_; lean_object* v___x_3158_; 
v___x_3157_ = lean_io_get_num_heartbeats();
lean_inc(v_term_3004_);
v___x_3158_ = lp_aesop_Aesop_elabRuleTermForApplyLike(v_term_3004_, v_a_2996_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3158_) == 0)
{
lean_object* v_a_3159_; lean_object* v_pattern_x3f_3160_; lean_object* v___x_3161_; 
v_a_3159_ = lean_ctor_get(v___x_3158_, 0);
lean_inc_n(v_a_3159_, 2);
lean_dec_ref_known(v___x_3158_, 1);
v_pattern_x3f_3160_ = lean_ctor_get(v_options_3005_, 3);
lean_inc(v_pattern_x3f_3160_);
v___x_3161_ = lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(v_term_3004_, v_a_3159_);
if (lean_obj_tag(v_pattern_x3f_3160_) == 0)
{
lean_object* v___x_3162_; 
lean_dec(v_a_3159_);
v___x_3162_ = lean_box(0);
v___y_3119_ = v___x_3161_;
v___y_3120_ = v_a_3136_;
v___y_3121_ = v___x_3157_;
v_a_3122_ = v___x_3162_;
goto v___jp_3118_;
}
else
{
lean_object* v_val_3163_; lean_object* v___x_3165_; uint8_t v_isShared_3166_; uint8_t v_isSharedCheck_3173_; 
v_val_3163_ = lean_ctor_get(v_pattern_x3f_3160_, 0);
v_isSharedCheck_3173_ = !lean_is_exclusive(v_pattern_x3f_3160_);
if (v_isSharedCheck_3173_ == 0)
{
v___x_3165_ = v_pattern_x3f_3160_;
v_isShared_3166_ = v_isSharedCheck_3173_;
goto v_resetjp_3164_;
}
else
{
lean_inc(v_val_3163_);
lean_dec(v_pattern_x3f_3160_);
v___x_3165_ = lean_box(0);
v_isShared_3166_ = v_isSharedCheck_3173_;
goto v_resetjp_3164_;
}
v_resetjp_3164_:
{
lean_object* v___x_3167_; 
v___x_3167_ = lp_aesop_Aesop_RulePattern_elab(v_val_3163_, v_a_3159_, v_a_2997_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
if (lean_obj_tag(v___x_3167_) == 0)
{
lean_object* v_a_3168_; lean_object* v___x_3170_; 
v_a_3168_ = lean_ctor_get(v___x_3167_, 0);
lean_inc(v_a_3168_);
lean_dec_ref_known(v___x_3167_, 1);
if (v_isShared_3166_ == 0)
{
lean_ctor_set(v___x_3165_, 0, v_a_3168_);
v___x_3170_ = v___x_3165_;
goto v_reusejp_3169_;
}
else
{
lean_object* v_reuseFailAlloc_3171_; 
v_reuseFailAlloc_3171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3171_, 0, v_a_3168_);
v___x_3170_ = v_reuseFailAlloc_3171_;
goto v_reusejp_3169_;
}
v_reusejp_3169_:
{
v___y_3119_ = v___x_3161_;
v___y_3120_ = v_a_3136_;
v___y_3121_ = v___x_3157_;
v_a_3122_ = v___x_3170_;
goto v___jp_3118_;
}
}
else
{
lean_object* v_a_3172_; 
lean_del_object(v___x_3165_);
lean_dec_ref(v___x_3161_);
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
v_a_3172_ = lean_ctor_get(v___x_3167_, 0);
lean_inc(v_a_3172_);
lean_dec_ref_known(v___x_3167_, 1);
v___y_3114_ = v_a_3136_;
v___y_3115_ = v___x_3157_;
v_a_3116_ = v_a_3172_;
goto v___jp_3113_;
}
}
}
}
else
{
lean_object* v_a_3174_; 
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
lean_dec(v_term_3004_);
v_a_3174_ = lean_ctor_get(v___x_3158_, 0);
lean_inc(v_a_3174_);
lean_dec_ref_known(v___x_3158_, 1);
v___y_3114_ = v_a_3136_;
v___y_3115_ = v___x_3157_;
v_a_3116_ = v_a_3174_;
goto v___jp_3113_;
}
}
}
}
v___jp_3007_:
{
lean_object* v_immediatePremises_x3f_3014_; lean_object* v___x_3015_; 
v_immediatePremises_x3f_3014_ = lean_ctor_get(v_options_3005_, 0);
lean_inc(v_immediatePremises_x3f_3014_);
lean_dec_ref(v_options_3005_);
v___x_3015_ = lp_aesop_Aesop_RuleBuilder_forwardCore(v___y_3012_, v_immediatePremises_x3f_3014_, v_a_3013_, v_phase_3006_, v_isDestruct_2994_, v___y_3010_, v___y_3009_, v___y_3008_, v___y_3011_);
return v___x_3015_;
}
v___jp_3016_:
{
lean_object* v_pattern_x3f_3024_; lean_object* v_t_3025_; 
v_pattern_x3f_3024_ = lean_ctor_get(v_options_3005_, 3);
lean_inc(v_pattern_x3f_3024_);
lean_inc_ref(v_e_3017_);
v_t_3025_ = lp_aesop_Aesop_ElabRuleTerm_ofElaboratedTerm(v_term_3004_, v_e_3017_);
if (lean_obj_tag(v_pattern_x3f_3024_) == 0)
{
lean_object* v___x_3026_; 
lean_dec_ref(v_e_3017_);
v___x_3026_ = lean_box(0);
v___y_3008_ = v___y_3022_;
v___y_3009_ = v___y_3021_;
v___y_3010_ = v___y_3020_;
v___y_3011_ = v___y_3023_;
v___y_3012_ = v_t_3025_;
v_a_3013_ = v___x_3026_;
goto v___jp_3007_;
}
else
{
lean_object* v_val_3027_; lean_object* v___x_3029_; uint8_t v_isShared_3030_; uint8_t v_isSharedCheck_3044_; 
v_val_3027_ = lean_ctor_get(v_pattern_x3f_3024_, 0);
v_isSharedCheck_3044_ = !lean_is_exclusive(v_pattern_x3f_3024_);
if (v_isSharedCheck_3044_ == 0)
{
v___x_3029_ = v_pattern_x3f_3024_;
v_isShared_3030_ = v_isSharedCheck_3044_;
goto v_resetjp_3028_;
}
else
{
lean_inc(v_val_3027_);
lean_dec(v_pattern_x3f_3024_);
v___x_3029_ = lean_box(0);
v_isShared_3030_ = v_isSharedCheck_3044_;
goto v_resetjp_3028_;
}
v_resetjp_3028_:
{
lean_object* v___x_3031_; 
v___x_3031_ = lp_aesop_Aesop_RulePattern_elab(v_val_3027_, v_e_3017_, v___y_3018_, v___y_3019_, v___y_3020_, v___y_3021_, v___y_3022_, v___y_3023_);
if (lean_obj_tag(v___x_3031_) == 0)
{
lean_object* v_a_3032_; lean_object* v___x_3034_; 
v_a_3032_ = lean_ctor_get(v___x_3031_, 0);
lean_inc(v_a_3032_);
lean_dec_ref_known(v___x_3031_, 1);
if (v_isShared_3030_ == 0)
{
lean_ctor_set(v___x_3029_, 0, v_a_3032_);
v___x_3034_ = v___x_3029_;
goto v_reusejp_3033_;
}
else
{
lean_object* v_reuseFailAlloc_3035_; 
v_reuseFailAlloc_3035_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3035_, 0, v_a_3032_);
v___x_3034_ = v_reuseFailAlloc_3035_;
goto v_reusejp_3033_;
}
v_reusejp_3033_:
{
v___y_3008_ = v___y_3022_;
v___y_3009_ = v___y_3021_;
v___y_3010_ = v___y_3020_;
v___y_3011_ = v___y_3023_;
v___y_3012_ = v_t_3025_;
v_a_3013_ = v___x_3034_;
goto v___jp_3007_;
}
}
else
{
lean_object* v_a_3036_; lean_object* v___x_3038_; uint8_t v_isShared_3039_; uint8_t v_isSharedCheck_3043_; 
lean_del_object(v___x_3029_);
lean_dec_ref(v_t_3025_);
lean_dec_ref(v_phase_3006_);
lean_dec_ref(v_options_3005_);
v_a_3036_ = lean_ctor_get(v___x_3031_, 0);
v_isSharedCheck_3043_ = !lean_is_exclusive(v___x_3031_);
if (v_isSharedCheck_3043_ == 0)
{
v___x_3038_ = v___x_3031_;
v_isShared_3039_ = v_isSharedCheck_3043_;
goto v_resetjp_3037_;
}
else
{
lean_inc(v_a_3036_);
lean_dec(v___x_3031_);
v___x_3038_ = lean_box(0);
v_isShared_3039_ = v_isSharedCheck_3043_;
goto v_resetjp_3037_;
}
v_resetjp_3037_:
{
lean_object* v___x_3041_; 
if (v_isShared_3039_ == 0)
{
v___x_3041_ = v___x_3038_;
goto v_reusejp_3040_;
}
else
{
lean_object* v_reuseFailAlloc_3042_; 
v_reuseFailAlloc_3042_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3042_, 0, v_a_3036_);
v___x_3041_ = v_reuseFailAlloc_3042_;
goto v_reusejp_3040_;
}
v_reusejp_3040_:
{
return v___x_3041_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_forward___boxed(lean_object* v_isDestruct_3187_, lean_object* v_input_3188_, lean_object* v_a_3189_, lean_object* v_a_3190_, lean_object* v_a_3191_, lean_object* v_a_3192_, lean_object* v_a_3193_, lean_object* v_a_3194_, lean_object* v_a_3195_, lean_object* v_a_3196_){
_start:
{
uint8_t v_isDestruct_boxed_3197_; lean_object* v_res_3198_; 
v_isDestruct_boxed_3197_ = lean_unbox(v_isDestruct_3187_);
v_res_3198_ = lp_aesop_Aesop_RuleBuilder_forward(v_isDestruct_boxed_3197_, v_input_3188_, v_a_3189_, v_a_3190_, v_a_3191_, v_a_3192_, v_a_3193_, v_a_3194_, v_a_3195_);
lean_dec(v_a_3195_);
lean_dec_ref(v_a_3194_);
lean_dec(v_a_3193_);
lean_dec_ref(v_a_3192_);
lean_dec(v_a_3191_);
lean_dec_ref(v_a_3190_);
lean_dec_ref(v_a_3189_);
return v_res_3198_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2(lean_object* v_00_u03b1_3199_, lean_object* v_x_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_, lean_object* v___y_3206_, lean_object* v___y_3207_){
_start:
{
lean_object* v___x_3209_; 
v___x_3209_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___redArg(v_x_3200_);
return v___x_3209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2___boxed(lean_object* v_00_u03b1_3210_, lean_object* v_x_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_, lean_object* v___y_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_){
_start:
{
lean_object* v_res_3220_; 
v_res_3220_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__2(v_00_u03b1_3210_, v_x_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_, v___y_3217_, v___y_3218_);
lean_dec(v___y_3218_);
lean_dec_ref(v___y_3217_);
lean_dec(v___y_3216_);
lean_dec_ref(v___y_3215_);
lean_dec(v___y_3214_);
lean_dec_ref(v___y_3213_);
lean_dec_ref(v___y_3212_);
return v_res_3220_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1(lean_object* v_oldTraces_3221_, lean_object* v_data_3222_, lean_object* v_ref_3223_, lean_object* v_msg_3224_, lean_object* v___y_3225_, lean_object* v___y_3226_, lean_object* v___y_3227_, lean_object* v___y_3228_, lean_object* v___y_3229_, lean_object* v___y_3230_, lean_object* v___y_3231_){
_start:
{
lean_object* v___x_3233_; 
v___x_3233_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___redArg(v_oldTraces_3221_, v_data_3222_, v_ref_3223_, v_msg_3224_, v___y_3228_, v___y_3229_, v___y_3230_, v___y_3231_);
return v___x_3233_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1___boxed(lean_object* v_oldTraces_3234_, lean_object* v_data_3235_, lean_object* v_ref_3236_, lean_object* v_msg_3237_, lean_object* v___y_3238_, lean_object* v___y_3239_, lean_object* v___y_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_, lean_object* v___y_3244_, lean_object* v___y_3245_){
_start:
{
lean_object* v_res_3246_; 
v_res_3246_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_RuleBuilder_forward_spec__1_spec__1(v_oldTraces_3234_, v_data_3235_, v_ref_3236_, v_msg_3237_, v___y_3238_, v___y_3239_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_, v___y_3244_);
lean_dec(v___y_3244_);
lean_dec_ref(v___y_3243_);
lean_dec(v___y_3242_);
lean_dec_ref(v___y_3241_);
lean_dec(v___y_3240_);
lean_dec_ref(v___y_3239_);
lean_dec_ref(v___y_3238_);
return v_res_3246_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Builder_Forward(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix = _init_lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix();
lean_mark_persistent(lp_aesop___private_Aesop_Builder_Forward_0__Aesop_RuleBuilder_getImmediatePremises_errPrefix);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Builder_Forward(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Builder_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Builder_Forward(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Builder_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Builder_Forward(builtin);
}
#ifdef __cplusplus
}
#endif
