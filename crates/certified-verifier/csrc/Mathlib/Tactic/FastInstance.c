// Lean compiler output
// Module: Mathlib.Tactic.FastInstance
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
extern lean_object* l_Lean_Linter_linterMessageTag;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isClass_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_trySynthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
extern lean_object* l_Lean_Linter_linterSetsExt;
extern lean_object* l_Lean_Linter_instInhabitedLinterSetsState_default;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Linter_getLinterValue(lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t l_Lean_isStructure(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAuxTheorem(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
extern lean_object* l_Lean_trace_profiler;
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* lean_io_mono_nanos_now();
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_InternalExceptionId_getName(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_Elab_isAbortExceptionId(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "fast_instance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(37, 191, 65, 154, 14, 227, 101, 242)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__7_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__7_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__7_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__8_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__7_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__8_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__8_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "FastInstance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__10_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__8_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(74, 161, 4, 208, 158, 140, 150, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__10_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__10_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__11_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__10_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(203, 96, 133, 69, 60, 174, 4, 144)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__11_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__11_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__12_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__11_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(6, 54, 240, 36, 253, 102, 106, 45)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__12_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__12_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__13_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__12_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 226, 64, 36, 111, 108, 178, 109)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__13_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__13_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__14_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__13_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(157, 53, 47, 89, 181, 218, 217, 185)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__14_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__14_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__15_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__15_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__15_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__16_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__14_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__15_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(20, 185, 117, 144, 119, 157, 7, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__16_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__16_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__17_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__17_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__17_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__18_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__16_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__17_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(61, 230, 33, 215, 237, 196, 238, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__18_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__18_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__19_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__18_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(120, 134, 24, 95, 42, 241, 141, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__19_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__19_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__20_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__19_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__7_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(101, 201, 153, 37, 187, 33, 216, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__20_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__20_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__21_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__20_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(216, 103, 20, 110, 202, 199, 156, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__21_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__21_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__22_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__21_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)(((size_t)(414705255) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(22, 147, 23, 68, 226, 210, 4, 241)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__22_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__22_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__23_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__23_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__23_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__24_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__22_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__23_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(177, 26, 202, 119, 116, 148, 170, 79)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__24_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__24_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__25_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__25_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__25_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__26_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__24_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__25_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(73, 41, 122, 104, 221, 145, 114, 133)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__26_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__26_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__27_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__26_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(52, 123, 114, 73, 36, 108, 245, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__27_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__27_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "fast_instance_existing"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(218, 252, 252, 243, 214, 195, 171, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "Show a warning if `fast_instance%` can be replaced with `inferInstance`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__3_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__14_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(132, 37, 174, 195, 198, 45, 57, 198)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__1_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(188, 95, 110, 78, 131, 192, 110, 155)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 98, .m_capacity = 98, .m_length = 97, .m_data = "\n\nUse `set_option trace.Elab.fast_instance true` to analyze the error.\n\nTrace of fields visited: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "type: "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__1;
static const lean_string_object lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " false`"};
static const lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29_spec__34___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__20(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__20___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18_spec__24(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18_spec__24___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__1;
static lean_once_cell_t lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Proof `"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__0 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1;
static const lean_string_object lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "` does not have expected type `"};
static const lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__2 = (const lean_object*)&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Provided instance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "\nis not defeq to inferred instance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "replaced with synthesized instance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "An instance of `"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "` already exists.\nPlease use `inferInstance` instead of `fast_instance%`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__9_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 102, .m_capacity = 102, .m_length = 101, .m_data = "\n\nThis instance is not a structure and not canonical. Use a separate 'instance' command to define it."};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "` does not unify with the conclusion of `"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Incorrect number of arguments for constructor application `"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "`: "};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "Provided instance does not reduce to a constructor application"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__8 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "\nReduces to an application of "};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__10 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__12 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "\nis a proof, which does not need normalization."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Can only be used for classes, but type is"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "class is "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__17(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29_spec__34(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "fastInstance"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(228, 185, 96, 51, 222, 54, 124, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(237, 203, 126, 15, 73, 189, 28, 72)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 22, 164, 227, 31, 178, 69, 167)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "fast_instance% "};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_fastInstance = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4_spec__6(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "internal exception: "};
static const lean_object* lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "termInferInstanceAs%_"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__5_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(228, 185, 96, 51, 222, 54, 124, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__9_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(237, 203, 126, 15, 73, 189, 28, 72)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(206, 72, 11, 213, 146, 247, 106, 248)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "inferInstanceAs% "};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25__ = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "fast_instance%"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_<|_"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(152, 38, 96, 140, 215, 46, 31, 82)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "_root_.inferInstanceAs"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "_root_"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "inferInstanceAs"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(184, 175, 53, 50, 212, 152, 178, 8)}};
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(55, 231, 206, 76, 88, 206, 40, 142)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(120, 135, 37, 233, 184, 173, 222, 47)}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "<|"};
static const lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_66_; uint8_t v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_66_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_));
v___x_67_ = 0;
v___x_68_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__27_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_));
v___x_69_ = l_Lean_registerTraceClass(v___x_66_, v___x_67_, v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2____boxed(lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_();
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__spec__0(lean_object* v_name_72_, lean_object* v_decl_73_, lean_object* v_ref_74_){
_start:
{
lean_object* v_defValue_76_; lean_object* v_descr_77_; lean_object* v_deprecation_x3f_78_; lean_object* v___x_79_; uint8_t v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v_defValue_76_ = lean_ctor_get(v_decl_73_, 0);
v_descr_77_ = lean_ctor_get(v_decl_73_, 1);
v_deprecation_x3f_78_ = lean_ctor_get(v_decl_73_, 2);
v___x_79_ = lean_alloc_ctor(1, 0, 1);
v___x_80_ = lean_unbox(v_defValue_76_);
lean_ctor_set_uint8(v___x_79_, 0, v___x_80_);
lean_inc(v_deprecation_x3f_78_);
lean_inc_ref(v_descr_77_);
lean_inc_n(v_name_72_, 2);
v___x_81_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_81_, 0, v_name_72_);
lean_ctor_set(v___x_81_, 1, v_ref_74_);
lean_ctor_set(v___x_81_, 2, v___x_79_);
lean_ctor_set(v___x_81_, 3, v_descr_77_);
lean_ctor_set(v___x_81_, 4, v_deprecation_x3f_78_);
v___x_82_ = lean_register_option(v_name_72_, v___x_81_);
if (lean_obj_tag(v___x_82_) == 0)
{
lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_90_; 
v_isSharedCheck_90_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_90_ == 0)
{
lean_object* v_unused_91_; 
v_unused_91_ = lean_ctor_get(v___x_82_, 0);
lean_dec(v_unused_91_);
v___x_84_ = v___x_82_;
v_isShared_85_ = v_isSharedCheck_90_;
goto v_resetjp_83_;
}
else
{
lean_dec(v___x_82_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_90_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_86_; lean_object* v___x_88_; 
lean_inc(v_defValue_76_);
v___x_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_86_, 0, v_name_72_);
lean_ctor_set(v___x_86_, 1, v_defValue_76_);
if (v_isShared_85_ == 0)
{
lean_ctor_set(v___x_84_, 0, v___x_86_);
v___x_88_ = v___x_84_;
goto v_reusejp_87_;
}
else
{
lean_object* v_reuseFailAlloc_89_; 
v_reuseFailAlloc_89_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_89_, 0, v___x_86_);
v___x_88_ = v_reuseFailAlloc_89_;
goto v_reusejp_87_;
}
v_reusejp_87_:
{
return v___x_88_;
}
}
}
else
{
lean_object* v_a_92_; lean_object* v___x_94_; uint8_t v_isShared_95_; uint8_t v_isSharedCheck_99_; 
lean_dec(v_name_72_);
v_a_92_ = lean_ctor_get(v___x_82_, 0);
v_isSharedCheck_99_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_99_ == 0)
{
v___x_94_ = v___x_82_;
v_isShared_95_ = v_isSharedCheck_99_;
goto v_resetjp_93_;
}
else
{
lean_inc(v_a_92_);
lean_dec(v___x_82_);
v___x_94_ = lean_box(0);
v_isShared_95_ = v_isSharedCheck_99_;
goto v_resetjp_93_;
}
v_resetjp_93_:
{
lean_object* v___x_97_; 
if (v_isShared_95_ == 0)
{
v___x_97_ = v___x_94_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_a_92_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_100_, lean_object* v_decl_101_, lean_object* v_ref_102_, lean_object* v_a_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__spec__0(v_name_100_, v_decl_101_, v_ref_102_);
lean_dec_ref(v_decl_101_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_));
v___x_124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__4_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_));
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__6_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_));
v___x_126_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4__spec__0(v___x_123_, v___x_124_, v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4____boxed(lean_object* v_a_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_();
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(lean_object* v_msgData_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v___x_135_; lean_object* v_env_136_; lean_object* v___x_137_; lean_object* v_mctx_138_; lean_object* v_lctx_139_; lean_object* v_options_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_135_ = lean_st_ref_get(v___y_133_);
v_env_136_ = lean_ctor_get(v___x_135_, 0);
lean_inc_ref(v_env_136_);
lean_dec(v___x_135_);
v___x_137_ = lean_st_ref_get(v___y_131_);
v_mctx_138_ = lean_ctor_get(v___x_137_, 0);
lean_inc_ref(v_mctx_138_);
lean_dec(v___x_137_);
v_lctx_139_ = lean_ctor_get(v___y_130_, 2);
v_options_140_ = lean_ctor_get(v___y_132_, 2);
lean_inc_ref(v_options_140_);
lean_inc_ref(v_lctx_139_);
v___x_141_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_141_, 0, v_env_136_);
lean_ctor_set(v___x_141_, 1, v_mctx_138_);
lean_ctor_set(v___x_141_, 2, v_lctx_139_);
lean_ctor_set(v___x_141_, 3, v_options_140_);
v___x_142_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v_msgData_129_);
v___x_143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1___boxed(lean_object* v_msgData_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(v_msgData_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(lean_object* v_msg_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v_ref_157_; lean_object* v___x_158_; lean_object* v_a_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_167_; 
v_ref_157_ = lean_ctor_get(v___y_154_, 5);
v___x_158_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(v_msg_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_);
v_a_159_ = lean_ctor_get(v___x_158_, 0);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_158_);
if (v_isSharedCheck_167_ == 0)
{
v___x_161_ = v___x_158_;
v_isShared_162_ = v_isSharedCheck_167_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_a_159_);
lean_dec(v___x_158_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_167_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_163_; lean_object* v___x_165_; 
lean_inc(v_ref_157_);
v___x_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_163_, 0, v_ref_157_);
lean_ctor_set(v___x_163_, 1, v_a_159_);
if (v_isShared_162_ == 0)
{
lean_ctor_set_tag(v___x_161_, 1);
lean_ctor_set(v___x_161_, 0, v___x_163_);
v___x_165_ = v___x_161_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg___boxed(lean_object* v_msg_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v_msg_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__0(lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
if (lean_obj_tag(v_a_175_) == 0)
{
lean_object* v___x_177_; 
v___x_177_ = l_List_reverse___redArg(v_a_176_);
return v___x_177_;
}
else
{
lean_object* v_head_178_; lean_object* v_tail_179_; lean_object* v___x_181_; uint8_t v_isShared_182_; uint8_t v_isSharedCheck_188_; 
v_head_178_ = lean_ctor_get(v_a_175_, 0);
v_tail_179_ = lean_ctor_get(v_a_175_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v_a_175_);
if (v_isSharedCheck_188_ == 0)
{
v___x_181_ = v_a_175_;
v_isShared_182_ = v_isSharedCheck_188_;
goto v_resetjp_180_;
}
else
{
lean_inc(v_tail_179_);
lean_inc(v_head_178_);
lean_dec(v_a_175_);
v___x_181_ = lean_box(0);
v_isShared_182_ = v_isSharedCheck_188_;
goto v_resetjp_180_;
}
v_resetjp_180_:
{
lean_object* v___x_183_; lean_object* v___x_185_; 
v___x_183_ = l_Lean_MessageData_ofName(v_head_178_);
if (v_isShared_182_ == 0)
{
lean_ctor_set(v___x_181_, 1, v_a_176_);
lean_ctor_set(v___x_181_, 0, v___x_183_);
v___x_185_ = v___x_181_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_183_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_a_176_);
v___x_185_ = v_reuseFailAlloc_187_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
v_a_175_ = v_tail_179_;
v_a_176_ = v___x_185_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__1(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__0));
v___x_191_ = l_Lean_stringToMessageData(v___x_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(lean_object* v_trace_192_, lean_object* v_m_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_199_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___closed__1);
v___x_200_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_200_, 0, v_m_193_);
lean_ctor_set(v___x_200_, 1, v___x_199_);
v___x_201_ = lean_array_to_list(v_trace_192_);
v___x_202_ = lean_box(0);
v___x_203_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__0(v___x_201_, v___x_202_);
v___x_204_ = l_Lean_MessageData_ofList(v___x_203_);
v___x_205_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_205_, 0, v___x_200_);
lean_ctor_set(v___x_205_, 1, v___x_204_);
v___x_206_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_205_, v_a_194_, v_a_195_, v_a_196_, v_a_197_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg___boxed(lean_object* v_trace_207_, lean_object* v_m_208_, lean_object* v_a_209_, lean_object* v_a_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_207_, v_m_208_, v_a_209_, v_a_210_, v_a_211_, v_a_212_);
lean_dec(v_a_212_);
lean_dec_ref(v_a_211_);
lean_dec(v_a_210_);
lean_dec_ref(v_a_209_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error(lean_object* v_00_u03b1_215_, lean_object* v_trace_216_, lean_object* v_m_217_, lean_object* v_a_218_, lean_object* v_a_219_, lean_object* v_a_220_, lean_object* v_a_221_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_216_, v_m_217_, v_a_218_, v_a_219_, v_a_220_, v_a_221_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___boxed(lean_object* v_00_u03b1_224_, lean_object* v_trace_225_, lean_object* v_m_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error(v_00_u03b1_224_, v_trace_225_, v_m_226_, v_a_227_, v_a_228_, v_a_229_, v_a_230_);
lean_dec(v_a_230_);
lean_dec_ref(v_a_229_);
lean_dec(v_a_228_);
lean_dec_ref(v_a_227_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1(lean_object* v_00_u03b1_233_, lean_object* v_msg_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v_msg_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___boxed(lean_object* v_00_u03b1_241_, lean_object* v_msg_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_, lean_object* v___y_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1(v_00_u03b1_241_, v_msg_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
lean_dec(v___y_246_);
lean_dec_ref(v___y_245_);
lean_dec(v___y_244_);
lean_dec_ref(v___y_243_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(lean_object* v_e_249_, lean_object* v___y_250_){
_start:
{
uint8_t v___x_252_; 
v___x_252_ = l_Lean_Expr_hasMVar(v_e_249_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; 
v___x_253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_253_, 0, v_e_249_);
return v___x_253_;
}
else
{
lean_object* v___x_254_; lean_object* v_mctx_255_; lean_object* v___x_256_; lean_object* v_fst_257_; lean_object* v_snd_258_; lean_object* v___x_259_; lean_object* v_cache_260_; lean_object* v_zetaDeltaFVarIds_261_; lean_object* v_postponed_262_; lean_object* v_diag_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_272_; 
v___x_254_ = lean_st_ref_get(v___y_250_);
v_mctx_255_ = lean_ctor_get(v___x_254_, 0);
lean_inc_ref(v_mctx_255_);
lean_dec(v___x_254_);
v___x_256_ = l_Lean_instantiateMVarsCore(v_mctx_255_, v_e_249_);
v_fst_257_ = lean_ctor_get(v___x_256_, 0);
lean_inc(v_fst_257_);
v_snd_258_ = lean_ctor_get(v___x_256_, 1);
lean_inc(v_snd_258_);
lean_dec_ref(v___x_256_);
v___x_259_ = lean_st_ref_take(v___y_250_);
v_cache_260_ = lean_ctor_get(v___x_259_, 1);
v_zetaDeltaFVarIds_261_ = lean_ctor_get(v___x_259_, 2);
v_postponed_262_ = lean_ctor_get(v___x_259_, 3);
v_diag_263_ = lean_ctor_get(v___x_259_, 4);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_272_ == 0)
{
lean_object* v_unused_273_; 
v_unused_273_ = lean_ctor_get(v___x_259_, 0);
lean_dec(v_unused_273_);
v___x_265_ = v___x_259_;
v_isShared_266_ = v_isSharedCheck_272_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_diag_263_);
lean_inc(v_postponed_262_);
lean_inc(v_zetaDeltaFVarIds_261_);
lean_inc(v_cache_260_);
lean_dec(v___x_259_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_272_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_268_; 
if (v_isShared_266_ == 0)
{
lean_ctor_set(v___x_265_, 0, v_snd_258_);
v___x_268_ = v___x_265_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_snd_258_);
lean_ctor_set(v_reuseFailAlloc_271_, 1, v_cache_260_);
lean_ctor_set(v_reuseFailAlloc_271_, 2, v_zetaDeltaFVarIds_261_);
lean_ctor_set(v_reuseFailAlloc_271_, 3, v_postponed_262_);
lean_ctor_set(v_reuseFailAlloc_271_, 4, v_diag_263_);
v___x_268_ = v_reuseFailAlloc_271_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_269_ = lean_st_ref_set(v___y_250_, v___x_268_);
v___x_270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_270_, 0, v_fst_257_);
return v___x_270_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg___boxed(lean_object* v_e_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(v_e_274_, v___y_275_);
lean_dec(v___y_275_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3(lean_object* v_e_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(v_e_278_, v___y_280_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___boxed(lean_object* v_e_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3(v_e_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
lean_dec(v___y_289_);
lean_dec_ref(v___y_288_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___lam__0(lean_object* v_k_292_, lean_object* v_b_293_, lean_object* v_c_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_){
_start:
{
lean_object* v___x_300_; 
lean_inc(v___y_298_);
lean_inc_ref(v___y_297_);
lean_inc(v___y_296_);
lean_inc_ref(v___y_295_);
v___x_300_ = lean_apply_7(v_k_292_, v_b_293_, v_c_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_, lean_box(0));
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___lam__0___boxed(lean_object* v_k_301_, lean_object* v_b_302_, lean_object* v_c_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___lam__0(v_k_301_, v_b_302_, v_c_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_);
lean_dec(v___y_307_);
lean_dec_ref(v___y_306_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(lean_object* v_type_310_, lean_object* v_k_311_, uint8_t v_cleanupAnnotations_312_, uint8_t v_whnfType_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v___f_319_; lean_object* v___x_320_; 
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_319_, 0, v_k_311_);
v___x_320_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_310_, v___f_319_, v_cleanupAnnotations_312_, v_whnfType_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
if (lean_obj_tag(v___x_320_) == 0)
{
lean_object* v_a_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_328_; 
v_a_321_ = lean_ctor_get(v___x_320_, 0);
v_isSharedCheck_328_ = !lean_is_exclusive(v___x_320_);
if (v_isSharedCheck_328_ == 0)
{
v___x_323_ = v___x_320_;
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_a_321_);
lean_dec(v___x_320_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_326_; 
if (v_isShared_324_ == 0)
{
v___x_326_ = v___x_323_;
goto v_reusejp_325_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v_a_321_);
v___x_326_ = v_reuseFailAlloc_327_;
goto v_reusejp_325_;
}
v_reusejp_325_:
{
return v___x_326_;
}
}
}
else
{
lean_object* v_a_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_336_; 
v_a_329_ = lean_ctor_get(v___x_320_, 0);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_320_);
if (v_isSharedCheck_336_ == 0)
{
v___x_331_ = v___x_320_;
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_a_329_);
lean_dec(v___x_320_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_334_; 
if (v_isShared_332_ == 0)
{
v___x_334_ = v___x_331_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_a_329_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg___boxed(lean_object* v_type_337_, lean_object* v_k_338_, lean_object* v_cleanupAnnotations_339_, lean_object* v_whnfType_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_346_; uint8_t v_whnfType_boxed_347_; lean_object* v_res_348_; 
v_cleanupAnnotations_boxed_346_ = lean_unbox(v_cleanupAnnotations_339_);
v_whnfType_boxed_347_ = lean_unbox(v_whnfType_340_);
v_res_348_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(v_type_337_, v_k_338_, v_cleanupAnnotations_boxed_346_, v_whnfType_boxed_347_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
lean_dec(v___y_344_);
lean_dec_ref(v___y_343_);
lean_dec(v___y_342_);
lean_dec_ref(v___y_341_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5(lean_object* v_00_u03b1_349_, lean_object* v_type_350_, lean_object* v_k_351_, uint8_t v_cleanupAnnotations_352_, uint8_t v_whnfType_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(v_type_350_, v_k_351_, v_cleanupAnnotations_352_, v_whnfType_353_, v___y_354_, v___y_355_, v___y_356_, v___y_357_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___boxed(lean_object* v_00_u03b1_360_, lean_object* v_type_361_, lean_object* v_k_362_, lean_object* v_cleanupAnnotations_363_, lean_object* v_whnfType_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_370_; uint8_t v_whnfType_boxed_371_; lean_object* v_res_372_; 
v_cleanupAnnotations_boxed_370_ = lean_unbox(v_cleanupAnnotations_363_);
v_whnfType_boxed_371_ = lean_unbox(v_whnfType_364_);
v_res_372_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5(v_00_u03b1_360_, v_type_361_, v_k_362_, v_cleanupAnnotations_boxed_370_, v_whnfType_boxed_371_, v___y_365_, v___y_366_, v___y_367_, v___y_368_);
lean_dec(v___y_368_);
lean_dec_ref(v___y_367_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
return v_res_372_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__0(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_373_ = lean_unsigned_to_nat(32u);
v___x_374_ = lean_mk_empty_array_with_capacity(v___x_373_);
v___x_375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
return v___x_375_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__1(void){
_start:
{
size_t v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; 
v___x_376_ = ((size_t)5ULL);
v___x_377_ = lean_unsigned_to_nat(0u);
v___x_378_ = lean_unsigned_to_nat(32u);
v___x_379_ = lean_mk_empty_array_with_capacity(v___x_378_);
v___x_380_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__0, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__0_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__0);
v___x_381_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v___x_379_);
lean_ctor_set(v___x_381_, 2, v___x_377_);
lean_ctor_set(v___x_381_, 3, v___x_377_);
lean_ctor_set_usize(v___x_381_, 4, v___x_376_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg(lean_object* v___y_382_){
_start:
{
lean_object* v___x_384_; lean_object* v_traceState_385_; lean_object* v_traces_386_; lean_object* v___x_387_; lean_object* v_traceState_388_; lean_object* v_env_389_; lean_object* v_nextMacroScope_390_; lean_object* v_ngen_391_; lean_object* v_auxDeclNGen_392_; lean_object* v_cache_393_; lean_object* v_messages_394_; lean_object* v_infoState_395_; lean_object* v_snapshotTasks_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_415_; 
v___x_384_ = lean_st_ref_get(v___y_382_);
v_traceState_385_ = lean_ctor_get(v___x_384_, 4);
lean_inc_ref(v_traceState_385_);
lean_dec(v___x_384_);
v_traces_386_ = lean_ctor_get(v_traceState_385_, 0);
lean_inc_ref(v_traces_386_);
lean_dec_ref(v_traceState_385_);
v___x_387_ = lean_st_ref_take(v___y_382_);
v_traceState_388_ = lean_ctor_get(v___x_387_, 4);
v_env_389_ = lean_ctor_get(v___x_387_, 0);
v_nextMacroScope_390_ = lean_ctor_get(v___x_387_, 1);
v_ngen_391_ = lean_ctor_get(v___x_387_, 2);
v_auxDeclNGen_392_ = lean_ctor_get(v___x_387_, 3);
v_cache_393_ = lean_ctor_get(v___x_387_, 5);
v_messages_394_ = lean_ctor_get(v___x_387_, 6);
v_infoState_395_ = lean_ctor_get(v___x_387_, 7);
v_snapshotTasks_396_ = lean_ctor_get(v___x_387_, 8);
v_isSharedCheck_415_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_415_ == 0)
{
v___x_398_ = v___x_387_;
v_isShared_399_ = v_isSharedCheck_415_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_snapshotTasks_396_);
lean_inc(v_infoState_395_);
lean_inc(v_messages_394_);
lean_inc(v_cache_393_);
lean_inc(v_traceState_388_);
lean_inc(v_auxDeclNGen_392_);
lean_inc(v_ngen_391_);
lean_inc(v_nextMacroScope_390_);
lean_inc(v_env_389_);
lean_dec(v___x_387_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_415_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
uint64_t v_tid_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_413_; 
v_tid_400_ = lean_ctor_get_uint64(v_traceState_388_, sizeof(void*)*1);
v_isSharedCheck_413_ = !lean_is_exclusive(v_traceState_388_);
if (v_isSharedCheck_413_ == 0)
{
lean_object* v_unused_414_; 
v_unused_414_ = lean_ctor_get(v_traceState_388_, 0);
lean_dec(v_unused_414_);
v___x_402_ = v_traceState_388_;
v_isShared_403_ = v_isSharedCheck_413_;
goto v_resetjp_401_;
}
else
{
lean_dec(v_traceState_388_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_413_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_404_; lean_object* v___x_406_; 
v___x_404_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___closed__1);
if (v_isShared_403_ == 0)
{
lean_ctor_set(v___x_402_, 0, v___x_404_);
v___x_406_ = v___x_402_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_404_);
lean_ctor_set_uint64(v_reuseFailAlloc_412_, sizeof(void*)*1, v_tid_400_);
v___x_406_ = v_reuseFailAlloc_412_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
lean_object* v___x_408_; 
if (v_isShared_399_ == 0)
{
lean_ctor_set(v___x_398_, 4, v___x_406_);
v___x_408_ = v___x_398_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_env_389_);
lean_ctor_set(v_reuseFailAlloc_411_, 1, v_nextMacroScope_390_);
lean_ctor_set(v_reuseFailAlloc_411_, 2, v_ngen_391_);
lean_ctor_set(v_reuseFailAlloc_411_, 3, v_auxDeclNGen_392_);
lean_ctor_set(v_reuseFailAlloc_411_, 4, v___x_406_);
lean_ctor_set(v_reuseFailAlloc_411_, 5, v_cache_393_);
lean_ctor_set(v_reuseFailAlloc_411_, 6, v_messages_394_);
lean_ctor_set(v_reuseFailAlloc_411_, 7, v_infoState_395_);
lean_ctor_set(v_reuseFailAlloc_411_, 8, v_snapshotTasks_396_);
v___x_408_ = v_reuseFailAlloc_411_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_409_ = lean_st_ref_set(v___y_382_, v___x_408_);
v___x_410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_410_, 0, v_traces_386_);
return v___x_410_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg___boxed(lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg(v___y_416_);
lean_dec(v___y_416_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11(lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_, lean_object* v___y_422_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg(v___y_422_);
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___boxed(lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11(v___y_425_, v___y_426_, v___y_427_, v___y_428_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
return v_res_430_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(lean_object* v_opts_431_, lean_object* v_opt_432_){
_start:
{
lean_object* v_name_433_; lean_object* v_defValue_434_; lean_object* v_map_435_; lean_object* v___x_436_; 
v_name_433_ = lean_ctor_get(v_opt_432_, 0);
v_defValue_434_ = lean_ctor_get(v_opt_432_, 1);
v_map_435_ = lean_ctor_get(v_opts_431_, 0);
v___x_436_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_435_, v_name_433_);
if (lean_obj_tag(v___x_436_) == 0)
{
uint8_t v___x_437_; 
v___x_437_ = lean_unbox(v_defValue_434_);
return v___x_437_;
}
else
{
lean_object* v_val_438_; 
v_val_438_ = lean_ctor_get(v___x_436_, 0);
lean_inc(v_val_438_);
lean_dec_ref_known(v___x_436_, 1);
if (lean_obj_tag(v_val_438_) == 1)
{
uint8_t v_v_439_; 
v_v_439_ = lean_ctor_get_uint8(v_val_438_, 0);
lean_dec_ref_known(v_val_438_, 0);
return v_v_439_;
}
else
{
uint8_t v___x_440_; 
lean_dec(v_val_438_);
v___x_440_ = lean_unbox(v_defValue_434_);
return v___x_440_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12___boxed(lean_object* v_opts_441_, lean_object* v_opt_442_){
_start:
{
uint8_t v_res_443_; lean_object* v_r_444_; 
v_res_443_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_opts_441_, v_opt_442_);
lean_dec_ref(v_opt_442_);
lean_dec_ref(v_opts_441_);
v_r_444_ = lean_box(v_res_443_);
return v_r_444_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__1(void){
_start:
{
lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_446_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__0));
v___x_447_ = l_Lean_stringToMessageData(v___x_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0(lean_object* v_expectedType_448_, lean_object* v_x_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_455_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___closed__1);
v___x_456_ = l_Lean_MessageData_ofExpr(v_expectedType_448_);
v___x_457_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_457_, 0, v___x_455_);
lean_ctor_set(v___x_457_, 1, v___x_456_);
v___x_458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_458_, 0, v___x_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___boxed(lean_object* v_expectedType_459_, lean_object* v_x_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0(v_expectedType_459_, v_x_460_, v___y_461_, v___y_462_, v___y_463_, v___y_464_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
lean_dec_ref(v_x_460_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg(lean_object* v_o_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_470_; lean_object* v_env_471_; lean_object* v___x_472_; lean_object* v_toEnvExtension_473_; lean_object* v_asyncMode_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v_merged_478_; lean_object* v___x_480_; uint8_t v_isShared_481_; uint8_t v_isSharedCheck_486_; 
v___x_470_ = lean_st_ref_get(v___y_468_);
v_env_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc_ref(v_env_471_);
lean_dec(v___x_470_);
v___x_472_ = l_Lean_Linter_linterSetsExt;
v_toEnvExtension_473_ = lean_ctor_get(v___x_472_, 0);
v_asyncMode_474_ = lean_ctor_get(v_toEnvExtension_473_, 2);
v___x_475_ = l_Lean_Linter_instInhabitedLinterSetsState_default;
v___x_476_ = lean_box(0);
v___x_477_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_475_, v___x_472_, v_env_471_, v_asyncMode_474_, v___x_476_);
v_merged_478_ = lean_ctor_get(v___x_477_, 0);
v_isSharedCheck_486_ = !lean_is_exclusive(v___x_477_);
if (v_isSharedCheck_486_ == 0)
{
lean_object* v_unused_487_; 
v_unused_487_ = lean_ctor_get(v___x_477_, 1);
lean_dec(v_unused_487_);
v___x_480_ = v___x_477_;
v_isShared_481_ = v_isSharedCheck_486_;
goto v_resetjp_479_;
}
else
{
lean_inc(v_merged_478_);
lean_dec(v___x_477_);
v___x_480_ = lean_box(0);
v_isShared_481_ = v_isSharedCheck_486_;
goto v_resetjp_479_;
}
v_resetjp_479_:
{
lean_object* v___x_483_; 
if (v_isShared_481_ == 0)
{
lean_ctor_set(v___x_480_, 1, v_merged_478_);
lean_ctor_set(v___x_480_, 0, v_o_467_);
v___x_483_ = v___x_480_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_485_; 
v_reuseFailAlloc_485_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_485_, 0, v_o_467_);
lean_ctor_set(v_reuseFailAlloc_485_, 1, v_merged_478_);
v___x_483_ = v_reuseFailAlloc_485_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_484_; 
v___x_484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_484_, 0, v___x_483_);
return v___x_484_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg___boxed(lean_object* v_o_488_, lean_object* v___y_489_, lean_object* v___y_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg(v_o_488_, v___y_489_);
lean_dec(v___y_489_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1(lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_){
_start:
{
lean_object* v_options_497_; lean_object* v___x_498_; 
v_options_497_ = lean_ctor_get(v___y_494_, 2);
lean_inc_ref(v_options_497_);
v___x_498_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg(v_options_497_, v___y_495_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1___boxed(lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1(v___y_499_, v___y_500_, v___y_501_, v___y_502_);
lean_dec(v___y_502_);
lean_dec_ref(v___y_501_);
lean_dec(v___y_500_);
lean_dec_ref(v___y_499_);
return v_res_504_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0(uint8_t v___y_511_, uint8_t v_suppressElabErrors_512_, lean_object* v_x_513_){
_start:
{
if (lean_obj_tag(v_x_513_) == 1)
{
lean_object* v_pre_514_; 
v_pre_514_ = lean_ctor_get(v_x_513_, 0);
switch(lean_obj_tag(v_pre_514_))
{
case 1:
{
lean_object* v_pre_515_; 
v_pre_515_ = lean_ctor_get(v_pre_514_, 0);
switch(lean_obj_tag(v_pre_515_))
{
case 0:
{
lean_object* v_str_516_; lean_object* v_str_517_; lean_object* v___x_518_; uint8_t v___x_519_; 
v_str_516_ = lean_ctor_get(v_x_513_, 1);
v_str_517_ = lean_ctor_get(v_pre_514_, 1);
v___x_518_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__0_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_));
v___x_519_ = lean_string_dec_eq(v_str_517_, v___x_518_);
if (v___x_519_ == 0)
{
lean_object* v___x_520_; uint8_t v___x_521_; 
v___x_520_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__7_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_));
v___x_521_ = lean_string_dec_eq(v_str_517_, v___x_520_);
if (v___x_521_ == 0)
{
return v___y_511_;
}
else
{
lean_object* v___x_522_; uint8_t v___x_523_; 
v___x_522_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__0));
v___x_523_ = lean_string_dec_eq(v_str_516_, v___x_522_);
if (v___x_523_ == 0)
{
return v___y_511_;
}
else
{
return v_suppressElabErrors_512_;
}
}
}
else
{
lean_object* v___x_524_; uint8_t v___x_525_; 
v___x_524_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__1));
v___x_525_ = lean_string_dec_eq(v_str_516_, v___x_524_);
if (v___x_525_ == 0)
{
return v___y_511_;
}
else
{
return v_suppressElabErrors_512_;
}
}
}
case 1:
{
lean_object* v_pre_526_; 
v_pre_526_ = lean_ctor_get(v_pre_515_, 0);
if (lean_obj_tag(v_pre_526_) == 0)
{
lean_object* v_str_527_; lean_object* v_str_528_; lean_object* v_str_529_; lean_object* v___x_530_; uint8_t v___x_531_; 
v_str_527_ = lean_ctor_get(v_x_513_, 1);
v_str_528_ = lean_ctor_get(v_pre_514_, 1);
v_str_529_ = lean_ctor_get(v_pre_515_, 1);
v___x_530_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__2));
v___x_531_ = lean_string_dec_eq(v_str_529_, v___x_530_);
if (v___x_531_ == 0)
{
return v___y_511_;
}
else
{
lean_object* v___x_532_; uint8_t v___x_533_; 
v___x_532_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__3));
v___x_533_ = lean_string_dec_eq(v_str_528_, v___x_532_);
if (v___x_533_ == 0)
{
return v___y_511_;
}
else
{
lean_object* v___x_534_; uint8_t v___x_535_; 
v___x_534_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__4));
v___x_535_ = lean_string_dec_eq(v_str_527_, v___x_534_);
if (v___x_535_ == 0)
{
return v___y_511_;
}
else
{
return v_suppressElabErrors_512_;
}
}
}
}
else
{
return v___y_511_;
}
}
default: 
{
return v___y_511_;
}
}
}
case 0:
{
lean_object* v_str_536_; lean_object* v___x_537_; uint8_t v___x_538_; 
v_str_536_ = lean_ctor_get(v_x_513_, 1);
v___x_537_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___closed__5));
v___x_538_ = lean_string_dec_eq(v_str_536_, v___x_537_);
if (v___x_538_ == 0)
{
return v___y_511_;
}
else
{
return v_suppressElabErrors_512_;
}
}
default: 
{
return v___y_511_;
}
}
}
else
{
return v___y_511_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___boxed(lean_object* v___y_539_, lean_object* v_suppressElabErrors_540_, lean_object* v_x_541_){
_start:
{
uint8_t v___y_68151__boxed_542_; uint8_t v_suppressElabErrors_boxed_543_; uint8_t v_res_544_; lean_object* v_r_545_; 
v___y_68151__boxed_542_ = lean_unbox(v___y_539_);
v_suppressElabErrors_boxed_543_ = lean_unbox(v_suppressElabErrors_540_);
v_res_544_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0(v___y_68151__boxed_542_, v_suppressElabErrors_boxed_543_, v_x_541_);
lean_dec(v_x_541_);
v_r_545_ = lean_box(v_res_544_);
return v_r_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21(lean_object* v_ref_547_, lean_object* v_msgData_548_, uint8_t v_severity_549_, uint8_t v_isSilent_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_){
_start:
{
lean_object* v___y_557_; lean_object* v___y_558_; lean_object* v___y_559_; uint8_t v___y_560_; lean_object* v___y_561_; uint8_t v___y_562_; lean_object* v___y_563_; lean_object* v___y_564_; lean_object* v___y_565_; lean_object* v___y_593_; lean_object* v___y_594_; uint8_t v___y_595_; uint8_t v___y_596_; lean_object* v___y_597_; uint8_t v___y_598_; lean_object* v___y_599_; lean_object* v___y_600_; lean_object* v___y_618_; lean_object* v___y_619_; uint8_t v___y_620_; lean_object* v___y_621_; lean_object* v___y_622_; uint8_t v___y_623_; uint8_t v___y_624_; lean_object* v___y_625_; lean_object* v___y_629_; lean_object* v___y_630_; uint8_t v___y_631_; uint8_t v___y_632_; lean_object* v___y_633_; lean_object* v___y_634_; uint8_t v___y_635_; uint8_t v___x_640_; lean_object* v___y_642_; lean_object* v___y_643_; uint8_t v___y_644_; lean_object* v___y_645_; lean_object* v___y_646_; uint8_t v___y_647_; uint8_t v___y_648_; uint8_t v___y_650_; uint8_t v___x_665_; 
v___x_640_ = 2;
v___x_665_ = l_Lean_instBEqMessageSeverity_beq(v_severity_549_, v___x_640_);
if (v___x_665_ == 0)
{
v___y_650_ = v___x_665_;
goto v___jp_649_;
}
else
{
uint8_t v___x_666_; 
lean_inc_ref(v_msgData_548_);
v___x_666_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_548_);
v___y_650_ = v___x_666_;
goto v___jp_649_;
}
v___jp_556_:
{
lean_object* v___x_566_; lean_object* v_currNamespace_567_; lean_object* v_openDecls_568_; lean_object* v_env_569_; lean_object* v_nextMacroScope_570_; lean_object* v_ngen_571_; lean_object* v_auxDeclNGen_572_; lean_object* v_traceState_573_; lean_object* v_cache_574_; lean_object* v_messages_575_; lean_object* v_infoState_576_; lean_object* v_snapshotTasks_577_; lean_object* v___x_579_; uint8_t v_isShared_580_; uint8_t v_isSharedCheck_591_; 
v___x_566_ = lean_st_ref_take(v___y_565_);
v_currNamespace_567_ = lean_ctor_get(v___y_564_, 6);
v_openDecls_568_ = lean_ctor_get(v___y_564_, 7);
v_env_569_ = lean_ctor_get(v___x_566_, 0);
v_nextMacroScope_570_ = lean_ctor_get(v___x_566_, 1);
v_ngen_571_ = lean_ctor_get(v___x_566_, 2);
v_auxDeclNGen_572_ = lean_ctor_get(v___x_566_, 3);
v_traceState_573_ = lean_ctor_get(v___x_566_, 4);
v_cache_574_ = lean_ctor_get(v___x_566_, 5);
v_messages_575_ = lean_ctor_get(v___x_566_, 6);
v_infoState_576_ = lean_ctor_get(v___x_566_, 7);
v_snapshotTasks_577_ = lean_ctor_get(v___x_566_, 8);
v_isSharedCheck_591_ = !lean_is_exclusive(v___x_566_);
if (v_isSharedCheck_591_ == 0)
{
v___x_579_ = v___x_566_;
v_isShared_580_ = v_isSharedCheck_591_;
goto v_resetjp_578_;
}
else
{
lean_inc(v_snapshotTasks_577_);
lean_inc(v_infoState_576_);
lean_inc(v_messages_575_);
lean_inc(v_cache_574_);
lean_inc(v_traceState_573_);
lean_inc(v_auxDeclNGen_572_);
lean_inc(v_ngen_571_);
lean_inc(v_nextMacroScope_570_);
lean_inc(v_env_569_);
lean_dec(v___x_566_);
v___x_579_ = lean_box(0);
v_isShared_580_ = v_isSharedCheck_591_;
goto v_resetjp_578_;
}
v_resetjp_578_:
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_586_; 
lean_inc(v_openDecls_568_);
lean_inc(v_currNamespace_567_);
v___x_581_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_581_, 0, v_currNamespace_567_);
lean_ctor_set(v___x_581_, 1, v_openDecls_568_);
v___x_582_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set(v___x_582_, 1, v___y_558_);
lean_inc_ref(v___y_563_);
lean_inc_ref(v___y_557_);
v___x_583_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_583_, 0, v___y_557_);
lean_ctor_set(v___x_583_, 1, v___y_561_);
lean_ctor_set(v___x_583_, 2, v___y_559_);
lean_ctor_set(v___x_583_, 3, v___y_563_);
lean_ctor_set(v___x_583_, 4, v___x_582_);
lean_ctor_set_uint8(v___x_583_, sizeof(void*)*5, v___y_560_);
lean_ctor_set_uint8(v___x_583_, sizeof(void*)*5 + 1, v___y_562_);
lean_ctor_set_uint8(v___x_583_, sizeof(void*)*5 + 2, v_isSilent_550_);
v___x_584_ = l_Lean_MessageLog_add(v___x_583_, v_messages_575_);
if (v_isShared_580_ == 0)
{
lean_ctor_set(v___x_579_, 6, v___x_584_);
v___x_586_ = v___x_579_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_590_; 
v_reuseFailAlloc_590_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_590_, 0, v_env_569_);
lean_ctor_set(v_reuseFailAlloc_590_, 1, v_nextMacroScope_570_);
lean_ctor_set(v_reuseFailAlloc_590_, 2, v_ngen_571_);
lean_ctor_set(v_reuseFailAlloc_590_, 3, v_auxDeclNGen_572_);
lean_ctor_set(v_reuseFailAlloc_590_, 4, v_traceState_573_);
lean_ctor_set(v_reuseFailAlloc_590_, 5, v_cache_574_);
lean_ctor_set(v_reuseFailAlloc_590_, 6, v___x_584_);
lean_ctor_set(v_reuseFailAlloc_590_, 7, v_infoState_576_);
lean_ctor_set(v_reuseFailAlloc_590_, 8, v_snapshotTasks_577_);
v___x_586_ = v_reuseFailAlloc_590_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_587_ = lean_st_ref_set(v___y_565_, v___x_586_);
v___x_588_ = lean_box(0);
v___x_589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_589_, 0, v___x_588_);
return v___x_589_;
}
}
}
v___jp_592_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_616_; 
v___x_601_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_548_);
v___x_602_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(v___x_601_, v___y_551_, v___y_552_, v___y_553_, v___y_554_);
v_a_603_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_616_ == 0)
{
v___x_605_ = v___x_602_;
v_isShared_606_ = v_isSharedCheck_616_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_602_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_616_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
lean_inc_ref_n(v___y_597_, 2);
v___x_607_ = l_Lean_FileMap_toPosition(v___y_597_, v___y_599_);
lean_dec(v___y_599_);
v___x_608_ = l_Lean_FileMap_toPosition(v___y_597_, v___y_600_);
lean_dec(v___y_600_);
v___x_609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_609_, 0, v___x_608_);
v___x_610_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0));
if (v___y_595_ == 0)
{
lean_del_object(v___x_605_);
lean_dec_ref(v___y_593_);
v___y_557_ = v___y_594_;
v___y_558_ = v_a_603_;
v___y_559_ = v___x_609_;
v___y_560_ = v___y_596_;
v___y_561_ = v___x_607_;
v___y_562_ = v___y_598_;
v___y_563_ = v___x_610_;
v___y_564_ = v___y_553_;
v___y_565_ = v___y_554_;
goto v___jp_556_;
}
else
{
uint8_t v___x_611_; 
lean_inc(v_a_603_);
v___x_611_ = l_Lean_MessageData_hasTag(v___y_593_, v_a_603_);
if (v___x_611_ == 0)
{
lean_object* v___x_612_; lean_object* v___x_614_; 
lean_dec_ref_known(v___x_609_, 1);
lean_dec_ref(v___x_607_);
lean_dec(v_a_603_);
v___x_612_ = lean_box(0);
if (v_isShared_606_ == 0)
{
lean_ctor_set(v___x_605_, 0, v___x_612_);
v___x_614_ = v___x_605_;
goto v_reusejp_613_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v___x_612_);
v___x_614_ = v_reuseFailAlloc_615_;
goto v_reusejp_613_;
}
v_reusejp_613_:
{
return v___x_614_;
}
}
else
{
lean_del_object(v___x_605_);
v___y_557_ = v___y_594_;
v___y_558_ = v_a_603_;
v___y_559_ = v___x_609_;
v___y_560_ = v___y_596_;
v___y_561_ = v___x_607_;
v___y_562_ = v___y_598_;
v___y_563_ = v___x_610_;
v___y_564_ = v___y_553_;
v___y_565_ = v___y_554_;
goto v___jp_556_;
}
}
}
}
v___jp_617_:
{
lean_object* v___x_626_; 
v___x_626_ = l_Lean_Syntax_getTailPos_x3f(v___y_621_, v___y_623_);
lean_dec(v___y_621_);
if (lean_obj_tag(v___x_626_) == 0)
{
lean_inc(v___y_625_);
v___y_593_ = v___y_618_;
v___y_594_ = v___y_619_;
v___y_595_ = v___y_620_;
v___y_596_ = v___y_623_;
v___y_597_ = v___y_622_;
v___y_598_ = v___y_624_;
v___y_599_ = v___y_625_;
v___y_600_ = v___y_625_;
goto v___jp_592_;
}
else
{
lean_object* v_val_627_; 
v_val_627_ = lean_ctor_get(v___x_626_, 0);
lean_inc(v_val_627_);
lean_dec_ref_known(v___x_626_, 1);
v___y_593_ = v___y_618_;
v___y_594_ = v___y_619_;
v___y_595_ = v___y_620_;
v___y_596_ = v___y_623_;
v___y_597_ = v___y_622_;
v___y_598_ = v___y_624_;
v___y_599_ = v___y_625_;
v___y_600_ = v_val_627_;
goto v___jp_592_;
}
}
v___jp_628_:
{
lean_object* v_ref_636_; lean_object* v___x_637_; 
v_ref_636_ = l_Lean_replaceRef(v_ref_547_, v___y_634_);
v___x_637_ = l_Lean_Syntax_getPos_x3f(v_ref_636_, v___y_632_);
if (lean_obj_tag(v___x_637_) == 0)
{
lean_object* v___x_638_; 
v___x_638_ = lean_unsigned_to_nat(0u);
v___y_618_ = v___y_629_;
v___y_619_ = v___y_630_;
v___y_620_ = v___y_631_;
v___y_621_ = v_ref_636_;
v___y_622_ = v___y_633_;
v___y_623_ = v___y_632_;
v___y_624_ = v___y_635_;
v___y_625_ = v___x_638_;
goto v___jp_617_;
}
else
{
lean_object* v_val_639_; 
v_val_639_ = lean_ctor_get(v___x_637_, 0);
lean_inc(v_val_639_);
lean_dec_ref_known(v___x_637_, 1);
v___y_618_ = v___y_629_;
v___y_619_ = v___y_630_;
v___y_620_ = v___y_631_;
v___y_621_ = v_ref_636_;
v___y_622_ = v___y_633_;
v___y_623_ = v___y_632_;
v___y_624_ = v___y_635_;
v___y_625_ = v_val_639_;
goto v___jp_617_;
}
}
v___jp_641_:
{
if (v___y_648_ == 0)
{
v___y_629_ = v___y_642_;
v___y_630_ = v___y_643_;
v___y_631_ = v___y_644_;
v___y_632_ = v___y_647_;
v___y_633_ = v___y_645_;
v___y_634_ = v___y_646_;
v___y_635_ = v_severity_549_;
goto v___jp_628_;
}
else
{
v___y_629_ = v___y_642_;
v___y_630_ = v___y_643_;
v___y_631_ = v___y_644_;
v___y_632_ = v___y_647_;
v___y_633_ = v___y_645_;
v___y_634_ = v___y_646_;
v___y_635_ = v___x_640_;
goto v___jp_628_;
}
}
v___jp_649_:
{
if (v___y_650_ == 0)
{
lean_object* v_fileName_651_; lean_object* v_fileMap_652_; lean_object* v_options_653_; lean_object* v_ref_654_; uint8_t v_suppressElabErrors_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___f_658_; uint8_t v___x_659_; uint8_t v___x_660_; 
v_fileName_651_ = lean_ctor_get(v___y_553_, 0);
v_fileMap_652_ = lean_ctor_get(v___y_553_, 1);
v_options_653_ = lean_ctor_get(v___y_553_, 2);
v_ref_654_ = lean_ctor_get(v___y_553_, 5);
v_suppressElabErrors_655_ = lean_ctor_get_uint8(v___y_553_, sizeof(void*)*14 + 1);
v___x_656_ = lean_box(v___y_650_);
v___x_657_ = lean_box(v_suppressElabErrors_655_);
v___f_658_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___boxed), 3, 2);
lean_closure_set(v___f_658_, 0, v___x_656_);
lean_closure_set(v___f_658_, 1, v___x_657_);
v___x_659_ = 1;
v___x_660_ = l_Lean_instBEqMessageSeverity_beq(v_severity_549_, v___x_659_);
if (v___x_660_ == 0)
{
v___y_642_ = v___f_658_;
v___y_643_ = v_fileName_651_;
v___y_644_ = v_suppressElabErrors_655_;
v___y_645_ = v_fileMap_652_;
v___y_646_ = v_ref_654_;
v___y_647_ = v___y_650_;
v___y_648_ = v___x_660_;
goto v___jp_641_;
}
else
{
lean_object* v___x_661_; uint8_t v___x_662_; 
v___x_661_ = l_Lean_warningAsError;
v___x_662_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_options_653_, v___x_661_);
v___y_642_ = v___f_658_;
v___y_643_ = v_fileName_651_;
v___y_644_ = v_suppressElabErrors_655_;
v___y_645_ = v_fileMap_652_;
v___y_646_ = v_ref_654_;
v___y_647_ = v___y_650_;
v___y_648_ = v___x_662_;
goto v___jp_641_;
}
}
else
{
lean_object* v___x_663_; lean_object* v___x_664_; 
lean_dec_ref(v_msgData_548_);
v___x_663_ = lean_box(0);
v___x_664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_664_, 0, v___x_663_);
return v___x_664_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___boxed(lean_object* v_ref_667_, lean_object* v_msgData_668_, lean_object* v_severity_669_, lean_object* v_isSilent_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_){
_start:
{
uint8_t v_severity_boxed_676_; uint8_t v_isSilent_boxed_677_; lean_object* v_res_678_; 
v_severity_boxed_676_ = lean_unbox(v_severity_669_);
v_isSilent_boxed_677_ = lean_unbox(v_isSilent_670_);
v_res_678_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21(v_ref_667_, v_msgData_668_, v_severity_boxed_676_, v_isSilent_boxed_677_, v___y_671_, v___y_672_, v___y_673_, v___y_674_);
lean_dec(v___y_674_);
lean_dec_ref(v___y_673_);
lean_dec(v___y_672_);
lean_dec_ref(v___y_671_);
lean_dec(v_ref_667_);
return v_res_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2_spec__8(lean_object* v_ref_679_, lean_object* v_msgData_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_){
_start:
{
uint8_t v___x_686_; uint8_t v___x_687_; lean_object* v___x_688_; 
v___x_686_ = 1;
v___x_687_ = 0;
v___x_688_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21(v_ref_679_, v_msgData_680_, v___x_686_, v___x_687_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
return v___x_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2_spec__8___boxed(lean_object* v_ref_689_, lean_object* v_msgData_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2_spec__8(v_ref_689_, v_msgData_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_);
lean_dec(v___y_694_);
lean_dec_ref(v___y_693_);
lean_dec(v___y_692_);
lean_dec_ref(v___y_691_);
lean_dec(v_ref_689_);
return v_res_696_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_698_; lean_object* v___x_699_; 
v___x_698_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__0));
v___x_699_ = l_Lean_stringToMessageData(v___x_698_);
return v___x_699_;
}
}
static lean_object* _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_701_ = ((lean_object*)(lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__2));
v___x_702_ = l_Lean_stringToMessageData(v___x_701_);
return v___x_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2(lean_object* v_linterOption_703_, lean_object* v_stx_704_, lean_object* v_msg_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_){
_start:
{
lean_object* v_name_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_729_; 
v_name_711_ = lean_ctor_get(v_linterOption_703_, 0);
v_isSharedCheck_729_ = !lean_is_exclusive(v_linterOption_703_);
if (v_isSharedCheck_729_ == 0)
{
lean_object* v_unused_730_; 
v_unused_730_ = lean_ctor_get(v_linterOption_703_, 1);
lean_dec(v_unused_730_);
v___x_713_ = v_linterOption_703_;
v_isShared_714_ = v_isSharedCheck_729_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_name_711_);
lean_dec(v_linterOption_703_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_729_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_718_; 
v___x_715_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__1, &lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__1_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__1);
lean_inc(v_name_711_);
v___x_716_ = l_Lean_MessageData_ofName(v_name_711_);
if (v_isShared_714_ == 0)
{
lean_ctor_set_tag(v___x_713_, 7);
lean_ctor_set(v___x_713_, 1, v___x_716_);
lean_ctor_set(v___x_713_, 0, v___x_715_);
v___x_718_ = v___x_713_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v___x_715_);
lean_ctor_set(v_reuseFailAlloc_728_, 1, v___x_716_);
v___x_718_ = v_reuseFailAlloc_728_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v_disable_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; 
v___x_719_ = lean_obj_once(&lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__3, &lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__3_once, _init_lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___closed__3);
v___x_720_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_720_, 0, v___x_718_);
lean_ctor_set(v___x_720_, 1, v___x_719_);
v_disable_721_ = l_Lean_MessageData_note(v___x_720_);
v___x_722_ = l_Lean_Linter_linterMessageTag;
v___x_723_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_723_, 0, v_msg_705_);
lean_ctor_set(v___x_723_, 1, v_disable_721_);
v___x_724_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_722_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
v___x_725_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_725_, 0, v_name_711_);
lean_ctor_set(v___x_725_, 1, v___x_724_);
lean_inc(v_stx_704_);
v___x_726_ = lean_alloc_ctor(11, 2, 0);
lean_ctor_set(v___x_726_, 0, v_stx_704_);
lean_ctor_set(v___x_726_, 1, v___x_725_);
v___x_727_ = lp_mathlib_Lean_logWarningAt___at___00Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2_spec__8(v_stx_704_, v___x_726_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
lean_dec(v_stx_704_);
return v___x_727_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2___boxed(lean_object* v_linterOption_731_, lean_object* v_stx_732_, lean_object* v_msg_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2(v_linterOption_731_, v_stx_732_, v_msg_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
lean_dec(v___y_735_);
lean_dec_ref(v___y_734_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(lean_object* v_linterOption_740_, lean_object* v_stx_741_, lean_object* v_msg_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
lean_object* v___x_748_; lean_object* v_a_749_; lean_object* v___x_751_; uint8_t v_isShared_752_; uint8_t v_isSharedCheck_759_; 
v___x_748_ = lp_mathlib_Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1(v___y_743_, v___y_744_, v___y_745_, v___y_746_);
v_a_749_ = lean_ctor_get(v___x_748_, 0);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_748_);
if (v_isSharedCheck_759_ == 0)
{
v___x_751_ = v___x_748_;
v_isShared_752_ = v_isSharedCheck_759_;
goto v_resetjp_750_;
}
else
{
lean_inc(v_a_749_);
lean_dec(v___x_748_);
v___x_751_ = lean_box(0);
v_isShared_752_ = v_isSharedCheck_759_;
goto v_resetjp_750_;
}
v_resetjp_750_:
{
uint8_t v___x_753_; 
v___x_753_ = l_Lean_Linter_getLinterValue(v_linterOption_740_, v_a_749_);
lean_dec(v_a_749_);
if (v___x_753_ == 0)
{
lean_object* v___x_754_; lean_object* v___x_756_; 
lean_dec_ref(v_msg_742_);
lean_dec(v_stx_741_);
lean_dec_ref(v_linterOption_740_);
v___x_754_ = lean_box(0);
if (v_isShared_752_ == 0)
{
lean_ctor_set(v___x_751_, 0, v___x_754_);
v___x_756_ = v___x_751_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v___x_754_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
else
{
lean_object* v___x_758_; 
lean_del_object(v___x_751_);
v___x_758_ = lp_mathlib_Lean_Linter_logLint___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__2(v_linterOption_740_, v_stx_741_, v_msg_742_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
return v___x_758_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1___boxed(lean_object* v_linterOption_760_, lean_object* v_stx_761_, lean_object* v_msg_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(v_linterOption_760_, v_stx_761_, v_msg_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
lean_dec(v___y_766_);
lean_dec_ref(v___y_765_);
lean_dec(v___y_764_);
lean_dec_ref(v___y_763_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14(lean_object* v_msgData_769_, uint8_t v_severity_770_, uint8_t v_isSilent_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_){
_start:
{
lean_object* v_ref_777_; lean_object* v___x_778_; 
v_ref_777_ = lean_ctor_get(v___y_774_, 5);
v___x_778_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21(v_ref_777_, v_msgData_769_, v_severity_770_, v_isSilent_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14___boxed(lean_object* v_msgData_779_, lean_object* v_severity_780_, lean_object* v_isSilent_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
uint8_t v_severity_boxed_787_; uint8_t v_isSilent_boxed_788_; lean_object* v_res_789_; 
v_severity_boxed_787_ = lean_unbox(v_severity_780_);
v_isSilent_boxed_788_ = lean_unbox(v_isSilent_781_);
v_res_789_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14(v_msgData_779_, v_severity_boxed_787_, v_isSilent_boxed_788_, v___y_782_, v___y_783_, v___y_784_, v___y_785_);
lean_dec(v___y_785_);
lean_dec_ref(v___y_784_);
lean_dec(v___y_783_);
lean_dec_ref(v___y_782_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(lean_object* v_msgData_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_){
_start:
{
uint8_t v___x_796_; uint8_t v___x_797_; lean_object* v___x_798_; 
v___x_796_ = 1;
v___x_797_ = 0;
v___x_798_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14(v_msgData_790_, v___x_796_, v___x_797_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10___boxed(lean_object* v_msgData_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_){
_start:
{
lean_object* v_res_805_; 
v_res_805_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(v_msgData_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_);
lean_dec(v___y_803_);
lean_dec_ref(v___y_802_);
lean_dec(v___y_801_);
lean_dec_ref(v___y_800_);
return v_res_805_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0(void){
_start:
{
lean_object* v___x_806_; double v___x_807_; 
v___x_806_ = lean_unsigned_to_nat(0u);
v___x_807_ = lean_float_of_nat(v___x_806_);
return v___x_807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(lean_object* v_cls_810_, lean_object* v_msg_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_){
_start:
{
lean_object* v_ref_817_; lean_object* v___x_818_; lean_object* v_a_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_863_; 
v_ref_817_ = lean_ctor_get(v___y_814_, 5);
v___x_818_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(v_msg_811_, v___y_812_, v___y_813_, v___y_814_, v___y_815_);
v_a_819_ = lean_ctor_get(v___x_818_, 0);
v_isSharedCheck_863_ = !lean_is_exclusive(v___x_818_);
if (v_isSharedCheck_863_ == 0)
{
v___x_821_ = v___x_818_;
v_isShared_822_ = v_isSharedCheck_863_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_a_819_);
lean_dec(v___x_818_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_863_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v___x_823_; lean_object* v_traceState_824_; lean_object* v_env_825_; lean_object* v_nextMacroScope_826_; lean_object* v_ngen_827_; lean_object* v_auxDeclNGen_828_; lean_object* v_cache_829_; lean_object* v_messages_830_; lean_object* v_infoState_831_; lean_object* v_snapshotTasks_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_862_; 
v___x_823_ = lean_st_ref_take(v___y_815_);
v_traceState_824_ = lean_ctor_get(v___x_823_, 4);
v_env_825_ = lean_ctor_get(v___x_823_, 0);
v_nextMacroScope_826_ = lean_ctor_get(v___x_823_, 1);
v_ngen_827_ = lean_ctor_get(v___x_823_, 2);
v_auxDeclNGen_828_ = lean_ctor_get(v___x_823_, 3);
v_cache_829_ = lean_ctor_get(v___x_823_, 5);
v_messages_830_ = lean_ctor_get(v___x_823_, 6);
v_infoState_831_ = lean_ctor_get(v___x_823_, 7);
v_snapshotTasks_832_ = lean_ctor_get(v___x_823_, 8);
v_isSharedCheck_862_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_862_ == 0)
{
v___x_834_ = v___x_823_;
v_isShared_835_ = v_isSharedCheck_862_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_snapshotTasks_832_);
lean_inc(v_infoState_831_);
lean_inc(v_messages_830_);
lean_inc(v_cache_829_);
lean_inc(v_traceState_824_);
lean_inc(v_auxDeclNGen_828_);
lean_inc(v_ngen_827_);
lean_inc(v_nextMacroScope_826_);
lean_inc(v_env_825_);
lean_dec(v___x_823_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_862_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
uint64_t v_tid_836_; lean_object* v_traces_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_861_; 
v_tid_836_ = lean_ctor_get_uint64(v_traceState_824_, sizeof(void*)*1);
v_traces_837_ = lean_ctor_get(v_traceState_824_, 0);
v_isSharedCheck_861_ = !lean_is_exclusive(v_traceState_824_);
if (v_isSharedCheck_861_ == 0)
{
v___x_839_ = v_traceState_824_;
v_isShared_840_ = v_isSharedCheck_861_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_traces_837_);
lean_dec(v_traceState_824_);
v___x_839_ = lean_box(0);
v_isShared_840_ = v_isSharedCheck_861_;
goto v_resetjp_838_;
}
v_resetjp_838_:
{
lean_object* v___x_841_; double v___x_842_; uint8_t v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_851_; 
v___x_841_ = lean_box(0);
v___x_842_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0);
v___x_843_ = 0;
v___x_844_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0));
v___x_845_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_845_, 0, v_cls_810_);
lean_ctor_set(v___x_845_, 1, v___x_841_);
lean_ctor_set(v___x_845_, 2, v___x_844_);
lean_ctor_set_float(v___x_845_, sizeof(void*)*3, v___x_842_);
lean_ctor_set_float(v___x_845_, sizeof(void*)*3 + 8, v___x_842_);
lean_ctor_set_uint8(v___x_845_, sizeof(void*)*3 + 16, v___x_843_);
v___x_846_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__1));
v___x_847_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_847_, 0, v___x_845_);
lean_ctor_set(v___x_847_, 1, v_a_819_);
lean_ctor_set(v___x_847_, 2, v___x_846_);
lean_inc(v_ref_817_);
v___x_848_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_848_, 0, v_ref_817_);
lean_ctor_set(v___x_848_, 1, v___x_847_);
v___x_849_ = l_Lean_PersistentArray_push___redArg(v_traces_837_, v___x_848_);
if (v_isShared_840_ == 0)
{
lean_ctor_set(v___x_839_, 0, v___x_849_);
v___x_851_ = v___x_839_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_860_; 
v_reuseFailAlloc_860_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_860_, 0, v___x_849_);
lean_ctor_set_uint64(v_reuseFailAlloc_860_, sizeof(void*)*1, v_tid_836_);
v___x_851_ = v_reuseFailAlloc_860_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
lean_object* v___x_853_; 
if (v_isShared_835_ == 0)
{
lean_ctor_set(v___x_834_, 4, v___x_851_);
v___x_853_ = v___x_834_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v_env_825_);
lean_ctor_set(v_reuseFailAlloc_859_, 1, v_nextMacroScope_826_);
lean_ctor_set(v_reuseFailAlloc_859_, 2, v_ngen_827_);
lean_ctor_set(v_reuseFailAlloc_859_, 3, v_auxDeclNGen_828_);
lean_ctor_set(v_reuseFailAlloc_859_, 4, v___x_851_);
lean_ctor_set(v_reuseFailAlloc_859_, 5, v_cache_829_);
lean_ctor_set(v_reuseFailAlloc_859_, 6, v_messages_830_);
lean_ctor_set(v_reuseFailAlloc_859_, 7, v_infoState_831_);
lean_ctor_set(v_reuseFailAlloc_859_, 8, v_snapshotTasks_832_);
v___x_853_ = v_reuseFailAlloc_859_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_857_; 
v___x_854_ = lean_st_ref_set(v___y_815_, v___x_853_);
v___x_855_ = lean_box(0);
if (v_isShared_822_ == 0)
{
lean_ctor_set(v___x_821_, 0, v___x_855_);
v___x_857_ = v___x_821_;
goto v_reusejp_856_;
}
else
{
lean_object* v_reuseFailAlloc_858_; 
v_reuseFailAlloc_858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_858_, 0, v___x_855_);
v___x_857_ = v_reuseFailAlloc_858_;
goto v_reusejp_856_;
}
v_reusejp_856_:
{
return v___x_857_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___boxed(lean_object* v_cls_864_, lean_object* v_msg_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_){
_start:
{
lean_object* v_res_871_; 
v_res_871_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_864_, v_msg_865_, v___y_866_, v___y_867_, v___y_868_, v___y_869_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
lean_dec(v___y_867_);
lean_dec_ref(v___y_866_);
return v_res_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29_spec__34___redArg(lean_object* v_x_872_, lean_object* v_x_873_, lean_object* v_x_874_, lean_object* v_x_875_){
_start:
{
lean_object* v_ks_876_; lean_object* v_vs_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_901_; 
v_ks_876_ = lean_ctor_get(v_x_872_, 0);
v_vs_877_ = lean_ctor_get(v_x_872_, 1);
v_isSharedCheck_901_ = !lean_is_exclusive(v_x_872_);
if (v_isSharedCheck_901_ == 0)
{
v___x_879_ = v_x_872_;
v_isShared_880_ = v_isSharedCheck_901_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_vs_877_);
lean_inc(v_ks_876_);
lean_dec(v_x_872_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_901_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
lean_object* v___x_881_; uint8_t v___x_882_; 
v___x_881_ = lean_array_get_size(v_ks_876_);
v___x_882_ = lean_nat_dec_lt(v_x_873_, v___x_881_);
if (v___x_882_ == 0)
{
lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_886_; 
lean_dec(v_x_873_);
v___x_883_ = lean_array_push(v_ks_876_, v_x_874_);
v___x_884_ = lean_array_push(v_vs_877_, v_x_875_);
if (v_isShared_880_ == 0)
{
lean_ctor_set(v___x_879_, 1, v___x_884_);
lean_ctor_set(v___x_879_, 0, v___x_883_);
v___x_886_ = v___x_879_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v___x_883_);
lean_ctor_set(v_reuseFailAlloc_887_, 1, v___x_884_);
v___x_886_ = v_reuseFailAlloc_887_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
return v___x_886_;
}
}
else
{
lean_object* v_k_x27_888_; uint8_t v___x_889_; 
v_k_x27_888_ = lean_array_fget_borrowed(v_ks_876_, v_x_873_);
v___x_889_ = l_Lean_instBEqMVarId_beq(v_x_874_, v_k_x27_888_);
if (v___x_889_ == 0)
{
lean_object* v___x_891_; 
if (v_isShared_880_ == 0)
{
v___x_891_ = v___x_879_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v_ks_876_);
lean_ctor_set(v_reuseFailAlloc_895_, 1, v_vs_877_);
v___x_891_ = v_reuseFailAlloc_895_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
lean_object* v___x_892_; lean_object* v___x_893_; 
v___x_892_ = lean_unsigned_to_nat(1u);
v___x_893_ = lean_nat_add(v_x_873_, v___x_892_);
lean_dec(v_x_873_);
v_x_872_ = v___x_891_;
v_x_873_ = v___x_893_;
goto _start;
}
}
else
{
lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_899_; 
v___x_896_ = lean_array_fset(v_ks_876_, v_x_873_, v_x_874_);
v___x_897_ = lean_array_fset(v_vs_877_, v_x_873_, v_x_875_);
lean_dec(v_x_873_);
if (v_isShared_880_ == 0)
{
lean_ctor_set(v___x_879_, 1, v___x_897_);
lean_ctor_set(v___x_879_, 0, v___x_896_);
v___x_899_ = v___x_879_;
goto v_reusejp_898_;
}
else
{
lean_object* v_reuseFailAlloc_900_; 
v_reuseFailAlloc_900_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_900_, 0, v___x_896_);
lean_ctor_set(v_reuseFailAlloc_900_, 1, v___x_897_);
v___x_899_ = v_reuseFailAlloc_900_;
goto v_reusejp_898_;
}
v_reusejp_898_:
{
return v___x_899_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29___redArg(lean_object* v_n_902_, lean_object* v_k_903_, lean_object* v_v_904_){
_start:
{
lean_object* v___x_905_; lean_object* v___x_906_; 
v___x_905_ = lean_unsigned_to_nat(0u);
v___x_906_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29_spec__34___redArg(v_n_902_, v___x_905_, v_k_903_, v_v_904_);
return v___x_906_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___closed__0(void){
_start:
{
lean_object* v___x_907_; 
v___x_907_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(lean_object* v_x_908_, size_t v_x_909_, size_t v_x_910_, lean_object* v_x_911_, lean_object* v_x_912_){
_start:
{
if (lean_obj_tag(v_x_908_) == 0)
{
lean_object* v_es_913_; size_t v___x_914_; size_t v___x_915_; lean_object* v_j_916_; lean_object* v___x_917_; uint8_t v___x_918_; 
v_es_913_ = lean_ctor_get(v_x_908_, 0);
v___x_914_ = ((size_t)31ULL);
v___x_915_ = lean_usize_land(v_x_909_, v___x_914_);
v_j_916_ = lean_usize_to_nat(v___x_915_);
v___x_917_ = lean_array_get_size(v_es_913_);
v___x_918_ = lean_nat_dec_lt(v_j_916_, v___x_917_);
if (v___x_918_ == 0)
{
lean_dec(v_j_916_);
lean_dec(v_x_912_);
lean_dec(v_x_911_);
return v_x_908_;
}
else
{
lean_object* v___x_920_; uint8_t v_isShared_921_; uint8_t v_isSharedCheck_957_; 
lean_inc_ref(v_es_913_);
v_isSharedCheck_957_ = !lean_is_exclusive(v_x_908_);
if (v_isSharedCheck_957_ == 0)
{
lean_object* v_unused_958_; 
v_unused_958_ = lean_ctor_get(v_x_908_, 0);
lean_dec(v_unused_958_);
v___x_920_ = v_x_908_;
v_isShared_921_ = v_isSharedCheck_957_;
goto v_resetjp_919_;
}
else
{
lean_dec(v_x_908_);
v___x_920_ = lean_box(0);
v_isShared_921_ = v_isSharedCheck_957_;
goto v_resetjp_919_;
}
v_resetjp_919_:
{
lean_object* v_v_922_; lean_object* v___x_923_; lean_object* v_xs_x27_924_; lean_object* v___y_926_; 
v_v_922_ = lean_array_fget(v_es_913_, v_j_916_);
v___x_923_ = lean_box(0);
v_xs_x27_924_ = lean_array_fset(v_es_913_, v_j_916_, v___x_923_);
switch(lean_obj_tag(v_v_922_))
{
case 0:
{
lean_object* v_key_931_; lean_object* v_val_932_; lean_object* v___x_934_; uint8_t v_isShared_935_; uint8_t v_isSharedCheck_942_; 
v_key_931_ = lean_ctor_get(v_v_922_, 0);
v_val_932_ = lean_ctor_get(v_v_922_, 1);
v_isSharedCheck_942_ = !lean_is_exclusive(v_v_922_);
if (v_isSharedCheck_942_ == 0)
{
v___x_934_ = v_v_922_;
v_isShared_935_ = v_isSharedCheck_942_;
goto v_resetjp_933_;
}
else
{
lean_inc(v_val_932_);
lean_inc(v_key_931_);
lean_dec(v_v_922_);
v___x_934_ = lean_box(0);
v_isShared_935_ = v_isSharedCheck_942_;
goto v_resetjp_933_;
}
v_resetjp_933_:
{
uint8_t v___x_936_; 
v___x_936_ = l_Lean_instBEqMVarId_beq(v_x_911_, v_key_931_);
if (v___x_936_ == 0)
{
lean_object* v___x_937_; lean_object* v___x_938_; 
lean_del_object(v___x_934_);
v___x_937_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_931_, v_val_932_, v_x_911_, v_x_912_);
v___x_938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_938_, 0, v___x_937_);
v___y_926_ = v___x_938_;
goto v___jp_925_;
}
else
{
lean_object* v___x_940_; 
lean_dec(v_val_932_);
lean_dec(v_key_931_);
if (v_isShared_935_ == 0)
{
lean_ctor_set(v___x_934_, 1, v_x_912_);
lean_ctor_set(v___x_934_, 0, v_x_911_);
v___x_940_ = v___x_934_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v_x_911_);
lean_ctor_set(v_reuseFailAlloc_941_, 1, v_x_912_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
v___y_926_ = v___x_940_;
goto v___jp_925_;
}
}
}
}
case 1:
{
lean_object* v_node_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_955_; 
v_node_943_ = lean_ctor_get(v_v_922_, 0);
v_isSharedCheck_955_ = !lean_is_exclusive(v_v_922_);
if (v_isSharedCheck_955_ == 0)
{
v___x_945_ = v_v_922_;
v_isShared_946_ = v_isSharedCheck_955_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_node_943_);
lean_dec(v_v_922_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_955_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
size_t v___x_947_; size_t v___x_948_; size_t v___x_949_; size_t v___x_950_; lean_object* v___x_951_; lean_object* v___x_953_; 
v___x_947_ = ((size_t)5ULL);
v___x_948_ = lean_usize_shift_right(v_x_909_, v___x_947_);
v___x_949_ = ((size_t)1ULL);
v___x_950_ = lean_usize_add(v_x_910_, v___x_949_);
v___x_951_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(v_node_943_, v___x_948_, v___x_950_, v_x_911_, v_x_912_);
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 0, v___x_951_);
v___x_953_ = v___x_945_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v___x_951_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
v___y_926_ = v___x_953_;
goto v___jp_925_;
}
}
}
default: 
{
lean_object* v___x_956_; 
v___x_956_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_956_, 0, v_x_911_);
lean_ctor_set(v___x_956_, 1, v_x_912_);
v___y_926_ = v___x_956_;
goto v___jp_925_;
}
}
v___jp_925_:
{
lean_object* v___x_927_; lean_object* v___x_929_; 
v___x_927_ = lean_array_fset(v_xs_x27_924_, v_j_916_, v___y_926_);
lean_dec(v_j_916_);
if (v_isShared_921_ == 0)
{
lean_ctor_set(v___x_920_, 0, v___x_927_);
v___x_929_ = v___x_920_;
goto v_reusejp_928_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v___x_927_);
v___x_929_ = v_reuseFailAlloc_930_;
goto v_reusejp_928_;
}
v_reusejp_928_:
{
return v___x_929_;
}
}
}
}
}
else
{
lean_object* v_ks_959_; lean_object* v_vs_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_980_; 
v_ks_959_ = lean_ctor_get(v_x_908_, 0);
v_vs_960_ = lean_ctor_get(v_x_908_, 1);
v_isSharedCheck_980_ = !lean_is_exclusive(v_x_908_);
if (v_isSharedCheck_980_ == 0)
{
v___x_962_ = v_x_908_;
v_isShared_963_ = v_isSharedCheck_980_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_vs_960_);
lean_inc(v_ks_959_);
lean_dec(v_x_908_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_980_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_965_; 
if (v_isShared_963_ == 0)
{
v___x_965_ = v___x_962_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_979_; 
v_reuseFailAlloc_979_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_979_, 0, v_ks_959_);
lean_ctor_set(v_reuseFailAlloc_979_, 1, v_vs_960_);
v___x_965_ = v_reuseFailAlloc_979_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
lean_object* v_newNode_966_; uint8_t v___y_968_; size_t v___x_974_; uint8_t v___x_975_; 
v_newNode_966_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29___redArg(v___x_965_, v_x_911_, v_x_912_);
v___x_974_ = ((size_t)7ULL);
v___x_975_ = lean_usize_dec_le(v___x_974_, v_x_910_);
if (v___x_975_ == 0)
{
lean_object* v___x_976_; lean_object* v___x_977_; uint8_t v___x_978_; 
v___x_976_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_966_);
v___x_977_ = lean_unsigned_to_nat(4u);
v___x_978_ = lean_nat_dec_lt(v___x_976_, v___x_977_);
lean_dec(v___x_976_);
v___y_968_ = v___x_978_;
goto v___jp_967_;
}
else
{
v___y_968_ = v___x_975_;
goto v___jp_967_;
}
v___jp_967_:
{
if (v___y_968_ == 0)
{
lean_object* v_ks_969_; lean_object* v_vs_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; 
v_ks_969_ = lean_ctor_get(v_newNode_966_, 0);
lean_inc_ref(v_ks_969_);
v_vs_970_ = lean_ctor_get(v_newNode_966_, 1);
lean_inc_ref(v_vs_970_);
lean_dec_ref(v_newNode_966_);
v___x_971_ = lean_unsigned_to_nat(0u);
v___x_972_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___closed__0);
v___x_973_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg(v_x_910_, v_ks_969_, v_vs_970_, v___x_971_, v___x_972_);
lean_dec_ref(v_vs_970_);
lean_dec_ref(v_ks_969_);
return v___x_973_;
}
else
{
return v_newNode_966_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg(size_t v_depth_981_, lean_object* v_keys_982_, lean_object* v_vals_983_, lean_object* v_i_984_, lean_object* v_entries_985_){
_start:
{
lean_object* v___x_986_; uint8_t v___x_987_; 
v___x_986_ = lean_array_get_size(v_keys_982_);
v___x_987_ = lean_nat_dec_lt(v_i_984_, v___x_986_);
if (v___x_987_ == 0)
{
lean_dec(v_i_984_);
return v_entries_985_;
}
else
{
lean_object* v_k_988_; lean_object* v_v_989_; uint64_t v___x_990_; size_t v_h_991_; size_t v___x_992_; lean_object* v___x_993_; size_t v___x_994_; size_t v___x_995_; size_t v___x_996_; size_t v_h_997_; lean_object* v___x_998_; lean_object* v___x_999_; 
v_k_988_ = lean_array_fget_borrowed(v_keys_982_, v_i_984_);
v_v_989_ = lean_array_fget_borrowed(v_vals_983_, v_i_984_);
v___x_990_ = l_Lean_instHashableMVarId_hash(v_k_988_);
v_h_991_ = lean_uint64_to_usize(v___x_990_);
v___x_992_ = ((size_t)5ULL);
v___x_993_ = lean_unsigned_to_nat(1u);
v___x_994_ = ((size_t)1ULL);
v___x_995_ = lean_usize_sub(v_depth_981_, v___x_994_);
v___x_996_ = lean_usize_mul(v___x_992_, v___x_995_);
v_h_997_ = lean_usize_shift_right(v_h_991_, v___x_996_);
v___x_998_ = lean_nat_add(v_i_984_, v___x_993_);
lean_dec(v_i_984_);
lean_inc(v_v_989_);
lean_inc(v_k_988_);
v___x_999_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(v_entries_985_, v_h_997_, v_depth_981_, v_k_988_, v_v_989_);
v_i_984_ = v___x_998_;
v_entries_985_ = v___x_999_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg___boxed(lean_object* v_depth_1001_, lean_object* v_keys_1002_, lean_object* v_vals_1003_, lean_object* v_i_1004_, lean_object* v_entries_1005_){
_start:
{
size_t v_depth_boxed_1006_; lean_object* v_res_1007_; 
v_depth_boxed_1006_ = lean_unbox_usize(v_depth_1001_);
lean_dec(v_depth_1001_);
v_res_1007_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg(v_depth_boxed_1006_, v_keys_1002_, v_vals_1003_, v_i_1004_, v_entries_1005_);
lean_dec_ref(v_vals_1003_);
lean_dec_ref(v_keys_1002_);
return v_res_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg___boxed(lean_object* v_x_1008_, lean_object* v_x_1009_, lean_object* v_x_1010_, lean_object* v_x_1011_, lean_object* v_x_1012_){
_start:
{
size_t v_x_68772__boxed_1013_; size_t v_x_68773__boxed_1014_; lean_object* v_res_1015_; 
v_x_68772__boxed_1013_ = lean_unbox_usize(v_x_1009_);
lean_dec(v_x_1009_);
v_x_68773__boxed_1014_ = lean_unbox_usize(v_x_1010_);
lean_dec(v_x_1010_);
v_res_1015_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(v_x_1008_, v_x_68772__boxed_1013_, v_x_68773__boxed_1014_, v_x_1011_, v_x_1012_);
return v_res_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9___redArg(lean_object* v_x_1016_, lean_object* v_x_1017_, lean_object* v_x_1018_){
_start:
{
uint64_t v___x_1019_; size_t v___x_1020_; size_t v___x_1021_; lean_object* v___x_1022_; 
v___x_1019_ = l_Lean_instHashableMVarId_hash(v_x_1017_);
v___x_1020_ = lean_uint64_to_usize(v___x_1019_);
v___x_1021_ = ((size_t)1ULL);
v___x_1022_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(v_x_1016_, v___x_1020_, v___x_1021_, v_x_1017_, v_x_1018_);
return v___x_1022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(lean_object* v_mvarId_1023_, lean_object* v_val_1024_, lean_object* v___y_1025_){
_start:
{
lean_object* v___x_1027_; lean_object* v_mctx_1028_; lean_object* v_cache_1029_; lean_object* v_zetaDeltaFVarIds_1030_; lean_object* v_postponed_1031_; lean_object* v_diag_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1060_; 
v___x_1027_ = lean_st_ref_take(v___y_1025_);
v_mctx_1028_ = lean_ctor_get(v___x_1027_, 0);
v_cache_1029_ = lean_ctor_get(v___x_1027_, 1);
v_zetaDeltaFVarIds_1030_ = lean_ctor_get(v___x_1027_, 2);
v_postponed_1031_ = lean_ctor_get(v___x_1027_, 3);
v_diag_1032_ = lean_ctor_get(v___x_1027_, 4);
v_isSharedCheck_1060_ = !lean_is_exclusive(v___x_1027_);
if (v_isSharedCheck_1060_ == 0)
{
v___x_1034_ = v___x_1027_;
v_isShared_1035_ = v_isSharedCheck_1060_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_diag_1032_);
lean_inc(v_postponed_1031_);
lean_inc(v_zetaDeltaFVarIds_1030_);
lean_inc(v_cache_1029_);
lean_inc(v_mctx_1028_);
lean_dec(v___x_1027_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1060_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
lean_object* v_depth_1036_; lean_object* v_levelAssignDepth_1037_; lean_object* v_lmvarCounter_1038_; lean_object* v_mvarCounter_1039_; lean_object* v_lDecls_1040_; lean_object* v_decls_1041_; lean_object* v_userNames_1042_; lean_object* v_lAssignment_1043_; lean_object* v_eAssignment_1044_; lean_object* v_dAssignment_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1059_; 
v_depth_1036_ = lean_ctor_get(v_mctx_1028_, 0);
v_levelAssignDepth_1037_ = lean_ctor_get(v_mctx_1028_, 1);
v_lmvarCounter_1038_ = lean_ctor_get(v_mctx_1028_, 2);
v_mvarCounter_1039_ = lean_ctor_get(v_mctx_1028_, 3);
v_lDecls_1040_ = lean_ctor_get(v_mctx_1028_, 4);
v_decls_1041_ = lean_ctor_get(v_mctx_1028_, 5);
v_userNames_1042_ = lean_ctor_get(v_mctx_1028_, 6);
v_lAssignment_1043_ = lean_ctor_get(v_mctx_1028_, 7);
v_eAssignment_1044_ = lean_ctor_get(v_mctx_1028_, 8);
v_dAssignment_1045_ = lean_ctor_get(v_mctx_1028_, 9);
v_isSharedCheck_1059_ = !lean_is_exclusive(v_mctx_1028_);
if (v_isSharedCheck_1059_ == 0)
{
v___x_1047_ = v_mctx_1028_;
v_isShared_1048_ = v_isSharedCheck_1059_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_dAssignment_1045_);
lean_inc(v_eAssignment_1044_);
lean_inc(v_lAssignment_1043_);
lean_inc(v_userNames_1042_);
lean_inc(v_decls_1041_);
lean_inc(v_lDecls_1040_);
lean_inc(v_mvarCounter_1039_);
lean_inc(v_lmvarCounter_1038_);
lean_inc(v_levelAssignDepth_1037_);
lean_inc(v_depth_1036_);
lean_dec(v_mctx_1028_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1059_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v___x_1049_; lean_object* v___x_1051_; 
v___x_1049_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9___redArg(v_eAssignment_1044_, v_mvarId_1023_, v_val_1024_);
if (v_isShared_1048_ == 0)
{
lean_ctor_set(v___x_1047_, 8, v___x_1049_);
v___x_1051_ = v___x_1047_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1058_; 
v_reuseFailAlloc_1058_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1058_, 0, v_depth_1036_);
lean_ctor_set(v_reuseFailAlloc_1058_, 1, v_levelAssignDepth_1037_);
lean_ctor_set(v_reuseFailAlloc_1058_, 2, v_lmvarCounter_1038_);
lean_ctor_set(v_reuseFailAlloc_1058_, 3, v_mvarCounter_1039_);
lean_ctor_set(v_reuseFailAlloc_1058_, 4, v_lDecls_1040_);
lean_ctor_set(v_reuseFailAlloc_1058_, 5, v_decls_1041_);
lean_ctor_set(v_reuseFailAlloc_1058_, 6, v_userNames_1042_);
lean_ctor_set(v_reuseFailAlloc_1058_, 7, v_lAssignment_1043_);
lean_ctor_set(v_reuseFailAlloc_1058_, 8, v___x_1049_);
lean_ctor_set(v_reuseFailAlloc_1058_, 9, v_dAssignment_1045_);
v___x_1051_ = v_reuseFailAlloc_1058_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
lean_object* v___x_1053_; 
if (v_isShared_1035_ == 0)
{
lean_ctor_set(v___x_1034_, 0, v___x_1051_);
v___x_1053_ = v___x_1034_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v___x_1051_);
lean_ctor_set(v_reuseFailAlloc_1057_, 1, v_cache_1029_);
lean_ctor_set(v_reuseFailAlloc_1057_, 2, v_zetaDeltaFVarIds_1030_);
lean_ctor_set(v_reuseFailAlloc_1057_, 3, v_postponed_1031_);
lean_ctor_set(v_reuseFailAlloc_1057_, 4, v_diag_1032_);
v___x_1053_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; 
v___x_1054_ = lean_st_ref_set(v___y_1025_, v___x_1053_);
v___x_1055_ = lean_box(0);
v___x_1056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1055_);
return v___x_1056_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg___boxed(lean_object* v_mvarId_1061_, lean_object* v_val_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_){
_start:
{
lean_object* v_res_1065_; 
v_res_1065_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v_mvarId_1061_, v_val_1062_, v___y_1063_);
lean_dec(v___y_1063_);
return v_res_1065_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__20(lean_object* v_e_1066_){
_start:
{
if (lean_obj_tag(v_e_1066_) == 0)
{
uint8_t v___x_1067_; 
v___x_1067_ = 2;
return v___x_1067_;
}
else
{
lean_object* v_a_1068_; uint8_t v___x_1069_; 
v_a_1068_ = lean_ctor_get(v_e_1066_, 0);
v___x_1069_ = l_Lean_Expr_hasSyntheticSorry(v_a_1068_);
if (v___x_1069_ == 0)
{
uint8_t v___x_1070_; 
v___x_1070_ = 0;
return v___x_1070_;
}
else
{
uint8_t v___x_1071_; 
v___x_1071_ = 1;
return v___x_1071_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__20___boxed(lean_object* v_e_1072_){
_start:
{
uint8_t v_res_1073_; lean_object* v_r_1074_; 
v_res_1073_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__20(v_e_1072_);
lean_dec_ref(v_e_1072_);
v_r_1074_ = lean_box(v_res_1073_);
return v_r_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21(lean_object* v_opts_1075_, lean_object* v_opt_1076_){
_start:
{
lean_object* v_name_1077_; lean_object* v_defValue_1078_; lean_object* v_map_1079_; lean_object* v___x_1080_; 
v_name_1077_ = lean_ctor_get(v_opt_1076_, 0);
v_defValue_1078_ = lean_ctor_get(v_opt_1076_, 1);
v_map_1079_ = lean_ctor_get(v_opts_1075_, 0);
v___x_1080_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1079_, v_name_1077_);
if (lean_obj_tag(v___x_1080_) == 0)
{
lean_inc(v_defValue_1078_);
return v_defValue_1078_;
}
else
{
lean_object* v_val_1081_; 
v_val_1081_ = lean_ctor_get(v___x_1080_, 0);
lean_inc(v_val_1081_);
lean_dec_ref_known(v___x_1080_, 1);
if (lean_obj_tag(v_val_1081_) == 3)
{
lean_object* v_v_1082_; 
v_v_1082_ = lean_ctor_get(v_val_1081_, 0);
lean_inc(v_v_1082_);
lean_dec_ref_known(v_val_1081_, 1);
return v_v_1082_;
}
else
{
lean_dec(v_val_1081_);
lean_inc(v_defValue_1078_);
return v_defValue_1078_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21___boxed(lean_object* v_opts_1083_, lean_object* v_opt_1084_){
_start:
{
lean_object* v_res_1085_; 
v_res_1085_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21(v_opts_1083_, v_opt_1084_);
lean_dec_ref(v_opt_1084_);
lean_dec_ref(v_opts_1083_);
return v_res_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg(lean_object* v_x_1086_){
_start:
{
if (lean_obj_tag(v_x_1086_) == 0)
{
lean_object* v_a_1088_; lean_object* v___x_1090_; uint8_t v_isShared_1091_; uint8_t v_isSharedCheck_1095_; 
v_a_1088_ = lean_ctor_get(v_x_1086_, 0);
v_isSharedCheck_1095_ = !lean_is_exclusive(v_x_1086_);
if (v_isSharedCheck_1095_ == 0)
{
v___x_1090_ = v_x_1086_;
v_isShared_1091_ = v_isSharedCheck_1095_;
goto v_resetjp_1089_;
}
else
{
lean_inc(v_a_1088_);
lean_dec(v_x_1086_);
v___x_1090_ = lean_box(0);
v_isShared_1091_ = v_isSharedCheck_1095_;
goto v_resetjp_1089_;
}
v_resetjp_1089_:
{
lean_object* v___x_1093_; 
if (v_isShared_1091_ == 0)
{
lean_ctor_set_tag(v___x_1090_, 1);
v___x_1093_ = v___x_1090_;
goto v_reusejp_1092_;
}
else
{
lean_object* v_reuseFailAlloc_1094_; 
v_reuseFailAlloc_1094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1094_, 0, v_a_1088_);
v___x_1093_ = v_reuseFailAlloc_1094_;
goto v_reusejp_1092_;
}
v_reusejp_1092_:
{
return v___x_1093_;
}
}
}
else
{
lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1103_; 
v_a_1096_ = lean_ctor_get(v_x_1086_, 0);
v_isSharedCheck_1103_ = !lean_is_exclusive(v_x_1086_);
if (v_isSharedCheck_1103_ == 0)
{
v___x_1098_ = v_x_1086_;
v_isShared_1099_ = v_isSharedCheck_1103_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v_x_1086_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1103_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1101_; 
if (v_isShared_1099_ == 0)
{
lean_ctor_set_tag(v___x_1098_, 0);
v___x_1101_ = v___x_1098_;
goto v_reusejp_1100_;
}
else
{
lean_object* v_reuseFailAlloc_1102_; 
v_reuseFailAlloc_1102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1102_, 0, v_a_1096_);
v___x_1101_ = v_reuseFailAlloc_1102_;
goto v_reusejp_1100_;
}
v_reusejp_1100_:
{
return v___x_1101_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg___boxed(lean_object* v_x_1104_, lean_object* v___y_1105_){
_start:
{
lean_object* v_res_1106_; 
v_res_1106_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg(v_x_1104_);
return v_res_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18_spec__24(size_t v_sz_1107_, size_t v_i_1108_, lean_object* v_bs_1109_){
_start:
{
uint8_t v___x_1110_; 
v___x_1110_ = lean_usize_dec_lt(v_i_1108_, v_sz_1107_);
if (v___x_1110_ == 0)
{
return v_bs_1109_;
}
else
{
lean_object* v_v_1111_; lean_object* v_msg_1112_; lean_object* v___x_1113_; lean_object* v_bs_x27_1114_; size_t v___x_1115_; size_t v___x_1116_; lean_object* v___x_1117_; 
v_v_1111_ = lean_array_uget_borrowed(v_bs_1109_, v_i_1108_);
v_msg_1112_ = lean_ctor_get(v_v_1111_, 1);
lean_inc_ref(v_msg_1112_);
v___x_1113_ = lean_unsigned_to_nat(0u);
v_bs_x27_1114_ = lean_array_uset(v_bs_1109_, v_i_1108_, v___x_1113_);
v___x_1115_ = ((size_t)1ULL);
v___x_1116_ = lean_usize_add(v_i_1108_, v___x_1115_);
v___x_1117_ = lean_array_uset(v_bs_x27_1114_, v_i_1108_, v_msg_1112_);
v_i_1108_ = v___x_1116_;
v_bs_1109_ = v___x_1117_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18_spec__24___boxed(lean_object* v_sz_1119_, lean_object* v_i_1120_, lean_object* v_bs_1121_){
_start:
{
size_t v_sz_boxed_1122_; size_t v_i_boxed_1123_; lean_object* v_res_1124_; 
v_sz_boxed_1122_ = lean_unbox_usize(v_sz_1119_);
lean_dec(v_sz_1119_);
v_i_boxed_1123_ = lean_unbox_usize(v_i_1120_);
lean_dec(v_i_1120_);
v_res_1124_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18_spec__24(v_sz_boxed_1122_, v_i_boxed_1123_, v_bs_1121_);
return v_res_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18(lean_object* v_oldTraces_1125_, lean_object* v_data_1126_, lean_object* v_ref_1127_, lean_object* v_msg_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_){
_start:
{
lean_object* v_fileName_1134_; lean_object* v_fileMap_1135_; lean_object* v_options_1136_; lean_object* v_currRecDepth_1137_; lean_object* v_maxRecDepth_1138_; lean_object* v_ref_1139_; lean_object* v_currNamespace_1140_; lean_object* v_openDecls_1141_; lean_object* v_initHeartbeats_1142_; lean_object* v_maxHeartbeats_1143_; lean_object* v_quotContext_1144_; lean_object* v_currMacroScope_1145_; uint8_t v_diag_1146_; lean_object* v_cancelTk_x3f_1147_; uint8_t v_suppressElabErrors_1148_; lean_object* v_inheritedTraceOptions_1149_; lean_object* v___x_1150_; lean_object* v_traceState_1151_; lean_object* v_traces_1152_; lean_object* v_ref_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; size_t v_sz_1156_; size_t v___x_1157_; lean_object* v___x_1158_; lean_object* v_msg_1159_; lean_object* v___x_1160_; lean_object* v_a_1161_; lean_object* v___x_1163_; uint8_t v_isShared_1164_; uint8_t v_isSharedCheck_1198_; 
v_fileName_1134_ = lean_ctor_get(v___y_1131_, 0);
v_fileMap_1135_ = lean_ctor_get(v___y_1131_, 1);
v_options_1136_ = lean_ctor_get(v___y_1131_, 2);
v_currRecDepth_1137_ = lean_ctor_get(v___y_1131_, 3);
v_maxRecDepth_1138_ = lean_ctor_get(v___y_1131_, 4);
v_ref_1139_ = lean_ctor_get(v___y_1131_, 5);
v_currNamespace_1140_ = lean_ctor_get(v___y_1131_, 6);
v_openDecls_1141_ = lean_ctor_get(v___y_1131_, 7);
v_initHeartbeats_1142_ = lean_ctor_get(v___y_1131_, 8);
v_maxHeartbeats_1143_ = lean_ctor_get(v___y_1131_, 9);
v_quotContext_1144_ = lean_ctor_get(v___y_1131_, 10);
v_currMacroScope_1145_ = lean_ctor_get(v___y_1131_, 11);
v_diag_1146_ = lean_ctor_get_uint8(v___y_1131_, sizeof(void*)*14);
v_cancelTk_x3f_1147_ = lean_ctor_get(v___y_1131_, 12);
v_suppressElabErrors_1148_ = lean_ctor_get_uint8(v___y_1131_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1149_ = lean_ctor_get(v___y_1131_, 13);
v___x_1150_ = lean_st_ref_get(v___y_1132_);
v_traceState_1151_ = lean_ctor_get(v___x_1150_, 4);
lean_inc_ref(v_traceState_1151_);
lean_dec(v___x_1150_);
v_traces_1152_ = lean_ctor_get(v_traceState_1151_, 0);
lean_inc_ref(v_traces_1152_);
lean_dec_ref(v_traceState_1151_);
v_ref_1153_ = l_Lean_replaceRef(v_ref_1127_, v_ref_1139_);
lean_inc_ref(v_inheritedTraceOptions_1149_);
lean_inc(v_cancelTk_x3f_1147_);
lean_inc(v_currMacroScope_1145_);
lean_inc(v_quotContext_1144_);
lean_inc(v_maxHeartbeats_1143_);
lean_inc(v_initHeartbeats_1142_);
lean_inc(v_openDecls_1141_);
lean_inc(v_currNamespace_1140_);
lean_inc(v_maxRecDepth_1138_);
lean_inc(v_currRecDepth_1137_);
lean_inc_ref(v_options_1136_);
lean_inc_ref(v_fileMap_1135_);
lean_inc_ref(v_fileName_1134_);
v___x_1154_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1154_, 0, v_fileName_1134_);
lean_ctor_set(v___x_1154_, 1, v_fileMap_1135_);
lean_ctor_set(v___x_1154_, 2, v_options_1136_);
lean_ctor_set(v___x_1154_, 3, v_currRecDepth_1137_);
lean_ctor_set(v___x_1154_, 4, v_maxRecDepth_1138_);
lean_ctor_set(v___x_1154_, 5, v_ref_1153_);
lean_ctor_set(v___x_1154_, 6, v_currNamespace_1140_);
lean_ctor_set(v___x_1154_, 7, v_openDecls_1141_);
lean_ctor_set(v___x_1154_, 8, v_initHeartbeats_1142_);
lean_ctor_set(v___x_1154_, 9, v_maxHeartbeats_1143_);
lean_ctor_set(v___x_1154_, 10, v_quotContext_1144_);
lean_ctor_set(v___x_1154_, 11, v_currMacroScope_1145_);
lean_ctor_set(v___x_1154_, 12, v_cancelTk_x3f_1147_);
lean_ctor_set(v___x_1154_, 13, v_inheritedTraceOptions_1149_);
lean_ctor_set_uint8(v___x_1154_, sizeof(void*)*14, v_diag_1146_);
lean_ctor_set_uint8(v___x_1154_, sizeof(void*)*14 + 1, v_suppressElabErrors_1148_);
v___x_1155_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1152_);
lean_dec_ref(v_traces_1152_);
v_sz_1156_ = lean_array_size(v___x_1155_);
v___x_1157_ = ((size_t)0ULL);
v___x_1158_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18_spec__24(v_sz_1156_, v___x_1157_, v___x_1155_);
v_msg_1159_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1159_, 0, v_data_1126_);
lean_ctor_set(v_msg_1159_, 1, v_msg_1128_);
lean_ctor_set(v_msg_1159_, 2, v___x_1158_);
v___x_1160_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(v_msg_1159_, v___y_1129_, v___y_1130_, v___x_1154_, v___y_1132_);
lean_dec_ref_known(v___x_1154_, 14);
v_a_1161_ = lean_ctor_get(v___x_1160_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1160_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1163_ = v___x_1160_;
v_isShared_1164_ = v_isSharedCheck_1198_;
goto v_resetjp_1162_;
}
else
{
lean_inc(v_a_1161_);
lean_dec(v___x_1160_);
v___x_1163_ = lean_box(0);
v_isShared_1164_ = v_isSharedCheck_1198_;
goto v_resetjp_1162_;
}
v_resetjp_1162_:
{
lean_object* v___x_1165_; lean_object* v_traceState_1166_; lean_object* v_env_1167_; lean_object* v_nextMacroScope_1168_; lean_object* v_ngen_1169_; lean_object* v_auxDeclNGen_1170_; lean_object* v_cache_1171_; lean_object* v_messages_1172_; lean_object* v_infoState_1173_; lean_object* v_snapshotTasks_1174_; lean_object* v___x_1176_; uint8_t v_isShared_1177_; uint8_t v_isSharedCheck_1197_; 
v___x_1165_ = lean_st_ref_take(v___y_1132_);
v_traceState_1166_ = lean_ctor_get(v___x_1165_, 4);
v_env_1167_ = lean_ctor_get(v___x_1165_, 0);
v_nextMacroScope_1168_ = lean_ctor_get(v___x_1165_, 1);
v_ngen_1169_ = lean_ctor_get(v___x_1165_, 2);
v_auxDeclNGen_1170_ = lean_ctor_get(v___x_1165_, 3);
v_cache_1171_ = lean_ctor_get(v___x_1165_, 5);
v_messages_1172_ = lean_ctor_get(v___x_1165_, 6);
v_infoState_1173_ = lean_ctor_get(v___x_1165_, 7);
v_snapshotTasks_1174_ = lean_ctor_get(v___x_1165_, 8);
v_isSharedCheck_1197_ = !lean_is_exclusive(v___x_1165_);
if (v_isSharedCheck_1197_ == 0)
{
v___x_1176_ = v___x_1165_;
v_isShared_1177_ = v_isSharedCheck_1197_;
goto v_resetjp_1175_;
}
else
{
lean_inc(v_snapshotTasks_1174_);
lean_inc(v_infoState_1173_);
lean_inc(v_messages_1172_);
lean_inc(v_cache_1171_);
lean_inc(v_traceState_1166_);
lean_inc(v_auxDeclNGen_1170_);
lean_inc(v_ngen_1169_);
lean_inc(v_nextMacroScope_1168_);
lean_inc(v_env_1167_);
lean_dec(v___x_1165_);
v___x_1176_ = lean_box(0);
v_isShared_1177_ = v_isSharedCheck_1197_;
goto v_resetjp_1175_;
}
v_resetjp_1175_:
{
uint64_t v_tid_1178_; lean_object* v___x_1180_; uint8_t v_isShared_1181_; uint8_t v_isSharedCheck_1195_; 
v_tid_1178_ = lean_ctor_get_uint64(v_traceState_1166_, sizeof(void*)*1);
v_isSharedCheck_1195_ = !lean_is_exclusive(v_traceState_1166_);
if (v_isSharedCheck_1195_ == 0)
{
lean_object* v_unused_1196_; 
v_unused_1196_ = lean_ctor_get(v_traceState_1166_, 0);
lean_dec(v_unused_1196_);
v___x_1180_ = v_traceState_1166_;
v_isShared_1181_ = v_isSharedCheck_1195_;
goto v_resetjp_1179_;
}
else
{
lean_dec(v_traceState_1166_);
v___x_1180_ = lean_box(0);
v_isShared_1181_ = v_isSharedCheck_1195_;
goto v_resetjp_1179_;
}
v_resetjp_1179_:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1185_; 
v___x_1182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1182_, 0, v_ref_1127_);
lean_ctor_set(v___x_1182_, 1, v_a_1161_);
v___x_1183_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1125_, v___x_1182_);
if (v_isShared_1181_ == 0)
{
lean_ctor_set(v___x_1180_, 0, v___x_1183_);
v___x_1185_ = v___x_1180_;
goto v_reusejp_1184_;
}
else
{
lean_object* v_reuseFailAlloc_1194_; 
v_reuseFailAlloc_1194_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1194_, 0, v___x_1183_);
lean_ctor_set_uint64(v_reuseFailAlloc_1194_, sizeof(void*)*1, v_tid_1178_);
v___x_1185_ = v_reuseFailAlloc_1194_;
goto v_reusejp_1184_;
}
v_reusejp_1184_:
{
lean_object* v___x_1187_; 
if (v_isShared_1177_ == 0)
{
lean_ctor_set(v___x_1176_, 4, v___x_1185_);
v___x_1187_ = v___x_1176_;
goto v_reusejp_1186_;
}
else
{
lean_object* v_reuseFailAlloc_1193_; 
v_reuseFailAlloc_1193_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1193_, 0, v_env_1167_);
lean_ctor_set(v_reuseFailAlloc_1193_, 1, v_nextMacroScope_1168_);
lean_ctor_set(v_reuseFailAlloc_1193_, 2, v_ngen_1169_);
lean_ctor_set(v_reuseFailAlloc_1193_, 3, v_auxDeclNGen_1170_);
lean_ctor_set(v_reuseFailAlloc_1193_, 4, v___x_1185_);
lean_ctor_set(v_reuseFailAlloc_1193_, 5, v_cache_1171_);
lean_ctor_set(v_reuseFailAlloc_1193_, 6, v_messages_1172_);
lean_ctor_set(v_reuseFailAlloc_1193_, 7, v_infoState_1173_);
lean_ctor_set(v_reuseFailAlloc_1193_, 8, v_snapshotTasks_1174_);
v___x_1187_ = v_reuseFailAlloc_1193_;
goto v_reusejp_1186_;
}
v_reusejp_1186_:
{
lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1191_; 
v___x_1188_ = lean_st_ref_set(v___y_1132_, v___x_1187_);
v___x_1189_ = lean_box(0);
if (v_isShared_1164_ == 0)
{
lean_ctor_set(v___x_1163_, 0, v___x_1189_);
v___x_1191_ = v___x_1163_;
goto v_reusejp_1190_;
}
else
{
lean_object* v_reuseFailAlloc_1192_; 
v_reuseFailAlloc_1192_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1192_, 0, v___x_1189_);
v___x_1191_ = v_reuseFailAlloc_1192_;
goto v_reusejp_1190_;
}
v_reusejp_1190_:
{
return v___x_1191_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18___boxed(lean_object* v_oldTraces_1199_, lean_object* v_data_1200_, lean_object* v_ref_1201_, lean_object* v_msg_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v_res_1208_; 
v_res_1208_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18(v_oldTraces_1199_, v_data_1200_, v_ref_1201_, v_msg_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_);
lean_dec(v___y_1206_);
lean_dec_ref(v___y_1205_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
return v_res_1208_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__1(void){
_start:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; 
v___x_1210_ = ((lean_object*)(lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__0));
v___x_1211_ = l_Lean_stringToMessageData(v___x_1210_);
return v___x_1211_;
}
}
static double _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__2(void){
_start:
{
lean_object* v___x_1212_; double v___x_1213_; 
v___x_1212_ = lean_unsigned_to_nat(1000u);
v___x_1213_ = lean_float_of_nat(v___x_1212_);
return v___x_1213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13(lean_object* v_cls_1214_, uint8_t v_collapsed_1215_, lean_object* v_tag_1216_, lean_object* v_opts_1217_, uint8_t v_clsEnabled_1218_, lean_object* v_oldTraces_1219_, lean_object* v_msg_1220_, lean_object* v_resStartStop_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_){
_start:
{
lean_object* v_fst_1227_; lean_object* v_snd_1228_; lean_object* v___y_1230_; lean_object* v___y_1231_; lean_object* v_data_1232_; lean_object* v_fst_1243_; lean_object* v_snd_1244_; lean_object* v___x_1245_; uint8_t v___x_1246_; lean_object* v___y_1248_; lean_object* v_a_1249_; uint8_t v___y_1264_; double v___y_1295_; 
v_fst_1227_ = lean_ctor_get(v_resStartStop_1221_, 0);
lean_inc(v_fst_1227_);
v_snd_1228_ = lean_ctor_get(v_resStartStop_1221_, 1);
lean_inc(v_snd_1228_);
lean_dec_ref(v_resStartStop_1221_);
v_fst_1243_ = lean_ctor_get(v_snd_1228_, 0);
lean_inc(v_fst_1243_);
v_snd_1244_ = lean_ctor_get(v_snd_1228_, 1);
lean_inc(v_snd_1244_);
lean_dec(v_snd_1228_);
v___x_1245_ = l_Lean_trace_profiler;
v___x_1246_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_opts_1217_, v___x_1245_);
if (v___x_1246_ == 0)
{
v___y_1264_ = v___x_1246_;
goto v___jp_1263_;
}
else
{
lean_object* v___x_1300_; uint8_t v___x_1301_; 
v___x_1300_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1301_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_opts_1217_, v___x_1300_);
if (v___x_1301_ == 0)
{
lean_object* v___x_1302_; lean_object* v___x_1303_; double v___x_1304_; double v___x_1305_; double v___x_1306_; 
v___x_1302_ = l_Lean_trace_profiler_threshold;
v___x_1303_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21(v_opts_1217_, v___x_1302_);
v___x_1304_ = lean_float_of_nat(v___x_1303_);
v___x_1305_ = lean_float_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__2, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__2_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__2);
v___x_1306_ = lean_float_div(v___x_1304_, v___x_1305_);
v___y_1295_ = v___x_1306_;
goto v___jp_1294_;
}
else
{
lean_object* v___x_1307_; lean_object* v___x_1308_; double v___x_1309_; 
v___x_1307_ = l_Lean_trace_profiler_threshold;
v___x_1308_ = lp_mathlib_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__21(v_opts_1217_, v___x_1307_);
v___x_1309_ = lean_float_of_nat(v___x_1308_);
v___y_1295_ = v___x_1309_;
goto v___jp_1294_;
}
}
v___jp_1229_:
{
lean_object* v___x_1233_; 
lean_inc(v___y_1230_);
v___x_1233_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__18(v_oldTraces_1219_, v_data_1232_, v___y_1230_, v___y_1231_, v___y_1222_, v___y_1223_, v___y_1224_, v___y_1225_);
if (lean_obj_tag(v___x_1233_) == 0)
{
lean_object* v___x_1234_; 
lean_dec_ref_known(v___x_1233_, 1);
v___x_1234_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg(v_fst_1227_);
return v___x_1234_;
}
else
{
lean_object* v_a_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1242_; 
lean_dec(v_fst_1227_);
v_a_1235_ = lean_ctor_get(v___x_1233_, 0);
v_isSharedCheck_1242_ = !lean_is_exclusive(v___x_1233_);
if (v_isSharedCheck_1242_ == 0)
{
v___x_1237_ = v___x_1233_;
v_isShared_1238_ = v_isSharedCheck_1242_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_a_1235_);
lean_dec(v___x_1233_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1242_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v___x_1240_; 
if (v_isShared_1238_ == 0)
{
v___x_1240_ = v___x_1237_;
goto v_reusejp_1239_;
}
else
{
lean_object* v_reuseFailAlloc_1241_; 
v_reuseFailAlloc_1241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1241_, 0, v_a_1235_);
v___x_1240_ = v_reuseFailAlloc_1241_;
goto v_reusejp_1239_;
}
v_reusejp_1239_:
{
return v___x_1240_;
}
}
}
}
v___jp_1247_:
{
uint8_t v_result_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; double v___x_1253_; lean_object* v_data_1254_; 
v_result_1250_ = lp_mathlib_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__20(v_fst_1227_);
v___x_1251_ = lean_box(v_result_1250_);
v___x_1252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1252_, 0, v___x_1251_);
v___x_1253_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0___closed__0);
lean_inc_ref(v_tag_1216_);
lean_inc_ref(v___x_1252_);
lean_inc(v_cls_1214_);
v_data_1254_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1254_, 0, v_cls_1214_);
lean_ctor_set(v_data_1254_, 1, v___x_1252_);
lean_ctor_set(v_data_1254_, 2, v_tag_1216_);
lean_ctor_set_float(v_data_1254_, sizeof(void*)*3, v___x_1253_);
lean_ctor_set_float(v_data_1254_, sizeof(void*)*3 + 8, v___x_1253_);
lean_ctor_set_uint8(v_data_1254_, sizeof(void*)*3 + 16, v_collapsed_1215_);
if (v___x_1246_ == 0)
{
lean_dec_ref_known(v___x_1252_, 1);
lean_dec(v_snd_1244_);
lean_dec(v_fst_1243_);
lean_dec_ref(v_tag_1216_);
lean_dec(v_cls_1214_);
v___y_1230_ = v___y_1248_;
v___y_1231_ = v_a_1249_;
v_data_1232_ = v_data_1254_;
goto v___jp_1229_;
}
else
{
lean_object* v_data_1255_; double v___x_1256_; double v___x_1257_; 
lean_dec_ref_known(v_data_1254_, 3);
v_data_1255_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1255_, 0, v_cls_1214_);
lean_ctor_set(v_data_1255_, 1, v___x_1252_);
lean_ctor_set(v_data_1255_, 2, v_tag_1216_);
v___x_1256_ = lean_unbox_float(v_fst_1243_);
lean_dec(v_fst_1243_);
lean_ctor_set_float(v_data_1255_, sizeof(void*)*3, v___x_1256_);
v___x_1257_ = lean_unbox_float(v_snd_1244_);
lean_dec(v_snd_1244_);
lean_ctor_set_float(v_data_1255_, sizeof(void*)*3 + 8, v___x_1257_);
lean_ctor_set_uint8(v_data_1255_, sizeof(void*)*3 + 16, v_collapsed_1215_);
v___y_1230_ = v___y_1248_;
v___y_1231_ = v_a_1249_;
v_data_1232_ = v_data_1255_;
goto v___jp_1229_;
}
}
v___jp_1258_:
{
lean_object* v_ref_1259_; lean_object* v___x_1260_; 
v_ref_1259_ = lean_ctor_get(v___y_1224_, 5);
lean_inc(v___y_1225_);
lean_inc_ref(v___y_1224_);
lean_inc(v___y_1223_);
lean_inc_ref(v___y_1222_);
lean_inc(v_fst_1227_);
v___x_1260_ = lean_apply_6(v_msg_1220_, v_fst_1227_, v___y_1222_, v___y_1223_, v___y_1224_, v___y_1225_, lean_box(0));
if (lean_obj_tag(v___x_1260_) == 0)
{
lean_object* v_a_1261_; 
v_a_1261_ = lean_ctor_get(v___x_1260_, 0);
lean_inc(v_a_1261_);
lean_dec_ref_known(v___x_1260_, 1);
v___y_1248_ = v_ref_1259_;
v_a_1249_ = v_a_1261_;
goto v___jp_1247_;
}
else
{
lean_object* v___x_1262_; 
lean_dec_ref_known(v___x_1260_, 1);
v___x_1262_ = lean_obj_once(&lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__1, &lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__1_once, _init_lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___closed__1);
v___y_1248_ = v_ref_1259_;
v_a_1249_ = v___x_1262_;
goto v___jp_1247_;
}
}
v___jp_1263_:
{
if (v_clsEnabled_1218_ == 0)
{
if (v___y_1264_ == 0)
{
lean_object* v___x_1265_; lean_object* v_traceState_1266_; lean_object* v_env_1267_; lean_object* v_nextMacroScope_1268_; lean_object* v_ngen_1269_; lean_object* v_auxDeclNGen_1270_; lean_object* v_cache_1271_; lean_object* v_messages_1272_; lean_object* v_infoState_1273_; lean_object* v_snapshotTasks_1274_; lean_object* v___x_1276_; uint8_t v_isShared_1277_; uint8_t v_isSharedCheck_1293_; 
lean_dec(v_snd_1244_);
lean_dec(v_fst_1243_);
lean_dec_ref(v_msg_1220_);
lean_dec_ref(v_tag_1216_);
lean_dec(v_cls_1214_);
v___x_1265_ = lean_st_ref_take(v___y_1225_);
v_traceState_1266_ = lean_ctor_get(v___x_1265_, 4);
v_env_1267_ = lean_ctor_get(v___x_1265_, 0);
v_nextMacroScope_1268_ = lean_ctor_get(v___x_1265_, 1);
v_ngen_1269_ = lean_ctor_get(v___x_1265_, 2);
v_auxDeclNGen_1270_ = lean_ctor_get(v___x_1265_, 3);
v_cache_1271_ = lean_ctor_get(v___x_1265_, 5);
v_messages_1272_ = lean_ctor_get(v___x_1265_, 6);
v_infoState_1273_ = lean_ctor_get(v___x_1265_, 7);
v_snapshotTasks_1274_ = lean_ctor_get(v___x_1265_, 8);
v_isSharedCheck_1293_ = !lean_is_exclusive(v___x_1265_);
if (v_isSharedCheck_1293_ == 0)
{
v___x_1276_ = v___x_1265_;
v_isShared_1277_ = v_isSharedCheck_1293_;
goto v_resetjp_1275_;
}
else
{
lean_inc(v_snapshotTasks_1274_);
lean_inc(v_infoState_1273_);
lean_inc(v_messages_1272_);
lean_inc(v_cache_1271_);
lean_inc(v_traceState_1266_);
lean_inc(v_auxDeclNGen_1270_);
lean_inc(v_ngen_1269_);
lean_inc(v_nextMacroScope_1268_);
lean_inc(v_env_1267_);
lean_dec(v___x_1265_);
v___x_1276_ = lean_box(0);
v_isShared_1277_ = v_isSharedCheck_1293_;
goto v_resetjp_1275_;
}
v_resetjp_1275_:
{
uint64_t v_tid_1278_; lean_object* v_traces_1279_; lean_object* v___x_1281_; uint8_t v_isShared_1282_; uint8_t v_isSharedCheck_1292_; 
v_tid_1278_ = lean_ctor_get_uint64(v_traceState_1266_, sizeof(void*)*1);
v_traces_1279_ = lean_ctor_get(v_traceState_1266_, 0);
v_isSharedCheck_1292_ = !lean_is_exclusive(v_traceState_1266_);
if (v_isSharedCheck_1292_ == 0)
{
v___x_1281_ = v_traceState_1266_;
v_isShared_1282_ = v_isSharedCheck_1292_;
goto v_resetjp_1280_;
}
else
{
lean_inc(v_traces_1279_);
lean_dec(v_traceState_1266_);
v___x_1281_ = lean_box(0);
v_isShared_1282_ = v_isSharedCheck_1292_;
goto v_resetjp_1280_;
}
v_resetjp_1280_:
{
lean_object* v___x_1283_; lean_object* v___x_1285_; 
v___x_1283_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1219_, v_traces_1279_);
lean_dec_ref(v_traces_1279_);
if (v_isShared_1282_ == 0)
{
lean_ctor_set(v___x_1281_, 0, v___x_1283_);
v___x_1285_ = v___x_1281_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1291_; 
v_reuseFailAlloc_1291_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1291_, 0, v___x_1283_);
lean_ctor_set_uint64(v_reuseFailAlloc_1291_, sizeof(void*)*1, v_tid_1278_);
v___x_1285_ = v_reuseFailAlloc_1291_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
lean_object* v___x_1287_; 
if (v_isShared_1277_ == 0)
{
lean_ctor_set(v___x_1276_, 4, v___x_1285_);
v___x_1287_ = v___x_1276_;
goto v_reusejp_1286_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v_env_1267_);
lean_ctor_set(v_reuseFailAlloc_1290_, 1, v_nextMacroScope_1268_);
lean_ctor_set(v_reuseFailAlloc_1290_, 2, v_ngen_1269_);
lean_ctor_set(v_reuseFailAlloc_1290_, 3, v_auxDeclNGen_1270_);
lean_ctor_set(v_reuseFailAlloc_1290_, 4, v___x_1285_);
lean_ctor_set(v_reuseFailAlloc_1290_, 5, v_cache_1271_);
lean_ctor_set(v_reuseFailAlloc_1290_, 6, v_messages_1272_);
lean_ctor_set(v_reuseFailAlloc_1290_, 7, v_infoState_1273_);
lean_ctor_set(v_reuseFailAlloc_1290_, 8, v_snapshotTasks_1274_);
v___x_1287_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1286_;
}
v_reusejp_1286_:
{
lean_object* v___x_1288_; lean_object* v___x_1289_; 
v___x_1288_ = lean_st_ref_set(v___y_1225_, v___x_1287_);
v___x_1289_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg(v_fst_1227_);
return v___x_1289_;
}
}
}
}
}
else
{
goto v___jp_1258_;
}
}
else
{
goto v___jp_1258_;
}
}
v___jp_1294_:
{
double v___x_1296_; double v___x_1297_; double v___x_1298_; uint8_t v___x_1299_; 
v___x_1296_ = lean_unbox_float(v_snd_1244_);
v___x_1297_ = lean_unbox_float(v_fst_1243_);
v___x_1298_ = lean_float_sub(v___x_1296_, v___x_1297_);
v___x_1299_ = lean_float_decLt(v___y_1295_, v___x_1298_);
v___y_1264_ = v___x_1299_;
goto v___jp_1263_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13___boxed(lean_object* v_cls_1310_, lean_object* v_collapsed_1311_, lean_object* v_tag_1312_, lean_object* v_opts_1313_, lean_object* v_clsEnabled_1314_, lean_object* v_oldTraces_1315_, lean_object* v_msg_1316_, lean_object* v_resStartStop_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
uint8_t v_collapsed_boxed_1323_; uint8_t v_clsEnabled_boxed_1324_; lean_object* v_res_1325_; 
v_collapsed_boxed_1323_ = lean_unbox(v_collapsed_1311_);
v_clsEnabled_boxed_1324_ = lean_unbox(v_clsEnabled_1314_);
v_res_1325_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13(v_cls_1310_, v_collapsed_boxed_1323_, v_tag_1312_, v_opts_1313_, v_clsEnabled_boxed_1324_, v_oldTraces_1315_, v_msg_1316_, v_resStartStop_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_);
lean_dec(v___y_1321_);
lean_dec_ref(v___y_1320_);
lean_dec(v___y_1319_);
lean_dec_ref(v___y_1318_);
lean_dec_ref(v_opts_1313_);
return v_res_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__8(lean_object* v_a_1326_, lean_object* v_a_1327_){
_start:
{
if (lean_obj_tag(v_a_1326_) == 0)
{
lean_object* v___x_1328_; 
v___x_1328_ = l_List_reverse___redArg(v_a_1327_);
return v___x_1328_;
}
else
{
lean_object* v_head_1329_; lean_object* v_tail_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1339_; 
v_head_1329_ = lean_ctor_get(v_a_1326_, 0);
v_tail_1330_ = lean_ctor_get(v_a_1326_, 1);
v_isSharedCheck_1339_ = !lean_is_exclusive(v_a_1326_);
if (v_isSharedCheck_1339_ == 0)
{
v___x_1332_ = v_a_1326_;
v_isShared_1333_ = v_isSharedCheck_1339_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_tail_1330_);
lean_inc(v_head_1329_);
lean_dec(v_a_1326_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1339_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
lean_object* v___x_1334_; lean_object* v___x_1336_; 
v___x_1334_ = l_Lean_MessageData_ofExpr(v_head_1329_);
if (v_isShared_1333_ == 0)
{
lean_ctor_set(v___x_1332_, 1, v_a_1327_);
lean_ctor_set(v___x_1332_, 0, v___x_1334_);
v___x_1336_ = v___x_1332_;
goto v_reusejp_1335_;
}
else
{
lean_object* v_reuseFailAlloc_1338_; 
v_reuseFailAlloc_1338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1338_, 0, v___x_1334_);
lean_ctor_set(v_reuseFailAlloc_1338_, 1, v_a_1327_);
v___x_1336_ = v_reuseFailAlloc_1338_;
goto v_reusejp_1335_;
}
v_reusejp_1335_:
{
v_a_1326_ = v_tail_1330_;
v_a_1327_ = v___x_1336_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0(lean_object* v___x_1340_, uint8_t v___x_1341_, uint8_t v___x_1342_, lean_object* v_xs_1343_, lean_object* v_x_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_){
_start:
{
lean_object* v___x_1350_; lean_object* v___x_1351_; 
v___x_1350_ = l_Lean_mkAppN(v___x_1340_, v_xs_1343_);
v___x_1351_ = l_Lean_Meta_whnfI(v___x_1350_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_);
if (lean_obj_tag(v___x_1351_) == 0)
{
lean_object* v_a_1352_; uint8_t v___x_1353_; lean_object* v___x_1354_; 
v_a_1352_ = lean_ctor_get(v___x_1351_, 0);
lean_inc(v_a_1352_);
lean_dec_ref_known(v___x_1351_, 1);
v___x_1353_ = 1;
v___x_1354_ = l_Lean_Meta_mkLambdaFVars(v_xs_1343_, v_a_1352_, v___x_1341_, v___x_1342_, v___x_1341_, v___x_1342_, v___x_1353_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_);
return v___x_1354_;
}
else
{
return v___x_1351_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0___boxed(lean_object* v___x_1355_, lean_object* v___x_1356_, lean_object* v___x_1357_, lean_object* v_xs_1358_, lean_object* v_x_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_){
_start:
{
uint8_t v___x_69333__boxed_1365_; uint8_t v___x_69334__boxed_1366_; lean_object* v_res_1367_; 
v___x_69333__boxed_1365_ = lean_unbox(v___x_1356_);
v___x_69334__boxed_1366_ = lean_unbox(v___x_1357_);
v_res_1367_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0(v___x_1355_, v___x_69333__boxed_1365_, v___x_69334__boxed_1366_, v_xs_1358_, v_x_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_);
lean_dec(v___y_1363_);
lean_dec_ref(v___y_1362_);
lean_dec(v___y_1361_);
lean_dec_ref(v___y_1360_);
lean_dec_ref(v_x_1359_);
lean_dec_ref(v_xs_1358_);
return v_res_1367_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__0(void){
_start:
{
lean_object* v___x_1368_; 
v___x_1368_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1368_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1(void){
_start:
{
lean_object* v___x_1369_; lean_object* v___x_1370_; 
v___x_1369_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__0);
v___x_1370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1370_, 0, v___x_1369_);
return v___x_1370_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__2(void){
_start:
{
lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; 
v___x_1371_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1);
v___x_1372_ = lean_unsigned_to_nat(0u);
v___x_1373_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1373_, 0, v___x_1372_);
lean_ctor_set(v___x_1373_, 1, v___x_1372_);
lean_ctor_set(v___x_1373_, 2, v___x_1372_);
lean_ctor_set(v___x_1373_, 3, v___x_1372_);
lean_ctor_set(v___x_1373_, 4, v___x_1371_);
lean_ctor_set(v___x_1373_, 5, v___x_1371_);
lean_ctor_set(v___x_1373_, 6, v___x_1371_);
lean_ctor_set(v___x_1373_, 7, v___x_1371_);
lean_ctor_set(v___x_1373_, 8, v___x_1371_);
lean_ctor_set(v___x_1373_, 9, v___x_1371_);
return v___x_1373_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__3(void){
_start:
{
lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1374_ = lean_unsigned_to_nat(32u);
v___x_1375_ = lean_mk_empty_array_with_capacity(v___x_1374_);
v___x_1376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1376_, 0, v___x_1375_);
return v___x_1376_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__4(void){
_start:
{
size_t v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; 
v___x_1377_ = ((size_t)5ULL);
v___x_1378_ = lean_unsigned_to_nat(0u);
v___x_1379_ = lean_unsigned_to_nat(32u);
v___x_1380_ = lean_mk_empty_array_with_capacity(v___x_1379_);
v___x_1381_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__3);
v___x_1382_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1382_, 0, v___x_1381_);
lean_ctor_set(v___x_1382_, 1, v___x_1380_);
lean_ctor_set(v___x_1382_, 2, v___x_1378_);
lean_ctor_set(v___x_1382_, 3, v___x_1378_);
lean_ctor_set_usize(v___x_1382_, 4, v___x_1377_);
return v___x_1382_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__5(void){
_start:
{
lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; 
v___x_1383_ = lean_box(1);
v___x_1384_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__4);
v___x_1385_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__1);
v___x_1386_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1386_, 0, v___x_1385_);
lean_ctor_set(v___x_1386_, 1, v___x_1384_);
lean_ctor_set(v___x_1386_, 2, v___x_1383_);
return v___x_1386_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7(void){
_start:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1388_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__6));
v___x_1389_ = l_Lean_stringToMessageData(v___x_1388_);
return v___x_1389_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__9(void){
_start:
{
lean_object* v___x_1391_; lean_object* v___x_1392_; 
v___x_1391_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__8));
v___x_1392_ = l_Lean_stringToMessageData(v___x_1391_);
return v___x_1392_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__11(void){
_start:
{
lean_object* v___x_1394_; lean_object* v___x_1395_; 
v___x_1394_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__10));
v___x_1395_ = l_Lean_stringToMessageData(v___x_1394_);
return v___x_1395_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__13(void){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__12));
v___x_1398_ = l_Lean_stringToMessageData(v___x_1397_);
return v___x_1398_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__15(void){
_start:
{
lean_object* v___x_1400_; lean_object* v___x_1401_; 
v___x_1400_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__14));
v___x_1401_ = l_Lean_stringToMessageData(v___x_1400_);
return v___x_1401_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__17(void){
_start:
{
lean_object* v___x_1403_; lean_object* v___x_1404_; 
v___x_1403_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__16));
v___x_1404_ = l_Lean_stringToMessageData(v___x_1403_);
return v___x_1404_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__19(void){
_start:
{
lean_object* v___x_1406_; lean_object* v___x_1407_; 
v___x_1406_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__18));
v___x_1407_ = l_Lean_stringToMessageData(v___x_1406_);
return v___x_1407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg(lean_object* v_msg_1408_, lean_object* v_declHint_1409_, lean_object* v___y_1410_){
_start:
{
lean_object* v___x_1412_; lean_object* v_env_1413_; uint8_t v___x_1414_; 
v___x_1412_ = lean_st_ref_get(v___y_1410_);
v_env_1413_ = lean_ctor_get(v___x_1412_, 0);
lean_inc_ref(v_env_1413_);
lean_dec(v___x_1412_);
v___x_1414_ = l_Lean_Name_isAnonymous(v_declHint_1409_);
if (v___x_1414_ == 0)
{
uint8_t v_isExporting_1415_; 
v_isExporting_1415_ = lean_ctor_get_uint8(v_env_1413_, sizeof(void*)*8);
if (v_isExporting_1415_ == 0)
{
lean_object* v___x_1416_; 
lean_dec_ref(v_env_1413_);
lean_dec(v_declHint_1409_);
v___x_1416_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1416_, 0, v_msg_1408_);
return v___x_1416_;
}
else
{
lean_object* v___x_1417_; uint8_t v___x_1418_; 
lean_inc_ref(v_env_1413_);
v___x_1417_ = l_Lean_Environment_setExporting(v_env_1413_, v___x_1414_);
lean_inc(v_declHint_1409_);
lean_inc_ref(v___x_1417_);
v___x_1418_ = l_Lean_Environment_contains(v___x_1417_, v_declHint_1409_, v_isExporting_1415_);
if (v___x_1418_ == 0)
{
lean_object* v___x_1419_; 
lean_dec_ref(v___x_1417_);
lean_dec_ref(v_env_1413_);
lean_dec(v_declHint_1409_);
v___x_1419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1419_, 0, v_msg_1408_);
return v___x_1419_;
}
else
{
lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v_c_1425_; lean_object* v___x_1426_; 
v___x_1420_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__2);
v___x_1421_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__5);
v___x_1422_ = l_Lean_Options_empty;
v___x_1423_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1417_);
lean_ctor_set(v___x_1423_, 1, v___x_1420_);
lean_ctor_set(v___x_1423_, 2, v___x_1421_);
lean_ctor_set(v___x_1423_, 3, v___x_1422_);
lean_inc(v_declHint_1409_);
v___x_1424_ = l_Lean_MessageData_ofConstName(v_declHint_1409_, v___x_1414_);
v_c_1425_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_1425_, 0, v___x_1423_);
lean_ctor_set(v_c_1425_, 1, v___x_1424_);
v___x_1426_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_1413_, v_declHint_1409_);
if (lean_obj_tag(v___x_1426_) == 0)
{
lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; 
lean_dec_ref(v_env_1413_);
lean_dec(v_declHint_1409_);
v___x_1427_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7);
v___x_1428_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1428_, 0, v___x_1427_);
lean_ctor_set(v___x_1428_, 1, v_c_1425_);
v___x_1429_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__9);
v___x_1430_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1428_);
lean_ctor_set(v___x_1430_, 1, v___x_1429_);
v___x_1431_ = l_Lean_MessageData_note(v___x_1430_);
v___x_1432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1432_, 0, v_msg_1408_);
lean_ctor_set(v___x_1432_, 1, v___x_1431_);
v___x_1433_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1433_, 0, v___x_1432_);
return v___x_1433_;
}
else
{
lean_object* v_val_1434_; lean_object* v___x_1436_; uint8_t v_isShared_1437_; uint8_t v_isSharedCheck_1469_; 
v_val_1434_ = lean_ctor_get(v___x_1426_, 0);
v_isSharedCheck_1469_ = !lean_is_exclusive(v___x_1426_);
if (v_isSharedCheck_1469_ == 0)
{
v___x_1436_ = v___x_1426_;
v_isShared_1437_ = v_isSharedCheck_1469_;
goto v_resetjp_1435_;
}
else
{
lean_inc(v_val_1434_);
lean_dec(v___x_1426_);
v___x_1436_ = lean_box(0);
v_isShared_1437_ = v_isSharedCheck_1469_;
goto v_resetjp_1435_;
}
v_resetjp_1435_:
{
lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v_mod_1441_; uint8_t v___x_1442_; 
v___x_1438_ = lean_box(0);
v___x_1439_ = l_Lean_Environment_header(v_env_1413_);
lean_dec_ref(v_env_1413_);
v___x_1440_ = l_Lean_EnvironmentHeader_moduleNames(v___x_1439_);
v_mod_1441_ = lean_array_get(v___x_1438_, v___x_1440_, v_val_1434_);
lean_dec(v_val_1434_);
lean_dec_ref(v___x_1440_);
v___x_1442_ = l_Lean_isPrivateName(v_declHint_1409_);
lean_dec(v_declHint_1409_);
if (v___x_1442_ == 0)
{
lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1454_; 
v___x_1443_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__11);
v___x_1444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1444_, 0, v___x_1443_);
lean_ctor_set(v___x_1444_, 1, v_c_1425_);
v___x_1445_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__13);
v___x_1446_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1446_, 0, v___x_1444_);
lean_ctor_set(v___x_1446_, 1, v___x_1445_);
v___x_1447_ = l_Lean_MessageData_ofName(v_mod_1441_);
v___x_1448_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1448_, 0, v___x_1446_);
lean_ctor_set(v___x_1448_, 1, v___x_1447_);
v___x_1449_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__15);
v___x_1450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1450_, 0, v___x_1448_);
lean_ctor_set(v___x_1450_, 1, v___x_1449_);
v___x_1451_ = l_Lean_MessageData_note(v___x_1450_);
v___x_1452_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1452_, 0, v_msg_1408_);
lean_ctor_set(v___x_1452_, 1, v___x_1451_);
if (v_isShared_1437_ == 0)
{
lean_ctor_set_tag(v___x_1436_, 0);
lean_ctor_set(v___x_1436_, 0, v___x_1452_);
v___x_1454_ = v___x_1436_;
goto v_reusejp_1453_;
}
else
{
lean_object* v_reuseFailAlloc_1455_; 
v_reuseFailAlloc_1455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1455_, 0, v___x_1452_);
v___x_1454_ = v_reuseFailAlloc_1455_;
goto v_reusejp_1453_;
}
v_reusejp_1453_:
{
return v___x_1454_;
}
}
else
{
lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1467_; 
v___x_1456_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__7);
v___x_1457_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1457_, 0, v___x_1456_);
lean_ctor_set(v___x_1457_, 1, v_c_1425_);
v___x_1458_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__17);
v___x_1459_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1459_, 0, v___x_1457_);
lean_ctor_set(v___x_1459_, 1, v___x_1458_);
v___x_1460_ = l_Lean_MessageData_ofName(v_mod_1441_);
v___x_1461_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1461_, 0, v___x_1459_);
lean_ctor_set(v___x_1461_, 1, v___x_1460_);
v___x_1462_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___closed__19);
v___x_1463_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1463_, 0, v___x_1461_);
lean_ctor_set(v___x_1463_, 1, v___x_1462_);
v___x_1464_ = l_Lean_MessageData_note(v___x_1463_);
v___x_1465_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1465_, 0, v_msg_1408_);
lean_ctor_set(v___x_1465_, 1, v___x_1464_);
if (v_isShared_1437_ == 0)
{
lean_ctor_set_tag(v___x_1436_, 0);
lean_ctor_set(v___x_1436_, 0, v___x_1465_);
v___x_1467_ = v___x_1436_;
goto v_reusejp_1466_;
}
else
{
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v___x_1465_);
v___x_1467_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1466_;
}
v_reusejp_1466_:
{
return v___x_1467_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1470_; 
lean_dec_ref(v_env_1413_);
lean_dec(v_declHint_1409_);
v___x_1470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1470_, 0, v_msg_1408_);
return v___x_1470_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg___boxed(lean_object* v_msg_1471_, lean_object* v_declHint_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_){
_start:
{
lean_object* v_res_1475_; 
v_res_1475_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg(v_msg_1471_, v_declHint_1472_, v___y_1473_);
lean_dec(v___y_1473_);
return v_res_1475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30(lean_object* v_msg_1476_, lean_object* v_declHint_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_){
_start:
{
lean_object* v___x_1483_; lean_object* v_a_1484_; lean_object* v___x_1486_; uint8_t v_isShared_1487_; uint8_t v_isSharedCheck_1493_; 
v___x_1483_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg(v_msg_1476_, v_declHint_1477_, v___y_1481_);
v_a_1484_ = lean_ctor_get(v___x_1483_, 0);
v_isSharedCheck_1493_ = !lean_is_exclusive(v___x_1483_);
if (v_isSharedCheck_1493_ == 0)
{
v___x_1486_ = v___x_1483_;
v_isShared_1487_ = v_isSharedCheck_1493_;
goto v_resetjp_1485_;
}
else
{
lean_inc(v_a_1484_);
lean_dec(v___x_1483_);
v___x_1486_ = lean_box(0);
v_isShared_1487_ = v_isSharedCheck_1493_;
goto v_resetjp_1485_;
}
v_resetjp_1485_:
{
lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1491_; 
v___x_1488_ = l_Lean_unknownIdentifierMessageTag;
v___x_1489_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_1489_, 0, v___x_1488_);
lean_ctor_set(v___x_1489_, 1, v_a_1484_);
if (v_isShared_1487_ == 0)
{
lean_ctor_set(v___x_1486_, 0, v___x_1489_);
v___x_1491_ = v___x_1486_;
goto v_reusejp_1490_;
}
else
{
lean_object* v_reuseFailAlloc_1492_; 
v_reuseFailAlloc_1492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1492_, 0, v___x_1489_);
v___x_1491_ = v_reuseFailAlloc_1492_;
goto v_reusejp_1490_;
}
v_reusejp_1490_:
{
return v___x_1491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30___boxed(lean_object* v_msg_1494_, lean_object* v_declHint_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_){
_start:
{
lean_object* v_res_1501_; 
v_res_1501_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30(v_msg_1494_, v_declHint_1495_, v___y_1496_, v___y_1497_, v___y_1498_, v___y_1499_);
lean_dec(v___y_1499_);
lean_dec_ref(v___y_1498_);
lean_dec(v___y_1497_);
lean_dec_ref(v___y_1496_);
return v_res_1501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg(lean_object* v_ref_1502_, lean_object* v_msg_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_){
_start:
{
lean_object* v_fileName_1509_; lean_object* v_fileMap_1510_; lean_object* v_options_1511_; lean_object* v_currRecDepth_1512_; lean_object* v_maxRecDepth_1513_; lean_object* v_ref_1514_; lean_object* v_currNamespace_1515_; lean_object* v_openDecls_1516_; lean_object* v_initHeartbeats_1517_; lean_object* v_maxHeartbeats_1518_; lean_object* v_quotContext_1519_; lean_object* v_currMacroScope_1520_; uint8_t v_diag_1521_; lean_object* v_cancelTk_x3f_1522_; uint8_t v_suppressElabErrors_1523_; lean_object* v_inheritedTraceOptions_1524_; lean_object* v_ref_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v_fileName_1509_ = lean_ctor_get(v___y_1506_, 0);
v_fileMap_1510_ = lean_ctor_get(v___y_1506_, 1);
v_options_1511_ = lean_ctor_get(v___y_1506_, 2);
v_currRecDepth_1512_ = lean_ctor_get(v___y_1506_, 3);
v_maxRecDepth_1513_ = lean_ctor_get(v___y_1506_, 4);
v_ref_1514_ = lean_ctor_get(v___y_1506_, 5);
v_currNamespace_1515_ = lean_ctor_get(v___y_1506_, 6);
v_openDecls_1516_ = lean_ctor_get(v___y_1506_, 7);
v_initHeartbeats_1517_ = lean_ctor_get(v___y_1506_, 8);
v_maxHeartbeats_1518_ = lean_ctor_get(v___y_1506_, 9);
v_quotContext_1519_ = lean_ctor_get(v___y_1506_, 10);
v_currMacroScope_1520_ = lean_ctor_get(v___y_1506_, 11);
v_diag_1521_ = lean_ctor_get_uint8(v___y_1506_, sizeof(void*)*14);
v_cancelTk_x3f_1522_ = lean_ctor_get(v___y_1506_, 12);
v_suppressElabErrors_1523_ = lean_ctor_get_uint8(v___y_1506_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1524_ = lean_ctor_get(v___y_1506_, 13);
v_ref_1525_ = l_Lean_replaceRef(v_ref_1502_, v_ref_1514_);
lean_inc_ref(v_inheritedTraceOptions_1524_);
lean_inc(v_cancelTk_x3f_1522_);
lean_inc(v_currMacroScope_1520_);
lean_inc(v_quotContext_1519_);
lean_inc(v_maxHeartbeats_1518_);
lean_inc(v_initHeartbeats_1517_);
lean_inc(v_openDecls_1516_);
lean_inc(v_currNamespace_1515_);
lean_inc(v_maxRecDepth_1513_);
lean_inc(v_currRecDepth_1512_);
lean_inc_ref(v_options_1511_);
lean_inc_ref(v_fileMap_1510_);
lean_inc_ref(v_fileName_1509_);
v___x_1526_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1526_, 0, v_fileName_1509_);
lean_ctor_set(v___x_1526_, 1, v_fileMap_1510_);
lean_ctor_set(v___x_1526_, 2, v_options_1511_);
lean_ctor_set(v___x_1526_, 3, v_currRecDepth_1512_);
lean_ctor_set(v___x_1526_, 4, v_maxRecDepth_1513_);
lean_ctor_set(v___x_1526_, 5, v_ref_1525_);
lean_ctor_set(v___x_1526_, 6, v_currNamespace_1515_);
lean_ctor_set(v___x_1526_, 7, v_openDecls_1516_);
lean_ctor_set(v___x_1526_, 8, v_initHeartbeats_1517_);
lean_ctor_set(v___x_1526_, 9, v_maxHeartbeats_1518_);
lean_ctor_set(v___x_1526_, 10, v_quotContext_1519_);
lean_ctor_set(v___x_1526_, 11, v_currMacroScope_1520_);
lean_ctor_set(v___x_1526_, 12, v_cancelTk_x3f_1522_);
lean_ctor_set(v___x_1526_, 13, v_inheritedTraceOptions_1524_);
lean_ctor_set_uint8(v___x_1526_, sizeof(void*)*14, v_diag_1521_);
lean_ctor_set_uint8(v___x_1526_, sizeof(void*)*14 + 1, v_suppressElabErrors_1523_);
v___x_1527_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v_msg_1503_, v___y_1504_, v___y_1505_, v___x_1526_, v___y_1507_);
lean_dec_ref_known(v___x_1526_, 14);
return v___x_1527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg___boxed(lean_object* v_ref_1528_, lean_object* v_msg_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
lean_object* v_res_1535_; 
v_res_1535_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg(v_ref_1528_, v_msg_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_);
lean_dec(v___y_1533_);
lean_dec_ref(v___y_1532_);
lean_dec(v___y_1531_);
lean_dec_ref(v___y_1530_);
lean_dec(v_ref_1528_);
return v_res_1535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg(lean_object* v_ref_1536_, lean_object* v_msg_1537_, lean_object* v_declHint_1538_, lean_object* v___y_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_){
_start:
{
lean_object* v___x_1544_; lean_object* v_a_1545_; lean_object* v___x_1546_; 
v___x_1544_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30(v_msg_1537_, v_declHint_1538_, v___y_1539_, v___y_1540_, v___y_1541_, v___y_1542_);
v_a_1545_ = lean_ctor_get(v___x_1544_, 0);
lean_inc(v_a_1545_);
lean_dec_ref(v___x_1544_);
v___x_1546_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg(v_ref_1536_, v_a_1545_, v___y_1539_, v___y_1540_, v___y_1541_, v___y_1542_);
return v___x_1546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg___boxed(lean_object* v_ref_1547_, lean_object* v_msg_1548_, lean_object* v_declHint_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_){
_start:
{
lean_object* v_res_1555_; 
v_res_1555_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg(v_ref_1547_, v_msg_1548_, v_declHint_1549_, v___y_1550_, v___y_1551_, v___y_1552_, v___y_1553_);
lean_dec(v___y_1553_);
lean_dec_ref(v___y_1552_);
lean_dec(v___y_1551_);
lean_dec_ref(v___y_1550_);
lean_dec(v_ref_1547_);
return v_res_1555_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__1(void){
_start:
{
lean_object* v___x_1557_; lean_object* v___x_1558_; 
v___x_1557_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__0));
v___x_1558_ = l_Lean_stringToMessageData(v___x_1557_);
return v___x_1558_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1560_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__2));
v___x_1561_ = l_Lean_stringToMessageData(v___x_1560_);
return v___x_1561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg(lean_object* v_ref_1562_, lean_object* v_constName_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_){
_start:
{
lean_object* v___x_1569_; uint8_t v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; 
v___x_1569_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__1);
v___x_1570_ = 0;
lean_inc(v_constName_1563_);
v___x_1571_ = l_Lean_MessageData_ofConstName(v_constName_1563_, v___x_1570_);
v___x_1572_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1572_, 0, v___x_1569_);
lean_ctor_set(v___x_1572_, 1, v___x_1571_);
v___x_1573_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_1574_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1574_, 0, v___x_1572_);
lean_ctor_set(v___x_1574_, 1, v___x_1573_);
v___x_1575_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg(v_ref_1562_, v___x_1574_, v_constName_1563_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_);
return v___x_1575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___boxed(lean_object* v_ref_1576_, lean_object* v_constName_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_){
_start:
{
lean_object* v_res_1583_; 
v_res_1583_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg(v_ref_1576_, v_constName_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_);
lean_dec(v___y_1581_);
lean_dec_ref(v___y_1580_);
lean_dec(v___y_1579_);
lean_dec_ref(v___y_1578_);
lean_dec(v_ref_1576_);
return v_res_1583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg(lean_object* v_constName_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_){
_start:
{
lean_object* v_ref_1590_; lean_object* v___x_1591_; 
v_ref_1590_ = lean_ctor_get(v___y_1587_, 5);
v___x_1591_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg(v_ref_1590_, v_constName_1584_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
return v___x_1591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg___boxed(lean_object* v_constName_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_){
_start:
{
lean_object* v_res_1598_; 
v_res_1598_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg(v_constName_1592_, v___y_1593_, v___y_1594_, v___y_1595_, v___y_1596_);
lean_dec(v___y_1596_);
lean_dec_ref(v___y_1595_);
lean_dec(v___y_1594_);
lean_dec_ref(v___y_1593_);
return v_res_1598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2(lean_object* v_constName_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_){
_start:
{
lean_object* v___x_1605_; lean_object* v_env_1606_; uint8_t v___x_1607_; lean_object* v___x_1608_; 
v___x_1605_ = lean_st_ref_get(v___y_1603_);
v_env_1606_ = lean_ctor_get(v___x_1605_, 0);
lean_inc_ref(v_env_1606_);
lean_dec(v___x_1605_);
v___x_1607_ = 0;
lean_inc(v_constName_1599_);
v___x_1608_ = l_Lean_Environment_find_x3f(v_env_1606_, v_constName_1599_, v___x_1607_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_object* v___x_1609_; 
v___x_1609_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg(v_constName_1599_, v___y_1600_, v___y_1601_, v___y_1602_, v___y_1603_);
return v___x_1609_;
}
else
{
lean_object* v_val_1610_; lean_object* v___x_1612_; uint8_t v_isShared_1613_; uint8_t v_isSharedCheck_1617_; 
lean_dec(v_constName_1599_);
v_val_1610_ = lean_ctor_get(v___x_1608_, 0);
v_isSharedCheck_1617_ = !lean_is_exclusive(v___x_1608_);
if (v_isSharedCheck_1617_ == 0)
{
v___x_1612_ = v___x_1608_;
v_isShared_1613_ = v_isSharedCheck_1617_;
goto v_resetjp_1611_;
}
else
{
lean_inc(v_val_1610_);
lean_dec(v___x_1608_);
v___x_1612_ = lean_box(0);
v_isShared_1613_ = v_isSharedCheck_1617_;
goto v_resetjp_1611_;
}
v_resetjp_1611_:
{
lean_object* v___x_1615_; 
if (v_isShared_1613_ == 0)
{
lean_ctor_set_tag(v___x_1612_, 0);
v___x_1615_ = v___x_1612_;
goto v_reusejp_1614_;
}
else
{
lean_object* v_reuseFailAlloc_1616_; 
v_reuseFailAlloc_1616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1616_, 0, v_val_1610_);
v___x_1615_ = v_reuseFailAlloc_1616_;
goto v_reusejp_1614_;
}
v_reusejp_1614_:
{
return v___x_1615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2___boxed(lean_object* v_constName_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_){
_start:
{
lean_object* v_res_1624_; 
v_res_1624_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2(v_constName_1618_, v___y_1619_, v___y_1620_, v___y_1621_, v___y_1622_);
lean_dec(v___y_1622_);
lean_dec_ref(v___y_1621_);
lean_dec(v___y_1620_);
lean_dec_ref(v___y_1619_);
return v_res_1624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4(size_t v_sz_1625_, size_t v_i_1626_, lean_object* v_bs_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_){
_start:
{
uint8_t v___x_1633_; 
v___x_1633_ = lean_usize_dec_lt(v_i_1626_, v_sz_1625_);
if (v___x_1633_ == 0)
{
lean_object* v___x_1634_; 
v___x_1634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1634_, 0, v_bs_1627_);
return v___x_1634_;
}
else
{
lean_object* v_v_1635_; lean_object* v___x_1636_; 
v_v_1635_ = lean_array_uget_borrowed(v_bs_1627_, v_i_1626_);
lean_inc(v_v_1635_);
v___x_1636_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(v_v_1635_, v___y_1629_);
if (lean_obj_tag(v___x_1636_) == 0)
{
lean_object* v_a_1637_; lean_object* v___x_1638_; lean_object* v_bs_x27_1639_; size_t v___x_1640_; size_t v___x_1641_; lean_object* v___x_1642_; 
v_a_1637_ = lean_ctor_get(v___x_1636_, 0);
lean_inc(v_a_1637_);
lean_dec_ref_known(v___x_1636_, 1);
v___x_1638_ = lean_unsigned_to_nat(0u);
v_bs_x27_1639_ = lean_array_uset(v_bs_1627_, v_i_1626_, v___x_1638_);
v___x_1640_ = ((size_t)1ULL);
v___x_1641_ = lean_usize_add(v_i_1626_, v___x_1640_);
v___x_1642_ = lean_array_uset(v_bs_x27_1639_, v_i_1626_, v_a_1637_);
v_i_1626_ = v___x_1641_;
v_bs_1627_ = v___x_1642_;
goto _start;
}
else
{
lean_object* v_a_1644_; lean_object* v___x_1646_; uint8_t v_isShared_1647_; uint8_t v_isSharedCheck_1651_; 
lean_dec_ref(v_bs_1627_);
v_a_1644_ = lean_ctor_get(v___x_1636_, 0);
v_isSharedCheck_1651_ = !lean_is_exclusive(v___x_1636_);
if (v_isSharedCheck_1651_ == 0)
{
v___x_1646_ = v___x_1636_;
v_isShared_1647_ = v_isSharedCheck_1651_;
goto v_resetjp_1645_;
}
else
{
lean_inc(v_a_1644_);
lean_dec(v___x_1636_);
v___x_1646_ = lean_box(0);
v_isShared_1647_ = v_isSharedCheck_1651_;
goto v_resetjp_1645_;
}
v_resetjp_1645_:
{
lean_object* v___x_1649_; 
if (v_isShared_1647_ == 0)
{
v___x_1649_ = v___x_1646_;
goto v_reusejp_1648_;
}
else
{
lean_object* v_reuseFailAlloc_1650_; 
v_reuseFailAlloc_1650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1650_, 0, v_a_1644_);
v___x_1649_ = v_reuseFailAlloc_1650_;
goto v_reusejp_1648_;
}
v_reusejp_1648_:
{
return v___x_1649_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4___boxed(lean_object* v_sz_1652_, lean_object* v_i_1653_, lean_object* v_bs_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_){
_start:
{
size_t v_sz_boxed_1660_; size_t v_i_boxed_1661_; lean_object* v_res_1662_; 
v_sz_boxed_1660_ = lean_unbox_usize(v_sz_1652_);
lean_dec(v_sz_1652_);
v_i_boxed_1661_ = lean_unbox_usize(v_i_1653_);
lean_dec(v_i_1653_);
v_res_1662_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4(v_sz_boxed_1660_, v_i_boxed_1661_, v_bs_1654_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_);
lean_dec(v___y_1658_);
lean_dec_ref(v___y_1657_);
lean_dec(v___y_1656_);
lean_dec_ref(v___y_1655_);
return v_res_1662_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1(void){
_start:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1664_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__0));
v___x_1665_ = l_Lean_stringToMessageData(v___x_1664_);
return v___x_1665_;
}
}
static lean_object* _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3(void){
_start:
{
lean_object* v___x_1667_; lean_object* v___x_1668_; 
v___x_1667_ = ((lean_object*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__2));
v___x_1668_ = l_Lean_stringToMessageData(v___x_1667_);
return v___x_1668_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1670_; lean_object* v___x_1671_; 
v___x_1670_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__0));
v___x_1671_ = l_Lean_stringToMessageData(v___x_1670_);
return v___x_1671_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_1673_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__2));
v___x_1674_ = l_Lean_stringToMessageData(v___x_1673_);
return v___x_1674_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0(void){
_start:
{
lean_object* v_cls_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; 
v_cls_1677_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_));
v___x_1678_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__4));
v___x_1679_ = l_Lean_Name_append(v___x_1678_, v_cls_1677_);
return v___x_1679_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6(void){
_start:
{
lean_object* v___x_1681_; lean_object* v___x_1682_; 
v___x_1681_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__5));
v___x_1682_ = l_Lean_stringToMessageData(v___x_1681_);
return v___x_1682_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8(void){
_start:
{
lean_object* v___x_1684_; lean_object* v___x_1685_; 
v___x_1684_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__7));
v___x_1685_ = l_Lean_stringToMessageData(v___x_1684_);
return v___x_1685_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10(void){
_start:
{
lean_object* v___x_1687_; lean_object* v___x_1688_; 
v___x_1687_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__9));
v___x_1688_ = l_Lean_stringToMessageData(v___x_1687_);
return v___x_1688_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1689_; lean_object* v_dummy_1690_; 
v___x_1689_ = lean_box(0);
v_dummy_1690_ = l_Lean_Expr_sort___override(v___x_1689_);
return v_dummy_1690_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1(void){
_start:
{
lean_object* v___x_1692_; lean_object* v___x_1693_; 
v___x_1692_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__0));
v___x_1693_ = l_Lean_stringToMessageData(v___x_1692_);
return v___x_1693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg(lean_object* v_upperBound_1694_, lean_object* v_fst_1695_, lean_object* v_args_1696_, uint8_t v___x_1697_, lean_object* v_fst_1698_, lean_object* v_val_1699_, lean_object* v_trace_1700_, lean_object* v_a_1701_, lean_object* v_b_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_){
_start:
{
lean_object* v_a_1709_; uint8_t v___x_1713_; 
v___x_1713_ = lean_nat_dec_lt(v_a_1701_, v_upperBound_1694_);
if (v___x_1713_ == 0)
{
lean_object* v___x_1714_; 
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v___x_1714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1714_, 0, v_b_1702_);
return v___x_1714_;
}
else
{
lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; 
v___x_1715_ = l_Lean_instInhabitedExpr;
v___x_1716_ = lean_array_get_borrowed(v___x_1715_, v_fst_1695_, v_a_1701_);
v___x_1717_ = l_Lean_Expr_mvarId_x21(v___x_1716_);
lean_inc(v___x_1717_);
v___x_1718_ = l_Lean_MVarId_getDecl(v___x_1717_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1718_) == 0)
{
lean_object* v_a_1719_; lean_object* v_userName_1720_; lean_object* v_type_1721_; lean_object* v___x_1722_; 
v_a_1719_ = lean_ctor_get(v___x_1718_, 0);
lean_inc(v_a_1719_);
lean_dec_ref_known(v___x_1718_, 1);
v_userName_1720_ = lean_ctor_get(v_a_1719_, 0);
lean_inc(v_userName_1720_);
v_type_1721_ = lean_ctor_get(v_a_1719_, 2);
lean_inc_ref(v_type_1721_);
lean_dec(v_a_1719_);
v___x_1722_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(v_type_1721_, v___y_1704_);
if (lean_obj_tag(v___x_1722_) == 0)
{
lean_object* v_a_1723_; lean_object* v___x_1724_; 
v_a_1723_ = lean_ctor_get(v___x_1722_, 0);
lean_inc_n(v_a_1723_, 2);
lean_dec_ref_known(v___x_1722_, 1);
v___x_1724_ = l_Lean_Meta_isProp(v_a_1723_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1724_) == 0)
{
lean_object* v_a_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; uint8_t v___x_1728_; 
v_a_1725_ = lean_ctor_get(v___x_1724_, 0);
lean_inc(v_a_1725_);
lean_dec_ref_known(v___x_1724_, 1);
v___x_1726_ = lean_box(0);
v___x_1727_ = lean_array_get_borrowed(v___x_1715_, v_args_1696_, v_a_1701_);
v___x_1728_ = lean_unbox(v_a_1725_);
lean_dec(v_a_1725_);
if (v___x_1728_ == 0)
{
uint8_t v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; uint8_t v___x_1732_; uint8_t v___x_1733_; 
v___x_1729_ = 0;
v___x_1730_ = lean_box(v___x_1729_);
v___x_1731_ = lean_array_get(v___x_1730_, v_fst_1698_, v_a_1701_);
lean_dec(v___x_1730_);
v___x_1732_ = lean_unbox(v___x_1731_);
lean_dec(v___x_1731_);
v___x_1733_ = l_Lean_BinderInfo_isInstImplicit(v___x_1732_);
if (v___x_1733_ == 0)
{
lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___f_1736_; lean_object* v___x_1737_; 
lean_dec(v_userName_1720_);
v___x_1734_ = lean_box(v___x_1697_);
v___x_1735_ = lean_box(v___x_1713_);
lean_inc(v___x_1727_);
v___f_1736_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1736_, 0, v___x_1727_);
lean_closure_set(v___f_1736_, 1, v___x_1734_);
lean_closure_set(v___f_1736_, 2, v___x_1735_);
v___x_1737_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(v_a_1723_, v___f_1736_, v___x_1697_, v___x_1697_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1737_) == 0)
{
lean_object* v_a_1738_; lean_object* v___x_1739_; 
v_a_1738_ = lean_ctor_get(v___x_1737_, 0);
lean_inc(v_a_1738_);
lean_dec_ref_known(v___x_1737_, 1);
v___x_1739_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_1717_, v_a_1738_, v___y_1704_);
if (lean_obj_tag(v___x_1739_) == 0)
{
lean_dec_ref_known(v___x_1739_, 1);
v_a_1709_ = v___x_1726_;
goto v___jp_1708_;
}
else
{
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
return v___x_1739_;
}
}
else
{
lean_object* v_a_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1747_; 
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1740_ = lean_ctor_get(v___x_1737_, 0);
v_isSharedCheck_1747_ = !lean_is_exclusive(v___x_1737_);
if (v_isSharedCheck_1747_ == 0)
{
v___x_1742_ = v___x_1737_;
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_a_1740_);
lean_dec(v___x_1737_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1747_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1745_; 
if (v_isShared_1743_ == 0)
{
v___x_1745_ = v___x_1742_;
goto v_reusejp_1744_;
}
else
{
lean_object* v_reuseFailAlloc_1746_; 
v_reuseFailAlloc_1746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1746_, 0, v_a_1740_);
v___x_1745_ = v_reuseFailAlloc_1746_;
goto v_reusejp_1744_;
}
v_reusejp_1744_:
{
return v___x_1745_;
}
}
}
}
else
{
lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; 
lean_inc(v_val_1699_);
v___x_1748_ = l_Lean_Name_append(v_val_1699_, v_userName_1720_);
lean_inc_ref(v_trace_1700_);
v___x_1749_ = lean_array_push(v_trace_1700_, v___x_1748_);
lean_inc(v___x_1727_);
v___x_1750_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(v___x_1727_, v_a_1723_, v___x_1697_, v___x_1749_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1750_) == 0)
{
lean_object* v_a_1751_; lean_object* v___x_1752_; 
v_a_1751_ = lean_ctor_get(v___x_1750_, 0);
lean_inc(v_a_1751_);
lean_dec_ref_known(v___x_1750_, 1);
v___x_1752_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_1717_, v_a_1751_, v___y_1704_);
if (lean_obj_tag(v___x_1752_) == 0)
{
lean_dec_ref_known(v___x_1752_, 1);
v_a_1709_ = v___x_1726_;
goto v___jp_1708_;
}
else
{
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
return v___x_1752_;
}
}
else
{
lean_object* v_a_1753_; lean_object* v___x_1755_; uint8_t v_isShared_1756_; uint8_t v_isSharedCheck_1760_; 
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1753_ = lean_ctor_get(v___x_1750_, 0);
v_isSharedCheck_1760_ = !lean_is_exclusive(v___x_1750_);
if (v_isSharedCheck_1760_ == 0)
{
v___x_1755_ = v___x_1750_;
v_isShared_1756_ = v_isSharedCheck_1760_;
goto v_resetjp_1754_;
}
else
{
lean_inc(v_a_1753_);
lean_dec(v___x_1750_);
v___x_1755_ = lean_box(0);
v_isShared_1756_ = v_isSharedCheck_1760_;
goto v_resetjp_1754_;
}
v_resetjp_1754_:
{
lean_object* v___x_1758_; 
if (v_isShared_1756_ == 0)
{
v___x_1758_ = v___x_1755_;
goto v_reusejp_1757_;
}
else
{
lean_object* v_reuseFailAlloc_1759_; 
v_reuseFailAlloc_1759_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1759_, 0, v_a_1753_);
v___x_1758_ = v_reuseFailAlloc_1759_;
goto v_reusejp_1757_;
}
v_reusejp_1757_:
{
return v___x_1758_;
}
}
}
}
}
else
{
lean_object* v___x_1761_; 
lean_dec(v_userName_1720_);
lean_inc(v___y_1706_);
lean_inc_ref(v___y_1705_);
lean_inc(v___y_1704_);
lean_inc_ref(v___y_1703_);
lean_inc(v___x_1727_);
v___x_1761_ = lean_infer_type(v___x_1727_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1761_) == 0)
{
lean_object* v_a_1762_; lean_object* v_keyedConfig_1763_; uint8_t v_trackZetaDelta_1764_; lean_object* v_zetaDeltaSet_1765_; lean_object* v_lctx_1766_; lean_object* v_localInstances_1767_; lean_object* v_defEqCtx_x3f_1768_; lean_object* v_synthPendingDepth_1769_; lean_object* v_customCanUnfoldPredicate_x3f_1770_; uint8_t v_univApprox_1771_; uint8_t v_inTypeClassResolution_1772_; uint8_t v_cacheInferType_1773_; uint8_t v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; 
v_a_1762_ = lean_ctor_get(v___x_1761_, 0);
lean_inc(v_a_1762_);
lean_dec_ref_known(v___x_1761_, 1);
v_keyedConfig_1763_ = lean_ctor_get(v___y_1703_, 0);
v_trackZetaDelta_1764_ = lean_ctor_get_uint8(v___y_1703_, sizeof(void*)*7);
v_zetaDeltaSet_1765_ = lean_ctor_get(v___y_1703_, 1);
v_lctx_1766_ = lean_ctor_get(v___y_1703_, 2);
v_localInstances_1767_ = lean_ctor_get(v___y_1703_, 3);
v_defEqCtx_x3f_1768_ = lean_ctor_get(v___y_1703_, 4);
v_synthPendingDepth_1769_ = lean_ctor_get(v___y_1703_, 5);
v_customCanUnfoldPredicate_x3f_1770_ = lean_ctor_get(v___y_1703_, 6);
v_univApprox_1771_ = lean_ctor_get_uint8(v___y_1703_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1772_ = lean_ctor_get_uint8(v___y_1703_, sizeof(void*)*7 + 2);
v_cacheInferType_1773_ = lean_ctor_get_uint8(v___y_1703_, sizeof(void*)*7 + 3);
v___x_1774_ = 1;
lean_inc_ref(v_keyedConfig_1763_);
v___x_1775_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1774_, v_keyedConfig_1763_);
lean_inc(v_customCanUnfoldPredicate_x3f_1770_);
lean_inc(v_synthPendingDepth_1769_);
lean_inc(v_defEqCtx_x3f_1768_);
lean_inc_ref(v_localInstances_1767_);
lean_inc_ref(v_lctx_1766_);
lean_inc(v_zetaDeltaSet_1765_);
v___x_1776_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1776_, 0, v___x_1775_);
lean_ctor_set(v___x_1776_, 1, v_zetaDeltaSet_1765_);
lean_ctor_set(v___x_1776_, 2, v_lctx_1766_);
lean_ctor_set(v___x_1776_, 3, v_localInstances_1767_);
lean_ctor_set(v___x_1776_, 4, v_defEqCtx_x3f_1768_);
lean_ctor_set(v___x_1776_, 5, v_synthPendingDepth_1769_);
lean_ctor_set(v___x_1776_, 6, v_customCanUnfoldPredicate_x3f_1770_);
lean_ctor_set_uint8(v___x_1776_, sizeof(void*)*7, v_trackZetaDelta_1764_);
lean_ctor_set_uint8(v___x_1776_, sizeof(void*)*7 + 1, v_univApprox_1771_);
lean_ctor_set_uint8(v___x_1776_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1772_);
lean_ctor_set_uint8(v___x_1776_, sizeof(void*)*7 + 3, v_cacheInferType_1773_);
lean_inc(v_a_1723_);
v___x_1777_ = l_Lean_Meta_isExprDefEq(v_a_1723_, v_a_1762_, v___x_1776_, v___y_1704_, v___y_1705_, v___y_1706_);
lean_dec_ref_known(v___x_1776_, 7);
if (lean_obj_tag(v___x_1777_) == 0)
{
lean_object* v_a_1778_; uint8_t v___x_1779_; 
v_a_1778_ = lean_ctor_get(v___x_1777_, 0);
lean_inc(v_a_1778_);
lean_dec_ref_known(v___x_1777_, 1);
v___x_1779_ = lean_unbox(v_a_1778_);
lean_dec(v_a_1778_);
if (v___x_1779_ == 0)
{
lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; 
lean_dec(v___x_1717_);
v___x_1780_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1);
lean_inc(v___x_1727_);
v___x_1781_ = l_Lean_MessageData_ofExpr(v___x_1727_);
v___x_1782_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1782_, 0, v___x_1780_);
lean_ctor_set(v___x_1782_, 1, v___x_1781_);
v___x_1783_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3);
v___x_1784_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1784_, 0, v___x_1782_);
lean_ctor_set(v___x_1784_, 1, v___x_1783_);
v___x_1785_ = l_Lean_MessageData_ofExpr(v_a_1723_);
v___x_1786_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1786_, 0, v___x_1784_);
lean_ctor_set(v___x_1786_, 1, v___x_1785_);
v___x_1787_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_1788_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1788_, 0, v___x_1786_);
lean_ctor_set(v___x_1788_, 1, v___x_1787_);
v___x_1789_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_1788_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1789_) == 0)
{
lean_dec_ref_known(v___x_1789_, 1);
v_a_1709_ = v___x_1726_;
goto v___jp_1708_;
}
else
{
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
return v___x_1789_;
}
}
else
{
lean_object* v___x_1790_; lean_object* v___x_1791_; 
v___x_1790_ = lean_box(0);
lean_inc(v___x_1727_);
v___x_1791_ = l_Lean_Meta_mkAuxTheorem(v_a_1723_, v___x_1727_, v___x_1713_, v___x_1790_, v___x_1713_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_);
if (lean_obj_tag(v___x_1791_) == 0)
{
lean_object* v_a_1792_; lean_object* v___x_1793_; 
v_a_1792_ = lean_ctor_get(v___x_1791_, 0);
lean_inc(v_a_1792_);
lean_dec_ref_known(v___x_1791_, 1);
v___x_1793_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_1717_, v_a_1792_, v___y_1704_);
if (lean_obj_tag(v___x_1793_) == 0)
{
lean_dec_ref_known(v___x_1793_, 1);
v_a_1709_ = v___x_1726_;
goto v___jp_1708_;
}
else
{
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
return v___x_1793_;
}
}
else
{
lean_object* v_a_1794_; lean_object* v___x_1796_; uint8_t v_isShared_1797_; uint8_t v_isSharedCheck_1801_; 
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1794_ = lean_ctor_get(v___x_1791_, 0);
v_isSharedCheck_1801_ = !lean_is_exclusive(v___x_1791_);
if (v_isSharedCheck_1801_ == 0)
{
v___x_1796_ = v___x_1791_;
v_isShared_1797_ = v_isSharedCheck_1801_;
goto v_resetjp_1795_;
}
else
{
lean_inc(v_a_1794_);
lean_dec(v___x_1791_);
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
else
{
lean_object* v_a_1802_; lean_object* v___x_1804_; uint8_t v_isShared_1805_; uint8_t v_isSharedCheck_1809_; 
lean_dec(v_a_1723_);
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1802_ = lean_ctor_get(v___x_1777_, 0);
v_isSharedCheck_1809_ = !lean_is_exclusive(v___x_1777_);
if (v_isSharedCheck_1809_ == 0)
{
v___x_1804_ = v___x_1777_;
v_isShared_1805_ = v_isSharedCheck_1809_;
goto v_resetjp_1803_;
}
else
{
lean_inc(v_a_1802_);
lean_dec(v___x_1777_);
v___x_1804_ = lean_box(0);
v_isShared_1805_ = v_isSharedCheck_1809_;
goto v_resetjp_1803_;
}
v_resetjp_1803_:
{
lean_object* v___x_1807_; 
if (v_isShared_1805_ == 0)
{
v___x_1807_ = v___x_1804_;
goto v_reusejp_1806_;
}
else
{
lean_object* v_reuseFailAlloc_1808_; 
v_reuseFailAlloc_1808_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1808_, 0, v_a_1802_);
v___x_1807_ = v_reuseFailAlloc_1808_;
goto v_reusejp_1806_;
}
v_reusejp_1806_:
{
return v___x_1807_;
}
}
}
}
else
{
lean_object* v_a_1810_; lean_object* v___x_1812_; uint8_t v_isShared_1813_; uint8_t v_isSharedCheck_1817_; 
lean_dec(v_a_1723_);
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1810_ = lean_ctor_get(v___x_1761_, 0);
v_isSharedCheck_1817_ = !lean_is_exclusive(v___x_1761_);
if (v_isSharedCheck_1817_ == 0)
{
v___x_1812_ = v___x_1761_;
v_isShared_1813_ = v_isSharedCheck_1817_;
goto v_resetjp_1811_;
}
else
{
lean_inc(v_a_1810_);
lean_dec(v___x_1761_);
v___x_1812_ = lean_box(0);
v_isShared_1813_ = v_isSharedCheck_1817_;
goto v_resetjp_1811_;
}
v_resetjp_1811_:
{
lean_object* v___x_1815_; 
if (v_isShared_1813_ == 0)
{
v___x_1815_ = v___x_1812_;
goto v_reusejp_1814_;
}
else
{
lean_object* v_reuseFailAlloc_1816_; 
v_reuseFailAlloc_1816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1816_, 0, v_a_1810_);
v___x_1815_ = v_reuseFailAlloc_1816_;
goto v_reusejp_1814_;
}
v_reusejp_1814_:
{
return v___x_1815_;
}
}
}
}
}
else
{
lean_object* v_a_1818_; lean_object* v___x_1820_; uint8_t v_isShared_1821_; uint8_t v_isSharedCheck_1825_; 
lean_dec(v_a_1723_);
lean_dec(v_userName_1720_);
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1818_ = lean_ctor_get(v___x_1724_, 0);
v_isSharedCheck_1825_ = !lean_is_exclusive(v___x_1724_);
if (v_isSharedCheck_1825_ == 0)
{
v___x_1820_ = v___x_1724_;
v_isShared_1821_ = v_isSharedCheck_1825_;
goto v_resetjp_1819_;
}
else
{
lean_inc(v_a_1818_);
lean_dec(v___x_1724_);
v___x_1820_ = lean_box(0);
v_isShared_1821_ = v_isSharedCheck_1825_;
goto v_resetjp_1819_;
}
v_resetjp_1819_:
{
lean_object* v___x_1823_; 
if (v_isShared_1821_ == 0)
{
v___x_1823_ = v___x_1820_;
goto v_reusejp_1822_;
}
else
{
lean_object* v_reuseFailAlloc_1824_; 
v_reuseFailAlloc_1824_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1824_, 0, v_a_1818_);
v___x_1823_ = v_reuseFailAlloc_1824_;
goto v_reusejp_1822_;
}
v_reusejp_1822_:
{
return v___x_1823_;
}
}
}
}
else
{
lean_object* v_a_1826_; lean_object* v___x_1828_; uint8_t v_isShared_1829_; uint8_t v_isSharedCheck_1833_; 
lean_dec(v_userName_1720_);
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1826_ = lean_ctor_get(v___x_1722_, 0);
v_isSharedCheck_1833_ = !lean_is_exclusive(v___x_1722_);
if (v_isSharedCheck_1833_ == 0)
{
v___x_1828_ = v___x_1722_;
v_isShared_1829_ = v_isSharedCheck_1833_;
goto v_resetjp_1827_;
}
else
{
lean_inc(v_a_1826_);
lean_dec(v___x_1722_);
v___x_1828_ = lean_box(0);
v_isShared_1829_ = v_isSharedCheck_1833_;
goto v_resetjp_1827_;
}
v_resetjp_1827_:
{
lean_object* v___x_1831_; 
if (v_isShared_1829_ == 0)
{
v___x_1831_ = v___x_1828_;
goto v_reusejp_1830_;
}
else
{
lean_object* v_reuseFailAlloc_1832_; 
v_reuseFailAlloc_1832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1832_, 0, v_a_1826_);
v___x_1831_ = v_reuseFailAlloc_1832_;
goto v_reusejp_1830_;
}
v_reusejp_1830_:
{
return v___x_1831_;
}
}
}
}
else
{
lean_object* v_a_1834_; lean_object* v___x_1836_; uint8_t v_isShared_1837_; uint8_t v_isSharedCheck_1841_; 
lean_dec(v___x_1717_);
lean_dec(v_a_1701_);
lean_dec_ref(v_trace_1700_);
lean_dec(v_val_1699_);
v_a_1834_ = lean_ctor_get(v___x_1718_, 0);
v_isSharedCheck_1841_ = !lean_is_exclusive(v___x_1718_);
if (v_isSharedCheck_1841_ == 0)
{
v___x_1836_ = v___x_1718_;
v_isShared_1837_ = v_isSharedCheck_1841_;
goto v_resetjp_1835_;
}
else
{
lean_inc(v_a_1834_);
lean_dec(v___x_1718_);
v___x_1836_ = lean_box(0);
v_isShared_1837_ = v_isSharedCheck_1841_;
goto v_resetjp_1835_;
}
v_resetjp_1835_:
{
lean_object* v___x_1839_; 
if (v_isShared_1837_ == 0)
{
v___x_1839_ = v___x_1836_;
goto v_reusejp_1838_;
}
else
{
lean_object* v_reuseFailAlloc_1840_; 
v_reuseFailAlloc_1840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1840_, 0, v_a_1834_);
v___x_1839_ = v_reuseFailAlloc_1840_;
goto v_reusejp_1838_;
}
v_reusejp_1838_:
{
return v___x_1839_;
}
}
}
}
v___jp_1708_:
{
lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1710_ = lean_unsigned_to_nat(1u);
v___x_1711_ = lean_nat_add(v_a_1701_, v___x_1710_);
lean_dec(v_a_1701_);
v_a_1701_ = v___x_1711_;
v_b_1702_ = v_a_1709_;
goto _start;
}
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3(void){
_start:
{
lean_object* v___x_1843_; lean_object* v___x_1844_; 
v___x_1843_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__2));
v___x_1844_ = l_Lean_stringToMessageData(v___x_1843_);
return v___x_1844_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5(void){
_start:
{
lean_object* v___x_1846_; lean_object* v___x_1847_; 
v___x_1846_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__4));
v___x_1847_ = l_Lean_stringToMessageData(v___x_1846_);
return v___x_1847_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7(void){
_start:
{
lean_object* v___x_1849_; lean_object* v___x_1850_; 
v___x_1849_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__6));
v___x_1850_ = l_Lean_stringToMessageData(v___x_1849_);
return v___x_1850_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9(void){
_start:
{
lean_object* v___x_1852_; lean_object* v___x_1853_; 
v___x_1852_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__8));
v___x_1853_ = l_Lean_stringToMessageData(v___x_1852_);
return v___x_1853_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11(void){
_start:
{
lean_object* v___x_1855_; lean_object* v___x_1856_; 
v___x_1855_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__10));
v___x_1856_ = l_Lean_stringToMessageData(v___x_1855_);
return v___x_1856_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13(void){
_start:
{
lean_object* v___x_1858_; lean_object* v___x_1859_; 
v___x_1858_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__12));
v___x_1859_ = l_Lean_stringToMessageData(v___x_1858_);
return v___x_1859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__9(lean_object* v_val_1860_, lean_object* v_trace_1861_, uint8_t v___x_1862_, lean_object* v_expectedType_1863_, lean_object* v_inst_1864_, lean_object* v_x_1865_, lean_object* v_x_1866_, lean_object* v_x_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_){
_start:
{
lean_object* v_m_1874_; lean_object* v___y_1875_; lean_object* v___y_1876_; lean_object* v___y_1877_; lean_object* v___y_1878_; 
if (lean_obj_tag(v_x_1865_) == 5)
{
lean_object* v_fn_1886_; lean_object* v_arg_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; 
v_fn_1886_ = lean_ctor_get(v_x_1865_, 0);
lean_inc_ref(v_fn_1886_);
v_arg_1887_ = lean_ctor_get(v_x_1865_, 1);
lean_inc_ref(v_arg_1887_);
lean_dec_ref_known(v_x_1865_, 2);
v___x_1888_ = lean_array_set(v_x_1866_, v_x_1867_, v_arg_1887_);
v___x_1889_ = lean_unsigned_to_nat(1u);
v___x_1890_ = lean_nat_sub(v_x_1867_, v___x_1889_);
lean_dec(v_x_1867_);
v_x_1865_ = v_fn_1886_;
v_x_1866_ = v___x_1888_;
v_x_1867_ = v___x_1890_;
goto _start;
}
else
{
lean_dec(v_x_1867_);
if (lean_obj_tag(v_x_1865_) == 4)
{
lean_object* v_declName_1892_; lean_object* v___x_1893_; 
v_declName_1892_ = lean_ctor_get(v_x_1865_, 0);
lean_inc(v_declName_1892_);
v___x_1893_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2(v_declName_1892_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
if (lean_obj_tag(v___x_1893_) == 0)
{
lean_object* v_a_1894_; 
v_a_1894_ = lean_ctor_get(v___x_1893_, 0);
lean_inc(v_a_1894_);
lean_dec_ref_known(v___x_1893_, 1);
if (lean_obj_tag(v_a_1894_) == 6)
{
lean_object* v_val_1895_; lean_object* v___x_1896_; 
lean_dec_ref(v_inst_1864_);
v_val_1895_ = lean_ctor_get(v_a_1894_, 0);
lean_inc_ref(v_val_1895_);
lean_dec_ref_known(v_a_1894_, 1);
lean_inc(v___y_1871_);
lean_inc_ref(v___y_1870_);
lean_inc(v___y_1869_);
lean_inc_ref(v___y_1868_);
lean_inc_ref(v_x_1865_);
v___x_1896_ = lean_infer_type(v_x_1865_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
if (lean_obj_tag(v___x_1896_) == 0)
{
lean_object* v_a_1897_; uint8_t v___x_1898_; lean_object* v___x_1899_; 
v_a_1897_ = lean_ctor_get(v___x_1896_, 0);
lean_inc(v_a_1897_);
lean_dec_ref_known(v___x_1896_, 1);
v___x_1898_ = 0;
v___x_1899_ = l_Lean_Meta_forallMetaTelescope(v_a_1897_, v___x_1898_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
if (lean_obj_tag(v___x_1899_) == 0)
{
lean_object* v_a_1900_; lean_object* v_snd_1901_; lean_object* v_fst_1902_; lean_object* v___x_1904_; uint8_t v_isShared_1905_; uint8_t v_isSharedCheck_2008_; 
v_a_1900_ = lean_ctor_get(v___x_1899_, 0);
lean_inc(v_a_1900_);
lean_dec_ref_known(v___x_1899_, 1);
v_snd_1901_ = lean_ctor_get(v_a_1900_, 1);
v_fst_1902_ = lean_ctor_get(v_a_1900_, 0);
v_isSharedCheck_2008_ = !lean_is_exclusive(v_a_1900_);
if (v_isSharedCheck_2008_ == 0)
{
v___x_1904_ = v_a_1900_;
v_isShared_1905_ = v_isSharedCheck_2008_;
goto v_resetjp_1903_;
}
else
{
lean_inc(v_snd_1901_);
lean_inc(v_fst_1902_);
lean_dec(v_a_1900_);
v___x_1904_ = lean_box(0);
v_isShared_1905_ = v_isSharedCheck_2008_;
goto v_resetjp_1903_;
}
v_resetjp_1903_:
{
lean_object* v_fst_1906_; lean_object* v_snd_1907_; lean_object* v___x_1909_; uint8_t v_isShared_1910_; uint8_t v_isSharedCheck_2007_; 
v_fst_1906_ = lean_ctor_get(v_snd_1901_, 0);
v_snd_1907_ = lean_ctor_get(v_snd_1901_, 1);
v_isSharedCheck_2007_ = !lean_is_exclusive(v_snd_1901_);
if (v_isSharedCheck_2007_ == 0)
{
v___x_1909_ = v_snd_1901_;
v_isShared_1910_ = v_isSharedCheck_2007_;
goto v_resetjp_1908_;
}
else
{
lean_inc(v_snd_1907_);
lean_inc(v_fst_1906_);
lean_dec(v_snd_1901_);
v___x_1909_ = lean_box(0);
v_isShared_1910_ = v_isSharedCheck_2007_;
goto v_resetjp_1908_;
}
v_resetjp_1908_:
{
lean_object* v___y_1912_; lean_object* v___y_1913_; lean_object* v___y_1914_; lean_object* v___y_1915_; lean_object* v___y_1949_; lean_object* v___y_1950_; lean_object* v___y_1951_; lean_object* v___y_1952_; lean_object* v___x_1985_; lean_object* v___x_1986_; uint8_t v___x_1987_; 
v___x_1985_ = lean_array_get_size(v_x_1866_);
v___x_1986_ = lean_array_get_size(v_fst_1902_);
v___x_1987_ = lean_nat_dec_eq(v___x_1985_, v___x_1986_);
if (v___x_1987_ == 0)
{
lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v_a_1999_; lean_object* v___x_2001_; uint8_t v_isShared_2002_; uint8_t v_isSharedCheck_2006_; 
lean_del_object(v___x_1909_);
lean_dec(v_snd_1907_);
lean_dec(v_fst_1906_);
lean_del_object(v___x_1904_);
lean_dec(v_fst_1902_);
lean_dec_ref(v_val_1895_);
lean_dec_ref(v_expectedType_1863_);
lean_dec_ref(v_trace_1861_);
lean_dec(v_val_1860_);
v___x_1988_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5);
v___x_1989_ = l_Lean_MessageData_ofExpr(v_x_1865_);
v___x_1990_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1990_, 0, v___x_1988_);
lean_ctor_set(v___x_1990_, 1, v___x_1989_);
v___x_1991_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7);
v___x_1992_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1992_, 0, v___x_1990_);
lean_ctor_set(v___x_1992_, 1, v___x_1991_);
v___x_1993_ = lean_array_to_list(v_x_1866_);
v___x_1994_ = lean_box(0);
v___x_1995_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__8(v___x_1993_, v___x_1994_);
v___x_1996_ = l_Lean_MessageData_ofList(v___x_1995_);
v___x_1997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1997_, 0, v___x_1992_);
lean_ctor_set(v___x_1997_, 1, v___x_1996_);
v___x_1998_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_1997_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_);
v_a_1999_ = lean_ctor_get(v___x_1998_, 0);
v_isSharedCheck_2006_ = !lean_is_exclusive(v___x_1998_);
if (v_isSharedCheck_2006_ == 0)
{
v___x_2001_ = v___x_1998_;
v_isShared_2002_ = v_isSharedCheck_2006_;
goto v_resetjp_2000_;
}
else
{
lean_inc(v_a_1999_);
lean_dec(v___x_1998_);
v___x_2001_ = lean_box(0);
v_isShared_2002_ = v_isSharedCheck_2006_;
goto v_resetjp_2000_;
}
v_resetjp_2000_:
{
lean_object* v___x_2004_; 
if (v_isShared_2002_ == 0)
{
v___x_2004_ = v___x_2001_;
goto v_reusejp_2003_;
}
else
{
lean_object* v_reuseFailAlloc_2005_; 
v_reuseFailAlloc_2005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2005_, 0, v_a_1999_);
v___x_2004_ = v_reuseFailAlloc_2005_;
goto v_reusejp_2003_;
}
v_reusejp_2003_:
{
return v___x_2004_;
}
}
}
else
{
v___y_1949_ = v___y_1868_;
v___y_1950_ = v___y_1869_;
v___y_1951_ = v___y_1870_;
v___y_1952_ = v___y_1871_;
goto v___jp_1948_;
}
v___jp_1911_:
{
lean_object* v_numParams_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; 
v_numParams_1916_ = lean_ctor_get(v_val_1895_, 3);
lean_inc(v_numParams_1916_);
lean_dec_ref(v_val_1895_);
v___x_1917_ = lean_array_get_size(v_x_1866_);
v___x_1918_ = lean_box(0);
v___x_1919_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg(v___x_1917_, v_fst_1902_, v_x_1866_, v___x_1862_, v_fst_1906_, v_val_1860_, v_trace_1861_, v_numParams_1916_, v___x_1918_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec(v_fst_1906_);
lean_dec_ref(v_x_1866_);
if (lean_obj_tag(v___x_1919_) == 0)
{
size_t v_sz_1920_; size_t v___x_1921_; lean_object* v___x_1922_; 
lean_dec_ref_known(v___x_1919_, 1);
v_sz_1920_ = lean_array_size(v_fst_1902_);
v___x_1921_ = ((size_t)0ULL);
v___x_1922_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4(v_sz_1920_, v___x_1921_, v_fst_1902_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
if (lean_obj_tag(v___x_1922_) == 0)
{
lean_object* v_a_1923_; lean_object* v___x_1925_; uint8_t v_isShared_1926_; uint8_t v_isSharedCheck_1931_; 
v_a_1923_ = lean_ctor_get(v___x_1922_, 0);
v_isSharedCheck_1931_ = !lean_is_exclusive(v___x_1922_);
if (v_isSharedCheck_1931_ == 0)
{
v___x_1925_ = v___x_1922_;
v_isShared_1926_ = v_isSharedCheck_1931_;
goto v_resetjp_1924_;
}
else
{
lean_inc(v_a_1923_);
lean_dec(v___x_1922_);
v___x_1925_ = lean_box(0);
v_isShared_1926_ = v_isSharedCheck_1931_;
goto v_resetjp_1924_;
}
v_resetjp_1924_:
{
lean_object* v___x_1927_; lean_object* v___x_1929_; 
v___x_1927_ = l_Lean_mkAppN(v_x_1865_, v_a_1923_);
lean_dec(v_a_1923_);
if (v_isShared_1926_ == 0)
{
lean_ctor_set(v___x_1925_, 0, v___x_1927_);
v___x_1929_ = v___x_1925_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_1930_; 
v_reuseFailAlloc_1930_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1930_, 0, v___x_1927_);
v___x_1929_ = v_reuseFailAlloc_1930_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
return v___x_1929_;
}
}
}
else
{
lean_object* v_a_1932_; lean_object* v___x_1934_; uint8_t v_isShared_1935_; uint8_t v_isSharedCheck_1939_; 
lean_dec_ref_known(v_x_1865_, 2);
v_a_1932_ = lean_ctor_get(v___x_1922_, 0);
v_isSharedCheck_1939_ = !lean_is_exclusive(v___x_1922_);
if (v_isSharedCheck_1939_ == 0)
{
v___x_1934_ = v___x_1922_;
v_isShared_1935_ = v_isSharedCheck_1939_;
goto v_resetjp_1933_;
}
else
{
lean_inc(v_a_1932_);
lean_dec(v___x_1922_);
v___x_1934_ = lean_box(0);
v_isShared_1935_ = v_isSharedCheck_1939_;
goto v_resetjp_1933_;
}
v_resetjp_1933_:
{
lean_object* v___x_1937_; 
if (v_isShared_1935_ == 0)
{
v___x_1937_ = v___x_1934_;
goto v_reusejp_1936_;
}
else
{
lean_object* v_reuseFailAlloc_1938_; 
v_reuseFailAlloc_1938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1938_, 0, v_a_1932_);
v___x_1937_ = v_reuseFailAlloc_1938_;
goto v_reusejp_1936_;
}
v_reusejp_1936_:
{
return v___x_1937_;
}
}
}
}
else
{
lean_object* v_a_1940_; lean_object* v___x_1942_; uint8_t v_isShared_1943_; uint8_t v_isSharedCheck_1947_; 
lean_dec(v_fst_1902_);
lean_dec_ref_known(v_x_1865_, 2);
v_a_1940_ = lean_ctor_get(v___x_1919_, 0);
v_isSharedCheck_1947_ = !lean_is_exclusive(v___x_1919_);
if (v_isSharedCheck_1947_ == 0)
{
v___x_1942_ = v___x_1919_;
v_isShared_1943_ = v_isSharedCheck_1947_;
goto v_resetjp_1941_;
}
else
{
lean_inc(v_a_1940_);
lean_dec(v___x_1919_);
v___x_1942_ = lean_box(0);
v_isShared_1943_ = v_isSharedCheck_1947_;
goto v_resetjp_1941_;
}
v_resetjp_1941_:
{
lean_object* v___x_1945_; 
if (v_isShared_1943_ == 0)
{
v___x_1945_ = v___x_1942_;
goto v_reusejp_1944_;
}
else
{
lean_object* v_reuseFailAlloc_1946_; 
v_reuseFailAlloc_1946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1946_, 0, v_a_1940_);
v___x_1945_ = v_reuseFailAlloc_1946_;
goto v_reusejp_1944_;
}
v_reusejp_1944_:
{
return v___x_1945_;
}
}
}
}
v___jp_1948_:
{
lean_object* v___x_1953_; 
lean_inc_ref(v_expectedType_1863_);
v___x_1953_ = l_Lean_Meta_isExprDefEq(v_expectedType_1863_, v_snd_1907_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_);
if (lean_obj_tag(v___x_1953_) == 0)
{
lean_object* v_a_1954_; uint8_t v___x_1955_; 
v_a_1954_ = lean_ctor_get(v___x_1953_, 0);
lean_inc(v_a_1954_);
lean_dec_ref_known(v___x_1953_, 1);
v___x_1955_ = lean_unbox(v_a_1954_);
lean_dec(v_a_1954_);
if (v___x_1955_ == 0)
{
lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1959_; 
lean_inc(v_declName_1892_);
lean_dec(v_fst_1906_);
lean_dec(v_fst_1902_);
lean_dec_ref(v_val_1895_);
lean_dec_ref_known(v_x_1865_, 2);
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_trace_1861_);
lean_dec(v_val_1860_);
v___x_1956_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_1957_ = l_Lean_MessageData_ofExpr(v_expectedType_1863_);
if (v_isShared_1910_ == 0)
{
lean_ctor_set_tag(v___x_1909_, 7);
lean_ctor_set(v___x_1909_, 1, v___x_1957_);
lean_ctor_set(v___x_1909_, 0, v___x_1956_);
v___x_1959_ = v___x_1909_;
goto v_reusejp_1958_;
}
else
{
lean_object* v_reuseFailAlloc_1976_; 
v_reuseFailAlloc_1976_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1976_, 0, v___x_1956_);
lean_ctor_set(v_reuseFailAlloc_1976_, 1, v___x_1957_);
v___x_1959_ = v_reuseFailAlloc_1976_;
goto v_reusejp_1958_;
}
v_reusejp_1958_:
{
lean_object* v___x_1960_; lean_object* v___x_1962_; 
v___x_1960_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3);
if (v_isShared_1905_ == 0)
{
lean_ctor_set_tag(v___x_1904_, 7);
lean_ctor_set(v___x_1904_, 1, v___x_1960_);
lean_ctor_set(v___x_1904_, 0, v___x_1959_);
v___x_1962_ = v___x_1904_;
goto v_reusejp_1961_;
}
else
{
lean_object* v_reuseFailAlloc_1975_; 
v_reuseFailAlloc_1975_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1975_, 0, v___x_1959_);
lean_ctor_set(v_reuseFailAlloc_1975_, 1, v___x_1960_);
v___x_1962_ = v_reuseFailAlloc_1975_;
goto v_reusejp_1961_;
}
v_reusejp_1961_:
{
lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v_a_1967_; lean_object* v___x_1969_; uint8_t v_isShared_1970_; uint8_t v_isSharedCheck_1974_; 
v___x_1963_ = l_Lean_MessageData_ofConstName(v_declName_1892_, v___x_1862_);
v___x_1964_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1964_, 0, v___x_1962_);
lean_ctor_set(v___x_1964_, 1, v___x_1963_);
v___x_1965_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1965_, 0, v___x_1964_);
lean_ctor_set(v___x_1965_, 1, v___x_1956_);
v___x_1966_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_1965_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_);
v_a_1967_ = lean_ctor_get(v___x_1966_, 0);
v_isSharedCheck_1974_ = !lean_is_exclusive(v___x_1966_);
if (v_isSharedCheck_1974_ == 0)
{
v___x_1969_ = v___x_1966_;
v_isShared_1970_ = v_isSharedCheck_1974_;
goto v_resetjp_1968_;
}
else
{
lean_inc(v_a_1967_);
lean_dec(v___x_1966_);
v___x_1969_ = lean_box(0);
v_isShared_1970_ = v_isSharedCheck_1974_;
goto v_resetjp_1968_;
}
v_resetjp_1968_:
{
lean_object* v___x_1972_; 
if (v_isShared_1970_ == 0)
{
v___x_1972_ = v___x_1969_;
goto v_reusejp_1971_;
}
else
{
lean_object* v_reuseFailAlloc_1973_; 
v_reuseFailAlloc_1973_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1973_, 0, v_a_1967_);
v___x_1972_ = v_reuseFailAlloc_1973_;
goto v_reusejp_1971_;
}
v_reusejp_1971_:
{
return v___x_1972_;
}
}
}
}
}
else
{
lean_del_object(v___x_1909_);
lean_del_object(v___x_1904_);
lean_dec_ref(v_expectedType_1863_);
v___y_1912_ = v___y_1949_;
v___y_1913_ = v___y_1950_;
v___y_1914_ = v___y_1951_;
v___y_1915_ = v___y_1952_;
goto v___jp_1911_;
}
}
else
{
lean_object* v_a_1977_; lean_object* v___x_1979_; uint8_t v_isShared_1980_; uint8_t v_isSharedCheck_1984_; 
lean_del_object(v___x_1909_);
lean_dec(v_fst_1906_);
lean_del_object(v___x_1904_);
lean_dec(v_fst_1902_);
lean_dec_ref(v_val_1895_);
lean_dec_ref_known(v_x_1865_, 2);
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_expectedType_1863_);
lean_dec_ref(v_trace_1861_);
lean_dec(v_val_1860_);
v_a_1977_ = lean_ctor_get(v___x_1953_, 0);
v_isSharedCheck_1984_ = !lean_is_exclusive(v___x_1953_);
if (v_isSharedCheck_1984_ == 0)
{
v___x_1979_ = v___x_1953_;
v_isShared_1980_ = v_isSharedCheck_1984_;
goto v_resetjp_1978_;
}
else
{
lean_inc(v_a_1977_);
lean_dec(v___x_1953_);
v___x_1979_ = lean_box(0);
v_isShared_1980_ = v_isSharedCheck_1984_;
goto v_resetjp_1978_;
}
v_resetjp_1978_:
{
lean_object* v___x_1982_; 
if (v_isShared_1980_ == 0)
{
v___x_1982_ = v___x_1979_;
goto v_reusejp_1981_;
}
else
{
lean_object* v_reuseFailAlloc_1983_; 
v_reuseFailAlloc_1983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1983_, 0, v_a_1977_);
v___x_1982_ = v_reuseFailAlloc_1983_;
goto v_reusejp_1981_;
}
v_reusejp_1981_:
{
return v___x_1982_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2009_; lean_object* v___x_2011_; uint8_t v_isShared_2012_; uint8_t v_isSharedCheck_2016_; 
lean_dec_ref(v_val_1895_);
lean_dec_ref_known(v_x_1865_, 2);
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_expectedType_1863_);
lean_dec_ref(v_trace_1861_);
lean_dec(v_val_1860_);
v_a_2009_ = lean_ctor_get(v___x_1899_, 0);
v_isSharedCheck_2016_ = !lean_is_exclusive(v___x_1899_);
if (v_isSharedCheck_2016_ == 0)
{
v___x_2011_ = v___x_1899_;
v_isShared_2012_ = v_isSharedCheck_2016_;
goto v_resetjp_2010_;
}
else
{
lean_inc(v_a_2009_);
lean_dec(v___x_1899_);
v___x_2011_ = lean_box(0);
v_isShared_2012_ = v_isSharedCheck_2016_;
goto v_resetjp_2010_;
}
v_resetjp_2010_:
{
lean_object* v___x_2014_; 
if (v_isShared_2012_ == 0)
{
v___x_2014_ = v___x_2011_;
goto v_reusejp_2013_;
}
else
{
lean_object* v_reuseFailAlloc_2015_; 
v_reuseFailAlloc_2015_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2015_, 0, v_a_2009_);
v___x_2014_ = v_reuseFailAlloc_2015_;
goto v_reusejp_2013_;
}
v_reusejp_2013_:
{
return v___x_2014_;
}
}
}
}
else
{
lean_dec_ref(v_val_1895_);
lean_dec_ref_known(v_x_1865_, 2);
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_expectedType_1863_);
lean_dec_ref(v_trace_1861_);
lean_dec(v_val_1860_);
return v___x_1896_;
}
}
else
{
lean_object* v___x_2017_; lean_object* v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; 
lean_inc(v_declName_1892_);
lean_dec(v_a_1894_);
lean_dec_ref_known(v_x_1865_, 2);
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_expectedType_1863_);
v___x_2017_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9);
v___x_2018_ = l_Lean_indentExpr(v_inst_1864_);
v___x_2019_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2019_, 0, v___x_2017_);
lean_ctor_set(v___x_2019_, 1, v___x_2018_);
v___x_2020_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11);
v___x_2021_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2021_, 0, v___x_2019_);
lean_ctor_set(v___x_2021_, 1, v___x_2020_);
v___x_2022_ = l_Lean_MessageData_ofName(v_declName_1892_);
v___x_2023_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2023_, 0, v___x_2021_);
lean_ctor_set(v___x_2023_, 1, v___x_2022_);
v___x_2024_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13);
v___x_2025_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2025_, 0, v___x_2023_);
lean_ctor_set(v___x_2025_, 1, v___x_2024_);
v_m_1874_ = v___x_2025_;
v___y_1875_ = v___y_1868_;
v___y_1876_ = v___y_1869_;
v___y_1877_ = v___y_1870_;
v___y_1878_ = v___y_1871_;
goto v___jp_1873_;
}
}
else
{
lean_object* v_a_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2033_; 
lean_dec_ref_known(v_x_1865_, 2);
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_inst_1864_);
lean_dec_ref(v_expectedType_1863_);
lean_dec_ref(v_trace_1861_);
lean_dec(v_val_1860_);
v_a_2026_ = lean_ctor_get(v___x_1893_, 0);
v_isSharedCheck_2033_ = !lean_is_exclusive(v___x_1893_);
if (v_isSharedCheck_2033_ == 0)
{
v___x_2028_ = v___x_1893_;
v_isShared_2029_ = v_isSharedCheck_2033_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_a_2026_);
lean_dec(v___x_1893_);
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
else
{
lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; 
lean_dec_ref(v_x_1866_);
lean_dec_ref(v_x_1865_);
lean_dec_ref(v_expectedType_1863_);
v___x_2034_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9);
v___x_2035_ = l_Lean_indentExpr(v_inst_1864_);
v___x_2036_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2036_, 0, v___x_2034_);
lean_ctor_set(v___x_2036_, 1, v___x_2035_);
v_m_1874_ = v___x_2036_;
v___y_1875_ = v___y_1868_;
v___y_1876_ = v___y_1869_;
v___y_1877_ = v___y_1870_;
v___y_1878_ = v___y_1871_;
goto v___jp_1873_;
}
}
v___jp_1873_:
{
lean_object* v___x_1879_; lean_object* v_env_1880_; uint8_t v___x_1881_; 
v___x_1879_ = lean_st_ref_get(v___y_1878_);
v_env_1880_ = lean_ctor_get(v___x_1879_, 0);
lean_inc_ref(v_env_1880_);
lean_dec(v___x_1879_);
v___x_1881_ = l_Lean_isStructure(v_env_1880_, v_val_1860_);
if (v___x_1881_ == 0)
{
lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; 
v___x_1882_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1);
v___x_1883_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1883_, 0, v_m_1874_);
lean_ctor_set(v___x_1883_, 1, v___x_1882_);
v___x_1884_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_1861_, v___x_1883_, v___y_1875_, v___y_1876_, v___y_1877_, v___y_1878_);
return v___x_1884_;
}
else
{
lean_object* v___x_1885_; 
v___x_1885_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_1861_, v_m_1874_, v___y_1875_, v___y_1876_, v___y_1877_, v___y_1878_);
return v___x_1885_;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13(void){
_start:
{
lean_object* v___x_2038_; lean_object* v___x_2039_; 
v___x_2038_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__12));
v___x_2039_ = l_Lean_stringToMessageData(v___x_2038_);
return v___x_2039_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2(void){
_start:
{
lean_object* v___x_2041_; lean_object* v___x_2042_; 
v___x_2041_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__1));
v___x_2042_ = l_Lean_stringToMessageData(v___x_2041_);
return v___x_2042_;
}
}
static double _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__3(void){
_start:
{
lean_object* v___x_2043_; double v___x_2044_; 
v___x_2043_ = lean_unsigned_to_nat(1000000000u);
v___x_2044_ = lean_float_of_nat(v___x_2043_);
return v___x_2044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15(lean_object* v_val_2045_, lean_object* v_trace_2046_, uint8_t v___x_2047_, uint8_t v___x_2048_, lean_object* v_expectedType_2049_, lean_object* v_inst_2050_, lean_object* v_x_2051_, lean_object* v_x_2052_, lean_object* v_x_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_){
_start:
{
lean_object* v_m_2060_; lean_object* v___y_2061_; lean_object* v___y_2062_; lean_object* v___y_2063_; lean_object* v___y_2064_; 
if (lean_obj_tag(v_x_2051_) == 5)
{
lean_object* v_fn_2072_; lean_object* v_arg_2073_; lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; 
v_fn_2072_ = lean_ctor_get(v_x_2051_, 0);
lean_inc_ref(v_fn_2072_);
v_arg_2073_ = lean_ctor_get(v_x_2051_, 1);
lean_inc_ref(v_arg_2073_);
lean_dec_ref_known(v_x_2051_, 2);
v___x_2074_ = lean_array_set(v_x_2052_, v_x_2053_, v_arg_2073_);
v___x_2075_ = lean_unsigned_to_nat(1u);
v___x_2076_ = lean_nat_sub(v_x_2053_, v___x_2075_);
lean_dec(v_x_2053_);
v_x_2051_ = v_fn_2072_;
v_x_2052_ = v___x_2074_;
v_x_2053_ = v___x_2076_;
goto _start;
}
else
{
lean_dec(v_x_2053_);
if (lean_obj_tag(v_x_2051_) == 4)
{
lean_object* v_declName_2078_; lean_object* v___x_2079_; 
v_declName_2078_ = lean_ctor_get(v_x_2051_, 0);
lean_inc(v_declName_2078_);
v___x_2079_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2(v_declName_2078_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_);
if (lean_obj_tag(v___x_2079_) == 0)
{
lean_object* v_a_2080_; 
v_a_2080_ = lean_ctor_get(v___x_2079_, 0);
lean_inc(v_a_2080_);
lean_dec_ref_known(v___x_2079_, 1);
if (lean_obj_tag(v_a_2080_) == 6)
{
lean_object* v_val_2081_; lean_object* v___x_2082_; 
lean_dec_ref(v_inst_2050_);
v_val_2081_ = lean_ctor_get(v_a_2080_, 0);
lean_inc_ref(v_val_2081_);
lean_dec_ref_known(v_a_2080_, 1);
lean_inc(v___y_2057_);
lean_inc_ref(v___y_2056_);
lean_inc(v___y_2055_);
lean_inc_ref(v___y_2054_);
lean_inc_ref(v_x_2051_);
v___x_2082_ = lean_infer_type(v_x_2051_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_);
if (lean_obj_tag(v___x_2082_) == 0)
{
lean_object* v_a_2083_; uint8_t v___x_2084_; lean_object* v___x_2085_; 
v_a_2083_ = lean_ctor_get(v___x_2082_, 0);
lean_inc(v_a_2083_);
lean_dec_ref_known(v___x_2082_, 1);
v___x_2084_ = 0;
v___x_2085_ = l_Lean_Meta_forallMetaTelescope(v_a_2083_, v___x_2084_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_);
if (lean_obj_tag(v___x_2085_) == 0)
{
lean_object* v_a_2086_; lean_object* v_snd_2087_; lean_object* v_fst_2088_; lean_object* v___x_2090_; uint8_t v_isShared_2091_; uint8_t v_isSharedCheck_2194_; 
v_a_2086_ = lean_ctor_get(v___x_2085_, 0);
lean_inc(v_a_2086_);
lean_dec_ref_known(v___x_2085_, 1);
v_snd_2087_ = lean_ctor_get(v_a_2086_, 1);
v_fst_2088_ = lean_ctor_get(v_a_2086_, 0);
v_isSharedCheck_2194_ = !lean_is_exclusive(v_a_2086_);
if (v_isSharedCheck_2194_ == 0)
{
v___x_2090_ = v_a_2086_;
v_isShared_2091_ = v_isSharedCheck_2194_;
goto v_resetjp_2089_;
}
else
{
lean_inc(v_snd_2087_);
lean_inc(v_fst_2088_);
lean_dec(v_a_2086_);
v___x_2090_ = lean_box(0);
v_isShared_2091_ = v_isSharedCheck_2194_;
goto v_resetjp_2089_;
}
v_resetjp_2089_:
{
lean_object* v_fst_2092_; lean_object* v_snd_2093_; lean_object* v___x_2095_; uint8_t v_isShared_2096_; uint8_t v_isSharedCheck_2193_; 
v_fst_2092_ = lean_ctor_get(v_snd_2087_, 0);
v_snd_2093_ = lean_ctor_get(v_snd_2087_, 1);
v_isSharedCheck_2193_ = !lean_is_exclusive(v_snd_2087_);
if (v_isSharedCheck_2193_ == 0)
{
v___x_2095_ = v_snd_2087_;
v_isShared_2096_ = v_isSharedCheck_2193_;
goto v_resetjp_2094_;
}
else
{
lean_inc(v_snd_2093_);
lean_inc(v_fst_2092_);
lean_dec(v_snd_2087_);
v___x_2095_ = lean_box(0);
v_isShared_2096_ = v_isSharedCheck_2193_;
goto v_resetjp_2094_;
}
v_resetjp_2094_:
{
lean_object* v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2135_; lean_object* v___y_2136_; lean_object* v___y_2137_; lean_object* v___y_2138_; lean_object* v___x_2171_; lean_object* v___x_2172_; uint8_t v___x_2173_; 
v___x_2171_ = lean_array_get_size(v_x_2052_);
v___x_2172_ = lean_array_get_size(v_fst_2088_);
v___x_2173_ = lean_nat_dec_eq(v___x_2171_, v___x_2172_);
if (v___x_2173_ == 0)
{
lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v_a_2185_; lean_object* v___x_2187_; uint8_t v_isShared_2188_; uint8_t v_isSharedCheck_2192_; 
lean_del_object(v___x_2095_);
lean_dec(v_snd_2093_);
lean_dec(v_fst_2092_);
lean_del_object(v___x_2090_);
lean_dec(v_fst_2088_);
lean_dec_ref(v_val_2081_);
lean_dec_ref(v_expectedType_2049_);
lean_dec_ref(v_trace_2046_);
lean_dec(v_val_2045_);
v___x_2174_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5);
v___x_2175_ = l_Lean_MessageData_ofExpr(v_x_2051_);
v___x_2176_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2176_, 0, v___x_2174_);
lean_ctor_set(v___x_2176_, 1, v___x_2175_);
v___x_2177_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7);
v___x_2178_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2178_, 0, v___x_2176_);
lean_ctor_set(v___x_2178_, 1, v___x_2177_);
v___x_2179_ = lean_array_to_list(v_x_2052_);
v___x_2180_ = lean_box(0);
v___x_2181_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__8(v___x_2179_, v___x_2180_);
v___x_2182_ = l_Lean_MessageData_ofList(v___x_2181_);
v___x_2183_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2183_, 0, v___x_2178_);
lean_ctor_set(v___x_2183_, 1, v___x_2182_);
v___x_2184_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_2183_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_);
v_a_2185_ = lean_ctor_get(v___x_2184_, 0);
v_isSharedCheck_2192_ = !lean_is_exclusive(v___x_2184_);
if (v_isSharedCheck_2192_ == 0)
{
v___x_2187_ = v___x_2184_;
v_isShared_2188_ = v_isSharedCheck_2192_;
goto v_resetjp_2186_;
}
else
{
lean_inc(v_a_2185_);
lean_dec(v___x_2184_);
v___x_2187_ = lean_box(0);
v_isShared_2188_ = v_isSharedCheck_2192_;
goto v_resetjp_2186_;
}
v_resetjp_2186_:
{
lean_object* v___x_2190_; 
if (v_isShared_2188_ == 0)
{
v___x_2190_ = v___x_2187_;
goto v_reusejp_2189_;
}
else
{
lean_object* v_reuseFailAlloc_2191_; 
v_reuseFailAlloc_2191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2191_, 0, v_a_2185_);
v___x_2190_ = v_reuseFailAlloc_2191_;
goto v_reusejp_2189_;
}
v_reusejp_2189_:
{
return v___x_2190_;
}
}
}
else
{
v___y_2135_ = v___y_2054_;
v___y_2136_ = v___y_2055_;
v___y_2137_ = v___y_2056_;
v___y_2138_ = v___y_2057_;
goto v___jp_2134_;
}
v___jp_2097_:
{
lean_object* v_numParams_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; 
v_numParams_2102_ = lean_ctor_get(v_val_2081_, 3);
lean_inc(v_numParams_2102_);
lean_dec_ref(v_val_2081_);
v___x_2103_ = lean_array_get_size(v_x_2052_);
v___x_2104_ = lean_box(0);
v___x_2105_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg(v___x_2103_, v_fst_2088_, v_x_2052_, v___x_2047_, v___x_2048_, v_fst_2092_, v_val_2045_, v_trace_2046_, v_numParams_2102_, v___x_2104_, v___y_2098_, v___y_2099_, v___y_2100_, v___y_2101_);
lean_dec(v_fst_2092_);
lean_dec_ref(v_x_2052_);
if (lean_obj_tag(v___x_2105_) == 0)
{
size_t v_sz_2106_; size_t v___x_2107_; lean_object* v___x_2108_; 
lean_dec_ref_known(v___x_2105_, 1);
v_sz_2106_ = lean_array_size(v_fst_2088_);
v___x_2107_ = ((size_t)0ULL);
v___x_2108_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4(v_sz_2106_, v___x_2107_, v_fst_2088_, v___y_2098_, v___y_2099_, v___y_2100_, v___y_2101_);
if (lean_obj_tag(v___x_2108_) == 0)
{
lean_object* v_a_2109_; lean_object* v___x_2111_; uint8_t v_isShared_2112_; uint8_t v_isSharedCheck_2117_; 
v_a_2109_ = lean_ctor_get(v___x_2108_, 0);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2108_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2111_ = v___x_2108_;
v_isShared_2112_ = v_isSharedCheck_2117_;
goto v_resetjp_2110_;
}
else
{
lean_inc(v_a_2109_);
lean_dec(v___x_2108_);
v___x_2111_ = lean_box(0);
v_isShared_2112_ = v_isSharedCheck_2117_;
goto v_resetjp_2110_;
}
v_resetjp_2110_:
{
lean_object* v___x_2113_; lean_object* v___x_2115_; 
v___x_2113_ = l_Lean_mkAppN(v_x_2051_, v_a_2109_);
lean_dec(v_a_2109_);
if (v_isShared_2112_ == 0)
{
lean_ctor_set(v___x_2111_, 0, v___x_2113_);
v___x_2115_ = v___x_2111_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2116_; 
v_reuseFailAlloc_2116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2116_, 0, v___x_2113_);
v___x_2115_ = v_reuseFailAlloc_2116_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
return v___x_2115_;
}
}
}
else
{
lean_object* v_a_2118_; lean_object* v___x_2120_; uint8_t v_isShared_2121_; uint8_t v_isSharedCheck_2125_; 
lean_dec_ref_known(v_x_2051_, 2);
v_a_2118_ = lean_ctor_get(v___x_2108_, 0);
v_isSharedCheck_2125_ = !lean_is_exclusive(v___x_2108_);
if (v_isSharedCheck_2125_ == 0)
{
v___x_2120_ = v___x_2108_;
v_isShared_2121_ = v_isSharedCheck_2125_;
goto v_resetjp_2119_;
}
else
{
lean_inc(v_a_2118_);
lean_dec(v___x_2108_);
v___x_2120_ = lean_box(0);
v_isShared_2121_ = v_isSharedCheck_2125_;
goto v_resetjp_2119_;
}
v_resetjp_2119_:
{
lean_object* v___x_2123_; 
if (v_isShared_2121_ == 0)
{
v___x_2123_ = v___x_2120_;
goto v_reusejp_2122_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v_a_2118_);
v___x_2123_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2122_;
}
v_reusejp_2122_:
{
return v___x_2123_;
}
}
}
}
else
{
lean_object* v_a_2126_; lean_object* v___x_2128_; uint8_t v_isShared_2129_; uint8_t v_isSharedCheck_2133_; 
lean_dec(v_fst_2088_);
lean_dec_ref_known(v_x_2051_, 2);
v_a_2126_ = lean_ctor_get(v___x_2105_, 0);
v_isSharedCheck_2133_ = !lean_is_exclusive(v___x_2105_);
if (v_isSharedCheck_2133_ == 0)
{
v___x_2128_ = v___x_2105_;
v_isShared_2129_ = v_isSharedCheck_2133_;
goto v_resetjp_2127_;
}
else
{
lean_inc(v_a_2126_);
lean_dec(v___x_2105_);
v___x_2128_ = lean_box(0);
v_isShared_2129_ = v_isSharedCheck_2133_;
goto v_resetjp_2127_;
}
v_resetjp_2127_:
{
lean_object* v___x_2131_; 
if (v_isShared_2129_ == 0)
{
v___x_2131_ = v___x_2128_;
goto v_reusejp_2130_;
}
else
{
lean_object* v_reuseFailAlloc_2132_; 
v_reuseFailAlloc_2132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2132_, 0, v_a_2126_);
v___x_2131_ = v_reuseFailAlloc_2132_;
goto v_reusejp_2130_;
}
v_reusejp_2130_:
{
return v___x_2131_;
}
}
}
}
v___jp_2134_:
{
lean_object* v___x_2139_; 
lean_inc_ref(v_expectedType_2049_);
v___x_2139_ = l_Lean_Meta_isExprDefEq(v_expectedType_2049_, v_snd_2093_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_);
if (lean_obj_tag(v___x_2139_) == 0)
{
lean_object* v_a_2140_; uint8_t v___x_2141_; 
v_a_2140_ = lean_ctor_get(v___x_2139_, 0);
lean_inc(v_a_2140_);
lean_dec_ref_known(v___x_2139_, 1);
v___x_2141_ = lean_unbox(v_a_2140_);
lean_dec(v_a_2140_);
if (v___x_2141_ == 0)
{
lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2145_; 
lean_inc(v_declName_2078_);
lean_dec(v_fst_2092_);
lean_dec(v_fst_2088_);
lean_dec_ref(v_val_2081_);
lean_dec_ref_known(v_x_2051_, 2);
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_trace_2046_);
lean_dec(v_val_2045_);
v___x_2142_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_2143_ = l_Lean_MessageData_ofExpr(v_expectedType_2049_);
if (v_isShared_2096_ == 0)
{
lean_ctor_set_tag(v___x_2095_, 7);
lean_ctor_set(v___x_2095_, 1, v___x_2143_);
lean_ctor_set(v___x_2095_, 0, v___x_2142_);
v___x_2145_ = v___x_2095_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2162_; 
v_reuseFailAlloc_2162_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2162_, 0, v___x_2142_);
lean_ctor_set(v_reuseFailAlloc_2162_, 1, v___x_2143_);
v___x_2145_ = v_reuseFailAlloc_2162_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
lean_object* v___x_2146_; lean_object* v___x_2148_; 
v___x_2146_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3);
if (v_isShared_2091_ == 0)
{
lean_ctor_set_tag(v___x_2090_, 7);
lean_ctor_set(v___x_2090_, 1, v___x_2146_);
lean_ctor_set(v___x_2090_, 0, v___x_2145_);
v___x_2148_ = v___x_2090_;
goto v_reusejp_2147_;
}
else
{
lean_object* v_reuseFailAlloc_2161_; 
v_reuseFailAlloc_2161_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2161_, 0, v___x_2145_);
lean_ctor_set(v_reuseFailAlloc_2161_, 1, v___x_2146_);
v___x_2148_ = v_reuseFailAlloc_2161_;
goto v_reusejp_2147_;
}
v_reusejp_2147_:
{
lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v_a_2153_; lean_object* v___x_2155_; uint8_t v_isShared_2156_; uint8_t v_isSharedCheck_2160_; 
v___x_2149_ = l_Lean_MessageData_ofConstName(v_declName_2078_, v___x_2047_);
v___x_2150_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2150_, 0, v___x_2148_);
lean_ctor_set(v___x_2150_, 1, v___x_2149_);
v___x_2151_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2151_, 0, v___x_2150_);
lean_ctor_set(v___x_2151_, 1, v___x_2142_);
v___x_2152_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_2151_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_);
v_a_2153_ = lean_ctor_get(v___x_2152_, 0);
v_isSharedCheck_2160_ = !lean_is_exclusive(v___x_2152_);
if (v_isSharedCheck_2160_ == 0)
{
v___x_2155_ = v___x_2152_;
v_isShared_2156_ = v_isSharedCheck_2160_;
goto v_resetjp_2154_;
}
else
{
lean_inc(v_a_2153_);
lean_dec(v___x_2152_);
v___x_2155_ = lean_box(0);
v_isShared_2156_ = v_isSharedCheck_2160_;
goto v_resetjp_2154_;
}
v_resetjp_2154_:
{
lean_object* v___x_2158_; 
if (v_isShared_2156_ == 0)
{
v___x_2158_ = v___x_2155_;
goto v_reusejp_2157_;
}
else
{
lean_object* v_reuseFailAlloc_2159_; 
v_reuseFailAlloc_2159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2159_, 0, v_a_2153_);
v___x_2158_ = v_reuseFailAlloc_2159_;
goto v_reusejp_2157_;
}
v_reusejp_2157_:
{
return v___x_2158_;
}
}
}
}
}
else
{
lean_del_object(v___x_2095_);
lean_del_object(v___x_2090_);
lean_dec_ref(v_expectedType_2049_);
v___y_2098_ = v___y_2135_;
v___y_2099_ = v___y_2136_;
v___y_2100_ = v___y_2137_;
v___y_2101_ = v___y_2138_;
goto v___jp_2097_;
}
}
else
{
lean_object* v_a_2163_; lean_object* v___x_2165_; uint8_t v_isShared_2166_; uint8_t v_isSharedCheck_2170_; 
lean_del_object(v___x_2095_);
lean_dec(v_fst_2092_);
lean_del_object(v___x_2090_);
lean_dec(v_fst_2088_);
lean_dec_ref(v_val_2081_);
lean_dec_ref_known(v_x_2051_, 2);
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_expectedType_2049_);
lean_dec_ref(v_trace_2046_);
lean_dec(v_val_2045_);
v_a_2163_ = lean_ctor_get(v___x_2139_, 0);
v_isSharedCheck_2170_ = !lean_is_exclusive(v___x_2139_);
if (v_isSharedCheck_2170_ == 0)
{
v___x_2165_ = v___x_2139_;
v_isShared_2166_ = v_isSharedCheck_2170_;
goto v_resetjp_2164_;
}
else
{
lean_inc(v_a_2163_);
lean_dec(v___x_2139_);
v___x_2165_ = lean_box(0);
v_isShared_2166_ = v_isSharedCheck_2170_;
goto v_resetjp_2164_;
}
v_resetjp_2164_:
{
lean_object* v___x_2168_; 
if (v_isShared_2166_ == 0)
{
v___x_2168_ = v___x_2165_;
goto v_reusejp_2167_;
}
else
{
lean_object* v_reuseFailAlloc_2169_; 
v_reuseFailAlloc_2169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2169_, 0, v_a_2163_);
v___x_2168_ = v_reuseFailAlloc_2169_;
goto v_reusejp_2167_;
}
v_reusejp_2167_:
{
return v___x_2168_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2195_; lean_object* v___x_2197_; uint8_t v_isShared_2198_; uint8_t v_isSharedCheck_2202_; 
lean_dec_ref(v_val_2081_);
lean_dec_ref_known(v_x_2051_, 2);
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_expectedType_2049_);
lean_dec_ref(v_trace_2046_);
lean_dec(v_val_2045_);
v_a_2195_ = lean_ctor_get(v___x_2085_, 0);
v_isSharedCheck_2202_ = !lean_is_exclusive(v___x_2085_);
if (v_isSharedCheck_2202_ == 0)
{
v___x_2197_ = v___x_2085_;
v_isShared_2198_ = v_isSharedCheck_2202_;
goto v_resetjp_2196_;
}
else
{
lean_inc(v_a_2195_);
lean_dec(v___x_2085_);
v___x_2197_ = lean_box(0);
v_isShared_2198_ = v_isSharedCheck_2202_;
goto v_resetjp_2196_;
}
v_resetjp_2196_:
{
lean_object* v___x_2200_; 
if (v_isShared_2198_ == 0)
{
v___x_2200_ = v___x_2197_;
goto v_reusejp_2199_;
}
else
{
lean_object* v_reuseFailAlloc_2201_; 
v_reuseFailAlloc_2201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2201_, 0, v_a_2195_);
v___x_2200_ = v_reuseFailAlloc_2201_;
goto v_reusejp_2199_;
}
v_reusejp_2199_:
{
return v___x_2200_;
}
}
}
}
else
{
lean_dec_ref(v_val_2081_);
lean_dec_ref_known(v_x_2051_, 2);
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_expectedType_2049_);
lean_dec_ref(v_trace_2046_);
lean_dec(v_val_2045_);
return v___x_2082_;
}
}
else
{
lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; 
lean_inc(v_declName_2078_);
lean_dec(v_a_2080_);
lean_dec_ref_known(v_x_2051_, 2);
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_expectedType_2049_);
v___x_2203_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9);
v___x_2204_ = l_Lean_indentExpr(v_inst_2050_);
v___x_2205_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2205_, 0, v___x_2203_);
lean_ctor_set(v___x_2205_, 1, v___x_2204_);
v___x_2206_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11);
v___x_2207_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2207_, 0, v___x_2205_);
lean_ctor_set(v___x_2207_, 1, v___x_2206_);
v___x_2208_ = l_Lean_MessageData_ofName(v_declName_2078_);
v___x_2209_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2209_, 0, v___x_2207_);
lean_ctor_set(v___x_2209_, 1, v___x_2208_);
v___x_2210_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13);
v___x_2211_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2211_, 0, v___x_2209_);
lean_ctor_set(v___x_2211_, 1, v___x_2210_);
v_m_2060_ = v___x_2211_;
v___y_2061_ = v___y_2054_;
v___y_2062_ = v___y_2055_;
v___y_2063_ = v___y_2056_;
v___y_2064_ = v___y_2057_;
goto v___jp_2059_;
}
}
else
{
lean_object* v_a_2212_; lean_object* v___x_2214_; uint8_t v_isShared_2215_; uint8_t v_isSharedCheck_2219_; 
lean_dec_ref_known(v_x_2051_, 2);
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_inst_2050_);
lean_dec_ref(v_expectedType_2049_);
lean_dec_ref(v_trace_2046_);
lean_dec(v_val_2045_);
v_a_2212_ = lean_ctor_get(v___x_2079_, 0);
v_isSharedCheck_2219_ = !lean_is_exclusive(v___x_2079_);
if (v_isSharedCheck_2219_ == 0)
{
v___x_2214_ = v___x_2079_;
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
else
{
lean_inc(v_a_2212_);
lean_dec(v___x_2079_);
v___x_2214_ = lean_box(0);
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
v_resetjp_2213_:
{
lean_object* v___x_2217_; 
if (v_isShared_2215_ == 0)
{
v___x_2217_ = v___x_2214_;
goto v_reusejp_2216_;
}
else
{
lean_object* v_reuseFailAlloc_2218_; 
v_reuseFailAlloc_2218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2218_, 0, v_a_2212_);
v___x_2217_ = v_reuseFailAlloc_2218_;
goto v_reusejp_2216_;
}
v_reusejp_2216_:
{
return v___x_2217_;
}
}
}
}
else
{
lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; 
lean_dec_ref(v_x_2052_);
lean_dec_ref(v_x_2051_);
lean_dec_ref(v_expectedType_2049_);
v___x_2220_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9);
v___x_2221_ = l_Lean_indentExpr(v_inst_2050_);
v___x_2222_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2222_, 0, v___x_2220_);
lean_ctor_set(v___x_2222_, 1, v___x_2221_);
v_m_2060_ = v___x_2222_;
v___y_2061_ = v___y_2054_;
v___y_2062_ = v___y_2055_;
v___y_2063_ = v___y_2056_;
v___y_2064_ = v___y_2057_;
goto v___jp_2059_;
}
}
v___jp_2059_:
{
lean_object* v___x_2065_; lean_object* v_env_2066_; uint8_t v___x_2067_; 
v___x_2065_ = lean_st_ref_get(v___y_2064_);
v_env_2066_ = lean_ctor_get(v___x_2065_, 0);
lean_inc_ref(v_env_2066_);
lean_dec(v___x_2065_);
v___x_2067_ = l_Lean_isStructure(v_env_2066_, v_val_2045_);
if (v___x_2067_ == 0)
{
lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; 
v___x_2068_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1);
v___x_2069_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2069_, 0, v_m_2060_);
lean_ctor_set(v___x_2069_, 1, v___x_2068_);
v___x_2070_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2046_, v___x_2069_, v___y_2061_, v___y_2062_, v___y_2063_, v___y_2064_);
return v___x_2070_;
}
else
{
lean_object* v___x_2071_; 
v___x_2071_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2046_, v_m_2060_, v___y_2061_, v___y_2062_, v___y_2063_, v___y_2064_);
return v___x_2071_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1(lean_object* v_expectedType_2223_, lean_object* v_inst_2224_, lean_object* v_trace_2225_, lean_object* v_cls_2226_, uint8_t v_root_2227_, lean_object* v_val_2228_, uint8_t v___x_2229_, uint8_t v_hasTrace_2230_, lean_object* v_____r_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_){
_start:
{
lean_object* v___x_2237_; 
lean_inc_ref(v_expectedType_2223_);
v___x_2237_ = l_Lean_Meta_isProp(v_expectedType_2223_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
if (lean_obj_tag(v___x_2237_) == 0)
{
lean_object* v_a_2238_; uint8_t v___x_2239_; 
v_a_2238_ = lean_ctor_get(v___x_2237_, 0);
lean_inc(v_a_2238_);
lean_dec_ref_known(v___x_2237_, 1);
v___x_2239_ = lean_unbox(v_a_2238_);
lean_dec(v_a_2238_);
if (v___x_2239_ == 0)
{
lean_object* v___x_2240_; lean_object* v___x_2241_; 
v___x_2240_ = lean_box(0);
lean_inc_ref(v_expectedType_2223_);
v___x_2241_ = l_Lean_Meta_trySynthInstance(v_expectedType_2223_, v___x_2240_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
if (lean_obj_tag(v___x_2241_) == 0)
{
lean_object* v_a_2242_; 
v_a_2242_ = lean_ctor_get(v___x_2241_, 0);
lean_inc(v_a_2242_);
lean_dec_ref_known(v___x_2241_, 1);
if (lean_obj_tag(v_a_2242_) == 1)
{
lean_object* v_a_2243_; lean_object* v___y_2245_; lean_object* v___y_2246_; lean_object* v___y_2247_; lean_object* v___y_2248_; 
lean_dec(v_val_2228_);
v_a_2243_ = lean_ctor_get(v_a_2242_, 0);
lean_inc(v_a_2243_);
lean_dec_ref_known(v_a_2242_, 1);
if (v_root_2227_ == 0)
{
lean_dec_ref(v_expectedType_2223_);
v___y_2245_ = v___y_2232_;
v___y_2246_ = v___y_2233_;
v___y_2247_ = v___y_2234_;
v___y_2248_ = v___y_2235_;
goto v___jp_2244_;
}
else
{
lean_object* v_ref_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; lean_object* v___x_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; 
v_ref_2316_ = lean_ctor_get(v___y_2234_, 5);
v___x_2317_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing;
v___x_2318_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8);
v___x_2319_ = l_Lean_MessageData_ofExpr(v_expectedType_2223_);
v___x_2320_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2320_, 0, v___x_2318_);
lean_ctor_set(v___x_2320_, 1, v___x_2319_);
v___x_2321_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10);
v___x_2322_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2322_, 0, v___x_2320_);
lean_ctor_set(v___x_2322_, 1, v___x_2321_);
lean_inc(v_ref_2316_);
v___x_2323_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(v___x_2317_, v_ref_2316_, v___x_2322_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
if (lean_obj_tag(v___x_2323_) == 0)
{
lean_dec_ref_known(v___x_2323_, 1);
v___y_2245_ = v___y_2232_;
v___y_2246_ = v___y_2233_;
v___y_2247_ = v___y_2234_;
v___y_2248_ = v___y_2235_;
goto v___jp_2244_;
}
else
{
lean_object* v_a_2324_; lean_object* v___x_2326_; uint8_t v_isShared_2327_; uint8_t v_isSharedCheck_2331_; 
lean_dec(v_a_2243_);
lean_dec(v_cls_2226_);
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_inst_2224_);
v_a_2324_ = lean_ctor_get(v___x_2323_, 0);
v_isSharedCheck_2331_ = !lean_is_exclusive(v___x_2323_);
if (v_isSharedCheck_2331_ == 0)
{
v___x_2326_ = v___x_2323_;
v_isShared_2327_ = v_isSharedCheck_2331_;
goto v_resetjp_2325_;
}
else
{
lean_inc(v_a_2324_);
lean_dec(v___x_2323_);
v___x_2326_ = lean_box(0);
v_isShared_2327_ = v_isSharedCheck_2331_;
goto v_resetjp_2325_;
}
v_resetjp_2325_:
{
lean_object* v___x_2329_; 
if (v_isShared_2327_ == 0)
{
v___x_2329_ = v___x_2326_;
goto v_reusejp_2328_;
}
else
{
lean_object* v_reuseFailAlloc_2330_; 
v_reuseFailAlloc_2330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2330_, 0, v_a_2324_);
v___x_2329_ = v_reuseFailAlloc_2330_;
goto v_reusejp_2328_;
}
v_reusejp_2328_:
{
return v___x_2329_;
}
}
}
}
v___jp_2244_:
{
lean_object* v_keyedConfig_2249_; uint8_t v_trackZetaDelta_2250_; lean_object* v_zetaDeltaSet_2251_; lean_object* v_lctx_2252_; lean_object* v_localInstances_2253_; lean_object* v_defEqCtx_x3f_2254_; lean_object* v_synthPendingDepth_2255_; lean_object* v_customCanUnfoldPredicate_x3f_2256_; uint8_t v_univApprox_2257_; uint8_t v_inTypeClassResolution_2258_; uint8_t v_cacheInferType_2259_; uint8_t v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; 
v_keyedConfig_2249_ = lean_ctor_get(v___y_2245_, 0);
v_trackZetaDelta_2250_ = lean_ctor_get_uint8(v___y_2245_, sizeof(void*)*7);
v_zetaDeltaSet_2251_ = lean_ctor_get(v___y_2245_, 1);
v_lctx_2252_ = lean_ctor_get(v___y_2245_, 2);
v_localInstances_2253_ = lean_ctor_get(v___y_2245_, 3);
v_defEqCtx_x3f_2254_ = lean_ctor_get(v___y_2245_, 4);
v_synthPendingDepth_2255_ = lean_ctor_get(v___y_2245_, 5);
v_customCanUnfoldPredicate_x3f_2256_ = lean_ctor_get(v___y_2245_, 6);
v_univApprox_2257_ = lean_ctor_get_uint8(v___y_2245_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2258_ = lean_ctor_get_uint8(v___y_2245_, sizeof(void*)*7 + 2);
v_cacheInferType_2259_ = lean_ctor_get_uint8(v___y_2245_, sizeof(void*)*7 + 3);
v___x_2260_ = 1;
lean_inc_ref(v_keyedConfig_2249_);
v___x_2261_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2260_, v_keyedConfig_2249_);
lean_inc(v_customCanUnfoldPredicate_x3f_2256_);
lean_inc(v_synthPendingDepth_2255_);
lean_inc(v_defEqCtx_x3f_2254_);
lean_inc_ref(v_localInstances_2253_);
lean_inc_ref(v_lctx_2252_);
lean_inc(v_zetaDeltaSet_2251_);
v___x_2262_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2262_, 0, v___x_2261_);
lean_ctor_set(v___x_2262_, 1, v_zetaDeltaSet_2251_);
lean_ctor_set(v___x_2262_, 2, v_lctx_2252_);
lean_ctor_set(v___x_2262_, 3, v_localInstances_2253_);
lean_ctor_set(v___x_2262_, 4, v_defEqCtx_x3f_2254_);
lean_ctor_set(v___x_2262_, 5, v_synthPendingDepth_2255_);
lean_ctor_set(v___x_2262_, 6, v_customCanUnfoldPredicate_x3f_2256_);
lean_ctor_set_uint8(v___x_2262_, sizeof(void*)*7, v_trackZetaDelta_2250_);
lean_ctor_set_uint8(v___x_2262_, sizeof(void*)*7 + 1, v_univApprox_2257_);
lean_ctor_set_uint8(v___x_2262_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2258_);
lean_ctor_set_uint8(v___x_2262_, sizeof(void*)*7 + 3, v_cacheInferType_2259_);
lean_inc(v_a_2243_);
lean_inc_ref(v_inst_2224_);
v___x_2263_ = l_Lean_Meta_isExprDefEq(v_inst_2224_, v_a_2243_, v___x_2262_, v___y_2246_, v___y_2247_, v___y_2248_);
lean_dec_ref_known(v___x_2262_, 7);
if (lean_obj_tag(v___x_2263_) == 0)
{
lean_object* v_a_2264_; lean_object* v___x_2266_; uint8_t v_isShared_2267_; uint8_t v_isSharedCheck_2307_; 
v_a_2264_ = lean_ctor_get(v___x_2263_, 0);
v_isSharedCheck_2307_ = !lean_is_exclusive(v___x_2263_);
if (v_isSharedCheck_2307_ == 0)
{
v___x_2266_ = v___x_2263_;
v_isShared_2267_ = v_isSharedCheck_2307_;
goto v_resetjp_2265_;
}
else
{
lean_inc(v_a_2264_);
lean_dec(v___x_2263_);
v___x_2266_ = lean_box(0);
v_isShared_2267_ = v_isSharedCheck_2307_;
goto v_resetjp_2265_;
}
v_resetjp_2265_:
{
uint8_t v___x_2268_; 
v___x_2268_ = lean_unbox(v_a_2264_);
lean_dec(v_a_2264_);
if (v___x_2268_ == 0)
{
lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; 
lean_del_object(v___x_2266_);
lean_dec(v_cls_2226_);
v___x_2269_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
v___x_2270_ = l_Lean_indentExpr(v_inst_2224_);
v___x_2271_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2271_, 0, v___x_2269_);
lean_ctor_set(v___x_2271_, 1, v___x_2270_);
v___x_2272_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3);
v___x_2273_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2273_, 0, v___x_2271_);
lean_ctor_set(v___x_2273_, 1, v___x_2272_);
v___x_2274_ = l_Lean_indentExpr(v_a_2243_);
v___x_2275_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2273_);
lean_ctor_set(v___x_2275_, 1, v___x_2274_);
v___x_2276_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2225_, v___x_2275_, v___y_2245_, v___y_2246_, v___y_2247_, v___y_2248_);
return v___x_2276_;
}
else
{
lean_object* v_options_2277_; uint8_t v_hasTrace_2278_; 
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_inst_2224_);
v_options_2277_ = lean_ctor_get(v___y_2247_, 2);
v_hasTrace_2278_ = lean_ctor_get_uint8(v_options_2277_, sizeof(void*)*1);
if (v_hasTrace_2278_ == 0)
{
lean_object* v___x_2280_; 
lean_dec(v_cls_2226_);
if (v_isShared_2267_ == 0)
{
lean_ctor_set(v___x_2266_, 0, v_a_2243_);
v___x_2280_ = v___x_2266_;
goto v_reusejp_2279_;
}
else
{
lean_object* v_reuseFailAlloc_2281_; 
v_reuseFailAlloc_2281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2281_, 0, v_a_2243_);
v___x_2280_ = v_reuseFailAlloc_2281_;
goto v_reusejp_2279_;
}
v_reusejp_2279_:
{
return v___x_2280_;
}
}
else
{
lean_object* v_inheritedTraceOptions_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; uint8_t v___x_2285_; 
v_inheritedTraceOptions_2282_ = lean_ctor_get(v___y_2247_, 13);
v___x_2283_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__4));
lean_inc(v_cls_2226_);
v___x_2284_ = l_Lean_Name_append(v___x_2283_, v_cls_2226_);
v___x_2285_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2282_, v_options_2277_, v___x_2284_);
lean_dec(v___x_2284_);
if (v___x_2285_ == 0)
{
lean_object* v___x_2287_; 
lean_dec(v_cls_2226_);
if (v_isShared_2267_ == 0)
{
lean_ctor_set(v___x_2266_, 0, v_a_2243_);
v___x_2287_ = v___x_2266_;
goto v_reusejp_2286_;
}
else
{
lean_object* v_reuseFailAlloc_2288_; 
v_reuseFailAlloc_2288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2288_, 0, v_a_2243_);
v___x_2287_ = v_reuseFailAlloc_2288_;
goto v_reusejp_2286_;
}
v_reusejp_2286_:
{
return v___x_2287_;
}
}
else
{
lean_object* v___x_2289_; lean_object* v___x_2290_; 
lean_del_object(v___x_2266_);
v___x_2289_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6);
v___x_2290_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2226_, v___x_2289_, v___y_2245_, v___y_2246_, v___y_2247_, v___y_2248_);
if (lean_obj_tag(v___x_2290_) == 0)
{
lean_object* v___x_2292_; uint8_t v_isShared_2293_; uint8_t v_isSharedCheck_2297_; 
v_isSharedCheck_2297_ = !lean_is_exclusive(v___x_2290_);
if (v_isSharedCheck_2297_ == 0)
{
lean_object* v_unused_2298_; 
v_unused_2298_ = lean_ctor_get(v___x_2290_, 0);
lean_dec(v_unused_2298_);
v___x_2292_ = v___x_2290_;
v_isShared_2293_ = v_isSharedCheck_2297_;
goto v_resetjp_2291_;
}
else
{
lean_dec(v___x_2290_);
v___x_2292_ = lean_box(0);
v_isShared_2293_ = v_isSharedCheck_2297_;
goto v_resetjp_2291_;
}
v_resetjp_2291_:
{
lean_object* v___x_2295_; 
if (v_isShared_2293_ == 0)
{
lean_ctor_set(v___x_2292_, 0, v_a_2243_);
v___x_2295_ = v___x_2292_;
goto v_reusejp_2294_;
}
else
{
lean_object* v_reuseFailAlloc_2296_; 
v_reuseFailAlloc_2296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2296_, 0, v_a_2243_);
v___x_2295_ = v_reuseFailAlloc_2296_;
goto v_reusejp_2294_;
}
v_reusejp_2294_:
{
return v___x_2295_;
}
}
}
else
{
lean_object* v_a_2299_; lean_object* v___x_2301_; uint8_t v_isShared_2302_; uint8_t v_isSharedCheck_2306_; 
lean_dec(v_a_2243_);
v_a_2299_ = lean_ctor_get(v___x_2290_, 0);
v_isSharedCheck_2306_ = !lean_is_exclusive(v___x_2290_);
if (v_isSharedCheck_2306_ == 0)
{
v___x_2301_ = v___x_2290_;
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
else
{
lean_inc(v_a_2299_);
lean_dec(v___x_2290_);
v___x_2301_ = lean_box(0);
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
v_resetjp_2300_:
{
lean_object* v___x_2304_; 
if (v_isShared_2302_ == 0)
{
v___x_2304_ = v___x_2301_;
goto v_reusejp_2303_;
}
else
{
lean_object* v_reuseFailAlloc_2305_; 
v_reuseFailAlloc_2305_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2305_, 0, v_a_2299_);
v___x_2304_ = v_reuseFailAlloc_2305_;
goto v_reusejp_2303_;
}
v_reusejp_2303_:
{
return v___x_2304_;
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
lean_object* v_a_2308_; lean_object* v___x_2310_; uint8_t v_isShared_2311_; uint8_t v_isSharedCheck_2315_; 
lean_dec(v_a_2243_);
lean_dec(v_cls_2226_);
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_inst_2224_);
v_a_2308_ = lean_ctor_get(v___x_2263_, 0);
v_isSharedCheck_2315_ = !lean_is_exclusive(v___x_2263_);
if (v_isSharedCheck_2315_ == 0)
{
v___x_2310_ = v___x_2263_;
v_isShared_2311_ = v_isSharedCheck_2315_;
goto v_resetjp_2309_;
}
else
{
lean_inc(v_a_2308_);
lean_dec(v___x_2263_);
v___x_2310_ = lean_box(0);
v_isShared_2311_ = v_isSharedCheck_2315_;
goto v_resetjp_2309_;
}
v_resetjp_2309_:
{
lean_object* v___x_2313_; 
if (v_isShared_2311_ == 0)
{
v___x_2313_ = v___x_2310_;
goto v_reusejp_2312_;
}
else
{
lean_object* v_reuseFailAlloc_2314_; 
v_reuseFailAlloc_2314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2314_, 0, v_a_2308_);
v___x_2313_ = v_reuseFailAlloc_2314_;
goto v_reusejp_2312_;
}
v_reusejp_2312_:
{
return v___x_2313_;
}
}
}
}
}
else
{
lean_object* v___x_2332_; 
lean_dec(v_a_2242_);
lean_dec(v_cls_2226_);
lean_inc_ref(v_inst_2224_);
v___x_2332_ = l_Lean_Meta_whnfI(v_inst_2224_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
if (lean_obj_tag(v___x_2332_) == 0)
{
lean_object* v_a_2333_; lean_object* v_dummy_2334_; lean_object* v_nargs_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; 
v_a_2333_ = lean_ctor_get(v___x_2332_, 0);
lean_inc(v_a_2333_);
lean_dec_ref_known(v___x_2332_, 1);
v_dummy_2334_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11);
v_nargs_2335_ = l_Lean_Expr_getAppNumArgs(v_a_2333_);
lean_inc(v_nargs_2335_);
v___x_2336_ = lean_mk_array(v_nargs_2335_, v_dummy_2334_);
v___x_2337_ = lean_unsigned_to_nat(1u);
v___x_2338_ = lean_nat_sub(v_nargs_2335_, v___x_2337_);
lean_dec(v_nargs_2335_);
v___x_2339_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15(v_val_2228_, v_trace_2225_, v___x_2229_, v_hasTrace_2230_, v_expectedType_2223_, v_inst_2224_, v_a_2333_, v___x_2336_, v___x_2338_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
return v___x_2339_;
}
else
{
lean_dec(v_val_2228_);
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_inst_2224_);
lean_dec_ref(v_expectedType_2223_);
return v___x_2332_;
}
}
}
else
{
lean_object* v_a_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2347_; 
lean_dec(v_val_2228_);
lean_dec(v_cls_2226_);
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_inst_2224_);
lean_dec_ref(v_expectedType_2223_);
v_a_2340_ = lean_ctor_get(v___x_2241_, 0);
v_isSharedCheck_2347_ = !lean_is_exclusive(v___x_2241_);
if (v_isSharedCheck_2347_ == 0)
{
v___x_2342_ = v___x_2241_;
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_a_2340_);
lean_dec(v___x_2241_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2345_; 
if (v_isShared_2343_ == 0)
{
v___x_2345_ = v___x_2342_;
goto v_reusejp_2344_;
}
else
{
lean_object* v_reuseFailAlloc_2346_; 
v_reuseFailAlloc_2346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2346_, 0, v_a_2340_);
v___x_2345_ = v_reuseFailAlloc_2346_;
goto v_reusejp_2344_;
}
v_reusejp_2344_:
{
return v___x_2345_;
}
}
}
}
else
{
lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; 
lean_dec(v_val_2228_);
lean_dec(v_cls_2226_);
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_expectedType_2223_);
v___x_2348_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
lean_inc_ref(v_inst_2224_);
v___x_2349_ = l_Lean_indentExpr(v_inst_2224_);
v___x_2350_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2350_, 0, v___x_2348_);
lean_ctor_set(v___x_2350_, 1, v___x_2349_);
v___x_2351_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13);
v___x_2352_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2352_, 0, v___x_2350_);
lean_ctor_set(v___x_2352_, 1, v___x_2351_);
v___x_2353_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(v___x_2352_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_);
if (lean_obj_tag(v___x_2353_) == 0)
{
lean_object* v___x_2355_; uint8_t v_isShared_2356_; uint8_t v_isSharedCheck_2360_; 
v_isSharedCheck_2360_ = !lean_is_exclusive(v___x_2353_);
if (v_isSharedCheck_2360_ == 0)
{
lean_object* v_unused_2361_; 
v_unused_2361_ = lean_ctor_get(v___x_2353_, 0);
lean_dec(v_unused_2361_);
v___x_2355_ = v___x_2353_;
v_isShared_2356_ = v_isSharedCheck_2360_;
goto v_resetjp_2354_;
}
else
{
lean_dec(v___x_2353_);
v___x_2355_ = lean_box(0);
v_isShared_2356_ = v_isSharedCheck_2360_;
goto v_resetjp_2354_;
}
v_resetjp_2354_:
{
lean_object* v___x_2358_; 
if (v_isShared_2356_ == 0)
{
lean_ctor_set(v___x_2355_, 0, v_inst_2224_);
v___x_2358_ = v___x_2355_;
goto v_reusejp_2357_;
}
else
{
lean_object* v_reuseFailAlloc_2359_; 
v_reuseFailAlloc_2359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2359_, 0, v_inst_2224_);
v___x_2358_ = v_reuseFailAlloc_2359_;
goto v_reusejp_2357_;
}
v_reusejp_2357_:
{
return v___x_2358_;
}
}
}
else
{
lean_object* v_a_2362_; lean_object* v___x_2364_; uint8_t v_isShared_2365_; uint8_t v_isSharedCheck_2369_; 
lean_dec_ref(v_inst_2224_);
v_a_2362_ = lean_ctor_get(v___x_2353_, 0);
v_isSharedCheck_2369_ = !lean_is_exclusive(v___x_2353_);
if (v_isSharedCheck_2369_ == 0)
{
v___x_2364_ = v___x_2353_;
v_isShared_2365_ = v_isSharedCheck_2369_;
goto v_resetjp_2363_;
}
else
{
lean_inc(v_a_2362_);
lean_dec(v___x_2353_);
v___x_2364_ = lean_box(0);
v_isShared_2365_ = v_isSharedCheck_2369_;
goto v_resetjp_2363_;
}
v_resetjp_2363_:
{
lean_object* v___x_2367_; 
if (v_isShared_2365_ == 0)
{
v___x_2367_ = v___x_2364_;
goto v_reusejp_2366_;
}
else
{
lean_object* v_reuseFailAlloc_2368_; 
v_reuseFailAlloc_2368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2368_, 0, v_a_2362_);
v___x_2367_ = v_reuseFailAlloc_2368_;
goto v_reusejp_2366_;
}
v_reusejp_2366_:
{
return v___x_2367_;
}
}
}
}
}
else
{
lean_object* v_a_2370_; lean_object* v___x_2372_; uint8_t v_isShared_2373_; uint8_t v_isSharedCheck_2377_; 
lean_dec(v_val_2228_);
lean_dec(v_cls_2226_);
lean_dec_ref(v_trace_2225_);
lean_dec_ref(v_inst_2224_);
lean_dec_ref(v_expectedType_2223_);
v_a_2370_ = lean_ctor_get(v___x_2237_, 0);
v_isSharedCheck_2377_ = !lean_is_exclusive(v___x_2237_);
if (v_isSharedCheck_2377_ == 0)
{
v___x_2372_ = v___x_2237_;
v_isShared_2373_ = v_isSharedCheck_2377_;
goto v_resetjp_2371_;
}
else
{
lean_inc(v_a_2370_);
lean_dec(v___x_2237_);
v___x_2372_ = lean_box(0);
v_isShared_2373_ = v_isSharedCheck_2377_;
goto v_resetjp_2371_;
}
v_resetjp_2371_:
{
lean_object* v___x_2375_; 
if (v_isShared_2373_ == 0)
{
v___x_2375_ = v___x_2372_;
goto v_reusejp_2374_;
}
else
{
lean_object* v_reuseFailAlloc_2376_; 
v_reuseFailAlloc_2376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2376_, 0, v_a_2370_);
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
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5(void){
_start:
{
lean_object* v___x_2379_; lean_object* v___x_2380_; 
v___x_2379_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__4));
v___x_2380_ = l_Lean_stringToMessageData(v___x_2379_);
return v___x_2380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg(lean_object* v_upperBound_2381_, lean_object* v_fst_2382_, lean_object* v_args_2383_, lean_object* v_fst_2384_, uint8_t v___x_2385_, lean_object* v_val_2386_, lean_object* v_trace_2387_, lean_object* v_a_2388_, lean_object* v_b_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_, lean_object* v___y_2393_){
_start:
{
lean_object* v_a_2396_; uint8_t v___x_2400_; 
v___x_2400_ = lean_nat_dec_lt(v_a_2388_, v_upperBound_2381_);
if (v___x_2400_ == 0)
{
lean_object* v___x_2401_; 
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v___x_2401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2401_, 0, v_b_2389_);
return v___x_2401_;
}
else
{
lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; lean_object* v___x_2405_; 
v___x_2402_ = l_Lean_instInhabitedExpr;
v___x_2403_ = lean_array_get_borrowed(v___x_2402_, v_fst_2382_, v_a_2388_);
v___x_2404_ = l_Lean_Expr_mvarId_x21(v___x_2403_);
lean_inc(v___x_2404_);
v___x_2405_ = l_Lean_MVarId_getDecl(v___x_2404_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2405_) == 0)
{
lean_object* v_a_2406_; lean_object* v_userName_2407_; lean_object* v_type_2408_; lean_object* v___x_2409_; 
v_a_2406_ = lean_ctor_get(v___x_2405_, 0);
lean_inc(v_a_2406_);
lean_dec_ref_known(v___x_2405_, 1);
v_userName_2407_ = lean_ctor_get(v_a_2406_, 0);
lean_inc(v_userName_2407_);
v_type_2408_ = lean_ctor_get(v_a_2406_, 2);
lean_inc_ref(v_type_2408_);
lean_dec(v_a_2406_);
v___x_2409_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(v_type_2408_, v___y_2391_);
if (lean_obj_tag(v___x_2409_) == 0)
{
lean_object* v_a_2410_; lean_object* v___x_2411_; 
v_a_2410_ = lean_ctor_get(v___x_2409_, 0);
lean_inc_n(v_a_2410_, 2);
lean_dec_ref_known(v___x_2409_, 1);
v___x_2411_ = l_Lean_Meta_isProp(v_a_2410_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2411_) == 0)
{
lean_object* v_a_2412_; lean_object* v___x_2413_; lean_object* v___x_2414_; uint8_t v___x_2415_; 
v_a_2412_ = lean_ctor_get(v___x_2411_, 0);
lean_inc(v_a_2412_);
lean_dec_ref_known(v___x_2411_, 1);
v___x_2413_ = lean_box(0);
v___x_2414_ = lean_array_get_borrowed(v___x_2402_, v_args_2383_, v_a_2388_);
v___x_2415_ = lean_unbox(v_a_2412_);
if (v___x_2415_ == 0)
{
uint8_t v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; uint8_t v___x_2419_; uint8_t v___x_2420_; 
v___x_2416_ = 0;
v___x_2417_ = lean_box(v___x_2416_);
v___x_2418_ = lean_array_get(v___x_2417_, v_fst_2384_, v_a_2388_);
lean_dec(v___x_2417_);
v___x_2419_ = lean_unbox(v___x_2418_);
lean_dec(v___x_2418_);
v___x_2420_ = l_Lean_BinderInfo_isInstImplicit(v___x_2419_);
if (v___x_2420_ == 0)
{
lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___f_2423_; lean_object* v___x_2424_; 
lean_dec(v_a_2412_);
lean_dec(v_userName_2407_);
v___x_2421_ = lean_box(v___x_2420_);
v___x_2422_ = lean_box(v___x_2385_);
lean_inc(v___x_2414_);
v___f_2423_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_2423_, 0, v___x_2414_);
lean_closure_set(v___f_2423_, 1, v___x_2421_);
lean_closure_set(v___f_2423_, 2, v___x_2422_);
v___x_2424_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(v_a_2410_, v___f_2423_, v___x_2420_, v___x_2420_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2424_) == 0)
{
lean_object* v_a_2425_; lean_object* v___x_2426_; 
v_a_2425_ = lean_ctor_get(v___x_2424_, 0);
lean_inc(v_a_2425_);
lean_dec_ref_known(v___x_2424_, 1);
v___x_2426_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_2404_, v_a_2425_, v___y_2391_);
if (lean_obj_tag(v___x_2426_) == 0)
{
lean_dec_ref_known(v___x_2426_, 1);
v_a_2396_ = v___x_2413_;
goto v___jp_2395_;
}
else
{
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
return v___x_2426_;
}
}
else
{
lean_object* v_a_2427_; lean_object* v___x_2429_; uint8_t v_isShared_2430_; uint8_t v_isSharedCheck_2434_; 
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2427_ = lean_ctor_get(v___x_2424_, 0);
v_isSharedCheck_2434_ = !lean_is_exclusive(v___x_2424_);
if (v_isSharedCheck_2434_ == 0)
{
v___x_2429_ = v___x_2424_;
v_isShared_2430_ = v_isSharedCheck_2434_;
goto v_resetjp_2428_;
}
else
{
lean_inc(v_a_2427_);
lean_dec(v___x_2424_);
v___x_2429_ = lean_box(0);
v_isShared_2430_ = v_isSharedCheck_2434_;
goto v_resetjp_2428_;
}
v_resetjp_2428_:
{
lean_object* v___x_2432_; 
if (v_isShared_2430_ == 0)
{
v___x_2432_ = v___x_2429_;
goto v_reusejp_2431_;
}
else
{
lean_object* v_reuseFailAlloc_2433_; 
v_reuseFailAlloc_2433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2433_, 0, v_a_2427_);
v___x_2432_ = v_reuseFailAlloc_2433_;
goto v_reusejp_2431_;
}
v_reusejp_2431_:
{
return v___x_2432_;
}
}
}
}
else
{
lean_object* v___x_2435_; lean_object* v___x_2436_; uint8_t v___x_2437_; lean_object* v___x_2438_; 
lean_inc(v_val_2386_);
v___x_2435_ = l_Lean_Name_append(v_val_2386_, v_userName_2407_);
lean_inc_ref(v_trace_2387_);
v___x_2436_ = lean_array_push(v_trace_2387_, v___x_2435_);
v___x_2437_ = lean_unbox(v_a_2412_);
lean_dec(v_a_2412_);
lean_inc(v___x_2414_);
v___x_2438_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(v___x_2414_, v_a_2410_, v___x_2437_, v___x_2436_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2438_) == 0)
{
lean_object* v_a_2439_; lean_object* v___x_2440_; 
v_a_2439_ = lean_ctor_get(v___x_2438_, 0);
lean_inc(v_a_2439_);
lean_dec_ref_known(v___x_2438_, 1);
v___x_2440_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_2404_, v_a_2439_, v___y_2391_);
if (lean_obj_tag(v___x_2440_) == 0)
{
lean_dec_ref_known(v___x_2440_, 1);
v_a_2396_ = v___x_2413_;
goto v___jp_2395_;
}
else
{
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
return v___x_2440_;
}
}
else
{
lean_object* v_a_2441_; lean_object* v___x_2443_; uint8_t v_isShared_2444_; uint8_t v_isSharedCheck_2448_; 
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2441_ = lean_ctor_get(v___x_2438_, 0);
v_isSharedCheck_2448_ = !lean_is_exclusive(v___x_2438_);
if (v_isSharedCheck_2448_ == 0)
{
v___x_2443_ = v___x_2438_;
v_isShared_2444_ = v_isSharedCheck_2448_;
goto v_resetjp_2442_;
}
else
{
lean_inc(v_a_2441_);
lean_dec(v___x_2438_);
v___x_2443_ = lean_box(0);
v_isShared_2444_ = v_isSharedCheck_2448_;
goto v_resetjp_2442_;
}
v_resetjp_2442_:
{
lean_object* v___x_2446_; 
if (v_isShared_2444_ == 0)
{
v___x_2446_ = v___x_2443_;
goto v_reusejp_2445_;
}
else
{
lean_object* v_reuseFailAlloc_2447_; 
v_reuseFailAlloc_2447_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2447_, 0, v_a_2441_);
v___x_2446_ = v_reuseFailAlloc_2447_;
goto v_reusejp_2445_;
}
v_reusejp_2445_:
{
return v___x_2446_;
}
}
}
}
}
else
{
lean_object* v___x_2449_; 
lean_dec(v_a_2412_);
lean_dec(v_userName_2407_);
lean_inc(v___y_2393_);
lean_inc_ref(v___y_2392_);
lean_inc(v___y_2391_);
lean_inc_ref(v___y_2390_);
lean_inc(v___x_2414_);
v___x_2449_ = lean_infer_type(v___x_2414_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2449_) == 0)
{
lean_object* v_a_2450_; lean_object* v_keyedConfig_2451_; uint8_t v_trackZetaDelta_2452_; lean_object* v_zetaDeltaSet_2453_; lean_object* v_lctx_2454_; lean_object* v_localInstances_2455_; lean_object* v_defEqCtx_x3f_2456_; lean_object* v_synthPendingDepth_2457_; lean_object* v_customCanUnfoldPredicate_x3f_2458_; uint8_t v_univApprox_2459_; uint8_t v_inTypeClassResolution_2460_; uint8_t v_cacheInferType_2461_; uint8_t v___x_2462_; lean_object* v___x_2463_; lean_object* v___x_2464_; lean_object* v___x_2465_; 
v_a_2450_ = lean_ctor_get(v___x_2449_, 0);
lean_inc(v_a_2450_);
lean_dec_ref_known(v___x_2449_, 1);
v_keyedConfig_2451_ = lean_ctor_get(v___y_2390_, 0);
v_trackZetaDelta_2452_ = lean_ctor_get_uint8(v___y_2390_, sizeof(void*)*7);
v_zetaDeltaSet_2453_ = lean_ctor_get(v___y_2390_, 1);
v_lctx_2454_ = lean_ctor_get(v___y_2390_, 2);
v_localInstances_2455_ = lean_ctor_get(v___y_2390_, 3);
v_defEqCtx_x3f_2456_ = lean_ctor_get(v___y_2390_, 4);
v_synthPendingDepth_2457_ = lean_ctor_get(v___y_2390_, 5);
v_customCanUnfoldPredicate_x3f_2458_ = lean_ctor_get(v___y_2390_, 6);
v_univApprox_2459_ = lean_ctor_get_uint8(v___y_2390_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2460_ = lean_ctor_get_uint8(v___y_2390_, sizeof(void*)*7 + 2);
v_cacheInferType_2461_ = lean_ctor_get_uint8(v___y_2390_, sizeof(void*)*7 + 3);
v___x_2462_ = 1;
lean_inc_ref(v_keyedConfig_2451_);
v___x_2463_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2462_, v_keyedConfig_2451_);
lean_inc(v_customCanUnfoldPredicate_x3f_2458_);
lean_inc(v_synthPendingDepth_2457_);
lean_inc(v_defEqCtx_x3f_2456_);
lean_inc_ref(v_localInstances_2455_);
lean_inc_ref(v_lctx_2454_);
lean_inc(v_zetaDeltaSet_2453_);
v___x_2464_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2464_, 0, v___x_2463_);
lean_ctor_set(v___x_2464_, 1, v_zetaDeltaSet_2453_);
lean_ctor_set(v___x_2464_, 2, v_lctx_2454_);
lean_ctor_set(v___x_2464_, 3, v_localInstances_2455_);
lean_ctor_set(v___x_2464_, 4, v_defEqCtx_x3f_2456_);
lean_ctor_set(v___x_2464_, 5, v_synthPendingDepth_2457_);
lean_ctor_set(v___x_2464_, 6, v_customCanUnfoldPredicate_x3f_2458_);
lean_ctor_set_uint8(v___x_2464_, sizeof(void*)*7, v_trackZetaDelta_2452_);
lean_ctor_set_uint8(v___x_2464_, sizeof(void*)*7 + 1, v_univApprox_2459_);
lean_ctor_set_uint8(v___x_2464_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2460_);
lean_ctor_set_uint8(v___x_2464_, sizeof(void*)*7 + 3, v_cacheInferType_2461_);
lean_inc(v_a_2410_);
v___x_2465_ = l_Lean_Meta_isExprDefEq(v_a_2410_, v_a_2450_, v___x_2464_, v___y_2391_, v___y_2392_, v___y_2393_);
lean_dec_ref_known(v___x_2464_, 7);
if (lean_obj_tag(v___x_2465_) == 0)
{
lean_object* v_a_2466_; uint8_t v___x_2467_; 
v_a_2466_ = lean_ctor_get(v___x_2465_, 0);
lean_inc(v_a_2466_);
lean_dec_ref_known(v___x_2465_, 1);
v___x_2467_ = lean_unbox(v_a_2466_);
lean_dec(v_a_2466_);
if (v___x_2467_ == 0)
{
lean_object* v___x_2468_; lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; 
lean_dec(v___x_2404_);
v___x_2468_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1);
lean_inc(v___x_2414_);
v___x_2469_ = l_Lean_MessageData_ofExpr(v___x_2414_);
v___x_2470_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2470_, 0, v___x_2468_);
lean_ctor_set(v___x_2470_, 1, v___x_2469_);
v___x_2471_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3);
v___x_2472_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2472_, 0, v___x_2470_);
lean_ctor_set(v___x_2472_, 1, v___x_2471_);
v___x_2473_ = l_Lean_MessageData_ofExpr(v_a_2410_);
v___x_2474_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2474_, 0, v___x_2472_);
lean_ctor_set(v___x_2474_, 1, v___x_2473_);
v___x_2475_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_2476_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2476_, 0, v___x_2474_);
lean_ctor_set(v___x_2476_, 1, v___x_2475_);
v___x_2477_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_2476_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2477_) == 0)
{
lean_dec_ref_known(v___x_2477_, 1);
v_a_2396_ = v___x_2413_;
goto v___jp_2395_;
}
else
{
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
return v___x_2477_;
}
}
else
{
lean_object* v___x_2478_; lean_object* v___x_2479_; 
v___x_2478_ = lean_box(0);
lean_inc(v___x_2414_);
v___x_2479_ = l_Lean_Meta_mkAuxTheorem(v_a_2410_, v___x_2414_, v___x_2385_, v___x_2478_, v___x_2385_, v___y_2390_, v___y_2391_, v___y_2392_, v___y_2393_);
if (lean_obj_tag(v___x_2479_) == 0)
{
lean_object* v_a_2480_; lean_object* v___x_2481_; 
v_a_2480_ = lean_ctor_get(v___x_2479_, 0);
lean_inc(v_a_2480_);
lean_dec_ref_known(v___x_2479_, 1);
v___x_2481_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_2404_, v_a_2480_, v___y_2391_);
if (lean_obj_tag(v___x_2481_) == 0)
{
lean_dec_ref_known(v___x_2481_, 1);
v_a_2396_ = v___x_2413_;
goto v___jp_2395_;
}
else
{
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
return v___x_2481_;
}
}
else
{
lean_object* v_a_2482_; lean_object* v___x_2484_; uint8_t v_isShared_2485_; uint8_t v_isSharedCheck_2489_; 
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2482_ = lean_ctor_get(v___x_2479_, 0);
v_isSharedCheck_2489_ = !lean_is_exclusive(v___x_2479_);
if (v_isSharedCheck_2489_ == 0)
{
v___x_2484_ = v___x_2479_;
v_isShared_2485_ = v_isSharedCheck_2489_;
goto v_resetjp_2483_;
}
else
{
lean_inc(v_a_2482_);
lean_dec(v___x_2479_);
v___x_2484_ = lean_box(0);
v_isShared_2485_ = v_isSharedCheck_2489_;
goto v_resetjp_2483_;
}
v_resetjp_2483_:
{
lean_object* v___x_2487_; 
if (v_isShared_2485_ == 0)
{
v___x_2487_ = v___x_2484_;
goto v_reusejp_2486_;
}
else
{
lean_object* v_reuseFailAlloc_2488_; 
v_reuseFailAlloc_2488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2488_, 0, v_a_2482_);
v___x_2487_ = v_reuseFailAlloc_2488_;
goto v_reusejp_2486_;
}
v_reusejp_2486_:
{
return v___x_2487_;
}
}
}
}
}
else
{
lean_object* v_a_2490_; lean_object* v___x_2492_; uint8_t v_isShared_2493_; uint8_t v_isSharedCheck_2497_; 
lean_dec(v_a_2410_);
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2490_ = lean_ctor_get(v___x_2465_, 0);
v_isSharedCheck_2497_ = !lean_is_exclusive(v___x_2465_);
if (v_isSharedCheck_2497_ == 0)
{
v___x_2492_ = v___x_2465_;
v_isShared_2493_ = v_isSharedCheck_2497_;
goto v_resetjp_2491_;
}
else
{
lean_inc(v_a_2490_);
lean_dec(v___x_2465_);
v___x_2492_ = lean_box(0);
v_isShared_2493_ = v_isSharedCheck_2497_;
goto v_resetjp_2491_;
}
v_resetjp_2491_:
{
lean_object* v___x_2495_; 
if (v_isShared_2493_ == 0)
{
v___x_2495_ = v___x_2492_;
goto v_reusejp_2494_;
}
else
{
lean_object* v_reuseFailAlloc_2496_; 
v_reuseFailAlloc_2496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2496_, 0, v_a_2490_);
v___x_2495_ = v_reuseFailAlloc_2496_;
goto v_reusejp_2494_;
}
v_reusejp_2494_:
{
return v___x_2495_;
}
}
}
}
else
{
lean_object* v_a_2498_; lean_object* v___x_2500_; uint8_t v_isShared_2501_; uint8_t v_isSharedCheck_2505_; 
lean_dec(v_a_2410_);
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2498_ = lean_ctor_get(v___x_2449_, 0);
v_isSharedCheck_2505_ = !lean_is_exclusive(v___x_2449_);
if (v_isSharedCheck_2505_ == 0)
{
v___x_2500_ = v___x_2449_;
v_isShared_2501_ = v_isSharedCheck_2505_;
goto v_resetjp_2499_;
}
else
{
lean_inc(v_a_2498_);
lean_dec(v___x_2449_);
v___x_2500_ = lean_box(0);
v_isShared_2501_ = v_isSharedCheck_2505_;
goto v_resetjp_2499_;
}
v_resetjp_2499_:
{
lean_object* v___x_2503_; 
if (v_isShared_2501_ == 0)
{
v___x_2503_ = v___x_2500_;
goto v_reusejp_2502_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v_a_2498_);
v___x_2503_ = v_reuseFailAlloc_2504_;
goto v_reusejp_2502_;
}
v_reusejp_2502_:
{
return v___x_2503_;
}
}
}
}
}
else
{
lean_object* v_a_2506_; lean_object* v___x_2508_; uint8_t v_isShared_2509_; uint8_t v_isSharedCheck_2513_; 
lean_dec(v_a_2410_);
lean_dec(v_userName_2407_);
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2506_ = lean_ctor_get(v___x_2411_, 0);
v_isSharedCheck_2513_ = !lean_is_exclusive(v___x_2411_);
if (v_isSharedCheck_2513_ == 0)
{
v___x_2508_ = v___x_2411_;
v_isShared_2509_ = v_isSharedCheck_2513_;
goto v_resetjp_2507_;
}
else
{
lean_inc(v_a_2506_);
lean_dec(v___x_2411_);
v___x_2508_ = lean_box(0);
v_isShared_2509_ = v_isSharedCheck_2513_;
goto v_resetjp_2507_;
}
v_resetjp_2507_:
{
lean_object* v___x_2511_; 
if (v_isShared_2509_ == 0)
{
v___x_2511_ = v___x_2508_;
goto v_reusejp_2510_;
}
else
{
lean_object* v_reuseFailAlloc_2512_; 
v_reuseFailAlloc_2512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2512_, 0, v_a_2506_);
v___x_2511_ = v_reuseFailAlloc_2512_;
goto v_reusejp_2510_;
}
v_reusejp_2510_:
{
return v___x_2511_;
}
}
}
}
else
{
lean_object* v_a_2514_; lean_object* v___x_2516_; uint8_t v_isShared_2517_; uint8_t v_isSharedCheck_2521_; 
lean_dec(v_userName_2407_);
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2514_ = lean_ctor_get(v___x_2409_, 0);
v_isSharedCheck_2521_ = !lean_is_exclusive(v___x_2409_);
if (v_isSharedCheck_2521_ == 0)
{
v___x_2516_ = v___x_2409_;
v_isShared_2517_ = v_isSharedCheck_2521_;
goto v_resetjp_2515_;
}
else
{
lean_inc(v_a_2514_);
lean_dec(v___x_2409_);
v___x_2516_ = lean_box(0);
v_isShared_2517_ = v_isSharedCheck_2521_;
goto v_resetjp_2515_;
}
v_resetjp_2515_:
{
lean_object* v___x_2519_; 
if (v_isShared_2517_ == 0)
{
v___x_2519_ = v___x_2516_;
goto v_reusejp_2518_;
}
else
{
lean_object* v_reuseFailAlloc_2520_; 
v_reuseFailAlloc_2520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2520_, 0, v_a_2514_);
v___x_2519_ = v_reuseFailAlloc_2520_;
goto v_reusejp_2518_;
}
v_reusejp_2518_:
{
return v___x_2519_;
}
}
}
}
else
{
lean_object* v_a_2522_; lean_object* v___x_2524_; uint8_t v_isShared_2525_; uint8_t v_isSharedCheck_2529_; 
lean_dec(v___x_2404_);
lean_dec(v_a_2388_);
lean_dec_ref(v_trace_2387_);
lean_dec(v_val_2386_);
v_a_2522_ = lean_ctor_get(v___x_2405_, 0);
v_isSharedCheck_2529_ = !lean_is_exclusive(v___x_2405_);
if (v_isSharedCheck_2529_ == 0)
{
v___x_2524_ = v___x_2405_;
v_isShared_2525_ = v_isSharedCheck_2529_;
goto v_resetjp_2523_;
}
else
{
lean_inc(v_a_2522_);
lean_dec(v___x_2405_);
v___x_2524_ = lean_box(0);
v_isShared_2525_ = v_isSharedCheck_2529_;
goto v_resetjp_2523_;
}
v_resetjp_2523_:
{
lean_object* v___x_2527_; 
if (v_isShared_2525_ == 0)
{
v___x_2527_ = v___x_2524_;
goto v_reusejp_2526_;
}
else
{
lean_object* v_reuseFailAlloc_2528_; 
v_reuseFailAlloc_2528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2528_, 0, v_a_2522_);
v___x_2527_ = v_reuseFailAlloc_2528_;
goto v_reusejp_2526_;
}
v_reusejp_2526_:
{
return v___x_2527_;
}
}
}
}
v___jp_2395_:
{
lean_object* v___x_2397_; lean_object* v___x_2398_; 
v___x_2397_ = lean_unsigned_to_nat(1u);
v___x_2398_ = lean_nat_add(v_a_2388_, v___x_2397_);
lean_dec(v_a_2388_);
v_a_2388_ = v___x_2398_;
v_b_2389_ = v_a_2396_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__17(lean_object* v_val_2530_, lean_object* v_trace_2531_, uint8_t v___x_2532_, lean_object* v_expectedType_2533_, lean_object* v_inst_2534_, lean_object* v_x_2535_, lean_object* v_x_2536_, lean_object* v_x_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_){
_start:
{
lean_object* v_m_2544_; lean_object* v___y_2545_; lean_object* v___y_2546_; lean_object* v___y_2547_; lean_object* v___y_2548_; 
if (lean_obj_tag(v_x_2535_) == 5)
{
lean_object* v_fn_2556_; lean_object* v_arg_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; 
v_fn_2556_ = lean_ctor_get(v_x_2535_, 0);
lean_inc_ref(v_fn_2556_);
v_arg_2557_ = lean_ctor_get(v_x_2535_, 1);
lean_inc_ref(v_arg_2557_);
lean_dec_ref_known(v_x_2535_, 2);
v___x_2558_ = lean_array_set(v_x_2536_, v_x_2537_, v_arg_2557_);
v___x_2559_ = lean_unsigned_to_nat(1u);
v___x_2560_ = lean_nat_sub(v_x_2537_, v___x_2559_);
lean_dec(v_x_2537_);
v_x_2535_ = v_fn_2556_;
v_x_2536_ = v___x_2558_;
v_x_2537_ = v___x_2560_;
goto _start;
}
else
{
lean_dec(v_x_2537_);
if (lean_obj_tag(v_x_2535_) == 4)
{
lean_object* v_declName_2562_; lean_object* v___x_2563_; 
v_declName_2562_ = lean_ctor_get(v_x_2535_, 0);
lean_inc(v_declName_2562_);
v___x_2563_ = lp_mathlib_Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2(v_declName_2562_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_);
if (lean_obj_tag(v___x_2563_) == 0)
{
lean_object* v_a_2564_; 
v_a_2564_ = lean_ctor_get(v___x_2563_, 0);
lean_inc(v_a_2564_);
lean_dec_ref_known(v___x_2563_, 1);
if (lean_obj_tag(v_a_2564_) == 6)
{
lean_object* v_val_2565_; lean_object* v___x_2566_; 
lean_dec_ref(v_inst_2534_);
v_val_2565_ = lean_ctor_get(v_a_2564_, 0);
lean_inc_ref(v_val_2565_);
lean_dec_ref_known(v_a_2564_, 1);
lean_inc(v___y_2541_);
lean_inc_ref(v___y_2540_);
lean_inc(v___y_2539_);
lean_inc_ref(v___y_2538_);
lean_inc_ref(v_x_2535_);
v___x_2566_ = lean_infer_type(v_x_2535_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_);
if (lean_obj_tag(v___x_2566_) == 0)
{
lean_object* v_a_2567_; uint8_t v___x_2568_; lean_object* v___x_2569_; 
v_a_2567_ = lean_ctor_get(v___x_2566_, 0);
lean_inc(v_a_2567_);
lean_dec_ref_known(v___x_2566_, 1);
v___x_2568_ = 0;
v___x_2569_ = l_Lean_Meta_forallMetaTelescope(v_a_2567_, v___x_2568_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_);
if (lean_obj_tag(v___x_2569_) == 0)
{
lean_object* v_a_2570_; lean_object* v_snd_2571_; lean_object* v_fst_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2679_; 
v_a_2570_ = lean_ctor_get(v___x_2569_, 0);
lean_inc(v_a_2570_);
lean_dec_ref_known(v___x_2569_, 1);
v_snd_2571_ = lean_ctor_get(v_a_2570_, 1);
v_fst_2572_ = lean_ctor_get(v_a_2570_, 0);
v_isSharedCheck_2679_ = !lean_is_exclusive(v_a_2570_);
if (v_isSharedCheck_2679_ == 0)
{
v___x_2574_ = v_a_2570_;
v_isShared_2575_ = v_isSharedCheck_2679_;
goto v_resetjp_2573_;
}
else
{
lean_inc(v_snd_2571_);
lean_inc(v_fst_2572_);
lean_dec(v_a_2570_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2679_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v_fst_2576_; lean_object* v_snd_2577_; lean_object* v___x_2579_; uint8_t v_isShared_2580_; uint8_t v_isSharedCheck_2678_; 
v_fst_2576_ = lean_ctor_get(v_snd_2571_, 0);
v_snd_2577_ = lean_ctor_get(v_snd_2571_, 1);
v_isSharedCheck_2678_ = !lean_is_exclusive(v_snd_2571_);
if (v_isSharedCheck_2678_ == 0)
{
v___x_2579_ = v_snd_2571_;
v_isShared_2580_ = v_isSharedCheck_2678_;
goto v_resetjp_2578_;
}
else
{
lean_inc(v_snd_2577_);
lean_inc(v_fst_2576_);
lean_dec(v_snd_2571_);
v___x_2579_ = lean_box(0);
v_isShared_2580_ = v_isSharedCheck_2678_;
goto v_resetjp_2578_;
}
v_resetjp_2578_:
{
lean_object* v___y_2582_; lean_object* v___y_2583_; lean_object* v___y_2584_; lean_object* v___y_2585_; lean_object* v___y_2619_; lean_object* v___y_2620_; lean_object* v___y_2621_; lean_object* v___y_2622_; lean_object* v___x_2656_; lean_object* v___x_2657_; uint8_t v___x_2658_; 
v___x_2656_ = lean_array_get_size(v_x_2536_);
v___x_2657_ = lean_array_get_size(v_fst_2572_);
v___x_2658_ = lean_nat_dec_eq(v___x_2656_, v___x_2657_);
if (v___x_2658_ == 0)
{
lean_object* v___x_2659_; lean_object* v___x_2660_; lean_object* v___x_2661_; lean_object* v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v_a_2670_; lean_object* v___x_2672_; uint8_t v_isShared_2673_; uint8_t v_isSharedCheck_2677_; 
lean_del_object(v___x_2579_);
lean_dec(v_snd_2577_);
lean_dec(v_fst_2576_);
lean_del_object(v___x_2574_);
lean_dec(v_fst_2572_);
lean_dec_ref(v_val_2565_);
lean_dec_ref(v_expectedType_2533_);
lean_dec_ref(v_trace_2531_);
lean_dec(v_val_2530_);
v___x_2659_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__5);
v___x_2660_ = l_Lean_MessageData_ofExpr(v_x_2535_);
v___x_2661_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2661_, 0, v___x_2659_);
lean_ctor_set(v___x_2661_, 1, v___x_2660_);
v___x_2662_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__7);
v___x_2663_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2663_, 0, v___x_2661_);
lean_ctor_set(v___x_2663_, 1, v___x_2662_);
v___x_2664_ = lean_array_to_list(v_x_2536_);
v___x_2665_ = lean_box(0);
v___x_2666_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__8(v___x_2664_, v___x_2665_);
v___x_2667_ = l_Lean_MessageData_ofList(v___x_2666_);
v___x_2668_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2668_, 0, v___x_2663_);
lean_ctor_set(v___x_2668_, 1, v___x_2667_);
v___x_2669_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_2668_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_);
v_a_2670_ = lean_ctor_get(v___x_2669_, 0);
v_isSharedCheck_2677_ = !lean_is_exclusive(v___x_2669_);
if (v_isSharedCheck_2677_ == 0)
{
v___x_2672_ = v___x_2669_;
v_isShared_2673_ = v_isSharedCheck_2677_;
goto v_resetjp_2671_;
}
else
{
lean_inc(v_a_2670_);
lean_dec(v___x_2669_);
v___x_2672_ = lean_box(0);
v_isShared_2673_ = v_isSharedCheck_2677_;
goto v_resetjp_2671_;
}
v_resetjp_2671_:
{
lean_object* v___x_2675_; 
if (v_isShared_2673_ == 0)
{
v___x_2675_ = v___x_2672_;
goto v_reusejp_2674_;
}
else
{
lean_object* v_reuseFailAlloc_2676_; 
v_reuseFailAlloc_2676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2676_, 0, v_a_2670_);
v___x_2675_ = v_reuseFailAlloc_2676_;
goto v_reusejp_2674_;
}
v_reusejp_2674_:
{
return v___x_2675_;
}
}
}
else
{
v___y_2619_ = v___y_2538_;
v___y_2620_ = v___y_2539_;
v___y_2621_ = v___y_2540_;
v___y_2622_ = v___y_2541_;
goto v___jp_2618_;
}
v___jp_2581_:
{
lean_object* v_numParams_2586_; lean_object* v___x_2587_; lean_object* v___x_2588_; lean_object* v___x_2589_; 
v_numParams_2586_ = lean_ctor_get(v_val_2565_, 3);
lean_inc(v_numParams_2586_);
lean_dec_ref(v_val_2565_);
v___x_2587_ = lean_array_get_size(v_x_2536_);
v___x_2588_ = lean_box(0);
v___x_2589_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg(v___x_2587_, v_fst_2572_, v_x_2536_, v_fst_2576_, v___x_2532_, v_val_2530_, v_trace_2531_, v_numParams_2586_, v___x_2588_, v___y_2582_, v___y_2583_, v___y_2584_, v___y_2585_);
lean_dec(v_fst_2576_);
lean_dec_ref(v_x_2536_);
if (lean_obj_tag(v___x_2589_) == 0)
{
size_t v_sz_2590_; size_t v___x_2591_; lean_object* v___x_2592_; 
lean_dec_ref_known(v___x_2589_, 1);
v_sz_2590_ = lean_array_size(v_fst_2572_);
v___x_2591_ = ((size_t)0ULL);
v___x_2592_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__4(v_sz_2590_, v___x_2591_, v_fst_2572_, v___y_2582_, v___y_2583_, v___y_2584_, v___y_2585_);
if (lean_obj_tag(v___x_2592_) == 0)
{
lean_object* v_a_2593_; lean_object* v___x_2595_; uint8_t v_isShared_2596_; uint8_t v_isSharedCheck_2601_; 
v_a_2593_ = lean_ctor_get(v___x_2592_, 0);
v_isSharedCheck_2601_ = !lean_is_exclusive(v___x_2592_);
if (v_isSharedCheck_2601_ == 0)
{
v___x_2595_ = v___x_2592_;
v_isShared_2596_ = v_isSharedCheck_2601_;
goto v_resetjp_2594_;
}
else
{
lean_inc(v_a_2593_);
lean_dec(v___x_2592_);
v___x_2595_ = lean_box(0);
v_isShared_2596_ = v_isSharedCheck_2601_;
goto v_resetjp_2594_;
}
v_resetjp_2594_:
{
lean_object* v___x_2597_; lean_object* v___x_2599_; 
v___x_2597_ = l_Lean_mkAppN(v_x_2535_, v_a_2593_);
lean_dec(v_a_2593_);
if (v_isShared_2596_ == 0)
{
lean_ctor_set(v___x_2595_, 0, v___x_2597_);
v___x_2599_ = v___x_2595_;
goto v_reusejp_2598_;
}
else
{
lean_object* v_reuseFailAlloc_2600_; 
v_reuseFailAlloc_2600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2600_, 0, v___x_2597_);
v___x_2599_ = v_reuseFailAlloc_2600_;
goto v_reusejp_2598_;
}
v_reusejp_2598_:
{
return v___x_2599_;
}
}
}
else
{
lean_object* v_a_2602_; lean_object* v___x_2604_; uint8_t v_isShared_2605_; uint8_t v_isSharedCheck_2609_; 
lean_dec_ref_known(v_x_2535_, 2);
v_a_2602_ = lean_ctor_get(v___x_2592_, 0);
v_isSharedCheck_2609_ = !lean_is_exclusive(v___x_2592_);
if (v_isSharedCheck_2609_ == 0)
{
v___x_2604_ = v___x_2592_;
v_isShared_2605_ = v_isSharedCheck_2609_;
goto v_resetjp_2603_;
}
else
{
lean_inc(v_a_2602_);
lean_dec(v___x_2592_);
v___x_2604_ = lean_box(0);
v_isShared_2605_ = v_isSharedCheck_2609_;
goto v_resetjp_2603_;
}
v_resetjp_2603_:
{
lean_object* v___x_2607_; 
if (v_isShared_2605_ == 0)
{
v___x_2607_ = v___x_2604_;
goto v_reusejp_2606_;
}
else
{
lean_object* v_reuseFailAlloc_2608_; 
v_reuseFailAlloc_2608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2608_, 0, v_a_2602_);
v___x_2607_ = v_reuseFailAlloc_2608_;
goto v_reusejp_2606_;
}
v_reusejp_2606_:
{
return v___x_2607_;
}
}
}
}
else
{
lean_object* v_a_2610_; lean_object* v___x_2612_; uint8_t v_isShared_2613_; uint8_t v_isSharedCheck_2617_; 
lean_dec(v_fst_2572_);
lean_dec_ref_known(v_x_2535_, 2);
v_a_2610_ = lean_ctor_get(v___x_2589_, 0);
v_isSharedCheck_2617_ = !lean_is_exclusive(v___x_2589_);
if (v_isSharedCheck_2617_ == 0)
{
v___x_2612_ = v___x_2589_;
v_isShared_2613_ = v_isSharedCheck_2617_;
goto v_resetjp_2611_;
}
else
{
lean_inc(v_a_2610_);
lean_dec(v___x_2589_);
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
v___jp_2618_:
{
lean_object* v___x_2623_; 
lean_inc_ref(v_expectedType_2533_);
v___x_2623_ = l_Lean_Meta_isExprDefEq(v_expectedType_2533_, v_snd_2577_, v___y_2619_, v___y_2620_, v___y_2621_, v___y_2622_);
if (lean_obj_tag(v___x_2623_) == 0)
{
lean_object* v_a_2624_; uint8_t v___x_2625_; 
v_a_2624_ = lean_ctor_get(v___x_2623_, 0);
lean_inc(v_a_2624_);
lean_dec_ref_known(v___x_2623_, 1);
v___x_2625_ = lean_unbox(v_a_2624_);
if (v___x_2625_ == 0)
{
lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2629_; 
lean_inc(v_declName_2562_);
lean_dec(v_fst_2576_);
lean_dec(v_fst_2572_);
lean_dec_ref(v_val_2565_);
lean_dec_ref_known(v_x_2535_, 2);
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_trace_2531_);
lean_dec(v_val_2530_);
v___x_2626_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_2627_ = l_Lean_MessageData_ofExpr(v_expectedType_2533_);
if (v_isShared_2580_ == 0)
{
lean_ctor_set_tag(v___x_2579_, 7);
lean_ctor_set(v___x_2579_, 1, v___x_2627_);
lean_ctor_set(v___x_2579_, 0, v___x_2626_);
v___x_2629_ = v___x_2579_;
goto v_reusejp_2628_;
}
else
{
lean_object* v_reuseFailAlloc_2647_; 
v_reuseFailAlloc_2647_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2647_, 0, v___x_2626_);
lean_ctor_set(v_reuseFailAlloc_2647_, 1, v___x_2627_);
v___x_2629_ = v_reuseFailAlloc_2647_;
goto v_reusejp_2628_;
}
v_reusejp_2628_:
{
lean_object* v___x_2630_; lean_object* v___x_2632_; 
v___x_2630_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__3);
if (v_isShared_2575_ == 0)
{
lean_ctor_set_tag(v___x_2574_, 7);
lean_ctor_set(v___x_2574_, 1, v___x_2630_);
lean_ctor_set(v___x_2574_, 0, v___x_2629_);
v___x_2632_ = v___x_2574_;
goto v_reusejp_2631_;
}
else
{
lean_object* v_reuseFailAlloc_2646_; 
v_reuseFailAlloc_2646_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2646_, 0, v___x_2629_);
lean_ctor_set(v_reuseFailAlloc_2646_, 1, v___x_2630_);
v___x_2632_ = v_reuseFailAlloc_2646_;
goto v_reusejp_2631_;
}
v_reusejp_2631_:
{
uint8_t v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v_a_2638_; lean_object* v___x_2640_; uint8_t v_isShared_2641_; uint8_t v_isSharedCheck_2645_; 
v___x_2633_ = lean_unbox(v_a_2624_);
lean_dec(v_a_2624_);
v___x_2634_ = l_Lean_MessageData_ofConstName(v_declName_2562_, v___x_2633_);
v___x_2635_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2635_, 0, v___x_2632_);
lean_ctor_set(v___x_2635_, 1, v___x_2634_);
v___x_2636_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2636_, 0, v___x_2635_);
lean_ctor_set(v___x_2636_, 1, v___x_2626_);
v___x_2637_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_2636_, v___y_2619_, v___y_2620_, v___y_2621_, v___y_2622_);
v_a_2638_ = lean_ctor_get(v___x_2637_, 0);
v_isSharedCheck_2645_ = !lean_is_exclusive(v___x_2637_);
if (v_isSharedCheck_2645_ == 0)
{
v___x_2640_ = v___x_2637_;
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
else
{
lean_inc(v_a_2638_);
lean_dec(v___x_2637_);
v___x_2640_ = lean_box(0);
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
v_resetjp_2639_:
{
lean_object* v___x_2643_; 
if (v_isShared_2641_ == 0)
{
v___x_2643_ = v___x_2640_;
goto v_reusejp_2642_;
}
else
{
lean_object* v_reuseFailAlloc_2644_; 
v_reuseFailAlloc_2644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2644_, 0, v_a_2638_);
v___x_2643_ = v_reuseFailAlloc_2644_;
goto v_reusejp_2642_;
}
v_reusejp_2642_:
{
return v___x_2643_;
}
}
}
}
}
else
{
lean_dec(v_a_2624_);
lean_del_object(v___x_2579_);
lean_del_object(v___x_2574_);
lean_dec_ref(v_expectedType_2533_);
v___y_2582_ = v___y_2619_;
v___y_2583_ = v___y_2620_;
v___y_2584_ = v___y_2621_;
v___y_2585_ = v___y_2622_;
goto v___jp_2581_;
}
}
else
{
lean_object* v_a_2648_; lean_object* v___x_2650_; uint8_t v_isShared_2651_; uint8_t v_isSharedCheck_2655_; 
lean_del_object(v___x_2579_);
lean_dec(v_fst_2576_);
lean_del_object(v___x_2574_);
lean_dec(v_fst_2572_);
lean_dec_ref(v_val_2565_);
lean_dec_ref_known(v_x_2535_, 2);
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_expectedType_2533_);
lean_dec_ref(v_trace_2531_);
lean_dec(v_val_2530_);
v_a_2648_ = lean_ctor_get(v___x_2623_, 0);
v_isSharedCheck_2655_ = !lean_is_exclusive(v___x_2623_);
if (v_isSharedCheck_2655_ == 0)
{
v___x_2650_ = v___x_2623_;
v_isShared_2651_ = v_isSharedCheck_2655_;
goto v_resetjp_2649_;
}
else
{
lean_inc(v_a_2648_);
lean_dec(v___x_2623_);
v___x_2650_ = lean_box(0);
v_isShared_2651_ = v_isSharedCheck_2655_;
goto v_resetjp_2649_;
}
v_resetjp_2649_:
{
lean_object* v___x_2653_; 
if (v_isShared_2651_ == 0)
{
v___x_2653_ = v___x_2650_;
goto v_reusejp_2652_;
}
else
{
lean_object* v_reuseFailAlloc_2654_; 
v_reuseFailAlloc_2654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2654_, 0, v_a_2648_);
v___x_2653_ = v_reuseFailAlloc_2654_;
goto v_reusejp_2652_;
}
v_reusejp_2652_:
{
return v___x_2653_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2680_; lean_object* v___x_2682_; uint8_t v_isShared_2683_; uint8_t v_isSharedCheck_2687_; 
lean_dec_ref(v_val_2565_);
lean_dec_ref_known(v_x_2535_, 2);
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_expectedType_2533_);
lean_dec_ref(v_trace_2531_);
lean_dec(v_val_2530_);
v_a_2680_ = lean_ctor_get(v___x_2569_, 0);
v_isSharedCheck_2687_ = !lean_is_exclusive(v___x_2569_);
if (v_isSharedCheck_2687_ == 0)
{
v___x_2682_ = v___x_2569_;
v_isShared_2683_ = v_isSharedCheck_2687_;
goto v_resetjp_2681_;
}
else
{
lean_inc(v_a_2680_);
lean_dec(v___x_2569_);
v___x_2682_ = lean_box(0);
v_isShared_2683_ = v_isSharedCheck_2687_;
goto v_resetjp_2681_;
}
v_resetjp_2681_:
{
lean_object* v___x_2685_; 
if (v_isShared_2683_ == 0)
{
v___x_2685_ = v___x_2682_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2686_; 
v_reuseFailAlloc_2686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2686_, 0, v_a_2680_);
v___x_2685_ = v_reuseFailAlloc_2686_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
return v___x_2685_;
}
}
}
}
else
{
lean_dec_ref(v_val_2565_);
lean_dec_ref_known(v_x_2535_, 2);
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_expectedType_2533_);
lean_dec_ref(v_trace_2531_);
lean_dec(v_val_2530_);
return v___x_2566_;
}
}
else
{
lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; 
lean_inc(v_declName_2562_);
lean_dec(v_a_2564_);
lean_dec_ref_known(v_x_2535_, 2);
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_expectedType_2533_);
v___x_2688_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9);
v___x_2689_ = l_Lean_indentExpr(v_inst_2534_);
v___x_2690_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2690_, 0, v___x_2688_);
lean_ctor_set(v___x_2690_, 1, v___x_2689_);
v___x_2691_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__11);
v___x_2692_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2692_, 0, v___x_2690_);
lean_ctor_set(v___x_2692_, 1, v___x_2691_);
v___x_2693_ = l_Lean_MessageData_ofName(v_declName_2562_);
v___x_2694_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2694_, 0, v___x_2692_);
lean_ctor_set(v___x_2694_, 1, v___x_2693_);
v___x_2695_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__13);
v___x_2696_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2696_, 0, v___x_2694_);
lean_ctor_set(v___x_2696_, 1, v___x_2695_);
v_m_2544_ = v___x_2696_;
v___y_2545_ = v___y_2538_;
v___y_2546_ = v___y_2539_;
v___y_2547_ = v___y_2540_;
v___y_2548_ = v___y_2541_;
goto v___jp_2543_;
}
}
else
{
lean_object* v_a_2697_; lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2704_; 
lean_dec_ref_known(v_x_2535_, 2);
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_inst_2534_);
lean_dec_ref(v_expectedType_2533_);
lean_dec_ref(v_trace_2531_);
lean_dec(v_val_2530_);
v_a_2697_ = lean_ctor_get(v___x_2563_, 0);
v_isSharedCheck_2704_ = !lean_is_exclusive(v___x_2563_);
if (v_isSharedCheck_2704_ == 0)
{
v___x_2699_ = v___x_2563_;
v_isShared_2700_ = v_isSharedCheck_2704_;
goto v_resetjp_2698_;
}
else
{
lean_inc(v_a_2697_);
lean_dec(v___x_2563_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2704_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
lean_object* v___x_2702_; 
if (v_isShared_2700_ == 0)
{
v___x_2702_ = v___x_2699_;
goto v_reusejp_2701_;
}
else
{
lean_object* v_reuseFailAlloc_2703_; 
v_reuseFailAlloc_2703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2703_, 0, v_a_2697_);
v___x_2702_ = v_reuseFailAlloc_2703_;
goto v_reusejp_2701_;
}
v_reusejp_2701_:
{
return v___x_2702_;
}
}
}
}
else
{
lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; 
lean_dec_ref(v_x_2536_);
lean_dec_ref(v_x_2535_);
lean_dec_ref(v_expectedType_2533_);
v___x_2705_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__9);
v___x_2706_ = l_Lean_indentExpr(v_inst_2534_);
v___x_2707_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2707_, 0, v___x_2705_);
lean_ctor_set(v___x_2707_, 1, v___x_2706_);
v_m_2544_ = v___x_2707_;
v___y_2545_ = v___y_2538_;
v___y_2546_ = v___y_2539_;
v___y_2547_ = v___y_2540_;
v___y_2548_ = v___y_2541_;
goto v___jp_2543_;
}
}
v___jp_2543_:
{
lean_object* v___x_2549_; lean_object* v_env_2550_; uint8_t v___x_2551_; 
v___x_2549_ = lean_st_ref_get(v___y_2548_);
v_env_2550_ = lean_ctor_get(v___x_2549_, 0);
lean_inc_ref(v_env_2550_);
lean_dec(v___x_2549_);
v___x_2551_ = l_Lean_isStructure(v_env_2550_, v_val_2530_);
if (v___x_2551_ == 0)
{
lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; 
v___x_2552_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___closed__1);
v___x_2553_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2553_, 0, v_m_2544_);
lean_ctor_set(v___x_2553_, 1, v___x_2552_);
v___x_2554_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2531_, v___x_2553_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_);
return v___x_2554_;
}
else
{
lean_object* v___x_2555_; 
v___x_2555_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2531_, v_m_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_);
return v___x_2555_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2(lean_object* v_expectedType_2708_, lean_object* v_inst_2709_, lean_object* v_trace_2710_, lean_object* v_cls_2711_, uint8_t v_root_2712_, lean_object* v_val_2713_, uint8_t v___x_2714_, lean_object* v_____r_2715_, lean_object* v___y_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_){
_start:
{
lean_object* v___x_2721_; 
lean_inc_ref(v_expectedType_2708_);
v___x_2721_ = l_Lean_Meta_isProp(v_expectedType_2708_, v___y_2716_, v___y_2717_, v___y_2718_, v___y_2719_);
if (lean_obj_tag(v___x_2721_) == 0)
{
lean_object* v_a_2722_; uint8_t v___x_2723_; 
v_a_2722_ = lean_ctor_get(v___x_2721_, 0);
lean_inc(v_a_2722_);
lean_dec_ref_known(v___x_2721_, 1);
v___x_2723_ = lean_unbox(v_a_2722_);
lean_dec(v_a_2722_);
if (v___x_2723_ == 0)
{
lean_object* v___x_2724_; lean_object* v___x_2725_; 
v___x_2724_ = lean_box(0);
lean_inc_ref(v_expectedType_2708_);
v___x_2725_ = l_Lean_Meta_trySynthInstance(v_expectedType_2708_, v___x_2724_, v___y_2716_, v___y_2717_, v___y_2718_, v___y_2719_);
if (lean_obj_tag(v___x_2725_) == 0)
{
lean_object* v_a_2726_; 
v_a_2726_ = lean_ctor_get(v___x_2725_, 0);
lean_inc(v_a_2726_);
lean_dec_ref_known(v___x_2725_, 1);
if (lean_obj_tag(v_a_2726_) == 1)
{
lean_object* v_a_2727_; lean_object* v___y_2729_; lean_object* v___y_2730_; lean_object* v___y_2731_; lean_object* v___y_2732_; 
lean_dec(v_val_2713_);
v_a_2727_ = lean_ctor_get(v_a_2726_, 0);
lean_inc(v_a_2727_);
lean_dec_ref_known(v_a_2726_, 1);
if (v_root_2712_ == 0)
{
lean_dec_ref(v_expectedType_2708_);
v___y_2729_ = v___y_2716_;
v___y_2730_ = v___y_2717_;
v___y_2731_ = v___y_2718_;
v___y_2732_ = v___y_2719_;
goto v___jp_2728_;
}
else
{
lean_object* v_ref_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; 
v_ref_2800_ = lean_ctor_get(v___y_2718_, 5);
v___x_2801_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing;
v___x_2802_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8);
v___x_2803_ = l_Lean_MessageData_ofExpr(v_expectedType_2708_);
v___x_2804_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2804_, 0, v___x_2802_);
lean_ctor_set(v___x_2804_, 1, v___x_2803_);
v___x_2805_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10);
v___x_2806_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2806_, 0, v___x_2804_);
lean_ctor_set(v___x_2806_, 1, v___x_2805_);
lean_inc(v_ref_2800_);
v___x_2807_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(v___x_2801_, v_ref_2800_, v___x_2806_, v___y_2716_, v___y_2717_, v___y_2718_, v___y_2719_);
if (lean_obj_tag(v___x_2807_) == 0)
{
lean_dec_ref_known(v___x_2807_, 1);
v___y_2729_ = v___y_2716_;
v___y_2730_ = v___y_2717_;
v___y_2731_ = v___y_2718_;
v___y_2732_ = v___y_2719_;
goto v___jp_2728_;
}
else
{
lean_object* v_a_2808_; lean_object* v___x_2810_; uint8_t v_isShared_2811_; uint8_t v_isSharedCheck_2815_; 
lean_dec(v_a_2727_);
lean_dec(v_cls_2711_);
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_inst_2709_);
v_a_2808_ = lean_ctor_get(v___x_2807_, 0);
v_isSharedCheck_2815_ = !lean_is_exclusive(v___x_2807_);
if (v_isSharedCheck_2815_ == 0)
{
v___x_2810_ = v___x_2807_;
v_isShared_2811_ = v_isSharedCheck_2815_;
goto v_resetjp_2809_;
}
else
{
lean_inc(v_a_2808_);
lean_dec(v___x_2807_);
v___x_2810_ = lean_box(0);
v_isShared_2811_ = v_isSharedCheck_2815_;
goto v_resetjp_2809_;
}
v_resetjp_2809_:
{
lean_object* v___x_2813_; 
if (v_isShared_2811_ == 0)
{
v___x_2813_ = v___x_2810_;
goto v_reusejp_2812_;
}
else
{
lean_object* v_reuseFailAlloc_2814_; 
v_reuseFailAlloc_2814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2814_, 0, v_a_2808_);
v___x_2813_ = v_reuseFailAlloc_2814_;
goto v_reusejp_2812_;
}
v_reusejp_2812_:
{
return v___x_2813_;
}
}
}
}
v___jp_2728_:
{
lean_object* v_keyedConfig_2733_; uint8_t v_trackZetaDelta_2734_; lean_object* v_zetaDeltaSet_2735_; lean_object* v_lctx_2736_; lean_object* v_localInstances_2737_; lean_object* v_defEqCtx_x3f_2738_; lean_object* v_synthPendingDepth_2739_; lean_object* v_customCanUnfoldPredicate_x3f_2740_; uint8_t v_univApprox_2741_; uint8_t v_inTypeClassResolution_2742_; uint8_t v_cacheInferType_2743_; uint8_t v___x_2744_; lean_object* v___x_2745_; lean_object* v___x_2746_; lean_object* v___x_2747_; 
v_keyedConfig_2733_ = lean_ctor_get(v___y_2729_, 0);
v_trackZetaDelta_2734_ = lean_ctor_get_uint8(v___y_2729_, sizeof(void*)*7);
v_zetaDeltaSet_2735_ = lean_ctor_get(v___y_2729_, 1);
v_lctx_2736_ = lean_ctor_get(v___y_2729_, 2);
v_localInstances_2737_ = lean_ctor_get(v___y_2729_, 3);
v_defEqCtx_x3f_2738_ = lean_ctor_get(v___y_2729_, 4);
v_synthPendingDepth_2739_ = lean_ctor_get(v___y_2729_, 5);
v_customCanUnfoldPredicate_x3f_2740_ = lean_ctor_get(v___y_2729_, 6);
v_univApprox_2741_ = lean_ctor_get_uint8(v___y_2729_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2742_ = lean_ctor_get_uint8(v___y_2729_, sizeof(void*)*7 + 2);
v_cacheInferType_2743_ = lean_ctor_get_uint8(v___y_2729_, sizeof(void*)*7 + 3);
v___x_2744_ = 1;
lean_inc_ref(v_keyedConfig_2733_);
v___x_2745_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2744_, v_keyedConfig_2733_);
lean_inc(v_customCanUnfoldPredicate_x3f_2740_);
lean_inc(v_synthPendingDepth_2739_);
lean_inc(v_defEqCtx_x3f_2738_);
lean_inc_ref(v_localInstances_2737_);
lean_inc_ref(v_lctx_2736_);
lean_inc(v_zetaDeltaSet_2735_);
v___x_2746_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2746_, 0, v___x_2745_);
lean_ctor_set(v___x_2746_, 1, v_zetaDeltaSet_2735_);
lean_ctor_set(v___x_2746_, 2, v_lctx_2736_);
lean_ctor_set(v___x_2746_, 3, v_localInstances_2737_);
lean_ctor_set(v___x_2746_, 4, v_defEqCtx_x3f_2738_);
lean_ctor_set(v___x_2746_, 5, v_synthPendingDepth_2739_);
lean_ctor_set(v___x_2746_, 6, v_customCanUnfoldPredicate_x3f_2740_);
lean_ctor_set_uint8(v___x_2746_, sizeof(void*)*7, v_trackZetaDelta_2734_);
lean_ctor_set_uint8(v___x_2746_, sizeof(void*)*7 + 1, v_univApprox_2741_);
lean_ctor_set_uint8(v___x_2746_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2742_);
lean_ctor_set_uint8(v___x_2746_, sizeof(void*)*7 + 3, v_cacheInferType_2743_);
lean_inc(v_a_2727_);
lean_inc_ref(v_inst_2709_);
v___x_2747_ = l_Lean_Meta_isExprDefEq(v_inst_2709_, v_a_2727_, v___x_2746_, v___y_2730_, v___y_2731_, v___y_2732_);
lean_dec_ref_known(v___x_2746_, 7);
if (lean_obj_tag(v___x_2747_) == 0)
{
lean_object* v_a_2748_; lean_object* v___x_2750_; uint8_t v_isShared_2751_; uint8_t v_isSharedCheck_2791_; 
v_a_2748_ = lean_ctor_get(v___x_2747_, 0);
v_isSharedCheck_2791_ = !lean_is_exclusive(v___x_2747_);
if (v_isSharedCheck_2791_ == 0)
{
v___x_2750_ = v___x_2747_;
v_isShared_2751_ = v_isSharedCheck_2791_;
goto v_resetjp_2749_;
}
else
{
lean_inc(v_a_2748_);
lean_dec(v___x_2747_);
v___x_2750_ = lean_box(0);
v_isShared_2751_ = v_isSharedCheck_2791_;
goto v_resetjp_2749_;
}
v_resetjp_2749_:
{
uint8_t v___x_2752_; 
v___x_2752_ = lean_unbox(v_a_2748_);
lean_dec(v_a_2748_);
if (v___x_2752_ == 0)
{
lean_object* v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; 
lean_del_object(v___x_2750_);
lean_dec(v_cls_2711_);
v___x_2753_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
v___x_2754_ = l_Lean_indentExpr(v_inst_2709_);
v___x_2755_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2755_, 0, v___x_2753_);
lean_ctor_set(v___x_2755_, 1, v___x_2754_);
v___x_2756_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3);
v___x_2757_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2757_, 0, v___x_2755_);
lean_ctor_set(v___x_2757_, 1, v___x_2756_);
v___x_2758_ = l_Lean_indentExpr(v_a_2727_);
v___x_2759_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2759_, 0, v___x_2757_);
lean_ctor_set(v___x_2759_, 1, v___x_2758_);
v___x_2760_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2710_, v___x_2759_, v___y_2729_, v___y_2730_, v___y_2731_, v___y_2732_);
return v___x_2760_;
}
else
{
lean_object* v_options_2761_; uint8_t v_hasTrace_2762_; 
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_inst_2709_);
v_options_2761_ = lean_ctor_get(v___y_2731_, 2);
v_hasTrace_2762_ = lean_ctor_get_uint8(v_options_2761_, sizeof(void*)*1);
if (v_hasTrace_2762_ == 0)
{
lean_object* v___x_2764_; 
lean_dec(v_cls_2711_);
if (v_isShared_2751_ == 0)
{
lean_ctor_set(v___x_2750_, 0, v_a_2727_);
v___x_2764_ = v___x_2750_;
goto v_reusejp_2763_;
}
else
{
lean_object* v_reuseFailAlloc_2765_; 
v_reuseFailAlloc_2765_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2765_, 0, v_a_2727_);
v___x_2764_ = v_reuseFailAlloc_2765_;
goto v_reusejp_2763_;
}
v_reusejp_2763_:
{
return v___x_2764_;
}
}
else
{
lean_object* v_inheritedTraceOptions_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; uint8_t v___x_2769_; 
v_inheritedTraceOptions_2766_ = lean_ctor_get(v___y_2731_, 13);
v___x_2767_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__4));
lean_inc(v_cls_2711_);
v___x_2768_ = l_Lean_Name_append(v___x_2767_, v_cls_2711_);
v___x_2769_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2766_, v_options_2761_, v___x_2768_);
lean_dec(v___x_2768_);
if (v___x_2769_ == 0)
{
lean_object* v___x_2771_; 
lean_dec(v_cls_2711_);
if (v_isShared_2751_ == 0)
{
lean_ctor_set(v___x_2750_, 0, v_a_2727_);
v___x_2771_ = v___x_2750_;
goto v_reusejp_2770_;
}
else
{
lean_object* v_reuseFailAlloc_2772_; 
v_reuseFailAlloc_2772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2772_, 0, v_a_2727_);
v___x_2771_ = v_reuseFailAlloc_2772_;
goto v_reusejp_2770_;
}
v_reusejp_2770_:
{
return v___x_2771_;
}
}
else
{
lean_object* v___x_2773_; lean_object* v___x_2774_; 
lean_del_object(v___x_2750_);
v___x_2773_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6);
v___x_2774_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2711_, v___x_2773_, v___y_2729_, v___y_2730_, v___y_2731_, v___y_2732_);
if (lean_obj_tag(v___x_2774_) == 0)
{
lean_object* v___x_2776_; uint8_t v_isShared_2777_; uint8_t v_isSharedCheck_2781_; 
v_isSharedCheck_2781_ = !lean_is_exclusive(v___x_2774_);
if (v_isSharedCheck_2781_ == 0)
{
lean_object* v_unused_2782_; 
v_unused_2782_ = lean_ctor_get(v___x_2774_, 0);
lean_dec(v_unused_2782_);
v___x_2776_ = v___x_2774_;
v_isShared_2777_ = v_isSharedCheck_2781_;
goto v_resetjp_2775_;
}
else
{
lean_dec(v___x_2774_);
v___x_2776_ = lean_box(0);
v_isShared_2777_ = v_isSharedCheck_2781_;
goto v_resetjp_2775_;
}
v_resetjp_2775_:
{
lean_object* v___x_2779_; 
if (v_isShared_2777_ == 0)
{
lean_ctor_set(v___x_2776_, 0, v_a_2727_);
v___x_2779_ = v___x_2776_;
goto v_reusejp_2778_;
}
else
{
lean_object* v_reuseFailAlloc_2780_; 
v_reuseFailAlloc_2780_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2780_, 0, v_a_2727_);
v___x_2779_ = v_reuseFailAlloc_2780_;
goto v_reusejp_2778_;
}
v_reusejp_2778_:
{
return v___x_2779_;
}
}
}
else
{
lean_object* v_a_2783_; lean_object* v___x_2785_; uint8_t v_isShared_2786_; uint8_t v_isSharedCheck_2790_; 
lean_dec(v_a_2727_);
v_a_2783_ = lean_ctor_get(v___x_2774_, 0);
v_isSharedCheck_2790_ = !lean_is_exclusive(v___x_2774_);
if (v_isSharedCheck_2790_ == 0)
{
v___x_2785_ = v___x_2774_;
v_isShared_2786_ = v_isSharedCheck_2790_;
goto v_resetjp_2784_;
}
else
{
lean_inc(v_a_2783_);
lean_dec(v___x_2774_);
v___x_2785_ = lean_box(0);
v_isShared_2786_ = v_isSharedCheck_2790_;
goto v_resetjp_2784_;
}
v_resetjp_2784_:
{
lean_object* v___x_2788_; 
if (v_isShared_2786_ == 0)
{
v___x_2788_ = v___x_2785_;
goto v_reusejp_2787_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v_a_2783_);
v___x_2788_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2787_;
}
v_reusejp_2787_:
{
return v___x_2788_;
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
lean_object* v_a_2792_; lean_object* v___x_2794_; uint8_t v_isShared_2795_; uint8_t v_isSharedCheck_2799_; 
lean_dec(v_a_2727_);
lean_dec(v_cls_2711_);
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_inst_2709_);
v_a_2792_ = lean_ctor_get(v___x_2747_, 0);
v_isSharedCheck_2799_ = !lean_is_exclusive(v___x_2747_);
if (v_isSharedCheck_2799_ == 0)
{
v___x_2794_ = v___x_2747_;
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
else
{
lean_inc(v_a_2792_);
lean_dec(v___x_2747_);
v___x_2794_ = lean_box(0);
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
v_resetjp_2793_:
{
lean_object* v___x_2797_; 
if (v_isShared_2795_ == 0)
{
v___x_2797_ = v___x_2794_;
goto v_reusejp_2796_;
}
else
{
lean_object* v_reuseFailAlloc_2798_; 
v_reuseFailAlloc_2798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2798_, 0, v_a_2792_);
v___x_2797_ = v_reuseFailAlloc_2798_;
goto v_reusejp_2796_;
}
v_reusejp_2796_:
{
return v___x_2797_;
}
}
}
}
}
else
{
lean_object* v___x_2816_; 
lean_dec(v_a_2726_);
lean_dec(v_cls_2711_);
lean_inc_ref(v_inst_2709_);
v___x_2816_ = l_Lean_Meta_whnfI(v_inst_2709_, v___y_2716_, v___y_2717_, v___y_2718_, v___y_2719_);
if (lean_obj_tag(v___x_2816_) == 0)
{
lean_object* v_a_2817_; lean_object* v_dummy_2818_; lean_object* v_nargs_2819_; lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v___x_2822_; lean_object* v___x_2823_; 
v_a_2817_ = lean_ctor_get(v___x_2816_, 0);
lean_inc(v_a_2817_);
lean_dec_ref_known(v___x_2816_, 1);
v_dummy_2818_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11);
v_nargs_2819_ = l_Lean_Expr_getAppNumArgs(v_a_2817_);
lean_inc(v_nargs_2819_);
v___x_2820_ = lean_mk_array(v_nargs_2819_, v_dummy_2818_);
v___x_2821_ = lean_unsigned_to_nat(1u);
v___x_2822_ = lean_nat_sub(v_nargs_2819_, v___x_2821_);
lean_dec(v_nargs_2819_);
v___x_2823_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__17(v_val_2713_, v_trace_2710_, v___x_2714_, v_expectedType_2708_, v_inst_2709_, v_a_2817_, v___x_2820_, v___x_2822_, v___y_2716_, v___y_2717_, v___y_2718_, v___y_2719_);
return v___x_2823_;
}
else
{
lean_dec(v_val_2713_);
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_inst_2709_);
lean_dec_ref(v_expectedType_2708_);
return v___x_2816_;
}
}
}
else
{
lean_object* v_a_2824_; lean_object* v___x_2826_; uint8_t v_isShared_2827_; uint8_t v_isSharedCheck_2831_; 
lean_dec(v_val_2713_);
lean_dec(v_cls_2711_);
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_inst_2709_);
lean_dec_ref(v_expectedType_2708_);
v_a_2824_ = lean_ctor_get(v___x_2725_, 0);
v_isSharedCheck_2831_ = !lean_is_exclusive(v___x_2725_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2826_ = v___x_2725_;
v_isShared_2827_ = v_isSharedCheck_2831_;
goto v_resetjp_2825_;
}
else
{
lean_inc(v_a_2824_);
lean_dec(v___x_2725_);
v___x_2826_ = lean_box(0);
v_isShared_2827_ = v_isSharedCheck_2831_;
goto v_resetjp_2825_;
}
v_resetjp_2825_:
{
lean_object* v___x_2829_; 
if (v_isShared_2827_ == 0)
{
v___x_2829_ = v___x_2826_;
goto v_reusejp_2828_;
}
else
{
lean_object* v_reuseFailAlloc_2830_; 
v_reuseFailAlloc_2830_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2830_, 0, v_a_2824_);
v___x_2829_ = v_reuseFailAlloc_2830_;
goto v_reusejp_2828_;
}
v_reusejp_2828_:
{
return v___x_2829_;
}
}
}
}
else
{
lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; 
lean_dec(v_val_2713_);
lean_dec(v_cls_2711_);
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_expectedType_2708_);
v___x_2832_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
lean_inc_ref(v_inst_2709_);
v___x_2833_ = l_Lean_indentExpr(v_inst_2709_);
v___x_2834_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2834_, 0, v___x_2832_);
lean_ctor_set(v___x_2834_, 1, v___x_2833_);
v___x_2835_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13);
v___x_2836_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2836_, 0, v___x_2834_);
lean_ctor_set(v___x_2836_, 1, v___x_2835_);
v___x_2837_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(v___x_2836_, v___y_2716_, v___y_2717_, v___y_2718_, v___y_2719_);
if (lean_obj_tag(v___x_2837_) == 0)
{
lean_object* v___x_2839_; uint8_t v_isShared_2840_; uint8_t v_isSharedCheck_2844_; 
v_isSharedCheck_2844_ = !lean_is_exclusive(v___x_2837_);
if (v_isSharedCheck_2844_ == 0)
{
lean_object* v_unused_2845_; 
v_unused_2845_ = lean_ctor_get(v___x_2837_, 0);
lean_dec(v_unused_2845_);
v___x_2839_ = v___x_2837_;
v_isShared_2840_ = v_isSharedCheck_2844_;
goto v_resetjp_2838_;
}
else
{
lean_dec(v___x_2837_);
v___x_2839_ = lean_box(0);
v_isShared_2840_ = v_isSharedCheck_2844_;
goto v_resetjp_2838_;
}
v_resetjp_2838_:
{
lean_object* v___x_2842_; 
if (v_isShared_2840_ == 0)
{
lean_ctor_set(v___x_2839_, 0, v_inst_2709_);
v___x_2842_ = v___x_2839_;
goto v_reusejp_2841_;
}
else
{
lean_object* v_reuseFailAlloc_2843_; 
v_reuseFailAlloc_2843_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2843_, 0, v_inst_2709_);
v___x_2842_ = v_reuseFailAlloc_2843_;
goto v_reusejp_2841_;
}
v_reusejp_2841_:
{
return v___x_2842_;
}
}
}
else
{
lean_object* v_a_2846_; lean_object* v___x_2848_; uint8_t v_isShared_2849_; uint8_t v_isSharedCheck_2853_; 
lean_dec_ref(v_inst_2709_);
v_a_2846_ = lean_ctor_get(v___x_2837_, 0);
v_isSharedCheck_2853_ = !lean_is_exclusive(v___x_2837_);
if (v_isSharedCheck_2853_ == 0)
{
v___x_2848_ = v___x_2837_;
v_isShared_2849_ = v_isSharedCheck_2853_;
goto v_resetjp_2847_;
}
else
{
lean_inc(v_a_2846_);
lean_dec(v___x_2837_);
v___x_2848_ = lean_box(0);
v_isShared_2849_ = v_isSharedCheck_2853_;
goto v_resetjp_2847_;
}
v_resetjp_2847_:
{
lean_object* v___x_2851_; 
if (v_isShared_2849_ == 0)
{
v___x_2851_ = v___x_2848_;
goto v_reusejp_2850_;
}
else
{
lean_object* v_reuseFailAlloc_2852_; 
v_reuseFailAlloc_2852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2852_, 0, v_a_2846_);
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
else
{
lean_object* v_a_2854_; lean_object* v___x_2856_; uint8_t v_isShared_2857_; uint8_t v_isSharedCheck_2861_; 
lean_dec(v_val_2713_);
lean_dec(v_cls_2711_);
lean_dec_ref(v_trace_2710_);
lean_dec_ref(v_inst_2709_);
lean_dec_ref(v_expectedType_2708_);
v_a_2854_ = lean_ctor_get(v___x_2721_, 0);
v_isSharedCheck_2861_ = !lean_is_exclusive(v___x_2721_);
if (v_isSharedCheck_2861_ == 0)
{
v___x_2856_ = v___x_2721_;
v_isShared_2857_ = v_isSharedCheck_2861_;
goto v_resetjp_2855_;
}
else
{
lean_inc(v_a_2854_);
lean_dec(v___x_2721_);
v___x_2856_ = lean_box(0);
v_isShared_2857_ = v_isSharedCheck_2861_;
goto v_resetjp_2855_;
}
v_resetjp_2855_:
{
lean_object* v___x_2859_; 
if (v_isShared_2857_ == 0)
{
v___x_2859_ = v___x_2856_;
goto v_reusejp_2858_;
}
else
{
lean_object* v_reuseFailAlloc_2860_; 
v_reuseFailAlloc_2860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2860_, 0, v_a_2854_);
v___x_2859_ = v_reuseFailAlloc_2860_;
goto v_reusejp_2858_;
}
v_reusejp_2858_:
{
return v___x_2859_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(lean_object* v_inst_2862_, lean_object* v_expectedType_2863_, uint8_t v_root_2864_, lean_object* v_trace_2865_, lean_object* v_a_2866_, lean_object* v_a_2867_, lean_object* v_a_2868_, lean_object* v_a_2869_){
_start:
{
lean_object* v_options_2871_; lean_object* v_keyedConfig_2872_; uint8_t v_trackZetaDelta_2873_; lean_object* v_zetaDeltaSet_2874_; lean_object* v_lctx_2875_; lean_object* v_localInstances_2876_; lean_object* v_defEqCtx_x3f_2877_; lean_object* v_synthPendingDepth_2878_; lean_object* v_customCanUnfoldPredicate_x3f_2879_; uint8_t v_univApprox_2880_; uint8_t v_inTypeClassResolution_2881_; uint8_t v_cacheInferType_2882_; lean_object* v_ref_2883_; lean_object* v_inheritedTraceOptions_2884_; uint8_t v_hasTrace_2885_; lean_object* v_cls_2886_; lean_object* v___y_2888_; lean_object* v___y_2889_; lean_object* v___y_2890_; lean_object* v___y_2891_; lean_object* v___y_2892_; uint8_t v___x_2959_; lean_object* v___x_2960_; lean_object* v___x_2961_; 
v_options_2871_ = lean_ctor_get(v_a_2868_, 2);
v_keyedConfig_2872_ = lean_ctor_get(v_a_2866_, 0);
v_trackZetaDelta_2873_ = lean_ctor_get_uint8(v_a_2866_, sizeof(void*)*7);
v_zetaDeltaSet_2874_ = lean_ctor_get(v_a_2866_, 1);
v_lctx_2875_ = lean_ctor_get(v_a_2866_, 2);
v_localInstances_2876_ = lean_ctor_get(v_a_2866_, 3);
v_defEqCtx_x3f_2877_ = lean_ctor_get(v_a_2866_, 4);
v_synthPendingDepth_2878_ = lean_ctor_get(v_a_2866_, 5);
v_customCanUnfoldPredicate_x3f_2879_ = lean_ctor_get(v_a_2866_, 6);
v_univApprox_2880_ = lean_ctor_get_uint8(v_a_2866_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2881_ = lean_ctor_get_uint8(v_a_2866_, sizeof(void*)*7 + 2);
v_cacheInferType_2882_ = lean_ctor_get_uint8(v_a_2866_, sizeof(void*)*7 + 3);
v_ref_2883_ = lean_ctor_get(v_a_2868_, 5);
v_inheritedTraceOptions_2884_ = lean_ctor_get(v_a_2868_, 13);
v_hasTrace_2885_ = lean_ctor_get_uint8(v_options_2871_, sizeof(void*)*1);
v_cls_2886_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn___closed__2_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_));
v___x_2959_ = 2;
lean_inc_ref(v_keyedConfig_2872_);
v___x_2960_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2959_, v_keyedConfig_2872_);
lean_inc(v_customCanUnfoldPredicate_x3f_2879_);
lean_inc(v_synthPendingDepth_2878_);
lean_inc(v_defEqCtx_x3f_2877_);
lean_inc_ref(v_localInstances_2876_);
lean_inc_ref(v_lctx_2875_);
lean_inc(v_zetaDeltaSet_2874_);
lean_inc_ref(v___x_2960_);
v___x_2961_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2961_, 0, v___x_2960_);
lean_ctor_set(v___x_2961_, 1, v_zetaDeltaSet_2874_);
lean_ctor_set(v___x_2961_, 2, v_lctx_2875_);
lean_ctor_set(v___x_2961_, 3, v_localInstances_2876_);
lean_ctor_set(v___x_2961_, 4, v_defEqCtx_x3f_2877_);
lean_ctor_set(v___x_2961_, 5, v_synthPendingDepth_2878_);
lean_ctor_set(v___x_2961_, 6, v_customCanUnfoldPredicate_x3f_2879_);
lean_ctor_set_uint8(v___x_2961_, sizeof(void*)*7, v_trackZetaDelta_2873_);
lean_ctor_set_uint8(v___x_2961_, sizeof(void*)*7 + 1, v_univApprox_2880_);
lean_ctor_set_uint8(v___x_2961_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2881_);
lean_ctor_set_uint8(v___x_2961_, sizeof(void*)*7 + 3, v_cacheInferType_2882_);
if (v_hasTrace_2885_ == 0)
{
lean_object* v___x_2962_; 
lean_inc_ref(v_expectedType_2863_);
v___x_2962_ = l_Lean_Meta_isClass_x3f(v_expectedType_2863_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_2962_) == 0)
{
lean_object* v_a_2963_; 
v_a_2963_ = lean_ctor_get(v___x_2962_, 0);
lean_inc(v_a_2963_);
lean_dec_ref_known(v___x_2962_, 1);
if (lean_obj_tag(v_a_2963_) == 1)
{
lean_object* v_val_2964_; lean_object* v___x_2965_; 
v_val_2964_ = lean_ctor_get(v_a_2963_, 0);
lean_inc(v_val_2964_);
lean_dec_ref_known(v_a_2963_, 1);
lean_inc_ref(v_expectedType_2863_);
v___x_2965_ = l_Lean_Meta_isProp(v_expectedType_2863_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_2965_) == 0)
{
lean_object* v_a_2966_; uint8_t v___x_2967_; 
v_a_2966_ = lean_ctor_get(v___x_2965_, 0);
lean_inc(v_a_2966_);
lean_dec_ref_known(v___x_2965_, 1);
v___x_2967_ = lean_unbox(v_a_2966_);
lean_dec(v_a_2966_);
if (v___x_2967_ == 0)
{
lean_object* v___x_2968_; lean_object* v___x_2969_; 
v___x_2968_ = lean_box(0);
lean_inc_ref(v_expectedType_2863_);
v___x_2969_ = l_Lean_Meta_trySynthInstance(v_expectedType_2863_, v___x_2968_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_2969_) == 0)
{
lean_object* v_a_2970_; 
v_a_2970_ = lean_ctor_get(v___x_2969_, 0);
lean_inc(v_a_2970_);
lean_dec_ref_known(v___x_2969_, 1);
if (lean_obj_tag(v_a_2970_) == 1)
{
lean_object* v_a_2971_; lean_object* v___y_2973_; lean_object* v_keyedConfig_2974_; uint8_t v_trackZetaDelta_2975_; lean_object* v_zetaDeltaSet_2976_; lean_object* v_lctx_2977_; lean_object* v_localInstances_2978_; lean_object* v_defEqCtx_x3f_2979_; lean_object* v_synthPendingDepth_2980_; lean_object* v_customCanUnfoldPredicate_x3f_2981_; uint8_t v_univApprox_2982_; uint8_t v_inTypeClassResolution_2983_; uint8_t v_cacheInferType_2984_; lean_object* v___y_2985_; lean_object* v___y_2986_; lean_object* v___y_2987_; 
lean_dec(v_val_2964_);
v_a_2971_ = lean_ctor_get(v_a_2970_, 0);
lean_inc(v_a_2971_);
lean_dec_ref_known(v_a_2970_, 1);
if (v_root_2864_ == 0)
{
lean_dec_ref(v_expectedType_2863_);
v___y_2973_ = v___x_2961_;
v_keyedConfig_2974_ = v___x_2960_;
v_trackZetaDelta_2975_ = v_trackZetaDelta_2873_;
v_zetaDeltaSet_2976_ = v_zetaDeltaSet_2874_;
v_lctx_2977_ = v_lctx_2875_;
v_localInstances_2978_ = v_localInstances_2876_;
v_defEqCtx_x3f_2979_ = v_defEqCtx_x3f_2877_;
v_synthPendingDepth_2980_ = v_synthPendingDepth_2878_;
v_customCanUnfoldPredicate_x3f_2981_ = v_customCanUnfoldPredicate_x3f_2879_;
v_univApprox_2982_ = v_univApprox_2880_;
v_inTypeClassResolution_2983_ = v_inTypeClassResolution_2881_;
v_cacheInferType_2984_ = v_cacheInferType_2882_;
v___y_2985_ = v_a_2867_;
v___y_2986_ = v_a_2868_;
v___y_2987_ = v_a_2869_;
goto v___jp_2972_;
}
else
{
lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; 
v___x_3043_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing;
v___x_3044_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8);
v___x_3045_ = l_Lean_MessageData_ofExpr(v_expectedType_2863_);
v___x_3046_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3046_, 0, v___x_3044_);
lean_ctor_set(v___x_3046_, 1, v___x_3045_);
v___x_3047_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10);
v___x_3048_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3048_, 0, v___x_3046_);
lean_ctor_set(v___x_3048_, 1, v___x_3047_);
lean_inc(v_ref_2883_);
v___x_3049_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(v___x_3043_, v_ref_2883_, v___x_3048_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3049_) == 0)
{
lean_dec_ref_known(v___x_3049_, 1);
v___y_2973_ = v___x_2961_;
v_keyedConfig_2974_ = v___x_2960_;
v_trackZetaDelta_2975_ = v_trackZetaDelta_2873_;
v_zetaDeltaSet_2976_ = v_zetaDeltaSet_2874_;
v_lctx_2977_ = v_lctx_2875_;
v_localInstances_2978_ = v_localInstances_2876_;
v_defEqCtx_x3f_2979_ = v_defEqCtx_x3f_2877_;
v_synthPendingDepth_2980_ = v_synthPendingDepth_2878_;
v_customCanUnfoldPredicate_x3f_2981_ = v_customCanUnfoldPredicate_x3f_2879_;
v_univApprox_2982_ = v_univApprox_2880_;
v_inTypeClassResolution_2983_ = v_inTypeClassResolution_2881_;
v_cacheInferType_2984_ = v_cacheInferType_2882_;
v___y_2985_ = v_a_2867_;
v___y_2986_ = v_a_2868_;
v___y_2987_ = v_a_2869_;
goto v___jp_2972_;
}
else
{
lean_object* v_a_3050_; lean_object* v___x_3052_; uint8_t v_isShared_3053_; uint8_t v_isSharedCheck_3057_; 
lean_dec(v_a_2971_);
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v___x_2960_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_inst_2862_);
v_a_3050_ = lean_ctor_get(v___x_3049_, 0);
v_isSharedCheck_3057_ = !lean_is_exclusive(v___x_3049_);
if (v_isSharedCheck_3057_ == 0)
{
v___x_3052_ = v___x_3049_;
v_isShared_3053_ = v_isSharedCheck_3057_;
goto v_resetjp_3051_;
}
else
{
lean_inc(v_a_3050_);
lean_dec(v___x_3049_);
v___x_3052_ = lean_box(0);
v_isShared_3053_ = v_isSharedCheck_3057_;
goto v_resetjp_3051_;
}
v_resetjp_3051_:
{
lean_object* v___x_3055_; 
if (v_isShared_3053_ == 0)
{
v___x_3055_ = v___x_3052_;
goto v_reusejp_3054_;
}
else
{
lean_object* v_reuseFailAlloc_3056_; 
v_reuseFailAlloc_3056_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3056_, 0, v_a_3050_);
v___x_3055_ = v_reuseFailAlloc_3056_;
goto v_reusejp_3054_;
}
v_reusejp_3054_:
{
return v___x_3055_;
}
}
}
}
v___jp_2972_:
{
uint8_t v___x_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; lean_object* v___x_2991_; 
v___x_2988_ = 1;
v___x_2989_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2988_, v_keyedConfig_2974_);
lean_inc(v_customCanUnfoldPredicate_x3f_2981_);
lean_inc(v_synthPendingDepth_2980_);
lean_inc(v_defEqCtx_x3f_2979_);
lean_inc_ref(v_localInstances_2978_);
lean_inc_ref(v_lctx_2977_);
lean_inc(v_zetaDeltaSet_2976_);
v___x_2990_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2990_, 0, v___x_2989_);
lean_ctor_set(v___x_2990_, 1, v_zetaDeltaSet_2976_);
lean_ctor_set(v___x_2990_, 2, v_lctx_2977_);
lean_ctor_set(v___x_2990_, 3, v_localInstances_2978_);
lean_ctor_set(v___x_2990_, 4, v_defEqCtx_x3f_2979_);
lean_ctor_set(v___x_2990_, 5, v_synthPendingDepth_2980_);
lean_ctor_set(v___x_2990_, 6, v_customCanUnfoldPredicate_x3f_2981_);
lean_ctor_set_uint8(v___x_2990_, sizeof(void*)*7, v_trackZetaDelta_2975_);
lean_ctor_set_uint8(v___x_2990_, sizeof(void*)*7 + 1, v_univApprox_2982_);
lean_ctor_set_uint8(v___x_2990_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2983_);
lean_ctor_set_uint8(v___x_2990_, sizeof(void*)*7 + 3, v_cacheInferType_2984_);
lean_inc(v_a_2971_);
lean_inc_ref(v_inst_2862_);
v___x_2991_ = l_Lean_Meta_isExprDefEq(v_inst_2862_, v_a_2971_, v___x_2990_, v___y_2985_, v___y_2986_, v___y_2987_);
lean_dec_ref_known(v___x_2990_, 7);
if (lean_obj_tag(v___x_2991_) == 0)
{
lean_object* v_a_2992_; lean_object* v___x_2994_; uint8_t v_isShared_2995_; uint8_t v_isSharedCheck_3034_; 
v_a_2992_ = lean_ctor_get(v___x_2991_, 0);
v_isSharedCheck_3034_ = !lean_is_exclusive(v___x_2991_);
if (v_isSharedCheck_3034_ == 0)
{
v___x_2994_ = v___x_2991_;
v_isShared_2995_ = v_isSharedCheck_3034_;
goto v_resetjp_2993_;
}
else
{
lean_inc(v_a_2992_);
lean_dec(v___x_2991_);
v___x_2994_ = lean_box(0);
v_isShared_2995_ = v_isSharedCheck_3034_;
goto v_resetjp_2993_;
}
v_resetjp_2993_:
{
uint8_t v___x_2996_; 
v___x_2996_ = lean_unbox(v_a_2992_);
lean_dec(v_a_2992_);
if (v___x_2996_ == 0)
{
lean_object* v___x_2997_; lean_object* v___x_2998_; lean_object* v___x_2999_; lean_object* v___x_3000_; lean_object* v___x_3001_; lean_object* v___x_3002_; lean_object* v___x_3003_; lean_object* v___x_3004_; 
lean_del_object(v___x_2994_);
v___x_2997_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
v___x_2998_ = l_Lean_indentExpr(v_inst_2862_);
v___x_2999_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2999_, 0, v___x_2997_);
lean_ctor_set(v___x_2999_, 1, v___x_2998_);
v___x_3000_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3);
v___x_3001_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3001_, 0, v___x_2999_);
lean_ctor_set(v___x_3001_, 1, v___x_3000_);
v___x_3002_ = l_Lean_indentExpr(v_a_2971_);
v___x_3003_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3003_, 0, v___x_3001_);
lean_ctor_set(v___x_3003_, 1, v___x_3002_);
v___x_3004_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2865_, v___x_3003_, v___y_2973_, v___y_2985_, v___y_2986_, v___y_2987_);
lean_dec_ref(v___y_2973_);
return v___x_3004_;
}
else
{
lean_object* v_options_3005_; uint8_t v_hasTrace_3006_; 
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_inst_2862_);
v_options_3005_ = lean_ctor_get(v___y_2986_, 2);
v_hasTrace_3006_ = lean_ctor_get_uint8(v_options_3005_, sizeof(void*)*1);
if (v_hasTrace_3006_ == 0)
{
lean_object* v___x_3008_; 
lean_dec_ref(v___y_2973_);
if (v_isShared_2995_ == 0)
{
lean_ctor_set(v___x_2994_, 0, v_a_2971_);
v___x_3008_ = v___x_2994_;
goto v_reusejp_3007_;
}
else
{
lean_object* v_reuseFailAlloc_3009_; 
v_reuseFailAlloc_3009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3009_, 0, v_a_2971_);
v___x_3008_ = v_reuseFailAlloc_3009_;
goto v_reusejp_3007_;
}
v_reusejp_3007_:
{
return v___x_3008_;
}
}
else
{
lean_object* v_inheritedTraceOptions_3010_; lean_object* v___x_3011_; uint8_t v___x_3012_; 
v_inheritedTraceOptions_3010_ = lean_ctor_get(v___y_2986_, 13);
v___x_3011_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0);
v___x_3012_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3010_, v_options_3005_, v___x_3011_);
if (v___x_3012_ == 0)
{
lean_object* v___x_3014_; 
lean_dec_ref(v___y_2973_);
if (v_isShared_2995_ == 0)
{
lean_ctor_set(v___x_2994_, 0, v_a_2971_);
v___x_3014_ = v___x_2994_;
goto v_reusejp_3013_;
}
else
{
lean_object* v_reuseFailAlloc_3015_; 
v_reuseFailAlloc_3015_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3015_, 0, v_a_2971_);
v___x_3014_ = v_reuseFailAlloc_3015_;
goto v_reusejp_3013_;
}
v_reusejp_3013_:
{
return v___x_3014_;
}
}
else
{
lean_object* v___x_3016_; lean_object* v___x_3017_; 
lean_del_object(v___x_2994_);
v___x_3016_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6);
v___x_3017_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2886_, v___x_3016_, v___y_2973_, v___y_2985_, v___y_2986_, v___y_2987_);
lean_dec_ref(v___y_2973_);
if (lean_obj_tag(v___x_3017_) == 0)
{
lean_object* v___x_3019_; uint8_t v_isShared_3020_; uint8_t v_isSharedCheck_3024_; 
v_isSharedCheck_3024_ = !lean_is_exclusive(v___x_3017_);
if (v_isSharedCheck_3024_ == 0)
{
lean_object* v_unused_3025_; 
v_unused_3025_ = lean_ctor_get(v___x_3017_, 0);
lean_dec(v_unused_3025_);
v___x_3019_ = v___x_3017_;
v_isShared_3020_ = v_isSharedCheck_3024_;
goto v_resetjp_3018_;
}
else
{
lean_dec(v___x_3017_);
v___x_3019_ = lean_box(0);
v_isShared_3020_ = v_isSharedCheck_3024_;
goto v_resetjp_3018_;
}
v_resetjp_3018_:
{
lean_object* v___x_3022_; 
if (v_isShared_3020_ == 0)
{
lean_ctor_set(v___x_3019_, 0, v_a_2971_);
v___x_3022_ = v___x_3019_;
goto v_reusejp_3021_;
}
else
{
lean_object* v_reuseFailAlloc_3023_; 
v_reuseFailAlloc_3023_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3023_, 0, v_a_2971_);
v___x_3022_ = v_reuseFailAlloc_3023_;
goto v_reusejp_3021_;
}
v_reusejp_3021_:
{
return v___x_3022_;
}
}
}
else
{
lean_object* v_a_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3033_; 
lean_dec(v_a_2971_);
v_a_3026_ = lean_ctor_get(v___x_3017_, 0);
v_isSharedCheck_3033_ = !lean_is_exclusive(v___x_3017_);
if (v_isSharedCheck_3033_ == 0)
{
v___x_3028_ = v___x_3017_;
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_a_3026_);
lean_dec(v___x_3017_);
v___x_3028_ = lean_box(0);
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
v_resetjp_3027_:
{
lean_object* v___x_3031_; 
if (v_isShared_3029_ == 0)
{
v___x_3031_ = v___x_3028_;
goto v_reusejp_3030_;
}
else
{
lean_object* v_reuseFailAlloc_3032_; 
v_reuseFailAlloc_3032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3032_, 0, v_a_3026_);
v___x_3031_ = v_reuseFailAlloc_3032_;
goto v_reusejp_3030_;
}
v_reusejp_3030_:
{
return v___x_3031_;
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
lean_object* v_a_3035_; lean_object* v___x_3037_; uint8_t v_isShared_3038_; uint8_t v_isSharedCheck_3042_; 
lean_dec_ref(v___y_2973_);
lean_dec(v_a_2971_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_inst_2862_);
v_a_3035_ = lean_ctor_get(v___x_2991_, 0);
v_isSharedCheck_3042_ = !lean_is_exclusive(v___x_2991_);
if (v_isSharedCheck_3042_ == 0)
{
v___x_3037_ = v___x_2991_;
v_isShared_3038_ = v_isSharedCheck_3042_;
goto v_resetjp_3036_;
}
else
{
lean_inc(v_a_3035_);
lean_dec(v___x_2991_);
v___x_3037_ = lean_box(0);
v_isShared_3038_ = v_isSharedCheck_3042_;
goto v_resetjp_3036_;
}
v_resetjp_3036_:
{
lean_object* v___x_3040_; 
if (v_isShared_3038_ == 0)
{
v___x_3040_ = v___x_3037_;
goto v_reusejp_3039_;
}
else
{
lean_object* v_reuseFailAlloc_3041_; 
v_reuseFailAlloc_3041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3041_, 0, v_a_3035_);
v___x_3040_ = v_reuseFailAlloc_3041_;
goto v_reusejp_3039_;
}
v_reusejp_3039_:
{
return v___x_3040_;
}
}
}
}
}
else
{
lean_object* v___x_3058_; 
lean_dec(v_a_2970_);
lean_dec_ref(v___x_2960_);
lean_inc_ref(v_inst_2862_);
v___x_3058_ = l_Lean_Meta_whnfI(v_inst_2862_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3058_) == 0)
{
lean_object* v_a_3059_; lean_object* v_dummy_3060_; lean_object* v_nargs_3061_; lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; 
v_a_3059_ = lean_ctor_get(v___x_3058_, 0);
lean_inc(v_a_3059_);
lean_dec_ref_known(v___x_3058_, 1);
v_dummy_3060_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11);
v_nargs_3061_ = l_Lean_Expr_getAppNumArgs(v_a_3059_);
lean_inc(v_nargs_3061_);
v___x_3062_ = lean_mk_array(v_nargs_3061_, v_dummy_3060_);
v___x_3063_ = lean_unsigned_to_nat(1u);
v___x_3064_ = lean_nat_sub(v_nargs_3061_, v___x_3063_);
lean_dec(v_nargs_3061_);
v___x_3065_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__9(v_val_2964_, v_trace_2865_, v_hasTrace_2885_, v_expectedType_2863_, v_inst_2862_, v_a_3059_, v___x_3062_, v___x_3064_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec_ref_known(v___x_2961_, 7);
return v___x_3065_;
}
else
{
lean_dec(v_val_2964_);
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
return v___x_3058_;
}
}
}
else
{
lean_object* v_a_3066_; lean_object* v___x_3068_; uint8_t v_isShared_3069_; uint8_t v_isSharedCheck_3073_; 
lean_dec(v_val_2964_);
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v___x_2960_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3066_ = lean_ctor_get(v___x_2969_, 0);
v_isSharedCheck_3073_ = !lean_is_exclusive(v___x_2969_);
if (v_isSharedCheck_3073_ == 0)
{
v___x_3068_ = v___x_2969_;
v_isShared_3069_ = v_isSharedCheck_3073_;
goto v_resetjp_3067_;
}
else
{
lean_inc(v_a_3066_);
lean_dec(v___x_2969_);
v___x_3068_ = lean_box(0);
v_isShared_3069_ = v_isSharedCheck_3073_;
goto v_resetjp_3067_;
}
v_resetjp_3067_:
{
lean_object* v___x_3071_; 
if (v_isShared_3069_ == 0)
{
v___x_3071_ = v___x_3068_;
goto v_reusejp_3070_;
}
else
{
lean_object* v_reuseFailAlloc_3072_; 
v_reuseFailAlloc_3072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3072_, 0, v_a_3066_);
v___x_3071_ = v_reuseFailAlloc_3072_;
goto v_reusejp_3070_;
}
v_reusejp_3070_:
{
return v___x_3071_;
}
}
}
}
else
{
lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; lean_object* v___x_3079_; 
lean_dec(v_val_2964_);
lean_dec_ref(v___x_2960_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
v___x_3074_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
lean_inc_ref(v_inst_2862_);
v___x_3075_ = l_Lean_indentExpr(v_inst_2862_);
v___x_3076_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3076_, 0, v___x_3074_);
lean_ctor_set(v___x_3076_, 1, v___x_3075_);
v___x_3077_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13);
v___x_3078_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3078_, 0, v___x_3076_);
lean_ctor_set(v___x_3078_, 1, v___x_3077_);
v___x_3079_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(v___x_3078_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec_ref_known(v___x_2961_, 7);
if (lean_obj_tag(v___x_3079_) == 0)
{
lean_object* v___x_3081_; uint8_t v_isShared_3082_; uint8_t v_isSharedCheck_3086_; 
v_isSharedCheck_3086_ = !lean_is_exclusive(v___x_3079_);
if (v_isSharedCheck_3086_ == 0)
{
lean_object* v_unused_3087_; 
v_unused_3087_ = lean_ctor_get(v___x_3079_, 0);
lean_dec(v_unused_3087_);
v___x_3081_ = v___x_3079_;
v_isShared_3082_ = v_isSharedCheck_3086_;
goto v_resetjp_3080_;
}
else
{
lean_dec(v___x_3079_);
v___x_3081_ = lean_box(0);
v_isShared_3082_ = v_isSharedCheck_3086_;
goto v_resetjp_3080_;
}
v_resetjp_3080_:
{
lean_object* v___x_3084_; 
if (v_isShared_3082_ == 0)
{
lean_ctor_set(v___x_3081_, 0, v_inst_2862_);
v___x_3084_ = v___x_3081_;
goto v_reusejp_3083_;
}
else
{
lean_object* v_reuseFailAlloc_3085_; 
v_reuseFailAlloc_3085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3085_, 0, v_inst_2862_);
v___x_3084_ = v_reuseFailAlloc_3085_;
goto v_reusejp_3083_;
}
v_reusejp_3083_:
{
return v___x_3084_;
}
}
}
else
{
lean_object* v_a_3088_; lean_object* v___x_3090_; uint8_t v_isShared_3091_; uint8_t v_isSharedCheck_3095_; 
lean_dec_ref(v_inst_2862_);
v_a_3088_ = lean_ctor_get(v___x_3079_, 0);
v_isSharedCheck_3095_ = !lean_is_exclusive(v___x_3079_);
if (v_isSharedCheck_3095_ == 0)
{
v___x_3090_ = v___x_3079_;
v_isShared_3091_ = v_isSharedCheck_3095_;
goto v_resetjp_3089_;
}
else
{
lean_inc(v_a_3088_);
lean_dec(v___x_3079_);
v___x_3090_ = lean_box(0);
v_isShared_3091_ = v_isSharedCheck_3095_;
goto v_resetjp_3089_;
}
v_resetjp_3089_:
{
lean_object* v___x_3093_; 
if (v_isShared_3091_ == 0)
{
v___x_3093_ = v___x_3090_;
goto v_reusejp_3092_;
}
else
{
lean_object* v_reuseFailAlloc_3094_; 
v_reuseFailAlloc_3094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3094_, 0, v_a_3088_);
v___x_3093_ = v_reuseFailAlloc_3094_;
goto v_reusejp_3092_;
}
v_reusejp_3092_:
{
return v___x_3093_;
}
}
}
}
}
else
{
lean_object* v_a_3096_; lean_object* v___x_3098_; uint8_t v_isShared_3099_; uint8_t v_isSharedCheck_3103_; 
lean_dec(v_val_2964_);
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v___x_2960_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3096_ = lean_ctor_get(v___x_2965_, 0);
v_isSharedCheck_3103_ = !lean_is_exclusive(v___x_2965_);
if (v_isSharedCheck_3103_ == 0)
{
v___x_3098_ = v___x_2965_;
v_isShared_3099_ = v_isSharedCheck_3103_;
goto v_resetjp_3097_;
}
else
{
lean_inc(v_a_3096_);
lean_dec(v___x_2965_);
v___x_3098_ = lean_box(0);
v_isShared_3099_ = v_isSharedCheck_3103_;
goto v_resetjp_3097_;
}
v_resetjp_3097_:
{
lean_object* v___x_3101_; 
if (v_isShared_3099_ == 0)
{
v___x_3101_ = v___x_3098_;
goto v_reusejp_3100_;
}
else
{
lean_object* v_reuseFailAlloc_3102_; 
v_reuseFailAlloc_3102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3102_, 0, v_a_3096_);
v___x_3101_ = v_reuseFailAlloc_3102_;
goto v_reusejp_3100_;
}
v_reusejp_3100_:
{
return v___x_3101_;
}
}
}
}
else
{
lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; 
lean_dec(v_a_2963_);
lean_dec_ref(v___x_2960_);
lean_dec_ref(v_inst_2862_);
v___x_3104_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2);
v___x_3105_ = l_Lean_indentExpr(v_expectedType_2863_);
v___x_3106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3106_, 0, v___x_3104_);
lean_ctor_set(v___x_3106_, 1, v___x_3105_);
v___x_3107_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2865_, v___x_3106_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec_ref_known(v___x_2961_, 7);
return v___x_3107_;
}
}
else
{
lean_object* v_a_3108_; lean_object* v___x_3110_; uint8_t v_isShared_3111_; uint8_t v_isSharedCheck_3115_; 
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v___x_2960_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3108_ = lean_ctor_get(v___x_2962_, 0);
v_isSharedCheck_3115_ = !lean_is_exclusive(v___x_2962_);
if (v_isSharedCheck_3115_ == 0)
{
v___x_3110_ = v___x_2962_;
v_isShared_3111_ = v_isSharedCheck_3115_;
goto v_resetjp_3109_;
}
else
{
lean_inc(v_a_3108_);
lean_dec(v___x_2962_);
v___x_3110_ = lean_box(0);
v_isShared_3111_ = v_isSharedCheck_3115_;
goto v_resetjp_3109_;
}
v_resetjp_3109_:
{
lean_object* v___x_3113_; 
if (v_isShared_3111_ == 0)
{
v___x_3113_ = v___x_3110_;
goto v_reusejp_3112_;
}
else
{
lean_object* v_reuseFailAlloc_3114_; 
v_reuseFailAlloc_3114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3114_, 0, v_a_3108_);
v___x_3113_ = v_reuseFailAlloc_3114_;
goto v_reusejp_3112_;
}
v_reusejp_3112_:
{
return v___x_3113_;
}
}
}
}
else
{
lean_object* v___f_3116_; lean_object* v___x_3117_; lean_object* v___x_3118_; uint8_t v___x_3119_; lean_object* v___y_3121_; lean_object* v___y_3122_; lean_object* v_a_3123_; lean_object* v___y_3133_; lean_object* v___y_3134_; lean_object* v_a_3135_; lean_object* v___y_3138_; lean_object* v___y_3139_; lean_object* v___y_3140_; lean_object* v___y_3151_; lean_object* v___y_3152_; lean_object* v_a_3153_; lean_object* v___y_3166_; lean_object* v___y_3167_; lean_object* v_a_3168_; lean_object* v___y_3171_; lean_object* v___y_3172_; lean_object* v___y_3173_; 
lean_dec_ref(v___x_2960_);
lean_inc_ref(v_expectedType_2863_);
v___f_3116_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3116_, 0, v_expectedType_2863_);
v___x_3117_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0));
v___x_3118_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0);
v___x_3119_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2884_, v_options_2871_, v___x_3118_);
if (v___x_3119_ == 0)
{
lean_object* v___x_3234_; uint8_t v___x_3235_; 
v___x_3234_ = l_Lean_trace_profiler;
v___x_3235_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_options_2871_, v___x_3234_);
if (v___x_3235_ == 0)
{
lean_object* v___x_3236_; 
lean_dec_ref(v___f_3116_);
lean_inc_ref(v_expectedType_2863_);
v___x_3236_ = l_Lean_Meta_isClass_x3f(v_expectedType_2863_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3236_) == 0)
{
lean_object* v_a_3237_; 
v_a_3237_ = lean_ctor_get(v___x_3236_, 0);
lean_inc(v_a_3237_);
lean_dec_ref_known(v___x_3236_, 1);
if (lean_obj_tag(v_a_3237_) == 1)
{
lean_object* v_val_3238_; lean_object* v___y_3240_; lean_object* v___y_3241_; lean_object* v___y_3242_; lean_object* v___y_3243_; 
v_val_3238_ = lean_ctor_get(v_a_3237_, 0);
lean_inc(v_val_3238_);
lean_dec_ref_known(v_a_3237_, 1);
if (v___x_3119_ == 0)
{
v___y_3240_ = v___x_2961_;
v___y_3241_ = v_a_2867_;
v___y_3242_ = v_a_2868_;
v___y_3243_ = v_a_2869_;
goto v___jp_3239_;
}
else
{
lean_object* v___x_3314_; lean_object* v___x_3315_; lean_object* v___x_3316_; lean_object* v___x_3317_; 
v___x_3314_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5);
lean_inc(v_val_3238_);
v___x_3315_ = l_Lean_MessageData_ofName(v_val_3238_);
v___x_3316_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3316_, 0, v___x_3314_);
lean_ctor_set(v___x_3316_, 1, v___x_3315_);
v___x_3317_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2886_, v___x_3316_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3317_) == 0)
{
lean_dec_ref_known(v___x_3317_, 1);
v___y_3240_ = v___x_2961_;
v___y_3241_ = v_a_2867_;
v___y_3242_ = v_a_2868_;
v___y_3243_ = v_a_2869_;
goto v___jp_3239_;
}
else
{
lean_object* v_a_3318_; lean_object* v___x_3320_; uint8_t v_isShared_3321_; uint8_t v_isSharedCheck_3325_; 
lean_dec(v_val_3238_);
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3318_ = lean_ctor_get(v___x_3317_, 0);
v_isSharedCheck_3325_ = !lean_is_exclusive(v___x_3317_);
if (v_isSharedCheck_3325_ == 0)
{
v___x_3320_ = v___x_3317_;
v_isShared_3321_ = v_isSharedCheck_3325_;
goto v_resetjp_3319_;
}
else
{
lean_inc(v_a_3318_);
lean_dec(v___x_3317_);
v___x_3320_ = lean_box(0);
v_isShared_3321_ = v_isSharedCheck_3325_;
goto v_resetjp_3319_;
}
v_resetjp_3319_:
{
lean_object* v___x_3323_; 
if (v_isShared_3321_ == 0)
{
v___x_3323_ = v___x_3320_;
goto v_reusejp_3322_;
}
else
{
lean_object* v_reuseFailAlloc_3324_; 
v_reuseFailAlloc_3324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3324_, 0, v_a_3318_);
v___x_3323_ = v_reuseFailAlloc_3324_;
goto v_reusejp_3322_;
}
v_reusejp_3322_:
{
return v___x_3323_;
}
}
}
}
v___jp_3239_:
{
lean_object* v___x_3244_; 
lean_inc_ref(v_expectedType_2863_);
v___x_3244_ = l_Lean_Meta_isProp(v_expectedType_2863_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
if (lean_obj_tag(v___x_3244_) == 0)
{
lean_object* v_a_3245_; uint8_t v___x_3246_; 
v_a_3245_ = lean_ctor_get(v___x_3244_, 0);
lean_inc(v_a_3245_);
lean_dec_ref_known(v___x_3244_, 1);
v___x_3246_ = lean_unbox(v_a_3245_);
lean_dec(v_a_3245_);
if (v___x_3246_ == 0)
{
lean_object* v___x_3247_; lean_object* v___x_3248_; 
v___x_3247_ = lean_box(0);
lean_inc_ref(v_expectedType_2863_);
v___x_3248_ = l_Lean_Meta_trySynthInstance(v_expectedType_2863_, v___x_3247_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
if (lean_obj_tag(v___x_3248_) == 0)
{
lean_object* v_a_3249_; 
v_a_3249_ = lean_ctor_get(v___x_3248_, 0);
lean_inc(v_a_3249_);
lean_dec_ref_known(v___x_3248_, 1);
if (lean_obj_tag(v_a_3249_) == 1)
{
lean_dec(v_val_3238_);
if (v_root_2864_ == 0)
{
lean_object* v_a_3250_; 
lean_dec_ref(v_expectedType_2863_);
v_a_3250_ = lean_ctor_get(v_a_3249_, 0);
lean_inc(v_a_3250_);
lean_dec_ref_known(v_a_3249_, 1);
v___y_2888_ = v_a_3250_;
v___y_2889_ = v___y_3240_;
v___y_2890_ = v___y_3241_;
v___y_2891_ = v___y_3242_;
v___y_2892_ = v___y_3243_;
goto v___jp_2887_;
}
else
{
lean_object* v_a_3251_; lean_object* v_ref_3252_; lean_object* v___x_3253_; lean_object* v___x_3254_; lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3257_; lean_object* v___x_3258_; lean_object* v___x_3259_; 
v_a_3251_ = lean_ctor_get(v_a_3249_, 0);
lean_inc(v_a_3251_);
lean_dec_ref_known(v_a_3249_, 1);
v_ref_3252_ = lean_ctor_get(v___y_3242_, 5);
v___x_3253_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing;
v___x_3254_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__8);
v___x_3255_ = l_Lean_MessageData_ofExpr(v_expectedType_2863_);
v___x_3256_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3256_, 0, v___x_3254_);
lean_ctor_set(v___x_3256_, 1, v___x_3255_);
v___x_3257_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__10);
v___x_3258_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3258_, 0, v___x_3256_);
lean_ctor_set(v___x_3258_, 1, v___x_3257_);
lean_inc(v_ref_3252_);
v___x_3259_ = lp_mathlib_Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1(v___x_3253_, v_ref_3252_, v___x_3258_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
if (lean_obj_tag(v___x_3259_) == 0)
{
lean_dec_ref_known(v___x_3259_, 1);
v___y_2888_ = v_a_3251_;
v___y_2889_ = v___y_3240_;
v___y_2890_ = v___y_3241_;
v___y_2891_ = v___y_3242_;
v___y_2892_ = v___y_3243_;
goto v___jp_2887_;
}
else
{
lean_object* v_a_3260_; lean_object* v___x_3262_; uint8_t v_isShared_3263_; uint8_t v_isSharedCheck_3267_; 
lean_dec(v_a_3251_);
lean_dec_ref(v___y_3240_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_inst_2862_);
v_a_3260_ = lean_ctor_get(v___x_3259_, 0);
v_isSharedCheck_3267_ = !lean_is_exclusive(v___x_3259_);
if (v_isSharedCheck_3267_ == 0)
{
v___x_3262_ = v___x_3259_;
v_isShared_3263_ = v_isSharedCheck_3267_;
goto v_resetjp_3261_;
}
else
{
lean_inc(v_a_3260_);
lean_dec(v___x_3259_);
v___x_3262_ = lean_box(0);
v_isShared_3263_ = v_isSharedCheck_3267_;
goto v_resetjp_3261_;
}
v_resetjp_3261_:
{
lean_object* v___x_3265_; 
if (v_isShared_3263_ == 0)
{
v___x_3265_ = v___x_3262_;
goto v_reusejp_3264_;
}
else
{
lean_object* v_reuseFailAlloc_3266_; 
v_reuseFailAlloc_3266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3266_, 0, v_a_3260_);
v___x_3265_ = v_reuseFailAlloc_3266_;
goto v_reusejp_3264_;
}
v_reusejp_3264_:
{
return v___x_3265_;
}
}
}
}
}
else
{
lean_object* v___x_3268_; 
lean_dec(v_a_3249_);
lean_inc_ref(v_inst_2862_);
v___x_3268_ = l_Lean_Meta_whnfI(v_inst_2862_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
if (lean_obj_tag(v___x_3268_) == 0)
{
lean_object* v_a_3269_; lean_object* v_dummy_3270_; lean_object* v_nargs_3271_; lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_3274_; lean_object* v___x_3275_; 
v_a_3269_ = lean_ctor_get(v___x_3268_, 0);
lean_inc(v_a_3269_);
lean_dec_ref_known(v___x_3268_, 1);
v_dummy_3270_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__11);
v_nargs_3271_ = l_Lean_Expr_getAppNumArgs(v_a_3269_);
lean_inc(v_nargs_3271_);
v___x_3272_ = lean_mk_array(v_nargs_3271_, v_dummy_3270_);
v___x_3273_ = lean_unsigned_to_nat(1u);
v___x_3274_ = lean_nat_sub(v_nargs_3271_, v___x_3273_);
lean_dec(v_nargs_3271_);
v___x_3275_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15(v_val_3238_, v_trace_2865_, v___x_3235_, v_hasTrace_2885_, v_expectedType_2863_, v_inst_2862_, v_a_3269_, v___x_3272_, v___x_3274_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
lean_dec_ref(v___y_3240_);
return v___x_3275_;
}
else
{
lean_dec_ref(v___y_3240_);
lean_dec(v_val_3238_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
return v___x_3268_;
}
}
}
else
{
lean_object* v_a_3276_; lean_object* v___x_3278_; uint8_t v_isShared_3279_; uint8_t v_isSharedCheck_3283_; 
lean_dec_ref(v___y_3240_);
lean_dec(v_val_3238_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3276_ = lean_ctor_get(v___x_3248_, 0);
v_isSharedCheck_3283_ = !lean_is_exclusive(v___x_3248_);
if (v_isSharedCheck_3283_ == 0)
{
v___x_3278_ = v___x_3248_;
v_isShared_3279_ = v_isSharedCheck_3283_;
goto v_resetjp_3277_;
}
else
{
lean_inc(v_a_3276_);
lean_dec(v___x_3248_);
v___x_3278_ = lean_box(0);
v_isShared_3279_ = v_isSharedCheck_3283_;
goto v_resetjp_3277_;
}
v_resetjp_3277_:
{
lean_object* v___x_3281_; 
if (v_isShared_3279_ == 0)
{
v___x_3281_ = v___x_3278_;
goto v_reusejp_3280_;
}
else
{
lean_object* v_reuseFailAlloc_3282_; 
v_reuseFailAlloc_3282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3282_, 0, v_a_3276_);
v___x_3281_ = v_reuseFailAlloc_3282_;
goto v_reusejp_3280_;
}
v_reusejp_3280_:
{
return v___x_3281_;
}
}
}
}
else
{
lean_object* v___x_3284_; lean_object* v___x_3285_; lean_object* v___x_3286_; lean_object* v___x_3287_; lean_object* v___x_3288_; lean_object* v___x_3289_; 
lean_dec(v_val_3238_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
v___x_3284_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
lean_inc_ref(v_inst_2862_);
v___x_3285_ = l_Lean_indentExpr(v_inst_2862_);
v___x_3286_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3286_, 0, v___x_3284_);
lean_ctor_set(v___x_3286_, 1, v___x_3285_);
v___x_3287_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__13);
v___x_3288_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3288_, 0, v___x_3286_);
lean_ctor_set(v___x_3288_, 1, v___x_3287_);
v___x_3289_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10(v___x_3288_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_);
lean_dec_ref(v___y_3240_);
if (lean_obj_tag(v___x_3289_) == 0)
{
lean_object* v___x_3291_; uint8_t v_isShared_3292_; uint8_t v_isSharedCheck_3296_; 
v_isSharedCheck_3296_ = !lean_is_exclusive(v___x_3289_);
if (v_isSharedCheck_3296_ == 0)
{
lean_object* v_unused_3297_; 
v_unused_3297_ = lean_ctor_get(v___x_3289_, 0);
lean_dec(v_unused_3297_);
v___x_3291_ = v___x_3289_;
v_isShared_3292_ = v_isSharedCheck_3296_;
goto v_resetjp_3290_;
}
else
{
lean_dec(v___x_3289_);
v___x_3291_ = lean_box(0);
v_isShared_3292_ = v_isSharedCheck_3296_;
goto v_resetjp_3290_;
}
v_resetjp_3290_:
{
lean_object* v___x_3294_; 
if (v_isShared_3292_ == 0)
{
lean_ctor_set(v___x_3291_, 0, v_inst_2862_);
v___x_3294_ = v___x_3291_;
goto v_reusejp_3293_;
}
else
{
lean_object* v_reuseFailAlloc_3295_; 
v_reuseFailAlloc_3295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3295_, 0, v_inst_2862_);
v___x_3294_ = v_reuseFailAlloc_3295_;
goto v_reusejp_3293_;
}
v_reusejp_3293_:
{
return v___x_3294_;
}
}
}
else
{
lean_object* v_a_3298_; lean_object* v___x_3300_; uint8_t v_isShared_3301_; uint8_t v_isSharedCheck_3305_; 
lean_dec_ref(v_inst_2862_);
v_a_3298_ = lean_ctor_get(v___x_3289_, 0);
v_isSharedCheck_3305_ = !lean_is_exclusive(v___x_3289_);
if (v_isSharedCheck_3305_ == 0)
{
v___x_3300_ = v___x_3289_;
v_isShared_3301_ = v_isSharedCheck_3305_;
goto v_resetjp_3299_;
}
else
{
lean_inc(v_a_3298_);
lean_dec(v___x_3289_);
v___x_3300_ = lean_box(0);
v_isShared_3301_ = v_isSharedCheck_3305_;
goto v_resetjp_3299_;
}
v_resetjp_3299_:
{
lean_object* v___x_3303_; 
if (v_isShared_3301_ == 0)
{
v___x_3303_ = v___x_3300_;
goto v_reusejp_3302_;
}
else
{
lean_object* v_reuseFailAlloc_3304_; 
v_reuseFailAlloc_3304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3304_, 0, v_a_3298_);
v___x_3303_ = v_reuseFailAlloc_3304_;
goto v_reusejp_3302_;
}
v_reusejp_3302_:
{
return v___x_3303_;
}
}
}
}
}
else
{
lean_object* v_a_3306_; lean_object* v___x_3308_; uint8_t v_isShared_3309_; uint8_t v_isSharedCheck_3313_; 
lean_dec_ref(v___y_3240_);
lean_dec(v_val_3238_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3306_ = lean_ctor_get(v___x_3244_, 0);
v_isSharedCheck_3313_ = !lean_is_exclusive(v___x_3244_);
if (v_isSharedCheck_3313_ == 0)
{
v___x_3308_ = v___x_3244_;
v_isShared_3309_ = v_isSharedCheck_3313_;
goto v_resetjp_3307_;
}
else
{
lean_inc(v_a_3306_);
lean_dec(v___x_3244_);
v___x_3308_ = lean_box(0);
v_isShared_3309_ = v_isSharedCheck_3313_;
goto v_resetjp_3307_;
}
v_resetjp_3307_:
{
lean_object* v___x_3311_; 
if (v_isShared_3309_ == 0)
{
v___x_3311_ = v___x_3308_;
goto v_reusejp_3310_;
}
else
{
lean_object* v_reuseFailAlloc_3312_; 
v_reuseFailAlloc_3312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3312_, 0, v_a_3306_);
v___x_3311_ = v_reuseFailAlloc_3312_;
goto v_reusejp_3310_;
}
v_reusejp_3310_:
{
return v___x_3311_;
}
}
}
}
}
else
{
lean_object* v___x_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; 
lean_dec(v_a_3237_);
lean_dec_ref(v_inst_2862_);
v___x_3326_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2);
v___x_3327_ = l_Lean_indentExpr(v_expectedType_2863_);
v___x_3328_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3328_, 0, v___x_3326_);
lean_ctor_set(v___x_3328_, 1, v___x_3327_);
v___x_3329_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2865_, v___x_3328_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec_ref_known(v___x_2961_, 7);
return v___x_3329_;
}
}
else
{
lean_object* v_a_3330_; lean_object* v___x_3332_; uint8_t v_isShared_3333_; uint8_t v_isSharedCheck_3337_; 
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3330_ = lean_ctor_get(v___x_3236_, 0);
v_isSharedCheck_3337_ = !lean_is_exclusive(v___x_3236_);
if (v_isSharedCheck_3337_ == 0)
{
v___x_3332_ = v___x_3236_;
v_isShared_3333_ = v_isSharedCheck_3337_;
goto v_resetjp_3331_;
}
else
{
lean_inc(v_a_3330_);
lean_dec(v___x_3236_);
v___x_3332_ = lean_box(0);
v_isShared_3333_ = v_isSharedCheck_3337_;
goto v_resetjp_3331_;
}
v_resetjp_3331_:
{
lean_object* v___x_3335_; 
if (v_isShared_3333_ == 0)
{
v___x_3335_ = v___x_3332_;
goto v_reusejp_3334_;
}
else
{
lean_object* v_reuseFailAlloc_3336_; 
v_reuseFailAlloc_3336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3336_, 0, v_a_3330_);
v___x_3335_ = v_reuseFailAlloc_3336_;
goto v_reusejp_3334_;
}
v_reusejp_3334_:
{
return v___x_3335_;
}
}
}
}
else
{
goto v___jp_3183_;
}
}
else
{
goto v___jp_3183_;
}
v___jp_3120_:
{
lean_object* v___x_3124_; double v___x_3125_; double v___x_3126_; lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; 
v___x_3124_ = lean_io_get_num_heartbeats();
v___x_3125_ = lean_float_of_nat(v___y_3121_);
v___x_3126_ = lean_float_of_nat(v___x_3124_);
v___x_3127_ = lean_box_float(v___x_3125_);
v___x_3128_ = lean_box_float(v___x_3126_);
v___x_3129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3129_, 0, v___x_3127_);
lean_ctor_set(v___x_3129_, 1, v___x_3128_);
v___x_3130_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3130_, 0, v_a_3123_);
lean_ctor_set(v___x_3130_, 1, v___x_3129_);
v___x_3131_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13(v_cls_2886_, v_hasTrace_2885_, v___x_3117_, v_options_2871_, v___x_3119_, v___y_3122_, v___f_3116_, v___x_3130_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec_ref_known(v___x_2961_, 7);
return v___x_3131_;
}
v___jp_3132_:
{
lean_object* v___x_3136_; 
v___x_3136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3136_, 0, v_a_3135_);
v___y_3121_ = v___y_3133_;
v___y_3122_ = v___y_3134_;
v_a_3123_ = v___x_3136_;
goto v___jp_3120_;
}
v___jp_3137_:
{
if (lean_obj_tag(v___y_3140_) == 0)
{
lean_object* v_a_3141_; lean_object* v___x_3143_; uint8_t v_isShared_3144_; uint8_t v_isSharedCheck_3148_; 
v_a_3141_ = lean_ctor_get(v___y_3140_, 0);
v_isSharedCheck_3148_ = !lean_is_exclusive(v___y_3140_);
if (v_isSharedCheck_3148_ == 0)
{
v___x_3143_ = v___y_3140_;
v_isShared_3144_ = v_isSharedCheck_3148_;
goto v_resetjp_3142_;
}
else
{
lean_inc(v_a_3141_);
lean_dec(v___y_3140_);
v___x_3143_ = lean_box(0);
v_isShared_3144_ = v_isSharedCheck_3148_;
goto v_resetjp_3142_;
}
v_resetjp_3142_:
{
lean_object* v___x_3146_; 
if (v_isShared_3144_ == 0)
{
lean_ctor_set_tag(v___x_3143_, 1);
v___x_3146_ = v___x_3143_;
goto v_reusejp_3145_;
}
else
{
lean_object* v_reuseFailAlloc_3147_; 
v_reuseFailAlloc_3147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3147_, 0, v_a_3141_);
v___x_3146_ = v_reuseFailAlloc_3147_;
goto v_reusejp_3145_;
}
v_reusejp_3145_:
{
v___y_3121_ = v___y_3138_;
v___y_3122_ = v___y_3139_;
v_a_3123_ = v___x_3146_;
goto v___jp_3120_;
}
}
}
else
{
lean_object* v_a_3149_; 
v_a_3149_ = lean_ctor_get(v___y_3140_, 0);
lean_inc(v_a_3149_);
lean_dec_ref_known(v___y_3140_, 1);
v___y_3133_ = v___y_3138_;
v___y_3134_ = v___y_3139_;
v_a_3135_ = v_a_3149_;
goto v___jp_3132_;
}
}
v___jp_3150_:
{
lean_object* v___x_3154_; double v___x_3155_; double v___x_3156_; double v___x_3157_; double v___x_3158_; double v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; 
v___x_3154_ = lean_io_mono_nanos_now();
v___x_3155_ = lean_float_of_nat(v___y_3151_);
v___x_3156_ = lean_float_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__3, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__3);
v___x_3157_ = lean_float_div(v___x_3155_, v___x_3156_);
v___x_3158_ = lean_float_of_nat(v___x_3154_);
v___x_3159_ = lean_float_div(v___x_3158_, v___x_3156_);
v___x_3160_ = lean_box_float(v___x_3157_);
v___x_3161_ = lean_box_float(v___x_3159_);
v___x_3162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3162_, 0, v___x_3160_);
lean_ctor_set(v___x_3162_, 1, v___x_3161_);
v___x_3163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3163_, 0, v_a_3153_);
lean_ctor_set(v___x_3163_, 1, v___x_3162_);
v___x_3164_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13(v_cls_2886_, v_hasTrace_2885_, v___x_3117_, v_options_2871_, v___x_3119_, v___y_3152_, v___f_3116_, v___x_3163_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
lean_dec_ref_known(v___x_2961_, 7);
return v___x_3164_;
}
v___jp_3165_:
{
lean_object* v___x_3169_; 
v___x_3169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3169_, 0, v_a_3168_);
v___y_3151_ = v___y_3166_;
v___y_3152_ = v___y_3167_;
v_a_3153_ = v___x_3169_;
goto v___jp_3150_;
}
v___jp_3170_:
{
if (lean_obj_tag(v___y_3173_) == 0)
{
lean_object* v_a_3174_; lean_object* v___x_3176_; uint8_t v_isShared_3177_; uint8_t v_isSharedCheck_3181_; 
v_a_3174_ = lean_ctor_get(v___y_3173_, 0);
v_isSharedCheck_3181_ = !lean_is_exclusive(v___y_3173_);
if (v_isSharedCheck_3181_ == 0)
{
v___x_3176_ = v___y_3173_;
v_isShared_3177_ = v_isSharedCheck_3181_;
goto v_resetjp_3175_;
}
else
{
lean_inc(v_a_3174_);
lean_dec(v___y_3173_);
v___x_3176_ = lean_box(0);
v_isShared_3177_ = v_isSharedCheck_3181_;
goto v_resetjp_3175_;
}
v_resetjp_3175_:
{
lean_object* v___x_3179_; 
if (v_isShared_3177_ == 0)
{
lean_ctor_set_tag(v___x_3176_, 1);
v___x_3179_ = v___x_3176_;
goto v_reusejp_3178_;
}
else
{
lean_object* v_reuseFailAlloc_3180_; 
v_reuseFailAlloc_3180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3180_, 0, v_a_3174_);
v___x_3179_ = v_reuseFailAlloc_3180_;
goto v_reusejp_3178_;
}
v_reusejp_3178_:
{
v___y_3151_ = v___y_3171_;
v___y_3152_ = v___y_3172_;
v_a_3153_ = v___x_3179_;
goto v___jp_3150_;
}
}
}
else
{
lean_object* v_a_3182_; 
v_a_3182_ = lean_ctor_get(v___y_3173_, 0);
lean_inc(v_a_3182_);
lean_dec_ref_known(v___y_3173_, 1);
v___y_3166_ = v___y_3171_;
v___y_3167_ = v___y_3172_;
v_a_3168_ = v_a_3182_;
goto v___jp_3165_;
}
}
v___jp_3183_:
{
lean_object* v___x_3184_; 
v___x_3184_ = lp_mathlib___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__11___redArg(v_a_2869_);
if (lean_obj_tag(v___x_3184_) == 0)
{
lean_object* v_a_3185_; lean_object* v___x_3186_; uint8_t v___x_3187_; 
v_a_3185_ = lean_ctor_get(v___x_3184_, 0);
lean_inc(v_a_3185_);
lean_dec_ref_known(v___x_3184_, 1);
v___x_3186_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3187_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_options_2871_, v___x_3186_);
if (v___x_3187_ == 0)
{
lean_object* v___x_3188_; lean_object* v___x_3189_; 
v___x_3188_ = lean_io_mono_nanos_now();
lean_inc_ref(v_expectedType_2863_);
v___x_3189_ = l_Lean_Meta_isClass_x3f(v_expectedType_2863_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3189_) == 0)
{
lean_object* v_a_3190_; 
v_a_3190_ = lean_ctor_get(v___x_3189_, 0);
lean_inc(v_a_3190_);
lean_dec_ref_known(v___x_3189_, 1);
if (lean_obj_tag(v_a_3190_) == 1)
{
if (v___x_3119_ == 0)
{
lean_object* v_val_3191_; lean_object* v___x_3192_; lean_object* v___x_3193_; 
v_val_3191_ = lean_ctor_get(v_a_3190_, 0);
lean_inc(v_val_3191_);
lean_dec_ref_known(v_a_3190_, 1);
v___x_3192_ = lean_box(0);
v___x_3193_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1(v_expectedType_2863_, v_inst_2862_, v_trace_2865_, v_cls_2886_, v_root_2864_, v_val_3191_, v___x_3187_, v_hasTrace_2885_, v___x_3192_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
v___y_3171_ = v___x_3188_;
v___y_3172_ = v_a_3185_;
v___y_3173_ = v___x_3193_;
goto v___jp_3170_;
}
else
{
lean_object* v_val_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; 
v_val_3194_ = lean_ctor_get(v_a_3190_, 0);
lean_inc_n(v_val_3194_, 2);
lean_dec_ref_known(v_a_3190_, 1);
v___x_3195_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5);
v___x_3196_ = l_Lean_MessageData_ofName(v_val_3194_);
v___x_3197_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3197_, 0, v___x_3195_);
lean_ctor_set(v___x_3197_, 1, v___x_3196_);
v___x_3198_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2886_, v___x_3197_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3198_) == 0)
{
lean_object* v_a_3199_; lean_object* v___x_3200_; 
v_a_3199_ = lean_ctor_get(v___x_3198_, 0);
lean_inc(v_a_3199_);
lean_dec_ref_known(v___x_3198_, 1);
v___x_3200_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1(v_expectedType_2863_, v_inst_2862_, v_trace_2865_, v_cls_2886_, v_root_2864_, v_val_3194_, v___x_3187_, v_hasTrace_2885_, v_a_3199_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
v___y_3171_ = v___x_3188_;
v___y_3172_ = v_a_3185_;
v___y_3173_ = v___x_3200_;
goto v___jp_3170_;
}
else
{
lean_object* v_a_3201_; 
lean_dec(v_val_3194_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3201_ = lean_ctor_get(v___x_3198_, 0);
lean_inc(v_a_3201_);
lean_dec_ref_known(v___x_3198_, 1);
v___y_3166_ = v___x_3188_;
v___y_3167_ = v_a_3185_;
v_a_3168_ = v_a_3201_;
goto v___jp_3165_;
}
}
}
else
{
lean_object* v___x_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; 
lean_dec(v_a_3190_);
lean_dec_ref(v_inst_2862_);
v___x_3202_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2);
v___x_3203_ = l_Lean_indentExpr(v_expectedType_2863_);
v___x_3204_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3204_, 0, v___x_3202_);
lean_ctor_set(v___x_3204_, 1, v___x_3203_);
v___x_3205_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2865_, v___x_3204_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
v___y_3171_ = v___x_3188_;
v___y_3172_ = v_a_3185_;
v___y_3173_ = v___x_3205_;
goto v___jp_3170_;
}
}
else
{
lean_object* v_a_3206_; 
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3206_ = lean_ctor_get(v___x_3189_, 0);
lean_inc(v_a_3206_);
lean_dec_ref_known(v___x_3189_, 1);
v___y_3166_ = v___x_3188_;
v___y_3167_ = v_a_3185_;
v_a_3168_ = v_a_3206_;
goto v___jp_3165_;
}
}
else
{
lean_object* v___x_3207_; lean_object* v___x_3208_; 
v___x_3207_ = lean_io_get_num_heartbeats();
lean_inc_ref(v_expectedType_2863_);
v___x_3208_ = l_Lean_Meta_isClass_x3f(v_expectedType_2863_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3208_) == 0)
{
lean_object* v_a_3209_; 
v_a_3209_ = lean_ctor_get(v___x_3208_, 0);
lean_inc(v_a_3209_);
lean_dec_ref_known(v___x_3208_, 1);
if (lean_obj_tag(v_a_3209_) == 1)
{
if (v___x_3119_ == 0)
{
lean_object* v_val_3210_; lean_object* v___x_3211_; lean_object* v___x_3212_; 
v_val_3210_ = lean_ctor_get(v_a_3209_, 0);
lean_inc(v_val_3210_);
lean_dec_ref_known(v_a_3209_, 1);
v___x_3211_ = lean_box(0);
v___x_3212_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2(v_expectedType_2863_, v_inst_2862_, v_trace_2865_, v_cls_2886_, v_root_2864_, v_val_3210_, v___x_3187_, v___x_3211_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
v___y_3138_ = v___x_3207_;
v___y_3139_ = v_a_3185_;
v___y_3140_ = v___x_3212_;
goto v___jp_3137_;
}
else
{
lean_object* v_val_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; lean_object* v___x_3216_; lean_object* v___x_3217_; 
v_val_3213_ = lean_ctor_get(v_a_3209_, 0);
lean_inc_n(v_val_3213_, 2);
lean_dec_ref_known(v_a_3209_, 1);
v___x_3214_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__5);
v___x_3215_ = l_Lean_MessageData_ofName(v_val_3213_);
v___x_3216_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3216_, 0, v___x_3214_);
lean_ctor_set(v___x_3216_, 1, v___x_3215_);
v___x_3217_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2886_, v___x_3216_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
if (lean_obj_tag(v___x_3217_) == 0)
{
lean_object* v_a_3218_; lean_object* v___x_3219_; 
v_a_3218_ = lean_ctor_get(v___x_3217_, 0);
lean_inc(v_a_3218_);
lean_dec_ref_known(v___x_3217_, 1);
v___x_3219_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2(v_expectedType_2863_, v_inst_2862_, v_trace_2865_, v_cls_2886_, v_root_2864_, v_val_3213_, v___x_3187_, v_a_3218_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
v___y_3138_ = v___x_3207_;
v___y_3139_ = v_a_3185_;
v___y_3140_ = v___x_3219_;
goto v___jp_3137_;
}
else
{
lean_object* v_a_3220_; 
lean_dec(v_val_3213_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3220_ = lean_ctor_get(v___x_3217_, 0);
lean_inc(v_a_3220_);
lean_dec_ref_known(v___x_3217_, 1);
v___y_3133_ = v___x_3207_;
v___y_3134_ = v_a_3185_;
v_a_3135_ = v_a_3220_;
goto v___jp_3132_;
}
}
}
else
{
lean_object* v___x_3221_; lean_object* v___x_3222_; lean_object* v___x_3223_; lean_object* v___x_3224_; 
lean_dec(v_a_3209_);
lean_dec_ref(v_inst_2862_);
v___x_3221_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__2);
v___x_3222_ = l_Lean_indentExpr(v_expectedType_2863_);
v___x_3223_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3223_, 0, v___x_3221_);
lean_ctor_set(v___x_3223_, 1, v___x_3222_);
v___x_3224_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2865_, v___x_3223_, v___x_2961_, v_a_2867_, v_a_2868_, v_a_2869_);
v___y_3138_ = v___x_3207_;
v___y_3139_ = v_a_3185_;
v___y_3140_ = v___x_3224_;
goto v___jp_3137_;
}
}
else
{
lean_object* v_a_3225_; 
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3225_ = lean_ctor_get(v___x_3208_, 0);
lean_inc(v_a_3225_);
lean_dec_ref_known(v___x_3208_, 1);
v___y_3133_ = v___x_3207_;
v___y_3134_ = v_a_3185_;
v_a_3135_ = v_a_3225_;
goto v___jp_3132_;
}
}
}
else
{
lean_object* v_a_3226_; lean_object* v___x_3228_; uint8_t v_isShared_3229_; uint8_t v_isSharedCheck_3233_; 
lean_dec_ref(v___f_3116_);
lean_dec_ref_known(v___x_2961_, 7);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_expectedType_2863_);
lean_dec_ref(v_inst_2862_);
v_a_3226_ = lean_ctor_get(v___x_3184_, 0);
v_isSharedCheck_3233_ = !lean_is_exclusive(v___x_3184_);
if (v_isSharedCheck_3233_ == 0)
{
v___x_3228_ = v___x_3184_;
v_isShared_3229_ = v_isSharedCheck_3233_;
goto v_resetjp_3227_;
}
else
{
lean_inc(v_a_3226_);
lean_dec(v___x_3184_);
v___x_3228_ = lean_box(0);
v_isShared_3229_ = v_isSharedCheck_3233_;
goto v_resetjp_3227_;
}
v_resetjp_3227_:
{
lean_object* v___x_3231_; 
if (v_isShared_3229_ == 0)
{
v___x_3231_ = v___x_3228_;
goto v_reusejp_3230_;
}
else
{
lean_object* v_reuseFailAlloc_3232_; 
v_reuseFailAlloc_3232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3232_, 0, v_a_3226_);
v___x_3231_ = v_reuseFailAlloc_3232_;
goto v_reusejp_3230_;
}
v_reusejp_3230_:
{
return v___x_3231_;
}
}
}
}
}
v___jp_2887_:
{
lean_object* v_keyedConfig_2893_; uint8_t v_trackZetaDelta_2894_; lean_object* v_zetaDeltaSet_2895_; lean_object* v_lctx_2896_; lean_object* v_localInstances_2897_; lean_object* v_defEqCtx_x3f_2898_; lean_object* v_synthPendingDepth_2899_; lean_object* v_customCanUnfoldPredicate_x3f_2900_; uint8_t v_univApprox_2901_; uint8_t v_inTypeClassResolution_2902_; uint8_t v_cacheInferType_2903_; uint8_t v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2906_; lean_object* v___x_2907_; 
v_keyedConfig_2893_ = lean_ctor_get(v___y_2889_, 0);
v_trackZetaDelta_2894_ = lean_ctor_get_uint8(v___y_2889_, sizeof(void*)*7);
v_zetaDeltaSet_2895_ = lean_ctor_get(v___y_2889_, 1);
v_lctx_2896_ = lean_ctor_get(v___y_2889_, 2);
v_localInstances_2897_ = lean_ctor_get(v___y_2889_, 3);
v_defEqCtx_x3f_2898_ = lean_ctor_get(v___y_2889_, 4);
v_synthPendingDepth_2899_ = lean_ctor_get(v___y_2889_, 5);
v_customCanUnfoldPredicate_x3f_2900_ = lean_ctor_get(v___y_2889_, 6);
v_univApprox_2901_ = lean_ctor_get_uint8(v___y_2889_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2902_ = lean_ctor_get_uint8(v___y_2889_, sizeof(void*)*7 + 2);
v_cacheInferType_2903_ = lean_ctor_get_uint8(v___y_2889_, sizeof(void*)*7 + 3);
v___x_2904_ = 1;
lean_inc_ref(v_keyedConfig_2893_);
v___x_2905_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2904_, v_keyedConfig_2893_);
lean_inc(v_customCanUnfoldPredicate_x3f_2900_);
lean_inc(v_synthPendingDepth_2899_);
lean_inc(v_defEqCtx_x3f_2898_);
lean_inc_ref(v_localInstances_2897_);
lean_inc_ref(v_lctx_2896_);
lean_inc(v_zetaDeltaSet_2895_);
v___x_2906_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2906_, 0, v___x_2905_);
lean_ctor_set(v___x_2906_, 1, v_zetaDeltaSet_2895_);
lean_ctor_set(v___x_2906_, 2, v_lctx_2896_);
lean_ctor_set(v___x_2906_, 3, v_localInstances_2897_);
lean_ctor_set(v___x_2906_, 4, v_defEqCtx_x3f_2898_);
lean_ctor_set(v___x_2906_, 5, v_synthPendingDepth_2899_);
lean_ctor_set(v___x_2906_, 6, v_customCanUnfoldPredicate_x3f_2900_);
lean_ctor_set_uint8(v___x_2906_, sizeof(void*)*7, v_trackZetaDelta_2894_);
lean_ctor_set_uint8(v___x_2906_, sizeof(void*)*7 + 1, v_univApprox_2901_);
lean_ctor_set_uint8(v___x_2906_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2902_);
lean_ctor_set_uint8(v___x_2906_, sizeof(void*)*7 + 3, v_cacheInferType_2903_);
lean_inc_ref(v___y_2888_);
lean_inc_ref(v_inst_2862_);
v___x_2907_ = l_Lean_Meta_isExprDefEq(v_inst_2862_, v___y_2888_, v___x_2906_, v___y_2890_, v___y_2891_, v___y_2892_);
lean_dec_ref_known(v___x_2906_, 7);
if (lean_obj_tag(v___x_2907_) == 0)
{
lean_object* v_a_2908_; lean_object* v___x_2910_; uint8_t v_isShared_2911_; uint8_t v_isSharedCheck_2950_; 
v_a_2908_ = lean_ctor_get(v___x_2907_, 0);
v_isSharedCheck_2950_ = !lean_is_exclusive(v___x_2907_);
if (v_isSharedCheck_2950_ == 0)
{
v___x_2910_ = v___x_2907_;
v_isShared_2911_ = v_isSharedCheck_2950_;
goto v_resetjp_2909_;
}
else
{
lean_inc(v_a_2908_);
lean_dec(v___x_2907_);
v___x_2910_ = lean_box(0);
v_isShared_2911_ = v_isSharedCheck_2950_;
goto v_resetjp_2909_;
}
v_resetjp_2909_:
{
uint8_t v___x_2912_; 
v___x_2912_ = lean_unbox(v_a_2908_);
lean_dec(v_a_2908_);
if (v___x_2912_ == 0)
{
lean_object* v___x_2913_; lean_object* v___x_2914_; lean_object* v___x_2915_; lean_object* v___x_2916_; lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; lean_object* v___x_2920_; 
lean_del_object(v___x_2910_);
v___x_2913_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__1);
v___x_2914_ = l_Lean_indentExpr(v_inst_2862_);
v___x_2915_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2915_, 0, v___x_2913_);
lean_ctor_set(v___x_2915_, 1, v___x_2914_);
v___x_2916_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__3);
v___x_2917_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2917_, 0, v___x_2915_);
lean_ctor_set(v___x_2917_, 1, v___x_2916_);
v___x_2918_ = l_Lean_indentExpr(v___y_2888_);
v___x_2919_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2919_, 0, v___x_2917_);
lean_ctor_set(v___x_2919_, 1, v___x_2918_);
v___x_2920_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error___redArg(v_trace_2865_, v___x_2919_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_);
lean_dec_ref(v___y_2889_);
return v___x_2920_;
}
else
{
lean_object* v_options_2921_; uint8_t v_hasTrace_2922_; 
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_inst_2862_);
v_options_2921_ = lean_ctor_get(v___y_2891_, 2);
v_hasTrace_2922_ = lean_ctor_get_uint8(v_options_2921_, sizeof(void*)*1);
if (v_hasTrace_2922_ == 0)
{
lean_object* v___x_2924_; 
lean_dec_ref(v___y_2889_);
if (v_isShared_2911_ == 0)
{
lean_ctor_set(v___x_2910_, 0, v___y_2888_);
v___x_2924_ = v___x_2910_;
goto v_reusejp_2923_;
}
else
{
lean_object* v_reuseFailAlloc_2925_; 
v_reuseFailAlloc_2925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2925_, 0, v___y_2888_);
v___x_2924_ = v_reuseFailAlloc_2925_;
goto v_reusejp_2923_;
}
v_reusejp_2923_:
{
return v___x_2924_;
}
}
else
{
lean_object* v_inheritedTraceOptions_2926_; lean_object* v___x_2927_; uint8_t v___x_2928_; 
v_inheritedTraceOptions_2926_ = lean_ctor_get(v___y_2891_, 13);
v___x_2927_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___closed__0);
v___x_2928_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2926_, v_options_2921_, v___x_2927_);
if (v___x_2928_ == 0)
{
lean_object* v___x_2930_; 
lean_dec_ref(v___y_2889_);
if (v_isShared_2911_ == 0)
{
lean_ctor_set(v___x_2910_, 0, v___y_2888_);
v___x_2930_ = v___x_2910_;
goto v_reusejp_2929_;
}
else
{
lean_object* v_reuseFailAlloc_2931_; 
v_reuseFailAlloc_2931_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2931_, 0, v___y_2888_);
v___x_2930_ = v_reuseFailAlloc_2931_;
goto v_reusejp_2929_;
}
v_reusejp_2929_:
{
return v___x_2930_;
}
}
else
{
lean_object* v___x_2932_; lean_object* v___x_2933_; 
lean_del_object(v___x_2910_);
v___x_2932_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6, &lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___closed__6);
v___x_2933_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__0(v_cls_2886_, v___x_2932_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_);
lean_dec_ref(v___y_2889_);
if (lean_obj_tag(v___x_2933_) == 0)
{
lean_object* v___x_2935_; uint8_t v_isShared_2936_; uint8_t v_isSharedCheck_2940_; 
v_isSharedCheck_2940_ = !lean_is_exclusive(v___x_2933_);
if (v_isSharedCheck_2940_ == 0)
{
lean_object* v_unused_2941_; 
v_unused_2941_ = lean_ctor_get(v___x_2933_, 0);
lean_dec(v_unused_2941_);
v___x_2935_ = v___x_2933_;
v_isShared_2936_ = v_isSharedCheck_2940_;
goto v_resetjp_2934_;
}
else
{
lean_dec(v___x_2933_);
v___x_2935_ = lean_box(0);
v_isShared_2936_ = v_isSharedCheck_2940_;
goto v_resetjp_2934_;
}
v_resetjp_2934_:
{
lean_object* v___x_2938_; 
if (v_isShared_2936_ == 0)
{
lean_ctor_set(v___x_2935_, 0, v___y_2888_);
v___x_2938_ = v___x_2935_;
goto v_reusejp_2937_;
}
else
{
lean_object* v_reuseFailAlloc_2939_; 
v_reuseFailAlloc_2939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2939_, 0, v___y_2888_);
v___x_2938_ = v_reuseFailAlloc_2939_;
goto v_reusejp_2937_;
}
v_reusejp_2937_:
{
return v___x_2938_;
}
}
}
else
{
lean_object* v_a_2942_; lean_object* v___x_2944_; uint8_t v_isShared_2945_; uint8_t v_isSharedCheck_2949_; 
lean_dec_ref(v___y_2888_);
v_a_2942_ = lean_ctor_get(v___x_2933_, 0);
v_isSharedCheck_2949_ = !lean_is_exclusive(v___x_2933_);
if (v_isSharedCheck_2949_ == 0)
{
v___x_2944_ = v___x_2933_;
v_isShared_2945_ = v_isSharedCheck_2949_;
goto v_resetjp_2943_;
}
else
{
lean_inc(v_a_2942_);
lean_dec(v___x_2933_);
v___x_2944_ = lean_box(0);
v_isShared_2945_ = v_isSharedCheck_2949_;
goto v_resetjp_2943_;
}
v_resetjp_2943_:
{
lean_object* v___x_2947_; 
if (v_isShared_2945_ == 0)
{
v___x_2947_ = v___x_2944_;
goto v_reusejp_2946_;
}
else
{
lean_object* v_reuseFailAlloc_2948_; 
v_reuseFailAlloc_2948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2948_, 0, v_a_2942_);
v___x_2947_ = v_reuseFailAlloc_2948_;
goto v_reusejp_2946_;
}
v_reusejp_2946_:
{
return v___x_2947_;
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
lean_object* v_a_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_2958_; 
lean_dec_ref(v___y_2889_);
lean_dec_ref(v___y_2888_);
lean_dec_ref(v_trace_2865_);
lean_dec_ref(v_inst_2862_);
v_a_2951_ = lean_ctor_get(v___x_2907_, 0);
v_isSharedCheck_2958_ = !lean_is_exclusive(v___x_2907_);
if (v_isSharedCheck_2958_ == 0)
{
v___x_2953_ = v___x_2907_;
v_isShared_2954_ = v_isSharedCheck_2958_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_a_2951_);
lean_dec(v___x_2907_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_2958_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
lean_object* v___x_2956_; 
if (v_isShared_2954_ == 0)
{
v___x_2956_ = v___x_2953_;
goto v_reusejp_2955_;
}
else
{
lean_object* v_reuseFailAlloc_2957_; 
v_reuseFailAlloc_2957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2957_, 0, v_a_2951_);
v___x_2956_ = v_reuseFailAlloc_2957_;
goto v_reusejp_2955_;
}
v_reusejp_2955_:
{
return v___x_2956_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg(lean_object* v_upperBound_3338_, lean_object* v_fst_3339_, lean_object* v_args_3340_, uint8_t v___x_3341_, uint8_t v___x_3342_, lean_object* v_fst_3343_, lean_object* v_val_3344_, lean_object* v_trace_3345_, lean_object* v_a_3346_, lean_object* v_b_3347_, lean_object* v___y_3348_, lean_object* v___y_3349_, lean_object* v___y_3350_, lean_object* v___y_3351_){
_start:
{
lean_object* v_a_3354_; uint8_t v___x_3358_; 
v___x_3358_ = lean_nat_dec_lt(v_a_3346_, v_upperBound_3338_);
if (v___x_3358_ == 0)
{
lean_object* v___x_3359_; 
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v___x_3359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3359_, 0, v_b_3347_);
return v___x_3359_;
}
else
{
lean_object* v___x_3360_; lean_object* v___x_3361_; lean_object* v___x_3362_; lean_object* v___x_3363_; 
v___x_3360_ = l_Lean_instInhabitedExpr;
v___x_3361_ = lean_array_get_borrowed(v___x_3360_, v_fst_3339_, v_a_3346_);
v___x_3362_ = l_Lean_Expr_mvarId_x21(v___x_3361_);
lean_inc(v___x_3362_);
v___x_3363_ = l_Lean_MVarId_getDecl(v___x_3362_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3363_) == 0)
{
lean_object* v_a_3364_; lean_object* v_userName_3365_; lean_object* v_type_3366_; lean_object* v___x_3367_; 
v_a_3364_ = lean_ctor_get(v___x_3363_, 0);
lean_inc(v_a_3364_);
lean_dec_ref_known(v___x_3363_, 1);
v_userName_3365_ = lean_ctor_get(v_a_3364_, 0);
lean_inc(v_userName_3365_);
v_type_3366_ = lean_ctor_get(v_a_3364_, 2);
lean_inc_ref(v_type_3366_);
lean_dec(v_a_3364_);
v___x_3367_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__3___redArg(v_type_3366_, v___y_3349_);
if (lean_obj_tag(v___x_3367_) == 0)
{
lean_object* v_a_3368_; lean_object* v___x_3369_; 
v_a_3368_ = lean_ctor_get(v___x_3367_, 0);
lean_inc_n(v_a_3368_, 2);
lean_dec_ref_known(v___x_3367_, 1);
v___x_3369_ = l_Lean_Meta_isProp(v_a_3368_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3369_) == 0)
{
lean_object* v_a_3370_; lean_object* v___x_3371_; lean_object* v___x_3372_; uint8_t v___x_3373_; 
v_a_3370_ = lean_ctor_get(v___x_3369_, 0);
lean_inc(v_a_3370_);
lean_dec_ref_known(v___x_3369_, 1);
v___x_3371_ = lean_box(0);
v___x_3372_ = lean_array_get_borrowed(v___x_3360_, v_args_3340_, v_a_3346_);
v___x_3373_ = lean_unbox(v_a_3370_);
lean_dec(v_a_3370_);
if (v___x_3373_ == 0)
{
uint8_t v___x_3374_; lean_object* v___x_3375_; lean_object* v___x_3376_; uint8_t v___x_3377_; uint8_t v___x_3378_; 
v___x_3374_ = 0;
v___x_3375_ = lean_box(v___x_3374_);
v___x_3376_ = lean_array_get(v___x_3375_, v_fst_3343_, v_a_3346_);
lean_dec(v___x_3375_);
v___x_3377_ = lean_unbox(v___x_3376_);
lean_dec(v___x_3376_);
v___x_3378_ = l_Lean_BinderInfo_isInstImplicit(v___x_3377_);
if (v___x_3378_ == 0)
{
lean_object* v___x_3379_; lean_object* v___x_3380_; lean_object* v___f_3381_; lean_object* v___x_3382_; 
lean_dec(v_userName_3365_);
v___x_3379_ = lean_box(v___x_3341_);
v___x_3380_ = lean_box(v___x_3342_);
lean_inc(v___x_3372_);
v___f_3381_ = lean_alloc_closure((void*)(lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_3381_, 0, v___x_3372_);
lean_closure_set(v___f_3381_, 1, v___x_3379_);
lean_closure_set(v___f_3381_, 2, v___x_3380_);
v___x_3382_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__5___redArg(v_a_3368_, v___f_3381_, v___x_3341_, v___x_3341_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3382_) == 0)
{
lean_object* v_a_3383_; lean_object* v___x_3384_; 
v_a_3383_ = lean_ctor_get(v___x_3382_, 0);
lean_inc(v_a_3383_);
lean_dec_ref_known(v___x_3382_, 1);
v___x_3384_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_3362_, v_a_3383_, v___y_3349_);
if (lean_obj_tag(v___x_3384_) == 0)
{
lean_dec_ref_known(v___x_3384_, 1);
v_a_3354_ = v___x_3371_;
goto v___jp_3353_;
}
else
{
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
return v___x_3384_;
}
}
else
{
lean_object* v_a_3385_; lean_object* v___x_3387_; uint8_t v_isShared_3388_; uint8_t v_isSharedCheck_3392_; 
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3385_ = lean_ctor_get(v___x_3382_, 0);
v_isSharedCheck_3392_ = !lean_is_exclusive(v___x_3382_);
if (v_isSharedCheck_3392_ == 0)
{
v___x_3387_ = v___x_3382_;
v_isShared_3388_ = v_isSharedCheck_3392_;
goto v_resetjp_3386_;
}
else
{
lean_inc(v_a_3385_);
lean_dec(v___x_3382_);
v___x_3387_ = lean_box(0);
v_isShared_3388_ = v_isSharedCheck_3392_;
goto v_resetjp_3386_;
}
v_resetjp_3386_:
{
lean_object* v___x_3390_; 
if (v_isShared_3388_ == 0)
{
v___x_3390_ = v___x_3387_;
goto v_reusejp_3389_;
}
else
{
lean_object* v_reuseFailAlloc_3391_; 
v_reuseFailAlloc_3391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3391_, 0, v_a_3385_);
v___x_3390_ = v_reuseFailAlloc_3391_;
goto v_reusejp_3389_;
}
v_reusejp_3389_:
{
return v___x_3390_;
}
}
}
}
else
{
lean_object* v___x_3393_; lean_object* v___x_3394_; lean_object* v___x_3395_; 
lean_inc(v_val_3344_);
v___x_3393_ = l_Lean_Name_append(v_val_3344_, v_userName_3365_);
lean_inc_ref(v_trace_3345_);
v___x_3394_ = lean_array_push(v_trace_3345_, v___x_3393_);
lean_inc(v___x_3372_);
v___x_3395_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(v___x_3372_, v_a_3368_, v___x_3341_, v___x_3394_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3395_) == 0)
{
lean_object* v_a_3396_; lean_object* v___x_3397_; 
v_a_3396_ = lean_ctor_get(v___x_3395_, 0);
lean_inc(v_a_3396_);
lean_dec_ref_known(v___x_3395_, 1);
v___x_3397_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_3362_, v_a_3396_, v___y_3349_);
if (lean_obj_tag(v___x_3397_) == 0)
{
lean_dec_ref_known(v___x_3397_, 1);
v_a_3354_ = v___x_3371_;
goto v___jp_3353_;
}
else
{
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
return v___x_3397_;
}
}
else
{
lean_object* v_a_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3405_; 
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3398_ = lean_ctor_get(v___x_3395_, 0);
v_isSharedCheck_3405_ = !lean_is_exclusive(v___x_3395_);
if (v_isSharedCheck_3405_ == 0)
{
v___x_3400_ = v___x_3395_;
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_a_3398_);
lean_dec(v___x_3395_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3403_; 
if (v_isShared_3401_ == 0)
{
v___x_3403_ = v___x_3400_;
goto v_reusejp_3402_;
}
else
{
lean_object* v_reuseFailAlloc_3404_; 
v_reuseFailAlloc_3404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3404_, 0, v_a_3398_);
v___x_3403_ = v_reuseFailAlloc_3404_;
goto v_reusejp_3402_;
}
v_reusejp_3402_:
{
return v___x_3403_;
}
}
}
}
}
else
{
lean_object* v___x_3406_; 
lean_dec(v_userName_3365_);
lean_inc(v___y_3351_);
lean_inc_ref(v___y_3350_);
lean_inc(v___y_3349_);
lean_inc_ref(v___y_3348_);
lean_inc(v___x_3372_);
v___x_3406_ = lean_infer_type(v___x_3372_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3406_) == 0)
{
lean_object* v_a_3407_; lean_object* v_keyedConfig_3408_; uint8_t v_trackZetaDelta_3409_; lean_object* v_zetaDeltaSet_3410_; lean_object* v_lctx_3411_; lean_object* v_localInstances_3412_; lean_object* v_defEqCtx_x3f_3413_; lean_object* v_synthPendingDepth_3414_; lean_object* v_customCanUnfoldPredicate_x3f_3415_; uint8_t v_univApprox_3416_; uint8_t v_inTypeClassResolution_3417_; uint8_t v_cacheInferType_3418_; uint8_t v___x_3419_; lean_object* v___x_3420_; lean_object* v___x_3421_; lean_object* v___x_3422_; 
v_a_3407_ = lean_ctor_get(v___x_3406_, 0);
lean_inc(v_a_3407_);
lean_dec_ref_known(v___x_3406_, 1);
v_keyedConfig_3408_ = lean_ctor_get(v___y_3348_, 0);
v_trackZetaDelta_3409_ = lean_ctor_get_uint8(v___y_3348_, sizeof(void*)*7);
v_zetaDeltaSet_3410_ = lean_ctor_get(v___y_3348_, 1);
v_lctx_3411_ = lean_ctor_get(v___y_3348_, 2);
v_localInstances_3412_ = lean_ctor_get(v___y_3348_, 3);
v_defEqCtx_x3f_3413_ = lean_ctor_get(v___y_3348_, 4);
v_synthPendingDepth_3414_ = lean_ctor_get(v___y_3348_, 5);
v_customCanUnfoldPredicate_x3f_3415_ = lean_ctor_get(v___y_3348_, 6);
v_univApprox_3416_ = lean_ctor_get_uint8(v___y_3348_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3417_ = lean_ctor_get_uint8(v___y_3348_, sizeof(void*)*7 + 2);
v_cacheInferType_3418_ = lean_ctor_get_uint8(v___y_3348_, sizeof(void*)*7 + 3);
v___x_3419_ = 1;
lean_inc_ref(v_keyedConfig_3408_);
v___x_3420_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3419_, v_keyedConfig_3408_);
lean_inc(v_customCanUnfoldPredicate_x3f_3415_);
lean_inc(v_synthPendingDepth_3414_);
lean_inc(v_defEqCtx_x3f_3413_);
lean_inc_ref(v_localInstances_3412_);
lean_inc_ref(v_lctx_3411_);
lean_inc(v_zetaDeltaSet_3410_);
v___x_3421_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3421_, 0, v___x_3420_);
lean_ctor_set(v___x_3421_, 1, v_zetaDeltaSet_3410_);
lean_ctor_set(v___x_3421_, 2, v_lctx_3411_);
lean_ctor_set(v___x_3421_, 3, v_localInstances_3412_);
lean_ctor_set(v___x_3421_, 4, v_defEqCtx_x3f_3413_);
lean_ctor_set(v___x_3421_, 5, v_synthPendingDepth_3414_);
lean_ctor_set(v___x_3421_, 6, v_customCanUnfoldPredicate_x3f_3415_);
lean_ctor_set_uint8(v___x_3421_, sizeof(void*)*7, v_trackZetaDelta_3409_);
lean_ctor_set_uint8(v___x_3421_, sizeof(void*)*7 + 1, v_univApprox_3416_);
lean_ctor_set_uint8(v___x_3421_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3417_);
lean_ctor_set_uint8(v___x_3421_, sizeof(void*)*7 + 3, v_cacheInferType_3418_);
lean_inc(v_a_3368_);
v___x_3422_ = l_Lean_Meta_isExprDefEq(v_a_3368_, v_a_3407_, v___x_3421_, v___y_3349_, v___y_3350_, v___y_3351_);
lean_dec_ref_known(v___x_3421_, 7);
if (lean_obj_tag(v___x_3422_) == 0)
{
lean_object* v_a_3423_; uint8_t v___x_3424_; 
v_a_3423_ = lean_ctor_get(v___x_3422_, 0);
lean_inc(v_a_3423_);
lean_dec_ref_known(v___x_3422_, 1);
v___x_3424_ = lean_unbox(v_a_3423_);
lean_dec(v_a_3423_);
if (v___x_3424_ == 0)
{
lean_object* v___x_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; 
lean_dec(v___x_3362_);
v___x_3425_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__1);
lean_inc(v___x_3372_);
v___x_3426_ = l_Lean_MessageData_ofExpr(v___x_3372_);
v___x_3427_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3427_, 0, v___x_3425_);
lean_ctor_set(v___x_3427_, 1, v___x_3426_);
v___x_3428_ = lean_obj_once(&lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3, &lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3_once, _init_lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___closed__3);
v___x_3429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3429_, 0, v___x_3427_);
lean_ctor_set(v___x_3429_, 1, v___x_3428_);
v___x_3430_ = l_Lean_MessageData_ofExpr(v_a_3368_);
v___x_3431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3431_, 0, v___x_3429_);
lean_ctor_set(v___x_3431_, 1, v___x_3430_);
v___x_3432_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg___closed__3);
v___x_3433_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3433_, 0, v___x_3431_);
lean_ctor_set(v___x_3433_, 1, v___x_3432_);
v___x_3434_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1___redArg(v___x_3433_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3434_) == 0)
{
lean_dec_ref_known(v___x_3434_, 1);
v_a_3354_ = v___x_3371_;
goto v___jp_3353_;
}
else
{
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
return v___x_3434_;
}
}
else
{
lean_object* v___x_3435_; lean_object* v___x_3436_; 
v___x_3435_ = lean_box(0);
lean_inc(v___x_3372_);
v___x_3436_ = l_Lean_Meta_mkAuxTheorem(v_a_3368_, v___x_3372_, v___x_3342_, v___x_3435_, v___x_3342_, v___y_3348_, v___y_3349_, v___y_3350_, v___y_3351_);
if (lean_obj_tag(v___x_3436_) == 0)
{
lean_object* v_a_3437_; lean_object* v___x_3438_; 
v_a_3437_ = lean_ctor_get(v___x_3436_, 0);
lean_inc(v_a_3437_);
lean_dec_ref_known(v___x_3436_, 1);
v___x_3438_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v___x_3362_, v_a_3437_, v___y_3349_);
if (lean_obj_tag(v___x_3438_) == 0)
{
lean_dec_ref_known(v___x_3438_, 1);
v_a_3354_ = v___x_3371_;
goto v___jp_3353_;
}
else
{
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
return v___x_3438_;
}
}
else
{
lean_object* v_a_3439_; lean_object* v___x_3441_; uint8_t v_isShared_3442_; uint8_t v_isSharedCheck_3446_; 
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3439_ = lean_ctor_get(v___x_3436_, 0);
v_isSharedCheck_3446_ = !lean_is_exclusive(v___x_3436_);
if (v_isSharedCheck_3446_ == 0)
{
v___x_3441_ = v___x_3436_;
v_isShared_3442_ = v_isSharedCheck_3446_;
goto v_resetjp_3440_;
}
else
{
lean_inc(v_a_3439_);
lean_dec(v___x_3436_);
v___x_3441_ = lean_box(0);
v_isShared_3442_ = v_isSharedCheck_3446_;
goto v_resetjp_3440_;
}
v_resetjp_3440_:
{
lean_object* v___x_3444_; 
if (v_isShared_3442_ == 0)
{
v___x_3444_ = v___x_3441_;
goto v_reusejp_3443_;
}
else
{
lean_object* v_reuseFailAlloc_3445_; 
v_reuseFailAlloc_3445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3445_, 0, v_a_3439_);
v___x_3444_ = v_reuseFailAlloc_3445_;
goto v_reusejp_3443_;
}
v_reusejp_3443_:
{
return v___x_3444_;
}
}
}
}
}
else
{
lean_object* v_a_3447_; lean_object* v___x_3449_; uint8_t v_isShared_3450_; uint8_t v_isSharedCheck_3454_; 
lean_dec(v_a_3368_);
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3447_ = lean_ctor_get(v___x_3422_, 0);
v_isSharedCheck_3454_ = !lean_is_exclusive(v___x_3422_);
if (v_isSharedCheck_3454_ == 0)
{
v___x_3449_ = v___x_3422_;
v_isShared_3450_ = v_isSharedCheck_3454_;
goto v_resetjp_3448_;
}
else
{
lean_inc(v_a_3447_);
lean_dec(v___x_3422_);
v___x_3449_ = lean_box(0);
v_isShared_3450_ = v_isSharedCheck_3454_;
goto v_resetjp_3448_;
}
v_resetjp_3448_:
{
lean_object* v___x_3452_; 
if (v_isShared_3450_ == 0)
{
v___x_3452_ = v___x_3449_;
goto v_reusejp_3451_;
}
else
{
lean_object* v_reuseFailAlloc_3453_; 
v_reuseFailAlloc_3453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3453_, 0, v_a_3447_);
v___x_3452_ = v_reuseFailAlloc_3453_;
goto v_reusejp_3451_;
}
v_reusejp_3451_:
{
return v___x_3452_;
}
}
}
}
else
{
lean_object* v_a_3455_; lean_object* v___x_3457_; uint8_t v_isShared_3458_; uint8_t v_isSharedCheck_3462_; 
lean_dec(v_a_3368_);
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3455_ = lean_ctor_get(v___x_3406_, 0);
v_isSharedCheck_3462_ = !lean_is_exclusive(v___x_3406_);
if (v_isSharedCheck_3462_ == 0)
{
v___x_3457_ = v___x_3406_;
v_isShared_3458_ = v_isSharedCheck_3462_;
goto v_resetjp_3456_;
}
else
{
lean_inc(v_a_3455_);
lean_dec(v___x_3406_);
v___x_3457_ = lean_box(0);
v_isShared_3458_ = v_isSharedCheck_3462_;
goto v_resetjp_3456_;
}
v_resetjp_3456_:
{
lean_object* v___x_3460_; 
if (v_isShared_3458_ == 0)
{
v___x_3460_ = v___x_3457_;
goto v_reusejp_3459_;
}
else
{
lean_object* v_reuseFailAlloc_3461_; 
v_reuseFailAlloc_3461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3461_, 0, v_a_3455_);
v___x_3460_ = v_reuseFailAlloc_3461_;
goto v_reusejp_3459_;
}
v_reusejp_3459_:
{
return v___x_3460_;
}
}
}
}
}
else
{
lean_object* v_a_3463_; lean_object* v___x_3465_; uint8_t v_isShared_3466_; uint8_t v_isSharedCheck_3470_; 
lean_dec(v_a_3368_);
lean_dec(v_userName_3365_);
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3463_ = lean_ctor_get(v___x_3369_, 0);
v_isSharedCheck_3470_ = !lean_is_exclusive(v___x_3369_);
if (v_isSharedCheck_3470_ == 0)
{
v___x_3465_ = v___x_3369_;
v_isShared_3466_ = v_isSharedCheck_3470_;
goto v_resetjp_3464_;
}
else
{
lean_inc(v_a_3463_);
lean_dec(v___x_3369_);
v___x_3465_ = lean_box(0);
v_isShared_3466_ = v_isSharedCheck_3470_;
goto v_resetjp_3464_;
}
v_resetjp_3464_:
{
lean_object* v___x_3468_; 
if (v_isShared_3466_ == 0)
{
v___x_3468_ = v___x_3465_;
goto v_reusejp_3467_;
}
else
{
lean_object* v_reuseFailAlloc_3469_; 
v_reuseFailAlloc_3469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3469_, 0, v_a_3463_);
v___x_3468_ = v_reuseFailAlloc_3469_;
goto v_reusejp_3467_;
}
v_reusejp_3467_:
{
return v___x_3468_;
}
}
}
}
else
{
lean_object* v_a_3471_; lean_object* v___x_3473_; uint8_t v_isShared_3474_; uint8_t v_isSharedCheck_3478_; 
lean_dec(v_userName_3365_);
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3471_ = lean_ctor_get(v___x_3367_, 0);
v_isSharedCheck_3478_ = !lean_is_exclusive(v___x_3367_);
if (v_isSharedCheck_3478_ == 0)
{
v___x_3473_ = v___x_3367_;
v_isShared_3474_ = v_isSharedCheck_3478_;
goto v_resetjp_3472_;
}
else
{
lean_inc(v_a_3471_);
lean_dec(v___x_3367_);
v___x_3473_ = lean_box(0);
v_isShared_3474_ = v_isSharedCheck_3478_;
goto v_resetjp_3472_;
}
v_resetjp_3472_:
{
lean_object* v___x_3476_; 
if (v_isShared_3474_ == 0)
{
v___x_3476_ = v___x_3473_;
goto v_reusejp_3475_;
}
else
{
lean_object* v_reuseFailAlloc_3477_; 
v_reuseFailAlloc_3477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3477_, 0, v_a_3471_);
v___x_3476_ = v_reuseFailAlloc_3477_;
goto v_reusejp_3475_;
}
v_reusejp_3475_:
{
return v___x_3476_;
}
}
}
}
else
{
lean_object* v_a_3479_; lean_object* v___x_3481_; uint8_t v_isShared_3482_; uint8_t v_isSharedCheck_3486_; 
lean_dec(v___x_3362_);
lean_dec(v_a_3346_);
lean_dec_ref(v_trace_3345_);
lean_dec(v_val_3344_);
v_a_3479_ = lean_ctor_get(v___x_3363_, 0);
v_isSharedCheck_3486_ = !lean_is_exclusive(v___x_3363_);
if (v_isSharedCheck_3486_ == 0)
{
v___x_3481_ = v___x_3363_;
v_isShared_3482_ = v_isSharedCheck_3486_;
goto v_resetjp_3480_;
}
else
{
lean_inc(v_a_3479_);
lean_dec(v___x_3363_);
v___x_3481_ = lean_box(0);
v_isShared_3482_ = v_isSharedCheck_3486_;
goto v_resetjp_3480_;
}
v_resetjp_3480_:
{
lean_object* v___x_3484_; 
if (v_isShared_3482_ == 0)
{
v___x_3484_ = v___x_3481_;
goto v_reusejp_3483_;
}
else
{
lean_object* v_reuseFailAlloc_3485_; 
v_reuseFailAlloc_3485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3485_, 0, v_a_3479_);
v___x_3484_ = v_reuseFailAlloc_3485_;
goto v_reusejp_3483_;
}
v_reusejp_3483_:
{
return v___x_3484_;
}
}
}
}
v___jp_3353_:
{
lean_object* v___x_3355_; lean_object* v___x_3356_; 
v___x_3355_ = lean_unsigned_to_nat(1u);
v___x_3356_ = lean_nat_add(v_a_3346_, v___x_3355_);
lean_dec(v_a_3346_);
v_a_3346_ = v___x_3356_;
v_b_3347_ = v_a_3354_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg___boxed(lean_object* v_upperBound_3487_, lean_object* v_fst_3488_, lean_object* v_args_3489_, lean_object* v___x_3490_, lean_object* v___x_3491_, lean_object* v_fst_3492_, lean_object* v_val_3493_, lean_object* v_trace_3494_, lean_object* v_a_3495_, lean_object* v_b_3496_, lean_object* v___y_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_){
_start:
{
uint8_t v___x_70115__boxed_3502_; uint8_t v___x_70116__boxed_3503_; lean_object* v_res_3504_; 
v___x_70115__boxed_3502_ = lean_unbox(v___x_3490_);
v___x_70116__boxed_3503_ = lean_unbox(v___x_3491_);
v_res_3504_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg(v_upperBound_3487_, v_fst_3488_, v_args_3489_, v___x_70115__boxed_3502_, v___x_70116__boxed_3503_, v_fst_3492_, v_val_3493_, v_trace_3494_, v_a_3495_, v_b_3496_, v___y_3497_, v___y_3498_, v___y_3499_, v___y_3500_);
lean_dec(v___y_3500_);
lean_dec_ref(v___y_3499_);
lean_dec(v___y_3498_);
lean_dec_ref(v___y_3497_);
lean_dec_ref(v_fst_3492_);
lean_dec_ref(v_args_3489_);
lean_dec_ref(v_fst_3488_);
lean_dec(v_upperBound_3487_);
return v_res_3504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg___boxed(lean_object* v_upperBound_3505_, lean_object* v_fst_3506_, lean_object* v_args_3507_, lean_object* v_fst_3508_, lean_object* v___x_3509_, lean_object* v_val_3510_, lean_object* v_trace_3511_, lean_object* v_a_3512_, lean_object* v_b_3513_, lean_object* v___y_3514_, lean_object* v___y_3515_, lean_object* v___y_3516_, lean_object* v___y_3517_, lean_object* v___y_3518_){
_start:
{
uint8_t v___x_70202__boxed_3519_; lean_object* v_res_3520_; 
v___x_70202__boxed_3519_ = lean_unbox(v___x_3509_);
v_res_3520_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg(v_upperBound_3505_, v_fst_3506_, v_args_3507_, v_fst_3508_, v___x_70202__boxed_3519_, v_val_3510_, v_trace_3511_, v_a_3512_, v_b_3513_, v___y_3514_, v___y_3515_, v___y_3516_, v___y_3517_);
lean_dec(v___y_3517_);
lean_dec_ref(v___y_3516_);
lean_dec(v___y_3515_);
lean_dec_ref(v___y_3514_);
lean_dec_ref(v_fst_3508_);
lean_dec_ref(v_args_3507_);
lean_dec_ref(v_fst_3506_);
lean_dec(v_upperBound_3505_);
return v_res_3520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg___boxed(lean_object* v_upperBound_3521_, lean_object* v_fst_3522_, lean_object* v_args_3523_, lean_object* v___x_3524_, lean_object* v_fst_3525_, lean_object* v_val_3526_, lean_object* v_trace_3527_, lean_object* v_a_3528_, lean_object* v_b_3529_, lean_object* v___y_3530_, lean_object* v___y_3531_, lean_object* v___y_3532_, lean_object* v___y_3533_, lean_object* v___y_3534_){
_start:
{
uint8_t v___x_70289__boxed_3535_; lean_object* v_res_3536_; 
v___x_70289__boxed_3535_ = lean_unbox(v___x_3524_);
v_res_3536_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg(v_upperBound_3521_, v_fst_3522_, v_args_3523_, v___x_70289__boxed_3535_, v_fst_3525_, v_val_3526_, v_trace_3527_, v_a_3528_, v_b_3529_, v___y_3530_, v___y_3531_, v___y_3532_, v___y_3533_);
lean_dec(v___y_3533_);
lean_dec_ref(v___y_3532_);
lean_dec(v___y_3531_);
lean_dec_ref(v___y_3530_);
lean_dec_ref(v_fst_3525_);
lean_dec_ref(v_args_3523_);
lean_dec_ref(v_fst_3522_);
lean_dec(v_upperBound_3521_);
return v_res_3536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1___boxed(lean_object* v_expectedType_3537_, lean_object* v_inst_3538_, lean_object* v_trace_3539_, lean_object* v_cls_3540_, lean_object* v_root_3541_, lean_object* v_val_3542_, lean_object* v___x_3543_, lean_object* v_hasTrace_3544_, lean_object* v_____r_3545_, lean_object* v___y_3546_, lean_object* v___y_3547_, lean_object* v___y_3548_, lean_object* v___y_3549_, lean_object* v___y_3550_){
_start:
{
uint8_t v_root_boxed_3551_; uint8_t v___x_70398__boxed_3552_; uint8_t v_hasTrace_boxed_3553_; lean_object* v_res_3554_; 
v_root_boxed_3551_ = lean_unbox(v_root_3541_);
v___x_70398__boxed_3552_ = lean_unbox(v___x_3543_);
v_hasTrace_boxed_3553_ = lean_unbox(v_hasTrace_3544_);
v_res_3554_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__1(v_expectedType_3537_, v_inst_3538_, v_trace_3539_, v_cls_3540_, v_root_boxed_3551_, v_val_3542_, v___x_70398__boxed_3552_, v_hasTrace_boxed_3553_, v_____r_3545_, v___y_3546_, v___y_3547_, v___y_3548_, v___y_3549_);
lean_dec(v___y_3549_);
lean_dec_ref(v___y_3548_);
lean_dec(v___y_3547_);
lean_dec_ref(v___y_3546_);
return v_res_3554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2___boxed(lean_object* v_expectedType_3555_, lean_object* v_inst_3556_, lean_object* v_trace_3557_, lean_object* v_cls_3558_, lean_object* v_root_3559_, lean_object* v_val_3560_, lean_object* v___x_3561_, lean_object* v_____r_3562_, lean_object* v___y_3563_, lean_object* v___y_3564_, lean_object* v___y_3565_, lean_object* v___y_3566_, lean_object* v___y_3567_){
_start:
{
uint8_t v_root_boxed_3568_; uint8_t v___x_70475__boxed_3569_; lean_object* v_res_3570_; 
v_root_boxed_3568_ = lean_unbox(v_root_3559_);
v___x_70475__boxed_3569_ = lean_unbox(v___x_3561_);
v_res_3570_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___lam__2(v_expectedType_3555_, v_inst_3556_, v_trace_3557_, v_cls_3558_, v_root_boxed_3568_, v_val_3560_, v___x_70475__boxed_3569_, v_____r_3562_, v___y_3563_, v___y_3564_, v___y_3565_, v___y_3566_);
lean_dec(v___y_3566_);
lean_dec_ref(v___y_3565_);
lean_dec(v___y_3564_);
lean_dec_ref(v___y_3563_);
return v_res_3570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15___boxed(lean_object* v_val_3571_, lean_object* v_trace_3572_, lean_object* v___x_3573_, lean_object* v___x_3574_, lean_object* v_expectedType_3575_, lean_object* v_inst_3576_, lean_object* v_x_3577_, lean_object* v_x_3578_, lean_object* v_x_3579_, lean_object* v___y_3580_, lean_object* v___y_3581_, lean_object* v___y_3582_, lean_object* v___y_3583_, lean_object* v___y_3584_){
_start:
{
uint8_t v___x_70581__boxed_3585_; uint8_t v___x_70582__boxed_3586_; lean_object* v_res_3587_; 
v___x_70581__boxed_3585_ = lean_unbox(v___x_3573_);
v___x_70582__boxed_3586_ = lean_unbox(v___x_3574_);
v_res_3587_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__15(v_val_3571_, v_trace_3572_, v___x_70581__boxed_3585_, v___x_70582__boxed_3586_, v_expectedType_3575_, v_inst_3576_, v_x_3577_, v_x_3578_, v_x_3579_, v___y_3580_, v___y_3581_, v___y_3582_, v___y_3583_);
lean_dec(v___y_3583_);
lean_dec_ref(v___y_3582_);
lean_dec(v___y_3581_);
lean_dec_ref(v___y_3580_);
return v_res_3587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__17___boxed(lean_object* v_val_3588_, lean_object* v_trace_3589_, lean_object* v___x_3590_, lean_object* v_expectedType_3591_, lean_object* v_inst_3592_, lean_object* v_x_3593_, lean_object* v_x_3594_, lean_object* v_x_3595_, lean_object* v___y_3596_, lean_object* v___y_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_, lean_object* v___y_3600_){
_start:
{
uint8_t v___x_70696__boxed_3601_; lean_object* v_res_3602_; 
v___x_70696__boxed_3601_ = lean_unbox(v___x_3590_);
v_res_3602_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__17(v_val_3588_, v_trace_3589_, v___x_70696__boxed_3601_, v_expectedType_3591_, v_inst_3592_, v_x_3593_, v_x_3594_, v_x_3595_, v___y_3596_, v___y_3597_, v___y_3598_, v___y_3599_);
lean_dec(v___y_3599_);
lean_dec_ref(v___y_3598_);
lean_dec(v___y_3597_);
lean_dec_ref(v___y_3596_);
return v_res_3602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__9___boxed(lean_object* v_val_3603_, lean_object* v_trace_3604_, lean_object* v___x_3605_, lean_object* v_expectedType_3606_, lean_object* v_inst_3607_, lean_object* v_x_3608_, lean_object* v_x_3609_, lean_object* v_x_3610_, lean_object* v___y_3611_, lean_object* v___y_3612_, lean_object* v___y_3613_, lean_object* v___y_3614_, lean_object* v___y_3615_){
_start:
{
uint8_t v___x_70819__boxed_3616_; lean_object* v_res_3617_; 
v___x_70819__boxed_3616_ = lean_unbox(v___x_3605_);
v_res_3617_ = lp_mathlib_Lean_Expr_withAppAux___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__9(v_val_3603_, v_trace_3604_, v___x_70819__boxed_3616_, v_expectedType_3606_, v_inst_3607_, v_x_3608_, v_x_3609_, v_x_3610_, v___y_3611_, v___y_3612_, v___y_3613_, v___y_3614_);
lean_dec(v___y_3614_);
lean_dec_ref(v___y_3613_);
lean_dec(v___y_3612_);
lean_dec_ref(v___y_3611_);
return v_res_3617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance___boxed(lean_object* v_inst_3618_, lean_object* v_expectedType_3619_, lean_object* v_root_3620_, lean_object* v_trace_3621_, lean_object* v_a_3622_, lean_object* v_a_3623_, lean_object* v_a_3624_, lean_object* v_a_3625_, lean_object* v_a_3626_){
_start:
{
uint8_t v_root_boxed_3627_; lean_object* v_res_3628_; 
v_root_boxed_3627_ = lean_unbox(v_root_3620_);
v_res_3628_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(v_inst_3618_, v_expectedType_3619_, v_root_boxed_3627_, v_trace_3621_, v_a_3622_, v_a_3623_, v_a_3624_, v_a_3625_);
lean_dec(v_a_3625_);
lean_dec_ref(v_a_3624_);
lean_dec(v_a_3623_);
lean_dec_ref(v_a_3622_);
return v_res_3628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6(lean_object* v_mvarId_3629_, lean_object* v_val_3630_, lean_object* v___y_3631_, lean_object* v___y_3632_, lean_object* v___y_3633_, lean_object* v___y_3634_){
_start:
{
lean_object* v___x_3636_; 
v___x_3636_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___redArg(v_mvarId_3629_, v_val_3630_, v___y_3632_);
return v___x_3636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6___boxed(lean_object* v_mvarId_3637_, lean_object* v_val_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_, lean_object* v___y_3642_, lean_object* v___y_3643_){
_start:
{
lean_object* v_res_3644_; 
v_res_3644_ = lp_mathlib_Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6(v_mvarId_3637_, v_val_3638_, v___y_3639_, v___y_3640_, v___y_3641_, v___y_3642_);
lean_dec(v___y_3642_);
lean_dec_ref(v___y_3641_);
lean_dec(v___y_3640_);
lean_dec_ref(v___y_3639_);
return v_res_3644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7(lean_object* v_upperBound_3645_, lean_object* v_fst_3646_, lean_object* v_args_3647_, uint8_t v___x_3648_, lean_object* v_fst_3649_, lean_object* v_val_3650_, lean_object* v_trace_3651_, lean_object* v_inst_3652_, lean_object* v_R_3653_, lean_object* v_a_3654_, lean_object* v_b_3655_, lean_object* v_c_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_){
_start:
{
lean_object* v___x_3662_; 
v___x_3662_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___redArg(v_upperBound_3645_, v_fst_3646_, v_args_3647_, v___x_3648_, v_fst_3649_, v_val_3650_, v_trace_3651_, v_a_3654_, v_b_3655_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_);
return v___x_3662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7___boxed(lean_object** _args){
lean_object* v_upperBound_3663_ = _args[0];
lean_object* v_fst_3664_ = _args[1];
lean_object* v_args_3665_ = _args[2];
lean_object* v___x_3666_ = _args[3];
lean_object* v_fst_3667_ = _args[4];
lean_object* v_val_3668_ = _args[5];
lean_object* v_trace_3669_ = _args[6];
lean_object* v_inst_3670_ = _args[7];
lean_object* v_R_3671_ = _args[8];
lean_object* v_a_3672_ = _args[9];
lean_object* v_b_3673_ = _args[10];
lean_object* v_c_3674_ = _args[11];
lean_object* v___y_3675_ = _args[12];
lean_object* v___y_3676_ = _args[13];
lean_object* v___y_3677_ = _args[14];
lean_object* v___y_3678_ = _args[15];
lean_object* v___y_3679_ = _args[16];
_start:
{
uint8_t v___x_73600__boxed_3680_; lean_object* v_res_3681_; 
v___x_73600__boxed_3680_ = lean_unbox(v___x_3666_);
v_res_3681_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__7(v_upperBound_3663_, v_fst_3664_, v_args_3665_, v___x_73600__boxed_3680_, v_fst_3667_, v_val_3668_, v_trace_3669_, v_inst_3670_, v_R_3671_, v_a_3672_, v_b_3673_, v_c_3674_, v___y_3675_, v___y_3676_, v___y_3677_, v___y_3678_);
lean_dec(v___y_3678_);
lean_dec_ref(v___y_3677_);
lean_dec(v___y_3676_);
lean_dec_ref(v___y_3675_);
lean_dec_ref(v_fst_3667_);
lean_dec_ref(v_args_3665_);
lean_dec_ref(v_fst_3664_);
lean_dec(v_upperBound_3663_);
return v_res_3681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19(lean_object* v_00_u03b1_3682_, lean_object* v_x_3683_, lean_object* v___y_3684_, lean_object* v___y_3685_, lean_object* v___y_3686_, lean_object* v___y_3687_){
_start:
{
lean_object* v___x_3689_; 
v___x_3689_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___redArg(v_x_3683_);
return v___x_3689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19___boxed(lean_object* v_00_u03b1_3690_, lean_object* v_x_3691_, lean_object* v___y_3692_, lean_object* v___y_3693_, lean_object* v___y_3694_, lean_object* v___y_3695_, lean_object* v___y_3696_){
_start:
{
lean_object* v_res_3697_; 
v_res_3697_ = lp_mathlib_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__13_spec__19(v_00_u03b1_3690_, v_x_3691_, v___y_3692_, v___y_3693_, v___y_3694_, v___y_3695_);
lean_dec(v___y_3695_);
lean_dec_ref(v___y_3694_);
lean_dec(v___y_3693_);
lean_dec_ref(v___y_3692_);
return v_res_3697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14(lean_object* v_upperBound_3698_, lean_object* v_fst_3699_, lean_object* v_args_3700_, uint8_t v___x_3701_, uint8_t v___x_3702_, lean_object* v_fst_3703_, lean_object* v_val_3704_, lean_object* v_trace_3705_, lean_object* v_inst_3706_, lean_object* v_R_3707_, lean_object* v_a_3708_, lean_object* v_b_3709_, lean_object* v_c_3710_, lean_object* v___y_3711_, lean_object* v___y_3712_, lean_object* v___y_3713_, lean_object* v___y_3714_){
_start:
{
lean_object* v___x_3716_; 
v___x_3716_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___redArg(v_upperBound_3698_, v_fst_3699_, v_args_3700_, v___x_3701_, v___x_3702_, v_fst_3703_, v_val_3704_, v_trace_3705_, v_a_3708_, v_b_3709_, v___y_3711_, v___y_3712_, v___y_3713_, v___y_3714_);
return v___x_3716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14___boxed(lean_object** _args){
lean_object* v_upperBound_3717_ = _args[0];
lean_object* v_fst_3718_ = _args[1];
lean_object* v_args_3719_ = _args[2];
lean_object* v___x_3720_ = _args[3];
lean_object* v___x_3721_ = _args[4];
lean_object* v_fst_3722_ = _args[5];
lean_object* v_val_3723_ = _args[6];
lean_object* v_trace_3724_ = _args[7];
lean_object* v_inst_3725_ = _args[8];
lean_object* v_R_3726_ = _args[9];
lean_object* v_a_3727_ = _args[10];
lean_object* v_b_3728_ = _args[11];
lean_object* v_c_3729_ = _args[12];
lean_object* v___y_3730_ = _args[13];
lean_object* v___y_3731_ = _args[14];
lean_object* v___y_3732_ = _args[15];
lean_object* v___y_3733_ = _args[16];
lean_object* v___y_3734_ = _args[17];
_start:
{
uint8_t v___x_73655__boxed_3735_; uint8_t v___x_73656__boxed_3736_; lean_object* v_res_3737_; 
v___x_73655__boxed_3735_ = lean_unbox(v___x_3720_);
v___x_73656__boxed_3736_ = lean_unbox(v___x_3721_);
v_res_3737_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__14(v_upperBound_3717_, v_fst_3718_, v_args_3719_, v___x_73655__boxed_3735_, v___x_73656__boxed_3736_, v_fst_3722_, v_val_3723_, v_trace_3724_, v_inst_3725_, v_R_3726_, v_a_3727_, v_b_3728_, v_c_3729_, v___y_3730_, v___y_3731_, v___y_3732_, v___y_3733_);
lean_dec(v___y_3733_);
lean_dec_ref(v___y_3732_);
lean_dec(v___y_3731_);
lean_dec_ref(v___y_3730_);
lean_dec_ref(v_fst_3722_);
lean_dec_ref(v_args_3719_);
lean_dec_ref(v_fst_3718_);
lean_dec(v_upperBound_3717_);
return v_res_3737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16(lean_object* v_upperBound_3738_, lean_object* v_fst_3739_, lean_object* v_args_3740_, lean_object* v_fst_3741_, uint8_t v___x_3742_, lean_object* v_val_3743_, lean_object* v_trace_3744_, lean_object* v_inst_3745_, lean_object* v_R_3746_, lean_object* v_a_3747_, lean_object* v_b_3748_, lean_object* v_c_3749_, lean_object* v___y_3750_, lean_object* v___y_3751_, lean_object* v___y_3752_, lean_object* v___y_3753_){
_start:
{
lean_object* v___x_3755_; 
v___x_3755_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___redArg(v_upperBound_3738_, v_fst_3739_, v_args_3740_, v_fst_3741_, v___x_3742_, v_val_3743_, v_trace_3744_, v_a_3747_, v_b_3748_, v___y_3750_, v___y_3751_, v___y_3752_, v___y_3753_);
return v___x_3755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16___boxed(lean_object** _args){
lean_object* v_upperBound_3756_ = _args[0];
lean_object* v_fst_3757_ = _args[1];
lean_object* v_args_3758_ = _args[2];
lean_object* v_fst_3759_ = _args[3];
lean_object* v___x_3760_ = _args[4];
lean_object* v_val_3761_ = _args[5];
lean_object* v_trace_3762_ = _args[6];
lean_object* v_inst_3763_ = _args[7];
lean_object* v_R_3764_ = _args[8];
lean_object* v_a_3765_ = _args[9];
lean_object* v_b_3766_ = _args[10];
lean_object* v_c_3767_ = _args[11];
lean_object* v___y_3768_ = _args[12];
lean_object* v___y_3769_ = _args[13];
lean_object* v___y_3770_ = _args[14];
lean_object* v___y_3771_ = _args[15];
lean_object* v___y_3772_ = _args[16];
_start:
{
uint8_t v___x_73694__boxed_3773_; lean_object* v_res_3774_; 
v___x_73694__boxed_3773_ = lean_unbox(v___x_3760_);
v_res_3774_ = lp_mathlib_WellFounded_opaqueFix_u2083___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__16(v_upperBound_3756_, v_fst_3757_, v_args_3758_, v_fst_3759_, v___x_73694__boxed_3773_, v_val_3761_, v_trace_3762_, v_inst_3763_, v_R_3764_, v_a_3765_, v_b_3766_, v_c_3767_, v___y_3768_, v___y_3769_, v___y_3770_, v___y_3771_);
lean_dec(v___y_3771_);
lean_dec_ref(v___y_3770_);
lean_dec(v___y_3769_);
lean_dec_ref(v___y_3768_);
lean_dec_ref(v_fst_3759_);
lean_dec_ref(v_args_3758_);
lean_dec_ref(v_fst_3757_);
lean_dec(v_upperBound_3756_);
return v_res_3774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6(lean_object* v_o_3775_, lean_object* v___y_3776_, lean_object* v___y_3777_, lean_object* v___y_3778_, lean_object* v___y_3779_){
_start:
{
lean_object* v___x_3781_; 
v___x_3781_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___redArg(v_o_3775_, v___y_3779_);
return v___x_3781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6___boxed(lean_object* v_o_3782_, lean_object* v___y_3783_, lean_object* v___y_3784_, lean_object* v___y_3785_, lean_object* v___y_3786_, lean_object* v___y_3787_){
_start:
{
lean_object* v_res_3788_; 
v_res_3788_ = lp_mathlib_Lean_Options_toLinterOptions___at___00Lean_Linter_getLinterOptions___at___00Lean_Linter_logLintIf___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__1_spec__1_spec__6(v_o_3782_, v___y_3783_, v___y_3784_, v___y_3785_, v___y_3786_);
lean_dec(v___y_3786_);
lean_dec_ref(v___y_3785_);
lean_dec(v___y_3784_);
lean_dec_ref(v___y_3783_);
return v_res_3788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4(lean_object* v_00_u03b1_3789_, lean_object* v_constName_3790_, lean_object* v___y_3791_, lean_object* v___y_3792_, lean_object* v___y_3793_, lean_object* v___y_3794_){
_start:
{
lean_object* v___x_3796_; 
v___x_3796_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___redArg(v_constName_3790_, v___y_3791_, v___y_3792_, v___y_3793_, v___y_3794_);
return v___x_3796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4___boxed(lean_object* v_00_u03b1_3797_, lean_object* v_constName_3798_, lean_object* v___y_3799_, lean_object* v___y_3800_, lean_object* v___y_3801_, lean_object* v___y_3802_, lean_object* v___y_3803_){
_start:
{
lean_object* v_res_3804_; 
v_res_3804_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4(v_00_u03b1_3797_, v_constName_3798_, v___y_3799_, v___y_3800_, v___y_3801_, v___y_3802_);
lean_dec(v___y_3802_);
lean_dec_ref(v___y_3801_);
lean_dec(v___y_3800_);
lean_dec_ref(v___y_3799_);
return v_res_3804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9(lean_object* v_00_u03b2_3805_, lean_object* v_x_3806_, lean_object* v_x_3807_, lean_object* v_x_3808_){
_start:
{
lean_object* v___x_3809_; 
v___x_3809_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9___redArg(v_x_3806_, v_x_3807_, v_x_3808_);
return v___x_3809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11(lean_object* v_00_u03b1_3810_, lean_object* v_ref_3811_, lean_object* v_constName_3812_, lean_object* v___y_3813_, lean_object* v___y_3814_, lean_object* v___y_3815_, lean_object* v___y_3816_){
_start:
{
lean_object* v___x_3818_; 
v___x_3818_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___redArg(v_ref_3811_, v_constName_3812_, v___y_3813_, v___y_3814_, v___y_3815_, v___y_3816_);
return v___x_3818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11___boxed(lean_object* v_00_u03b1_3819_, lean_object* v_ref_3820_, lean_object* v_constName_3821_, lean_object* v___y_3822_, lean_object* v___y_3823_, lean_object* v___y_3824_, lean_object* v___y_3825_, lean_object* v___y_3826_){
_start:
{
lean_object* v_res_3827_; 
v_res_3827_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11(v_00_u03b1_3819_, v_ref_3820_, v_constName_3821_, v___y_3822_, v___y_3823_, v___y_3824_, v___y_3825_);
lean_dec(v___y_3825_);
lean_dec_ref(v___y_3824_);
lean_dec(v___y_3823_);
lean_dec_ref(v___y_3822_);
lean_dec(v_ref_3820_);
return v_res_3827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15(lean_object* v_00_u03b2_3828_, lean_object* v_x_3829_, size_t v_x_3830_, size_t v_x_3831_, lean_object* v_x_3832_, lean_object* v_x_3833_){
_start:
{
lean_object* v___x_3834_; 
v___x_3834_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___redArg(v_x_3829_, v_x_3830_, v_x_3831_, v_x_3832_, v_x_3833_);
return v___x_3834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15___boxed(lean_object* v_00_u03b2_3835_, lean_object* v_x_3836_, lean_object* v_x_3837_, lean_object* v_x_3838_, lean_object* v_x_3839_, lean_object* v_x_3840_){
_start:
{
size_t v_x_73786__boxed_3841_; size_t v_x_73787__boxed_3842_; lean_object* v_res_3843_; 
v_x_73786__boxed_3841_ = lean_unbox_usize(v_x_3837_);
lean_dec(v_x_3837_);
v_x_73787__boxed_3842_ = lean_unbox_usize(v_x_3838_);
lean_dec(v_x_3838_);
v_res_3843_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15(v_00_u03b2_3835_, v_x_3836_, v_x_73786__boxed_3841_, v_x_73787__boxed_3842_, v_x_3839_, v_x_3840_);
return v_res_3843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26(lean_object* v_00_u03b1_3844_, lean_object* v_ref_3845_, lean_object* v_msg_3846_, lean_object* v_declHint_3847_, lean_object* v___y_3848_, lean_object* v___y_3849_, lean_object* v___y_3850_, lean_object* v___y_3851_){
_start:
{
lean_object* v___x_3853_; 
v___x_3853_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___redArg(v_ref_3845_, v_msg_3846_, v_declHint_3847_, v___y_3848_, v___y_3849_, v___y_3850_, v___y_3851_);
return v___x_3853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26___boxed(lean_object* v_00_u03b1_3854_, lean_object* v_ref_3855_, lean_object* v_msg_3856_, lean_object* v_declHint_3857_, lean_object* v___y_3858_, lean_object* v___y_3859_, lean_object* v___y_3860_, lean_object* v___y_3861_, lean_object* v___y_3862_){
_start:
{
lean_object* v_res_3863_; 
v_res_3863_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26(v_00_u03b1_3854_, v_ref_3855_, v_msg_3856_, v_declHint_3857_, v___y_3858_, v___y_3859_, v___y_3860_, v___y_3861_);
lean_dec(v___y_3861_);
lean_dec_ref(v___y_3860_);
lean_dec(v___y_3859_);
lean_dec_ref(v___y_3858_);
lean_dec(v_ref_3855_);
return v_res_3863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29(lean_object* v_00_u03b2_3864_, lean_object* v_n_3865_, lean_object* v_k_3866_, lean_object* v_v_3867_){
_start:
{
lean_object* v___x_3868_; 
v___x_3868_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29___redArg(v_n_3865_, v_k_3866_, v_v_3867_);
return v___x_3868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30(lean_object* v_00_u03b2_3869_, size_t v_depth_3870_, lean_object* v_keys_3871_, lean_object* v_vals_3872_, lean_object* v_heq_3873_, lean_object* v_i_3874_, lean_object* v_entries_3875_){
_start:
{
lean_object* v___x_3876_; 
v___x_3876_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___redArg(v_depth_3870_, v_keys_3871_, v_vals_3872_, v_i_3874_, v_entries_3875_);
return v___x_3876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30___boxed(lean_object* v_00_u03b2_3877_, lean_object* v_depth_3878_, lean_object* v_keys_3879_, lean_object* v_vals_3880_, lean_object* v_heq_3881_, lean_object* v_i_3882_, lean_object* v_entries_3883_){
_start:
{
size_t v_depth_boxed_3884_; lean_object* v_res_3885_; 
v_depth_boxed_3884_ = lean_unbox_usize(v_depth_3878_);
lean_dec(v_depth_3878_);
v_res_3885_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__30(v_00_u03b2_3877_, v_depth_boxed_3884_, v_keys_3879_, v_vals_3880_, v_heq_3881_, v_i_3882_, v_entries_3883_);
lean_dec_ref(v_vals_3880_);
lean_dec_ref(v_keys_3879_);
return v_res_3885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34(lean_object* v_msg_3886_, lean_object* v_declHint_3887_, lean_object* v___y_3888_, lean_object* v___y_3889_, lean_object* v___y_3890_, lean_object* v___y_3891_){
_start:
{
lean_object* v___x_3893_; 
v___x_3893_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___redArg(v_msg_3886_, v_declHint_3887_, v___y_3891_);
return v___x_3893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34___boxed(lean_object* v_msg_3894_, lean_object* v_declHint_3895_, lean_object* v___y_3896_, lean_object* v___y_3897_, lean_object* v___y_3898_, lean_object* v___y_3899_, lean_object* v___y_3900_){
_start:
{
lean_object* v_res_3901_; 
v_res_3901_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__30_spec__34(v_msg_3894_, v_declHint_3895_, v___y_3896_, v___y_3897_, v___y_3898_, v___y_3899_);
lean_dec(v___y_3899_);
lean_dec_ref(v___y_3898_);
lean_dec(v___y_3897_);
lean_dec_ref(v___y_3896_);
return v_res_3901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31(lean_object* v_00_u03b1_3902_, lean_object* v_ref_3903_, lean_object* v_msg_3904_, lean_object* v___y_3905_, lean_object* v___y_3906_, lean_object* v___y_3907_, lean_object* v___y_3908_){
_start:
{
lean_object* v___x_3910_; 
v___x_3910_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___redArg(v_ref_3903_, v_msg_3904_, v___y_3905_, v___y_3906_, v___y_3907_, v___y_3908_);
return v___x_3910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31___boxed(lean_object* v_00_u03b1_3911_, lean_object* v_ref_3912_, lean_object* v_msg_3913_, lean_object* v___y_3914_, lean_object* v___y_3915_, lean_object* v___y_3916_, lean_object* v___y_3917_, lean_object* v___y_3918_){
_start:
{
lean_object* v_res_3919_; 
v_res_3919_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__2_spec__4_spec__11_spec__26_spec__31(v_00_u03b1_3911_, v_ref_3912_, v_msg_3913_, v___y_3914_, v___y_3915_, v___y_3916_, v___y_3917_);
lean_dec(v___y_3917_);
lean_dec_ref(v___y_3916_);
lean_dec(v___y_3915_);
lean_dec_ref(v___y_3914_);
lean_dec(v_ref_3912_);
return v_res_3919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29_spec__34(lean_object* v_00_u03b2_3920_, lean_object* v_x_3921_, lean_object* v_x_3922_, lean_object* v_x_3923_, lean_object* v_x_3924_){
_start:
{
lean_object* v___x_3925_; 
v___x_3925_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__6_spec__9_spec__15_spec__29_spec__34___redArg(v_x_3921_, v_x_3922_, v_x_3923_, v_x_3924_);
return v___x_3925_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3953_; lean_object* v___x_3954_; lean_object* v___x_3955_; 
v___x_3953_ = lean_box(0);
v___x_3954_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3955_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3955_, 0, v___x_3954_);
lean_ctor_set(v___x_3955_, 1, v___x_3953_);
return v___x_3955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg(){
_start:
{
lean_object* v___x_3957_; lean_object* v___x_3958_; 
v___x_3957_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___closed__0);
v___x_3958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3958_, 0, v___x_3957_);
return v___x_3958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg___boxed(lean_object* v___y_3959_){
_start:
{
lean_object* v_res_3960_; 
v_res_3960_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg();
return v_res_3960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0(lean_object* v_00_u03b1_3961_, lean_object* v___y_3962_, lean_object* v___y_3963_, lean_object* v___y_3964_, lean_object* v___y_3965_, lean_object* v___y_3966_, lean_object* v___y_3967_){
_start:
{
lean_object* v___x_3969_; 
v___x_3969_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg();
return v___x_3969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___boxed(lean_object* v_00_u03b1_3970_, lean_object* v___y_3971_, lean_object* v___y_3972_, lean_object* v___y_3973_, lean_object* v___y_3974_, lean_object* v___y_3975_, lean_object* v___y_3976_, lean_object* v___y_3977_){
_start:
{
lean_object* v_res_3978_; 
v_res_3978_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0(v_00_u03b1_3970_, v___y_3971_, v___y_3972_, v___y_3973_, v___y_3974_, v___y_3975_, v___y_3976_);
lean_dec(v___y_3976_);
lean_dec_ref(v___y_3975_);
lean_dec(v___y_3974_);
lean_dec_ref(v___y_3973_);
lean_dec(v___y_3972_);
lean_dec_ref(v___y_3971_);
return v_res_3978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___lam__0(lean_object* v_k_3979_, lean_object* v___y_3980_, lean_object* v___y_3981_, lean_object* v___y_3982_, lean_object* v___y_3983_, lean_object* v___y_3984_, lean_object* v___y_3985_){
_start:
{
lean_object* v___x_3987_; 
lean_inc(v___y_3981_);
lean_inc_ref(v___y_3980_);
v___x_3987_ = lean_apply_7(v_k_3979_, v___y_3980_, v___y_3981_, v___y_3982_, v___y_3983_, v___y_3984_, v___y_3985_, lean_box(0));
return v___x_3987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___lam__0___boxed(lean_object* v_k_3988_, lean_object* v___y_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_, lean_object* v___y_3993_, lean_object* v___y_3994_, lean_object* v___y_3995_){
_start:
{
lean_object* v_res_3996_; 
v_res_3996_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___lam__0(v_k_3988_, v___y_3989_, v___y_3990_, v___y_3991_, v___y_3992_, v___y_3993_, v___y_3994_);
lean_dec(v___y_3990_);
lean_dec_ref(v___y_3989_);
return v_res_3996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg(lean_object* v_k_3997_, uint8_t v_allowLevelAssignments_3998_, lean_object* v___y_3999_, lean_object* v___y_4000_, lean_object* v___y_4001_, lean_object* v___y_4002_, lean_object* v___y_4003_, lean_object* v___y_4004_){
_start:
{
lean_object* v___f_4006_; lean_object* v___x_4007_; 
lean_inc(v___y_4000_);
lean_inc_ref(v___y_3999_);
v___f_4006_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_4006_, 0, v_k_3997_);
lean_closure_set(v___f_4006_, 1, v___y_3999_);
lean_closure_set(v___f_4006_, 2, v___y_4000_);
v___x_4007_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_3998_, v___f_4006_, v___y_4001_, v___y_4002_, v___y_4003_, v___y_4004_);
if (lean_obj_tag(v___x_4007_) == 0)
{
return v___x_4007_;
}
else
{
lean_object* v_a_4008_; lean_object* v___x_4010_; uint8_t v_isShared_4011_; uint8_t v_isSharedCheck_4015_; 
v_a_4008_ = lean_ctor_get(v___x_4007_, 0);
v_isSharedCheck_4015_ = !lean_is_exclusive(v___x_4007_);
if (v_isSharedCheck_4015_ == 0)
{
v___x_4010_ = v___x_4007_;
v_isShared_4011_ = v_isSharedCheck_4015_;
goto v_resetjp_4009_;
}
else
{
lean_inc(v_a_4008_);
lean_dec(v___x_4007_);
v___x_4010_ = lean_box(0);
v_isShared_4011_ = v_isSharedCheck_4015_;
goto v_resetjp_4009_;
}
v_resetjp_4009_:
{
lean_object* v___x_4013_; 
if (v_isShared_4011_ == 0)
{
v___x_4013_ = v___x_4010_;
goto v_reusejp_4012_;
}
else
{
lean_object* v_reuseFailAlloc_4014_; 
v_reuseFailAlloc_4014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4014_, 0, v_a_4008_);
v___x_4013_ = v_reuseFailAlloc_4014_;
goto v_reusejp_4012_;
}
v_reusejp_4012_:
{
return v___x_4013_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg___boxed(lean_object* v_k_4016_, lean_object* v_allowLevelAssignments_4017_, lean_object* v___y_4018_, lean_object* v___y_4019_, lean_object* v___y_4020_, lean_object* v___y_4021_, lean_object* v___y_4022_, lean_object* v___y_4023_, lean_object* v___y_4024_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_4025_; lean_object* v_res_4026_; 
v_allowLevelAssignments_boxed_4025_ = lean_unbox(v_allowLevelAssignments_4017_);
v_res_4026_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg(v_k_4016_, v_allowLevelAssignments_boxed_4025_, v___y_4018_, v___y_4019_, v___y_4020_, v___y_4021_, v___y_4022_, v___y_4023_);
lean_dec(v___y_4023_);
lean_dec_ref(v___y_4022_);
lean_dec(v___y_4021_);
lean_dec_ref(v___y_4020_);
lean_dec(v___y_4019_);
lean_dec_ref(v___y_4018_);
return v_res_4026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1(lean_object* v_00_u03b1_4027_, lean_object* v_k_4028_, uint8_t v_allowLevelAssignments_4029_, lean_object* v___y_4030_, lean_object* v___y_4031_, lean_object* v___y_4032_, lean_object* v___y_4033_, lean_object* v___y_4034_, lean_object* v___y_4035_){
_start:
{
lean_object* v___x_4037_; 
v___x_4037_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg(v_k_4028_, v_allowLevelAssignments_4029_, v___y_4030_, v___y_4031_, v___y_4032_, v___y_4033_, v___y_4034_, v___y_4035_);
return v___x_4037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___boxed(lean_object* v_00_u03b1_4038_, lean_object* v_k_4039_, lean_object* v_allowLevelAssignments_4040_, lean_object* v___y_4041_, lean_object* v___y_4042_, lean_object* v___y_4043_, lean_object* v___y_4044_, lean_object* v___y_4045_, lean_object* v___y_4046_, lean_object* v___y_4047_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_4048_; lean_object* v_res_4049_; 
v_allowLevelAssignments_boxed_4048_ = lean_unbox(v_allowLevelAssignments_4040_);
v_res_4049_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1(v_00_u03b1_4038_, v_k_4039_, v_allowLevelAssignments_boxed_4048_, v___y_4041_, v___y_4042_, v___y_4043_, v___y_4044_, v___y_4045_, v___y_4046_);
lean_dec(v___y_4046_);
lean_dec_ref(v___y_4045_);
lean_dec(v___y_4044_);
lean_dec_ref(v___y_4043_);
lean_dec(v___y_4042_);
lean_dec_ref(v___y_4041_);
return v_res_4049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___lam__0(lean_object* v_k_4050_, lean_object* v___y_4051_, lean_object* v___y_4052_, lean_object* v_b_4053_, lean_object* v_c_4054_, lean_object* v___y_4055_, lean_object* v___y_4056_, lean_object* v___y_4057_, lean_object* v___y_4058_){
_start:
{
lean_object* v___x_4060_; 
lean_inc(v___y_4058_);
lean_inc_ref(v___y_4057_);
lean_inc(v___y_4056_);
lean_inc_ref(v___y_4055_);
lean_inc(v___y_4052_);
lean_inc_ref(v___y_4051_);
v___x_4060_ = lean_apply_9(v_k_4050_, v_b_4053_, v_c_4054_, v___y_4051_, v___y_4052_, v___y_4055_, v___y_4056_, v___y_4057_, v___y_4058_, lean_box(0));
return v___x_4060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___lam__0___boxed(lean_object* v_k_4061_, lean_object* v___y_4062_, lean_object* v___y_4063_, lean_object* v_b_4064_, lean_object* v_c_4065_, lean_object* v___y_4066_, lean_object* v___y_4067_, lean_object* v___y_4068_, lean_object* v___y_4069_, lean_object* v___y_4070_){
_start:
{
lean_object* v_res_4071_; 
v_res_4071_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___lam__0(v_k_4061_, v___y_4062_, v___y_4063_, v_b_4064_, v_c_4065_, v___y_4066_, v___y_4067_, v___y_4068_, v___y_4069_);
lean_dec(v___y_4069_);
lean_dec_ref(v___y_4068_);
lean_dec(v___y_4067_);
lean_dec_ref(v___y_4066_);
lean_dec(v___y_4063_);
lean_dec_ref(v___y_4062_);
return v_res_4071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg(lean_object* v_type_4072_, lean_object* v_k_4073_, uint8_t v_cleanupAnnotations_4074_, uint8_t v_whnfType_4075_, lean_object* v___y_4076_, lean_object* v___y_4077_, lean_object* v___y_4078_, lean_object* v___y_4079_, lean_object* v___y_4080_, lean_object* v___y_4081_){
_start:
{
lean_object* v___f_4083_; lean_object* v___x_4084_; 
lean_inc(v___y_4077_);
lean_inc_ref(v___y_4076_);
v___f_4083_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_4083_, 0, v_k_4073_);
lean_closure_set(v___f_4083_, 1, v___y_4076_);
lean_closure_set(v___f_4083_, 2, v___y_4077_);
v___x_4084_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_4072_, v___f_4083_, v_cleanupAnnotations_4074_, v_whnfType_4075_, v___y_4078_, v___y_4079_, v___y_4080_, v___y_4081_);
if (lean_obj_tag(v___x_4084_) == 0)
{
return v___x_4084_;
}
else
{
lean_object* v_a_4085_; lean_object* v___x_4087_; uint8_t v_isShared_4088_; uint8_t v_isSharedCheck_4092_; 
v_a_4085_ = lean_ctor_get(v___x_4084_, 0);
v_isSharedCheck_4092_ = !lean_is_exclusive(v___x_4084_);
if (v_isSharedCheck_4092_ == 0)
{
v___x_4087_ = v___x_4084_;
v_isShared_4088_ = v_isSharedCheck_4092_;
goto v_resetjp_4086_;
}
else
{
lean_inc(v_a_4085_);
lean_dec(v___x_4084_);
v___x_4087_ = lean_box(0);
v_isShared_4088_ = v_isSharedCheck_4092_;
goto v_resetjp_4086_;
}
v_resetjp_4086_:
{
lean_object* v___x_4090_; 
if (v_isShared_4088_ == 0)
{
v___x_4090_ = v___x_4087_;
goto v_reusejp_4089_;
}
else
{
lean_object* v_reuseFailAlloc_4091_; 
v_reuseFailAlloc_4091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4091_, 0, v_a_4085_);
v___x_4090_ = v_reuseFailAlloc_4091_;
goto v_reusejp_4089_;
}
v_reusejp_4089_:
{
return v___x_4090_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg___boxed(lean_object* v_type_4093_, lean_object* v_k_4094_, lean_object* v_cleanupAnnotations_4095_, lean_object* v_whnfType_4096_, lean_object* v___y_4097_, lean_object* v___y_4098_, lean_object* v___y_4099_, lean_object* v___y_4100_, lean_object* v___y_4101_, lean_object* v___y_4102_, lean_object* v___y_4103_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_4104_; uint8_t v_whnfType_boxed_4105_; lean_object* v_res_4106_; 
v_cleanupAnnotations_boxed_4104_ = lean_unbox(v_cleanupAnnotations_4095_);
v_whnfType_boxed_4105_ = lean_unbox(v_whnfType_4096_);
v_res_4106_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg(v_type_4093_, v_k_4094_, v_cleanupAnnotations_boxed_4104_, v_whnfType_boxed_4105_, v___y_4097_, v___y_4098_, v___y_4099_, v___y_4100_, v___y_4101_, v___y_4102_);
lean_dec(v___y_4102_);
lean_dec_ref(v___y_4101_);
lean_dec(v___y_4100_);
lean_dec_ref(v___y_4099_);
lean_dec(v___y_4098_);
lean_dec_ref(v___y_4097_);
return v_res_4106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2(lean_object* v_00_u03b1_4107_, lean_object* v_type_4108_, lean_object* v_k_4109_, uint8_t v_cleanupAnnotations_4110_, uint8_t v_whnfType_4111_, lean_object* v___y_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_, lean_object* v___y_4117_){
_start:
{
lean_object* v___x_4119_; 
v___x_4119_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg(v_type_4108_, v_k_4109_, v_cleanupAnnotations_4110_, v_whnfType_4111_, v___y_4112_, v___y_4113_, v___y_4114_, v___y_4115_, v___y_4116_, v___y_4117_);
return v___x_4119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___boxed(lean_object* v_00_u03b1_4120_, lean_object* v_type_4121_, lean_object* v_k_4122_, lean_object* v_cleanupAnnotations_4123_, lean_object* v_whnfType_4124_, lean_object* v___y_4125_, lean_object* v___y_4126_, lean_object* v___y_4127_, lean_object* v___y_4128_, lean_object* v___y_4129_, lean_object* v___y_4130_, lean_object* v___y_4131_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_4132_; uint8_t v_whnfType_boxed_4133_; lean_object* v_res_4134_; 
v_cleanupAnnotations_boxed_4132_ = lean_unbox(v_cleanupAnnotations_4123_);
v_whnfType_boxed_4133_ = lean_unbox(v_whnfType_4124_);
v_res_4134_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2(v_00_u03b1_4120_, v_type_4121_, v_k_4122_, v_cleanupAnnotations_boxed_4132_, v_whnfType_boxed_4133_, v___y_4125_, v___y_4126_, v___y_4127_, v___y_4128_, v___y_4129_, v___y_4130_);
lean_dec(v___y_4130_);
lean_dec_ref(v___y_4129_);
lean_dec(v___y_4128_);
lean_dec_ref(v___y_4127_);
lean_dec(v___y_4126_);
lean_dec_ref(v___y_4125_);
return v_res_4134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__0(lean_object* v_a_4135_, lean_object* v_expectedType_4136_, uint8_t v___x_4137_, lean_object* v___x_4138_, lean_object* v___y_4139_, lean_object* v___y_4140_, lean_object* v___y_4141_, lean_object* v___y_4142_, lean_object* v___y_4143_, lean_object* v___y_4144_){
_start:
{
lean_object* v___x_4146_; 
v___x_4146_ = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance(v_a_4135_, v_expectedType_4136_, v___x_4137_, v___x_4138_, v___y_4141_, v___y_4142_, v___y_4143_, v___y_4144_);
return v___x_4146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__0___boxed(lean_object* v_a_4147_, lean_object* v_expectedType_4148_, lean_object* v___x_4149_, lean_object* v___x_4150_, lean_object* v___y_4151_, lean_object* v___y_4152_, lean_object* v___y_4153_, lean_object* v___y_4154_, lean_object* v___y_4155_, lean_object* v___y_4156_, lean_object* v___y_4157_){
_start:
{
uint8_t v___x_8856__boxed_4158_; lean_object* v_res_4159_; 
v___x_8856__boxed_4158_ = lean_unbox(v___x_4149_);
v_res_4159_ = lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__0(v_a_4147_, v_expectedType_4148_, v___x_8856__boxed_4158_, v___x_4150_, v___y_4151_, v___y_4152_, v___y_4153_, v___y_4154_, v___y_4155_, v___y_4156_);
lean_dec(v___y_4156_);
lean_dec_ref(v___y_4155_);
lean_dec(v___y_4154_);
lean_dec_ref(v___y_4153_);
lean_dec(v___y_4152_);
lean_dec_ref(v___y_4151_);
return v_res_4159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1(lean_object* v_a_4162_, uint8_t v___x_4163_, lean_object* v_xs_4164_, lean_object* v_expectedType_4165_, lean_object* v___y_4166_, lean_object* v___y_4167_, lean_object* v___y_4168_, lean_object* v___y_4169_, lean_object* v___y_4170_, lean_object* v___y_4171_){
_start:
{
lean_object* v___x_4173_; lean_object* v___x_4174_; lean_object* v___f_4175_; uint8_t v___x_4176_; lean_object* v___x_4177_; 
v___x_4173_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___closed__0));
v___x_4174_ = lean_box(v___x_4163_);
v___f_4175_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__0___boxed), 11, 4);
lean_closure_set(v___f_4175_, 0, v_a_4162_);
lean_closure_set(v___f_4175_, 1, v_expectedType_4165_);
lean_closure_set(v___f_4175_, 2, v___x_4174_);
lean_closure_set(v___f_4175_, 3, v___x_4173_);
v___x_4176_ = 0;
v___x_4177_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__1___redArg(v___f_4175_, v___x_4176_, v___y_4166_, v___y_4167_, v___y_4168_, v___y_4169_, v___y_4170_, v___y_4171_);
if (lean_obj_tag(v___x_4177_) == 0)
{
lean_object* v_a_4178_; uint8_t v___x_4179_; lean_object* v___x_4180_; 
v_a_4178_ = lean_ctor_get(v___x_4177_, 0);
lean_inc(v_a_4178_);
lean_dec_ref_known(v___x_4177_, 1);
v___x_4179_ = 1;
v___x_4180_ = l_Lean_Meta_mkLambdaFVars(v_xs_4164_, v_a_4178_, v___x_4176_, v___x_4163_, v___x_4176_, v___x_4163_, v___x_4179_, v___y_4168_, v___y_4169_, v___y_4170_, v___y_4171_);
return v___x_4180_;
}
else
{
return v___x_4177_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___boxed(lean_object* v_a_4181_, lean_object* v___x_4182_, lean_object* v_xs_4183_, lean_object* v_expectedType_4184_, lean_object* v___y_4185_, lean_object* v___y_4186_, lean_object* v___y_4187_, lean_object* v___y_4188_, lean_object* v___y_4189_, lean_object* v___y_4190_, lean_object* v___y_4191_){
_start:
{
uint8_t v___x_8892__boxed_4192_; lean_object* v_res_4193_; 
v___x_8892__boxed_4192_ = lean_unbox(v___x_4182_);
v_res_4193_ = lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1(v_a_4181_, v___x_8892__boxed_4192_, v_xs_4183_, v_expectedType_4184_, v___y_4185_, v___y_4186_, v___y_4187_, v___y_4188_, v___y_4189_, v___y_4190_);
lean_dec(v___y_4190_);
lean_dec_ref(v___y_4189_);
lean_dec(v___y_4188_);
lean_dec_ref(v___y_4187_);
lean_dec(v___y_4186_);
lean_dec_ref(v___y_4185_);
lean_dec_ref(v_xs_4183_);
return v_res_4193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg(lean_object* v_ref_4194_, lean_object* v_msgData_4195_, uint8_t v_severity_4196_, uint8_t v_isSilent_4197_, lean_object* v___y_4198_, lean_object* v___y_4199_, lean_object* v___y_4200_, lean_object* v___y_4201_){
_start:
{
lean_object* v___y_4204_; lean_object* v___y_4205_; uint8_t v___y_4206_; lean_object* v___y_4207_; uint8_t v___y_4208_; lean_object* v___y_4209_; lean_object* v___y_4210_; lean_object* v___y_4211_; lean_object* v___y_4212_; lean_object* v___y_4240_; lean_object* v___y_4241_; uint8_t v___y_4242_; uint8_t v___y_4243_; lean_object* v___y_4244_; lean_object* v___y_4245_; uint8_t v___y_4246_; lean_object* v___y_4247_; lean_object* v___y_4265_; lean_object* v___y_4266_; uint8_t v___y_4267_; uint8_t v___y_4268_; lean_object* v___y_4269_; lean_object* v___y_4270_; uint8_t v___y_4271_; lean_object* v___y_4272_; lean_object* v___y_4276_; lean_object* v___y_4277_; uint8_t v___y_4278_; lean_object* v___y_4279_; lean_object* v___y_4280_; uint8_t v___y_4281_; uint8_t v___y_4282_; uint8_t v___x_4287_; lean_object* v___y_4289_; lean_object* v___y_4290_; lean_object* v___y_4291_; lean_object* v___y_4292_; uint8_t v___y_4293_; uint8_t v___y_4294_; uint8_t v___y_4295_; uint8_t v___y_4297_; uint8_t v___x_4312_; 
v___x_4287_ = 2;
v___x_4312_ = l_Lean_instBEqMessageSeverity_beq(v_severity_4196_, v___x_4287_);
if (v___x_4312_ == 0)
{
v___y_4297_ = v___x_4312_;
goto v___jp_4296_;
}
else
{
uint8_t v___x_4313_; 
lean_inc_ref(v_msgData_4195_);
v___x_4313_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_4195_);
v___y_4297_ = v___x_4313_;
goto v___jp_4296_;
}
v___jp_4203_:
{
lean_object* v___x_4213_; lean_object* v_currNamespace_4214_; lean_object* v_openDecls_4215_; lean_object* v_env_4216_; lean_object* v_nextMacroScope_4217_; lean_object* v_ngen_4218_; lean_object* v_auxDeclNGen_4219_; lean_object* v_traceState_4220_; lean_object* v_cache_4221_; lean_object* v_messages_4222_; lean_object* v_infoState_4223_; lean_object* v_snapshotTasks_4224_; lean_object* v___x_4226_; uint8_t v_isShared_4227_; uint8_t v_isSharedCheck_4238_; 
v___x_4213_ = lean_st_ref_take(v___y_4212_);
v_currNamespace_4214_ = lean_ctor_get(v___y_4211_, 6);
v_openDecls_4215_ = lean_ctor_get(v___y_4211_, 7);
v_env_4216_ = lean_ctor_get(v___x_4213_, 0);
v_nextMacroScope_4217_ = lean_ctor_get(v___x_4213_, 1);
v_ngen_4218_ = lean_ctor_get(v___x_4213_, 2);
v_auxDeclNGen_4219_ = lean_ctor_get(v___x_4213_, 3);
v_traceState_4220_ = lean_ctor_get(v___x_4213_, 4);
v_cache_4221_ = lean_ctor_get(v___x_4213_, 5);
v_messages_4222_ = lean_ctor_get(v___x_4213_, 6);
v_infoState_4223_ = lean_ctor_get(v___x_4213_, 7);
v_snapshotTasks_4224_ = lean_ctor_get(v___x_4213_, 8);
v_isSharedCheck_4238_ = !lean_is_exclusive(v___x_4213_);
if (v_isSharedCheck_4238_ == 0)
{
v___x_4226_ = v___x_4213_;
v_isShared_4227_ = v_isSharedCheck_4238_;
goto v_resetjp_4225_;
}
else
{
lean_inc(v_snapshotTasks_4224_);
lean_inc(v_infoState_4223_);
lean_inc(v_messages_4222_);
lean_inc(v_cache_4221_);
lean_inc(v_traceState_4220_);
lean_inc(v_auxDeclNGen_4219_);
lean_inc(v_ngen_4218_);
lean_inc(v_nextMacroScope_4217_);
lean_inc(v_env_4216_);
lean_dec(v___x_4213_);
v___x_4226_ = lean_box(0);
v_isShared_4227_ = v_isSharedCheck_4238_;
goto v_resetjp_4225_;
}
v_resetjp_4225_:
{
lean_object* v___x_4228_; lean_object* v___x_4229_; lean_object* v___x_4230_; lean_object* v___x_4231_; lean_object* v___x_4233_; 
lean_inc(v_openDecls_4215_);
lean_inc(v_currNamespace_4214_);
v___x_4228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4228_, 0, v_currNamespace_4214_);
lean_ctor_set(v___x_4228_, 1, v_openDecls_4215_);
v___x_4229_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_4229_, 0, v___x_4228_);
lean_ctor_set(v___x_4229_, 1, v___y_4207_);
lean_inc_ref(v___y_4210_);
lean_inc_ref(v___y_4204_);
v___x_4230_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_4230_, 0, v___y_4204_);
lean_ctor_set(v___x_4230_, 1, v___y_4205_);
lean_ctor_set(v___x_4230_, 2, v___y_4209_);
lean_ctor_set(v___x_4230_, 3, v___y_4210_);
lean_ctor_set(v___x_4230_, 4, v___x_4229_);
lean_ctor_set_uint8(v___x_4230_, sizeof(void*)*5, v___y_4208_);
lean_ctor_set_uint8(v___x_4230_, sizeof(void*)*5 + 1, v___y_4206_);
lean_ctor_set_uint8(v___x_4230_, sizeof(void*)*5 + 2, v_isSilent_4197_);
v___x_4231_ = l_Lean_MessageLog_add(v___x_4230_, v_messages_4222_);
if (v_isShared_4227_ == 0)
{
lean_ctor_set(v___x_4226_, 6, v___x_4231_);
v___x_4233_ = v___x_4226_;
goto v_reusejp_4232_;
}
else
{
lean_object* v_reuseFailAlloc_4237_; 
v_reuseFailAlloc_4237_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4237_, 0, v_env_4216_);
lean_ctor_set(v_reuseFailAlloc_4237_, 1, v_nextMacroScope_4217_);
lean_ctor_set(v_reuseFailAlloc_4237_, 2, v_ngen_4218_);
lean_ctor_set(v_reuseFailAlloc_4237_, 3, v_auxDeclNGen_4219_);
lean_ctor_set(v_reuseFailAlloc_4237_, 4, v_traceState_4220_);
lean_ctor_set(v_reuseFailAlloc_4237_, 5, v_cache_4221_);
lean_ctor_set(v_reuseFailAlloc_4237_, 6, v___x_4231_);
lean_ctor_set(v_reuseFailAlloc_4237_, 7, v_infoState_4223_);
lean_ctor_set(v_reuseFailAlloc_4237_, 8, v_snapshotTasks_4224_);
v___x_4233_ = v_reuseFailAlloc_4237_;
goto v_reusejp_4232_;
}
v_reusejp_4232_:
{
lean_object* v___x_4234_; lean_object* v___x_4235_; lean_object* v___x_4236_; 
v___x_4234_ = lean_st_ref_set(v___y_4212_, v___x_4233_);
v___x_4235_ = lean_box(0);
v___x_4236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4236_, 0, v___x_4235_);
return v___x_4236_;
}
}
}
v___jp_4239_:
{
lean_object* v___x_4248_; lean_object* v___x_4249_; lean_object* v_a_4250_; lean_object* v___x_4252_; uint8_t v_isShared_4253_; uint8_t v_isSharedCheck_4263_; 
v___x_4248_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_4195_);
v___x_4249_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_error_spec__1_spec__1(v___x_4248_, v___y_4198_, v___y_4199_, v___y_4200_, v___y_4201_);
v_a_4250_ = lean_ctor_get(v___x_4249_, 0);
v_isSharedCheck_4263_ = !lean_is_exclusive(v___x_4249_);
if (v_isSharedCheck_4263_ == 0)
{
v___x_4252_ = v___x_4249_;
v_isShared_4253_ = v_isSharedCheck_4263_;
goto v_resetjp_4251_;
}
else
{
lean_inc(v_a_4250_);
lean_dec(v___x_4249_);
v___x_4252_ = lean_box(0);
v_isShared_4253_ = v_isSharedCheck_4263_;
goto v_resetjp_4251_;
}
v_resetjp_4251_:
{
lean_object* v___x_4254_; lean_object* v___x_4255_; lean_object* v___x_4256_; lean_object* v___x_4257_; 
lean_inc_ref_n(v___y_4244_, 2);
v___x_4254_ = l_Lean_FileMap_toPosition(v___y_4244_, v___y_4245_);
lean_dec(v___y_4245_);
v___x_4255_ = l_Lean_FileMap_toPosition(v___y_4244_, v___y_4247_);
lean_dec(v___y_4247_);
v___x_4256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4256_, 0, v___x_4255_);
v___x_4257_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___closed__0));
if (v___y_4246_ == 0)
{
lean_del_object(v___x_4252_);
lean_dec_ref(v___y_4240_);
v___y_4204_ = v___y_4241_;
v___y_4205_ = v___x_4254_;
v___y_4206_ = v___y_4242_;
v___y_4207_ = v_a_4250_;
v___y_4208_ = v___y_4243_;
v___y_4209_ = v___x_4256_;
v___y_4210_ = v___x_4257_;
v___y_4211_ = v___y_4200_;
v___y_4212_ = v___y_4201_;
goto v___jp_4203_;
}
else
{
uint8_t v___x_4258_; 
lean_inc(v_a_4250_);
v___x_4258_ = l_Lean_MessageData_hasTag(v___y_4240_, v_a_4250_);
if (v___x_4258_ == 0)
{
lean_object* v___x_4259_; lean_object* v___x_4261_; 
lean_dec_ref_known(v___x_4256_, 1);
lean_dec_ref(v___x_4254_);
lean_dec(v_a_4250_);
v___x_4259_ = lean_box(0);
if (v_isShared_4253_ == 0)
{
lean_ctor_set(v___x_4252_, 0, v___x_4259_);
v___x_4261_ = v___x_4252_;
goto v_reusejp_4260_;
}
else
{
lean_object* v_reuseFailAlloc_4262_; 
v_reuseFailAlloc_4262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4262_, 0, v___x_4259_);
v___x_4261_ = v_reuseFailAlloc_4262_;
goto v_reusejp_4260_;
}
v_reusejp_4260_:
{
return v___x_4261_;
}
}
else
{
lean_del_object(v___x_4252_);
v___y_4204_ = v___y_4241_;
v___y_4205_ = v___x_4254_;
v___y_4206_ = v___y_4242_;
v___y_4207_ = v_a_4250_;
v___y_4208_ = v___y_4243_;
v___y_4209_ = v___x_4256_;
v___y_4210_ = v___x_4257_;
v___y_4211_ = v___y_4200_;
v___y_4212_ = v___y_4201_;
goto v___jp_4203_;
}
}
}
}
v___jp_4264_:
{
lean_object* v___x_4273_; 
v___x_4273_ = l_Lean_Syntax_getTailPos_x3f(v___y_4269_, v___y_4268_);
lean_dec(v___y_4269_);
if (lean_obj_tag(v___x_4273_) == 0)
{
lean_inc(v___y_4272_);
v___y_4240_ = v___y_4265_;
v___y_4241_ = v___y_4266_;
v___y_4242_ = v___y_4267_;
v___y_4243_ = v___y_4268_;
v___y_4244_ = v___y_4270_;
v___y_4245_ = v___y_4272_;
v___y_4246_ = v___y_4271_;
v___y_4247_ = v___y_4272_;
goto v___jp_4239_;
}
else
{
lean_object* v_val_4274_; 
v_val_4274_ = lean_ctor_get(v___x_4273_, 0);
lean_inc(v_val_4274_);
lean_dec_ref_known(v___x_4273_, 1);
v___y_4240_ = v___y_4265_;
v___y_4241_ = v___y_4266_;
v___y_4242_ = v___y_4267_;
v___y_4243_ = v___y_4268_;
v___y_4244_ = v___y_4270_;
v___y_4245_ = v___y_4272_;
v___y_4246_ = v___y_4271_;
v___y_4247_ = v_val_4274_;
goto v___jp_4239_;
}
}
v___jp_4275_:
{
lean_object* v_ref_4283_; lean_object* v___x_4284_; 
v_ref_4283_ = l_Lean_replaceRef(v_ref_4194_, v___y_4280_);
v___x_4284_ = l_Lean_Syntax_getPos_x3f(v_ref_4283_, v___y_4278_);
if (lean_obj_tag(v___x_4284_) == 0)
{
lean_object* v___x_4285_; 
v___x_4285_ = lean_unsigned_to_nat(0u);
v___y_4265_ = v___y_4276_;
v___y_4266_ = v___y_4277_;
v___y_4267_ = v___y_4282_;
v___y_4268_ = v___y_4278_;
v___y_4269_ = v_ref_4283_;
v___y_4270_ = v___y_4279_;
v___y_4271_ = v___y_4281_;
v___y_4272_ = v___x_4285_;
goto v___jp_4264_;
}
else
{
lean_object* v_val_4286_; 
v_val_4286_ = lean_ctor_get(v___x_4284_, 0);
lean_inc(v_val_4286_);
lean_dec_ref_known(v___x_4284_, 1);
v___y_4265_ = v___y_4276_;
v___y_4266_ = v___y_4277_;
v___y_4267_ = v___y_4282_;
v___y_4268_ = v___y_4278_;
v___y_4269_ = v_ref_4283_;
v___y_4270_ = v___y_4279_;
v___y_4271_ = v___y_4281_;
v___y_4272_ = v_val_4286_;
goto v___jp_4264_;
}
}
v___jp_4288_:
{
if (v___y_4295_ == 0)
{
v___y_4276_ = v___y_4290_;
v___y_4277_ = v___y_4289_;
v___y_4278_ = v___y_4294_;
v___y_4279_ = v___y_4291_;
v___y_4280_ = v___y_4292_;
v___y_4281_ = v___y_4293_;
v___y_4282_ = v_severity_4196_;
goto v___jp_4275_;
}
else
{
v___y_4276_ = v___y_4290_;
v___y_4277_ = v___y_4289_;
v___y_4278_ = v___y_4294_;
v___y_4279_ = v___y_4291_;
v___y_4280_ = v___y_4292_;
v___y_4281_ = v___y_4293_;
v___y_4282_ = v___x_4287_;
goto v___jp_4275_;
}
}
v___jp_4296_:
{
if (v___y_4297_ == 0)
{
lean_object* v_fileName_4298_; lean_object* v_fileMap_4299_; lean_object* v_options_4300_; lean_object* v_ref_4301_; uint8_t v_suppressElabErrors_4302_; lean_object* v___x_4303_; lean_object* v___x_4304_; lean_object* v___f_4305_; uint8_t v___x_4306_; uint8_t v___x_4307_; 
v_fileName_4298_ = lean_ctor_get(v___y_4200_, 0);
v_fileMap_4299_ = lean_ctor_get(v___y_4200_, 1);
v_options_4300_ = lean_ctor_get(v___y_4200_, 2);
v_ref_4301_ = lean_ctor_get(v___y_4200_, 5);
v_suppressElabErrors_4302_ = lean_ctor_get_uint8(v___y_4200_, sizeof(void*)*14 + 1);
v___x_4303_ = lean_box(v___y_4297_);
v___x_4304_ = lean_box(v_suppressElabErrors_4302_);
v___f_4305_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__10_spec__14_spec__21___lam__0___boxed), 3, 2);
lean_closure_set(v___f_4305_, 0, v___x_4303_);
lean_closure_set(v___f_4305_, 1, v___x_4304_);
v___x_4306_ = 1;
v___x_4307_ = l_Lean_instBEqMessageSeverity_beq(v_severity_4196_, v___x_4306_);
if (v___x_4307_ == 0)
{
v___y_4289_ = v_fileName_4298_;
v___y_4290_ = v___f_4305_;
v___y_4291_ = v_fileMap_4299_;
v___y_4292_ = v_ref_4301_;
v___y_4293_ = v_suppressElabErrors_4302_;
v___y_4294_ = v___y_4297_;
v___y_4295_ = v___x_4307_;
goto v___jp_4288_;
}
else
{
lean_object* v___x_4308_; uint8_t v___x_4309_; 
v___x_4308_ = l_Lean_warningAsError;
v___x_4309_ = lp_mathlib_Lean_Option_get___at___00__private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_makeFastInstance_spec__12(v_options_4300_, v___x_4308_);
v___y_4289_ = v_fileName_4298_;
v___y_4290_ = v___f_4305_;
v___y_4291_ = v_fileMap_4299_;
v___y_4292_ = v_ref_4301_;
v___y_4293_ = v_suppressElabErrors_4302_;
v___y_4294_ = v___y_4297_;
v___y_4295_ = v___x_4309_;
goto v___jp_4288_;
}
}
else
{
lean_object* v___x_4310_; lean_object* v___x_4311_; 
lean_dec_ref(v_msgData_4195_);
v___x_4310_ = lean_box(0);
v___x_4311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4311_, 0, v___x_4310_);
return v___x_4311_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg___boxed(lean_object* v_ref_4314_, lean_object* v_msgData_4315_, lean_object* v_severity_4316_, lean_object* v_isSilent_4317_, lean_object* v___y_4318_, lean_object* v___y_4319_, lean_object* v___y_4320_, lean_object* v___y_4321_, lean_object* v___y_4322_){
_start:
{
uint8_t v_severity_boxed_4323_; uint8_t v_isSilent_boxed_4324_; lean_object* v_res_4325_; 
v_severity_boxed_4323_ = lean_unbox(v_severity_4316_);
v_isSilent_boxed_4324_ = lean_unbox(v_isSilent_4317_);
v_res_4325_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg(v_ref_4314_, v_msgData_4315_, v_severity_boxed_4323_, v_isSilent_boxed_4324_, v___y_4318_, v___y_4319_, v___y_4320_, v___y_4321_);
lean_dec(v___y_4321_);
lean_dec_ref(v___y_4320_);
lean_dec(v___y_4319_);
lean_dec_ref(v___y_4318_);
lean_dec(v_ref_4314_);
return v_res_4325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3(lean_object* v_ref_4326_, lean_object* v_msgData_4327_, lean_object* v___y_4328_, lean_object* v___y_4329_, lean_object* v___y_4330_, lean_object* v___y_4331_, lean_object* v___y_4332_, lean_object* v___y_4333_){
_start:
{
uint8_t v___x_4335_; uint8_t v___x_4336_; lean_object* v___x_4337_; 
v___x_4335_ = 2;
v___x_4336_ = 0;
v___x_4337_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg(v_ref_4326_, v_msgData_4327_, v___x_4335_, v___x_4336_, v___y_4330_, v___y_4331_, v___y_4332_, v___y_4333_);
return v___x_4337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3___boxed(lean_object* v_ref_4338_, lean_object* v_msgData_4339_, lean_object* v___y_4340_, lean_object* v___y_4341_, lean_object* v___y_4342_, lean_object* v___y_4343_, lean_object* v___y_4344_, lean_object* v___y_4345_, lean_object* v___y_4346_){
_start:
{
lean_object* v_res_4347_; 
v_res_4347_ = lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3(v_ref_4338_, v_msgData_4339_, v___y_4340_, v___y_4341_, v___y_4342_, v___y_4343_, v___y_4344_, v___y_4345_);
lean_dec(v___y_4345_);
lean_dec_ref(v___y_4344_);
lean_dec(v___y_4343_);
lean_dec_ref(v___y_4342_);
lean_dec(v___y_4341_);
lean_dec_ref(v___y_4340_);
lean_dec(v_ref_4338_);
return v_res_4347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4_spec__6(lean_object* v_msgData_4348_, uint8_t v_severity_4349_, uint8_t v_isSilent_4350_, lean_object* v___y_4351_, lean_object* v___y_4352_, lean_object* v___y_4353_, lean_object* v___y_4354_, lean_object* v___y_4355_, lean_object* v___y_4356_){
_start:
{
lean_object* v_ref_4358_; lean_object* v___x_4359_; 
v_ref_4358_ = lean_ctor_get(v___y_4355_, 5);
v___x_4359_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg(v_ref_4358_, v_msgData_4348_, v_severity_4349_, v_isSilent_4350_, v___y_4353_, v___y_4354_, v___y_4355_, v___y_4356_);
return v___x_4359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4_spec__6___boxed(lean_object* v_msgData_4360_, lean_object* v_severity_4361_, lean_object* v_isSilent_4362_, lean_object* v___y_4363_, lean_object* v___y_4364_, lean_object* v___y_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_){
_start:
{
uint8_t v_severity_boxed_4370_; uint8_t v_isSilent_boxed_4371_; lean_object* v_res_4372_; 
v_severity_boxed_4370_ = lean_unbox(v_severity_4361_);
v_isSilent_boxed_4371_ = lean_unbox(v_isSilent_4362_);
v_res_4372_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4_spec__6(v_msgData_4360_, v_severity_boxed_4370_, v_isSilent_boxed_4371_, v___y_4363_, v___y_4364_, v___y_4365_, v___y_4366_, v___y_4367_, v___y_4368_);
lean_dec(v___y_4368_);
lean_dec_ref(v___y_4367_);
lean_dec(v___y_4366_);
lean_dec_ref(v___y_4365_);
lean_dec(v___y_4364_);
lean_dec_ref(v___y_4363_);
return v_res_4372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4(lean_object* v_msgData_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_, lean_object* v___y_4376_, lean_object* v___y_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_){
_start:
{
uint8_t v___x_4381_; uint8_t v___x_4382_; lean_object* v___x_4383_; 
v___x_4381_ = 2;
v___x_4382_ = 0;
v___x_4383_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4_spec__6(v_msgData_4373_, v___x_4381_, v___x_4382_, v___y_4374_, v___y_4375_, v___y_4376_, v___y_4377_, v___y_4378_, v___y_4379_);
return v___x_4383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4___boxed(lean_object* v_msgData_4384_, lean_object* v___y_4385_, lean_object* v___y_4386_, lean_object* v___y_4387_, lean_object* v___y_4388_, lean_object* v___y_4389_, lean_object* v___y_4390_, lean_object* v___y_4391_){
_start:
{
lean_object* v_res_4392_; 
v_res_4392_ = lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4(v_msgData_4384_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_);
lean_dec(v___y_4390_);
lean_dec_ref(v___y_4389_);
lean_dec(v___y_4388_);
lean_dec_ref(v___y_4387_);
lean_dec(v___y_4386_);
lean_dec_ref(v___y_4385_);
return v_res_4392_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__1(void){
_start:
{
lean_object* v___x_4394_; lean_object* v___x_4395_; 
v___x_4394_ = ((lean_object*)(lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__0));
v___x_4395_ = l_Lean_stringToMessageData(v___x_4394_);
return v___x_4395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3(lean_object* v_ex_4396_, lean_object* v___y_4397_, lean_object* v___y_4398_, lean_object* v___y_4399_, lean_object* v___y_4400_, lean_object* v___y_4401_, lean_object* v___y_4402_){
_start:
{
if (lean_obj_tag(v_ex_4396_) == 0)
{
lean_object* v_ref_4404_; lean_object* v_msg_4405_; lean_object* v___x_4406_; 
v_ref_4404_ = lean_ctor_get(v_ex_4396_, 0);
lean_inc(v_ref_4404_);
v_msg_4405_ = lean_ctor_get(v_ex_4396_, 1);
lean_inc_ref(v_msg_4405_);
lean_dec_ref_known(v_ex_4396_, 2);
v___x_4406_ = lp_mathlib_Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3(v_ref_4404_, v_msg_4405_, v___y_4397_, v___y_4398_, v___y_4399_, v___y_4400_, v___y_4401_, v___y_4402_);
lean_dec(v_ref_4404_);
return v___x_4406_;
}
else
{
lean_object* v_id_4407_; uint8_t v___y_4409_; uint8_t v___x_4431_; 
v_id_4407_ = lean_ctor_get(v_ex_4396_, 0);
lean_inc(v_id_4407_);
v___x_4431_ = l_Lean_Elab_isAbortExceptionId(v_id_4407_);
if (v___x_4431_ == 0)
{
uint8_t v___x_4432_; 
v___x_4432_ = l_Lean_Exception_isInterrupt(v_ex_4396_);
lean_dec_ref_known(v_ex_4396_, 2);
v___y_4409_ = v___x_4432_;
goto v___jp_4408_;
}
else
{
lean_dec_ref_known(v_ex_4396_, 2);
v___y_4409_ = v___x_4431_;
goto v___jp_4408_;
}
v___jp_4408_:
{
if (v___y_4409_ == 0)
{
lean_object* v___x_4410_; 
v___x_4410_ = l_Lean_InternalExceptionId_getName(v_id_4407_);
lean_dec(v_id_4407_);
if (lean_obj_tag(v___x_4410_) == 0)
{
lean_object* v_a_4411_; lean_object* v___x_4412_; lean_object* v___x_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; 
v_a_4411_ = lean_ctor_get(v___x_4410_, 0);
lean_inc(v_a_4411_);
lean_dec_ref_known(v___x_4410_, 1);
v___x_4412_ = lean_obj_once(&lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__1, &lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__1_once, _init_lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___closed__1);
v___x_4413_ = l_Lean_MessageData_ofName(v_a_4411_);
v___x_4414_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4414_, 0, v___x_4412_);
lean_ctor_set(v___x_4414_, 1, v___x_4413_);
v___x_4415_ = lp_mathlib_Lean_logError___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__4(v___x_4414_, v___y_4397_, v___y_4398_, v___y_4399_, v___y_4400_, v___y_4401_, v___y_4402_);
return v___x_4415_;
}
else
{
lean_object* v_a_4416_; lean_object* v___x_4418_; uint8_t v_isShared_4419_; uint8_t v_isSharedCheck_4428_; 
v_a_4416_ = lean_ctor_get(v___x_4410_, 0);
v_isSharedCheck_4428_ = !lean_is_exclusive(v___x_4410_);
if (v_isSharedCheck_4428_ == 0)
{
v___x_4418_ = v___x_4410_;
v_isShared_4419_ = v_isSharedCheck_4428_;
goto v_resetjp_4417_;
}
else
{
lean_inc(v_a_4416_);
lean_dec(v___x_4410_);
v___x_4418_ = lean_box(0);
v_isShared_4419_ = v_isSharedCheck_4428_;
goto v_resetjp_4417_;
}
v_resetjp_4417_:
{
lean_object* v_ref_4420_; lean_object* v___x_4421_; lean_object* v___x_4422_; lean_object* v___x_4423_; lean_object* v___x_4424_; lean_object* v___x_4426_; 
v_ref_4420_ = lean_ctor_get(v___y_4401_, 5);
v___x_4421_ = lean_io_error_to_string(v_a_4416_);
v___x_4422_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4422_, 0, v___x_4421_);
v___x_4423_ = l_Lean_MessageData_ofFormat(v___x_4422_);
lean_inc(v_ref_4420_);
v___x_4424_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4424_, 0, v_ref_4420_);
lean_ctor_set(v___x_4424_, 1, v___x_4423_);
if (v_isShared_4419_ == 0)
{
lean_ctor_set(v___x_4418_, 0, v___x_4424_);
v___x_4426_ = v___x_4418_;
goto v_reusejp_4425_;
}
else
{
lean_object* v_reuseFailAlloc_4427_; 
v_reuseFailAlloc_4427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4427_, 0, v___x_4424_);
v___x_4426_ = v_reuseFailAlloc_4427_;
goto v_reusejp_4425_;
}
v_reusejp_4425_:
{
return v___x_4426_;
}
}
}
}
else
{
lean_object* v___x_4429_; lean_object* v___x_4430_; 
lean_dec(v_id_4407_);
v___x_4429_ = lean_box(0);
v___x_4430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4430_, 0, v___x_4429_);
return v___x_4430_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3___boxed(lean_object* v_ex_4433_, lean_object* v___y_4434_, lean_object* v___y_4435_, lean_object* v___y_4436_, lean_object* v___y_4437_, lean_object* v___y_4438_, lean_object* v___y_4439_, lean_object* v___y_4440_){
_start:
{
lean_object* v_res_4441_; 
v_res_4441_ = lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3(v_ex_4433_, v___y_4434_, v___y_4435_, v___y_4436_, v___y_4437_, v___y_4438_, v___y_4439_);
lean_dec(v___y_4439_);
lean_dec_ref(v___y_4438_);
lean_dec(v___y_4437_);
lean_dec_ref(v___y_4436_);
lean_dec(v___y_4435_);
lean_dec_ref(v___y_4434_);
return v_res_4441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance(lean_object* v_x_4442_, lean_object* v_x_4443_, lean_object* v_a_4444_, lean_object* v_a_4445_, lean_object* v_a_4446_, lean_object* v_a_4447_, lean_object* v_a_4448_, lean_object* v_a_4449_){
_start:
{
lean_object* v___x_4451_; uint8_t v___x_4452_; 
v___x_4451_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1));
lean_inc(v_x_4442_);
v___x_4452_ = l_Lean_Syntax_isOfKind(v_x_4442_, v___x_4451_);
if (v___x_4452_ == 0)
{
lean_object* v___x_4453_; 
lean_dec(v_x_4443_);
lean_dec(v_x_4442_);
v___x_4453_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__0___redArg();
return v___x_4453_;
}
else
{
lean_object* v___x_4454_; lean_object* v___x_4455_; lean_object* v___x_4456_; lean_object* v___x_4457_; lean_object* v___x_4458_; uint8_t v___x_4459_; lean_object* v___x_4460_; 
v___x_4454_ = lean_unsigned_to_nat(1u);
v___x_4455_ = l_Lean_Syntax_getArg(v_x_4442_, v___x_4454_);
lean_dec(v_x_4442_);
v___x_4456_ = lean_box(v___x_4452_);
v___x_4457_ = lean_box(v___x_4452_);
lean_inc(v_x_4443_);
v___x_4458_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTerm___boxed), 11, 4);
lean_closure_set(v___x_4458_, 0, v___x_4455_);
lean_closure_set(v___x_4458_, 1, v_x_4443_);
lean_closure_set(v___x_4458_, 2, v___x_4456_);
lean_closure_set(v___x_4458_, 3, v___x_4457_);
v___x_4459_ = 1;
v___x_4460_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_4458_, v___x_4459_, v_a_4444_, v_a_4445_, v_a_4446_, v_a_4447_, v_a_4448_, v_a_4449_);
if (lean_obj_tag(v___x_4460_) == 0)
{
lean_object* v_a_4461_; lean_object* v___y_4463_; lean_object* v___y_4464_; uint8_t v___y_4465_; lean_object* v___x_4483_; lean_object* v___f_4484_; lean_object* v_a_4486_; 
v_a_4461_ = lean_ctor_get(v___x_4460_, 0);
lean_inc_n(v_a_4461_, 2);
lean_dec_ref_known(v___x_4460_, 1);
v___x_4483_ = lean_box(v___x_4452_);
v___f_4484_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___lam__1___boxed), 11, 2);
lean_closure_set(v___f_4484_, 0, v_a_4461_);
lean_closure_set(v___f_4484_, 1, v___x_4483_);
if (lean_obj_tag(v_x_4443_) == 0)
{
lean_object* v___x_4492_; 
lean_inc(v_a_4449_);
lean_inc_ref(v_a_4448_);
lean_inc(v_a_4447_);
lean_inc_ref(v_a_4446_);
lean_inc(v_a_4461_);
v___x_4492_ = lean_infer_type(v_a_4461_, v_a_4446_, v_a_4447_, v_a_4448_, v_a_4449_);
if (lean_obj_tag(v___x_4492_) == 0)
{
lean_object* v_a_4493_; 
v_a_4493_ = lean_ctor_get(v___x_4492_, 0);
lean_inc(v_a_4493_);
lean_dec_ref_known(v___x_4492_, 1);
v_a_4486_ = v_a_4493_;
goto v___jp_4485_;
}
else
{
lean_dec_ref(v___f_4484_);
lean_dec(v_a_4461_);
return v___x_4492_;
}
}
else
{
lean_object* v_val_4494_; 
v_val_4494_ = lean_ctor_get(v_x_4443_, 0);
lean_inc(v_val_4494_);
lean_dec_ref_known(v_x_4443_, 1);
v_a_4486_ = v_val_4494_;
goto v___jp_4485_;
}
v___jp_4462_:
{
if (v___y_4465_ == 0)
{
lean_object* v___x_4466_; 
lean_dec_ref(v___y_4464_);
v___x_4466_ = lp_mathlib_Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3(v___y_4463_, v_a_4444_, v_a_4445_, v_a_4446_, v_a_4447_, v_a_4448_, v_a_4449_);
if (lean_obj_tag(v___x_4466_) == 0)
{
lean_object* v___x_4468_; uint8_t v_isShared_4469_; uint8_t v_isSharedCheck_4473_; 
v_isSharedCheck_4473_ = !lean_is_exclusive(v___x_4466_);
if (v_isSharedCheck_4473_ == 0)
{
lean_object* v_unused_4474_; 
v_unused_4474_ = lean_ctor_get(v___x_4466_, 0);
lean_dec(v_unused_4474_);
v___x_4468_ = v___x_4466_;
v_isShared_4469_ = v_isSharedCheck_4473_;
goto v_resetjp_4467_;
}
else
{
lean_dec(v___x_4466_);
v___x_4468_ = lean_box(0);
v_isShared_4469_ = v_isSharedCheck_4473_;
goto v_resetjp_4467_;
}
v_resetjp_4467_:
{
lean_object* v___x_4471_; 
if (v_isShared_4469_ == 0)
{
lean_ctor_set(v___x_4468_, 0, v_a_4461_);
v___x_4471_ = v___x_4468_;
goto v_reusejp_4470_;
}
else
{
lean_object* v_reuseFailAlloc_4472_; 
v_reuseFailAlloc_4472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4472_, 0, v_a_4461_);
v___x_4471_ = v_reuseFailAlloc_4472_;
goto v_reusejp_4470_;
}
v_reusejp_4470_:
{
return v___x_4471_;
}
}
}
else
{
lean_object* v_a_4475_; lean_object* v___x_4477_; uint8_t v_isShared_4478_; uint8_t v_isSharedCheck_4482_; 
lean_dec(v_a_4461_);
v_a_4475_ = lean_ctor_get(v___x_4466_, 0);
v_isSharedCheck_4482_ = !lean_is_exclusive(v___x_4466_);
if (v_isSharedCheck_4482_ == 0)
{
v___x_4477_ = v___x_4466_;
v_isShared_4478_ = v_isSharedCheck_4482_;
goto v_resetjp_4476_;
}
else
{
lean_inc(v_a_4475_);
lean_dec(v___x_4466_);
v___x_4477_ = lean_box(0);
v_isShared_4478_ = v_isSharedCheck_4482_;
goto v_resetjp_4476_;
}
v_resetjp_4476_:
{
lean_object* v___x_4480_; 
if (v_isShared_4478_ == 0)
{
v___x_4480_ = v___x_4477_;
goto v_reusejp_4479_;
}
else
{
lean_object* v_reuseFailAlloc_4481_; 
v_reuseFailAlloc_4481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4481_, 0, v_a_4475_);
v___x_4480_ = v_reuseFailAlloc_4481_;
goto v_reusejp_4479_;
}
v_reusejp_4479_:
{
return v___x_4480_;
}
}
}
}
else
{
lean_dec_ref(v___y_4463_);
lean_dec(v_a_4461_);
return v___y_4464_;
}
}
v___jp_4485_:
{
uint8_t v___x_4487_; lean_object* v___x_4488_; 
v___x_4487_ = 0;
v___x_4488_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__2___redArg(v_a_4486_, v___f_4484_, v___x_4487_, v___x_4487_, v_a_4444_, v_a_4445_, v_a_4446_, v_a_4447_, v_a_4448_, v_a_4449_);
if (lean_obj_tag(v___x_4488_) == 0)
{
lean_dec(v_a_4461_);
return v___x_4488_;
}
else
{
lean_object* v_a_4489_; uint8_t v___x_4490_; 
v_a_4489_ = lean_ctor_get(v___x_4488_, 0);
lean_inc(v_a_4489_);
v___x_4490_ = l_Lean_Exception_isInterrupt(v_a_4489_);
if (v___x_4490_ == 0)
{
uint8_t v___x_4491_; 
lean_inc(v_a_4489_);
v___x_4491_ = l_Lean_Exception_isRuntime(v_a_4489_);
v___y_4463_ = v_a_4489_;
v___y_4464_ = v___x_4488_;
v___y_4465_ = v___x_4491_;
goto v___jp_4462_;
}
else
{
v___y_4463_ = v_a_4489_;
v___y_4464_ = v___x_4488_;
v___y_4465_ = v___x_4490_;
goto v___jp_4462_;
}
}
}
}
else
{
lean_dec(v_x_4443_);
return v___x_4460_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance___boxed(lean_object* v_x_4495_, lean_object* v_x_4496_, lean_object* v_a_4497_, lean_object* v_a_4498_, lean_object* v_a_4499_, lean_object* v_a_4500_, lean_object* v_a_4501_, lean_object* v_a_4502_, lean_object* v_a_4503_){
_start:
{
lean_object* v_res_4504_; 
v_res_4504_ = lp_mathlib_Mathlib_Elab_FastInstance_elabFastInstance(v_x_4495_, v_x_4496_, v_a_4497_, v_a_4498_, v_a_4499_, v_a_4500_, v_a_4501_, v_a_4502_);
lean_dec(v_a_4502_);
lean_dec_ref(v_a_4501_);
lean_dec(v_a_4500_);
lean_dec_ref(v_a_4499_);
lean_dec(v_a_4498_);
lean_dec_ref(v_a_4497_);
return v_res_4504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4(lean_object* v_ref_4505_, lean_object* v_msgData_4506_, uint8_t v_severity_4507_, uint8_t v_isSilent_4508_, lean_object* v___y_4509_, lean_object* v___y_4510_, lean_object* v___y_4511_, lean_object* v___y_4512_, lean_object* v___y_4513_, lean_object* v___y_4514_){
_start:
{
lean_object* v___x_4516_; 
v___x_4516_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___redArg(v_ref_4505_, v_msgData_4506_, v_severity_4507_, v_isSilent_4508_, v___y_4511_, v___y_4512_, v___y_4513_, v___y_4514_);
return v___x_4516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4___boxed(lean_object* v_ref_4517_, lean_object* v_msgData_4518_, lean_object* v_severity_4519_, lean_object* v_isSilent_4520_, lean_object* v___y_4521_, lean_object* v___y_4522_, lean_object* v___y_4523_, lean_object* v___y_4524_, lean_object* v___y_4525_, lean_object* v___y_4526_, lean_object* v___y_4527_){
_start:
{
uint8_t v_severity_boxed_4528_; uint8_t v_isSilent_boxed_4529_; lean_object* v_res_4530_; 
v_severity_boxed_4528_ = lean_unbox(v_severity_4519_);
v_isSilent_boxed_4529_ = lean_unbox(v_isSilent_4520_);
v_res_4530_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Lean_Elab_logException___at___00Mathlib_Elab_FastInstance_elabFastInstance_spec__3_spec__3_spec__4(v_ref_4517_, v_msgData_4518_, v_severity_boxed_4528_, v_isSilent_boxed_4529_, v___y_4521_, v___y_4522_, v___y_4523_, v___y_4524_, v___y_4525_, v___y_4526_);
lean_dec(v___y_4526_);
lean_dec_ref(v___y_4525_);
lean_dec(v___y_4524_);
lean_dec_ref(v___y_4523_);
lean_dec(v___y_4522_);
lean_dec_ref(v___y_4521_);
lean_dec(v_ref_4517_);
return v_res_4530_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__4(void){
_start:
{
lean_object* v___x_4554_; lean_object* v___x_4555_; 
v___x_4554_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__3));
v___x_4555_ = l_String_toRawSubstring_x27(v___x_4554_);
return v___x_4555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1(lean_object* v_x_4570_, lean_object* v_a_4571_, lean_object* v_a_4572_){
_start:
{
lean_object* v___x_4573_; uint8_t v___x_4574_; 
v___x_4573_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance_termInferInstanceAs_x25___00__closed__1));
lean_inc(v_x_4570_);
v___x_4574_ = l_Lean_Syntax_isOfKind(v_x_4570_, v___x_4573_);
if (v___x_4574_ == 0)
{
lean_object* v___x_4575_; lean_object* v___x_4576_; 
lean_dec(v_x_4570_);
v___x_4575_ = lean_box(1);
v___x_4576_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4576_, 0, v___x_4575_);
lean_ctor_set(v___x_4576_, 1, v_a_4572_);
return v___x_4576_;
}
else
{
lean_object* v_quotContext_4577_; lean_object* v_currMacroScope_4578_; lean_object* v_ref_4579_; lean_object* v___x_4580_; lean_object* v___x_4581_; uint8_t v___x_4582_; lean_object* v___x_4583_; lean_object* v___x_4584_; lean_object* v___x_4585_; lean_object* v___x_4586_; lean_object* v___x_4587_; lean_object* v___x_4588_; lean_object* v___x_4589_; lean_object* v___x_4590_; lean_object* v___x_4591_; lean_object* v___x_4592_; lean_object* v___x_4593_; lean_object* v___x_4594_; lean_object* v___x_4595_; lean_object* v___x_4596_; lean_object* v___x_4597_; 
v_quotContext_4577_ = lean_ctor_get(v_a_4571_, 1);
v_currMacroScope_4578_ = lean_ctor_get(v_a_4571_, 2);
v_ref_4579_ = lean_ctor_get(v_a_4571_, 5);
v___x_4580_ = lean_unsigned_to_nat(1u);
v___x_4581_ = l_Lean_Syntax_getArg(v_x_4570_, v___x_4580_);
lean_dec(v_x_4570_);
v___x_4582_ = 0;
v___x_4583_ = l_Lean_SourceInfo_fromRef(v_ref_4579_, v___x_4582_);
v___x_4584_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance_fastInstance___closed__1));
v___x_4585_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__0));
lean_inc_n(v___x_4583_, 4);
v___x_4586_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4586_, 0, v___x_4583_);
lean_ctor_set(v___x_4586_, 1, v___x_4585_);
v___x_4587_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__2));
v___x_4588_ = lean_obj_once(&lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__4, &lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__4_once, _init_lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__4);
v___x_4589_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__7));
lean_inc(v_currMacroScope_4578_);
lean_inc(v_quotContext_4577_);
v___x_4590_ = l_Lean_addMacroScope(v_quotContext_4577_, v___x_4589_, v_currMacroScope_4578_);
v___x_4591_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__10));
v___x_4592_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4592_, 0, v___x_4583_);
lean_ctor_set(v___x_4592_, 1, v___x_4588_);
lean_ctor_set(v___x_4592_, 2, v___x_4590_);
lean_ctor_set(v___x_4592_, 3, v___x_4591_);
v___x_4593_ = ((lean_object*)(lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___closed__11));
v___x_4594_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4594_, 0, v___x_4583_);
lean_ctor_set(v___x_4594_, 1, v___x_4593_);
v___x_4595_ = l_Lean_Syntax_node3(v___x_4583_, v___x_4587_, v___x_4592_, v___x_4594_, v___x_4581_);
v___x_4596_ = l_Lean_Syntax_node2(v___x_4583_, v___x_4584_, v___x_4586_, v___x_4595_);
v___x_4597_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4597_, 0, v___x_4596_);
lean_ctor_set(v___x_4597_, 1, v_a_4572_);
return v___x_4597_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1___boxed(lean_object* v_x_4598_, lean_object* v_a_4599_, lean_object* v_a_4600_){
_start:
{
lean_object* v_res_4601_; 
v_res_4601_ = lp_mathlib_Mathlib_Elab_FastInstance___aux__Mathlib__Tactic__FastInstance______macroRules__Mathlib__Elab__FastInstance__termInferInstanceAs_x25____1(v_x_4598_, v_a_4599_, v_a_4600_);
lean_dec_ref(v_a_4599_);
return v_res_4601_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_414705255____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_initFn_00___x40_Mathlib_Tactic_FastInstance_1741628746____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_FastInstance_0__Mathlib_Elab_FastInstance_linter_fast__instance__existing);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
}
#ifdef __cplusplus
}
#endif
