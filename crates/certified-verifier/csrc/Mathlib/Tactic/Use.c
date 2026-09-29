// Lean compiler output
// Module: Mathlib.Tactic.Use
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Util public meta import Lean.Elab.Tactic.Basic public import Mathlib.Init
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
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_Lean_BinderInfo_isExplicit(uint8_t);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_findAsync_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_AsyncConstantInfo_toConstantInfo(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_saveState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_SavedState_restore(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_pruneSolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_discharger;
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_focus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "use"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(31, 151, 162, 50, 1, 244, 84, 6)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Use"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(159, 221, 242, 105, 161, 2, 157, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(210, 99, 4, 67, 212, 175, 70, 18)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(67, 43, 35, 11, 4, 122, 232, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(242, 81, 97, 52, 132, 202, 185, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(135, 117, 253, 145, 76, 66, 131, 58)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(162, 111, 242, 194, 134, 41, 74, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(19, 166, 84, 194, 77, 63, 55, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(130, 55, 81, 193, 40, 82, 179, 132)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(182, 168, 58, 7, 104, 50, 135, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)(((size_t)(721082523) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(187, 55, 40, 38, 218, 119, 212, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 205, 72, 56, 171, 74, 77, 48)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(228, 224, 213, 25, 69, 126, 100, 225)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(253, 37, 158, 150, 233, 207, 200, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__1;
static const lean_string_object lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "` is not a constructor"};
static const lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__3;
static const lean_string_object lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.MonadEnv"};
static const lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Lean.isCtor\?"};
static const lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "target is not an inductive datatype"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "target inductive type does not have exactly one constructor"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__3;
static const lean_array_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "type mismatch"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constructor"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__0_value),LEAN_SCALAR_PTR_LITERAL(209, 67, 1, 221, 34, 155, 196, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Constructor. "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__5(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_useLoop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_useLoop___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "too many arguments supplied to `use`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "gs = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "\nargs = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\nacc = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__12;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__15_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__24_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__18_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__18_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__15_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(7, 212, 55, 101, 104, 194, 19, 213)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__35_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "expl.length = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__38_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__39;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = ", impl.length = "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__40_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__41;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "argument is not definitionally equal to inferred value"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__42_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__43;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__0;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "running discharger on "};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "tacticUse_discharger"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__0_value),LEAN_SCALAR_PTR_LITERAL(143, 128, 238, 160, 112, 212, 210, 95)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "use_discharger"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse__discharger = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "exists_prop.mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "exists_prop"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(169, 132, 191, 43, 249, 116, 95, 104)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(92, 232, 53, 252, 95, 154, 86, 80)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(169, 132, 191, 43, 249, 116, 95, 104)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "And.intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(58, 46, 244, 208, 18, 71, 77, 162)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "True.intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__2_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 152, 123, 219, 220, 182, 189, 250)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "discharger"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 186, 255, 143, 150, 72, 152, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__2_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__4_value),LEAN_SCALAR_PTR_LITERAL(34, 109, 187, 155, 23, 130, 33, 152)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__7_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__9_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__11_value),LEAN_SCALAR_PTR_LITERAL(197, 44, 223, 192, 8, 197, 146, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "useSyntax"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 44, 121, 254, 252, 237, 50, 187)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useSyntax___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useSyntax___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__9_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useSyntax___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__16_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_useSyntax___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__21_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_useSyntax___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useSyntax___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_useSyntax___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_useSyntax___closed__24;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useSyntax;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__useSyntax__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__useSyntax__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "tacticUse!___,,"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(33, 236, 78, 157, 250, 30, 94, 97)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "use!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__tacticUse_x21_______x2c_x2c__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__tacticUse_x21_______x2c_x2c__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_63_; uint8_t v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_63_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_));
v___x_64_ = 0;
v___x_65_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__26_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_));
v___x_66_ = l_Lean_registerTraceClass(v___x_63_, v___x_64_, v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2____boxed(lean_object* v_a_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_();
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg(lean_object* v_mvarId_69_, lean_object* v_x_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_69_, v_x_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_);
if (lean_obj_tag(v___x_76_) == 0)
{
lean_object* v_a_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_84_; 
v_a_77_ = lean_ctor_get(v___x_76_, 0);
v_isSharedCheck_84_ = !lean_is_exclusive(v___x_76_);
if (v_isSharedCheck_84_ == 0)
{
v___x_79_ = v___x_76_;
v_isShared_80_ = v_isSharedCheck_84_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_a_77_);
lean_dec(v___x_76_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_84_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_82_; 
if (v_isShared_80_ == 0)
{
v___x_82_ = v___x_79_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v_a_77_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
}
else
{
lean_object* v_a_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_92_; 
v_a_85_ = lean_ctor_get(v___x_76_, 0);
v_isSharedCheck_92_ = !lean_is_exclusive(v___x_76_);
if (v_isSharedCheck_92_ == 0)
{
v___x_87_ = v___x_76_;
v_isShared_88_ = v_isSharedCheck_92_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_a_85_);
lean_dec(v___x_76_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg___boxed(lean_object* v_mvarId_93_, lean_object* v_x_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg(v_mvarId_93_, v_x_94_, v___y_95_, v___y_96_, v___y_97_, v___y_98_);
lean_dec(v___y_98_);
lean_dec_ref(v___y_97_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4(lean_object* v_00_u03b1_101_, lean_object* v_mvarId_102_, lean_object* v_x_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg(v_mvarId_102_, v_x_103_, v___y_104_, v___y_105_, v___y_106_, v___y_107_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___boxed(lean_object* v_00_u03b1_110_, lean_object* v_mvarId_111_, lean_object* v_x_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4(v_00_u03b1_110_, v_mvarId_111_, v_x_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
lean_dec(v___y_116_);
lean_dec_ref(v___y_115_);
lean_dec(v___y_114_);
lean_dec_ref(v___y_113_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(lean_object* v_msgData_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_){
_start:
{
lean_object* v___x_125_; lean_object* v_env_126_; lean_object* v___x_127_; lean_object* v_mctx_128_; lean_object* v_lctx_129_; lean_object* v_options_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_125_ = lean_st_ref_get(v___y_123_);
v_env_126_ = lean_ctor_get(v___x_125_, 0);
lean_inc_ref(v_env_126_);
lean_dec(v___x_125_);
v___x_127_ = lean_st_ref_get(v___y_121_);
v_mctx_128_ = lean_ctor_get(v___x_127_, 0);
lean_inc_ref(v_mctx_128_);
lean_dec(v___x_127_);
v_lctx_129_ = lean_ctor_get(v___y_120_, 2);
v_options_130_ = lean_ctor_get(v___y_122_, 2);
lean_inc_ref(v_options_130_);
lean_inc_ref(v_lctx_129_);
v___x_131_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_131_, 0, v_env_126_);
lean_ctor_set(v___x_131_, 1, v_mctx_128_);
lean_ctor_set(v___x_131_, 2, v_lctx_129_);
lean_ctor_set(v___x_131_, 3, v_options_130_);
v___x_132_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v_msgData_119_);
v___x_133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5___boxed(lean_object* v_msgData_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(v_msgData_134_, v___y_135_, v___y_136_, v___y_137_, v___y_138_);
lean_dec(v___y_138_);
lean_dec_ref(v___y_137_);
lean_dec(v___y_136_);
lean_dec_ref(v___y_135_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg(lean_object* v_msg_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v_ref_147_; lean_object* v___x_148_; lean_object* v_a_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_157_; 
v_ref_147_ = lean_ctor_get(v___y_144_, 5);
v___x_148_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(v_msg_141_, v___y_142_, v___y_143_, v___y_144_, v___y_145_);
v_a_149_ = lean_ctor_get(v___x_148_, 0);
v_isSharedCheck_157_ = !lean_is_exclusive(v___x_148_);
if (v_isSharedCheck_157_ == 0)
{
v___x_151_ = v___x_148_;
v_isShared_152_ = v_isSharedCheck_157_;
goto v_resetjp_150_;
}
else
{
lean_inc(v_a_149_);
lean_dec(v___x_148_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_157_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v___x_153_; lean_object* v___x_155_; 
lean_inc(v_ref_147_);
v___x_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_153_, 0, v_ref_147_);
lean_ctor_set(v___x_153_, 1, v_a_149_);
if (v_isShared_152_ == 0)
{
lean_ctor_set_tag(v___x_151_, 1);
lean_ctor_set(v___x_151_, 0, v___x_153_);
v___x_155_ = v___x_151_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_156_; 
v_reuseFailAlloc_156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_156_, 0, v___x_153_);
v___x_155_ = v_reuseFailAlloc_156_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
return v___x_155_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg___boxed(lean_object* v_msg_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg(v_msg_158_, v___y_159_, v___y_160_, v___y_161_, v___y_162_);
lean_dec(v___y_162_);
lean_dec_ref(v___y_161_);
lean_dec(v___y_160_);
lean_dec_ref(v___y_159_);
return v_res_164_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = l_instMonadEIO(lean_box(0));
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0(lean_object* v_msg_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v_toApplicative_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_239_; 
v___x_176_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__0, &lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__0_once, _init_lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__0);
v___x_177_ = l_StateRefT_x27_instMonad___redArg(v___x_176_);
v_toApplicative_178_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_239_ == 0)
{
lean_object* v_unused_240_; 
v_unused_240_ = lean_ctor_get(v___x_177_, 1);
lean_dec(v_unused_240_);
v___x_180_ = v___x_177_;
v_isShared_181_ = v_isSharedCheck_239_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_toApplicative_178_);
lean_dec(v___x_177_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_239_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v_toFunctor_182_; lean_object* v_toSeq_183_; lean_object* v_toSeqLeft_184_; lean_object* v_toSeqRight_185_; lean_object* v___x_187_; uint8_t v_isShared_188_; uint8_t v_isSharedCheck_237_; 
v_toFunctor_182_ = lean_ctor_get(v_toApplicative_178_, 0);
v_toSeq_183_ = lean_ctor_get(v_toApplicative_178_, 2);
v_toSeqLeft_184_ = lean_ctor_get(v_toApplicative_178_, 3);
v_toSeqRight_185_ = lean_ctor_get(v_toApplicative_178_, 4);
v_isSharedCheck_237_ = !lean_is_exclusive(v_toApplicative_178_);
if (v_isSharedCheck_237_ == 0)
{
lean_object* v_unused_238_; 
v_unused_238_ = lean_ctor_get(v_toApplicative_178_, 1);
lean_dec(v_unused_238_);
v___x_187_ = v_toApplicative_178_;
v_isShared_188_ = v_isSharedCheck_237_;
goto v_resetjp_186_;
}
else
{
lean_inc(v_toSeqRight_185_);
lean_inc(v_toSeqLeft_184_);
lean_inc(v_toSeq_183_);
lean_inc(v_toFunctor_182_);
lean_dec(v_toApplicative_178_);
v___x_187_ = lean_box(0);
v_isShared_188_ = v_isSharedCheck_237_;
goto v_resetjp_186_;
}
v_resetjp_186_:
{
lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___f_191_; lean_object* v___f_192_; lean_object* v___x_193_; lean_object* v___f_194_; lean_object* v___f_195_; lean_object* v___f_196_; lean_object* v___x_198_; 
v___f_189_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__1));
v___f_190_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__2));
lean_inc_ref(v_toFunctor_182_);
v___f_191_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_191_, 0, v_toFunctor_182_);
v___f_192_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_192_, 0, v_toFunctor_182_);
v___x_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_193_, 0, v___f_191_);
lean_ctor_set(v___x_193_, 1, v___f_192_);
v___f_194_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_194_, 0, v_toSeqRight_185_);
v___f_195_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_195_, 0, v_toSeqLeft_184_);
v___f_196_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_196_, 0, v_toSeq_183_);
if (v_isShared_188_ == 0)
{
lean_ctor_set(v___x_187_, 4, v___f_194_);
lean_ctor_set(v___x_187_, 3, v___f_195_);
lean_ctor_set(v___x_187_, 2, v___f_196_);
lean_ctor_set(v___x_187_, 1, v___f_189_);
lean_ctor_set(v___x_187_, 0, v___x_193_);
v___x_198_ = v___x_187_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v___x_193_);
lean_ctor_set(v_reuseFailAlloc_236_, 1, v___f_189_);
lean_ctor_set(v_reuseFailAlloc_236_, 2, v___f_196_);
lean_ctor_set(v_reuseFailAlloc_236_, 3, v___f_195_);
lean_ctor_set(v_reuseFailAlloc_236_, 4, v___f_194_);
v___x_198_ = v_reuseFailAlloc_236_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
lean_object* v___x_200_; 
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 1, v___f_190_);
lean_ctor_set(v___x_180_, 0, v___x_198_);
v___x_200_ = v___x_180_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_235_; 
v_reuseFailAlloc_235_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_235_, 0, v___x_198_);
lean_ctor_set(v_reuseFailAlloc_235_, 1, v___f_190_);
v___x_200_ = v_reuseFailAlloc_235_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
lean_object* v___x_201_; lean_object* v_toApplicative_202_; lean_object* v___x_204_; uint8_t v_isShared_205_; uint8_t v_isSharedCheck_233_; 
v___x_201_ = l_StateRefT_x27_instMonad___redArg(v___x_200_);
v_toApplicative_202_ = lean_ctor_get(v___x_201_, 0);
v_isSharedCheck_233_ = !lean_is_exclusive(v___x_201_);
if (v_isSharedCheck_233_ == 0)
{
lean_object* v_unused_234_; 
v_unused_234_ = lean_ctor_get(v___x_201_, 1);
lean_dec(v_unused_234_);
v___x_204_ = v___x_201_;
v_isShared_205_ = v_isSharedCheck_233_;
goto v_resetjp_203_;
}
else
{
lean_inc(v_toApplicative_202_);
lean_dec(v___x_201_);
v___x_204_ = lean_box(0);
v_isShared_205_ = v_isSharedCheck_233_;
goto v_resetjp_203_;
}
v_resetjp_203_:
{
lean_object* v_toFunctor_206_; lean_object* v_toSeq_207_; lean_object* v_toSeqLeft_208_; lean_object* v_toSeqRight_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_231_; 
v_toFunctor_206_ = lean_ctor_get(v_toApplicative_202_, 0);
v_toSeq_207_ = lean_ctor_get(v_toApplicative_202_, 2);
v_toSeqLeft_208_ = lean_ctor_get(v_toApplicative_202_, 3);
v_toSeqRight_209_ = lean_ctor_get(v_toApplicative_202_, 4);
v_isSharedCheck_231_ = !lean_is_exclusive(v_toApplicative_202_);
if (v_isSharedCheck_231_ == 0)
{
lean_object* v_unused_232_; 
v_unused_232_ = lean_ctor_get(v_toApplicative_202_, 1);
lean_dec(v_unused_232_);
v___x_211_ = v_toApplicative_202_;
v_isShared_212_ = v_isSharedCheck_231_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_toSeqRight_209_);
lean_inc(v_toSeqLeft_208_);
lean_inc(v_toSeq_207_);
lean_inc(v_toFunctor_206_);
lean_dec(v_toApplicative_202_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_231_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___f_213_; lean_object* v___f_214_; lean_object* v___f_215_; lean_object* v___f_216_; lean_object* v___x_217_; lean_object* v___f_218_; lean_object* v___f_219_; lean_object* v___f_220_; lean_object* v___x_222_; 
v___f_213_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__3));
v___f_214_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___closed__4));
lean_inc_ref(v_toFunctor_206_);
v___f_215_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_215_, 0, v_toFunctor_206_);
v___f_216_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_216_, 0, v_toFunctor_206_);
v___x_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_217_, 0, v___f_215_);
lean_ctor_set(v___x_217_, 1, v___f_216_);
v___f_218_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_218_, 0, v_toSeqRight_209_);
v___f_219_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_219_, 0, v_toSeqLeft_208_);
v___f_220_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_220_, 0, v_toSeq_207_);
if (v_isShared_212_ == 0)
{
lean_ctor_set(v___x_211_, 4, v___f_218_);
lean_ctor_set(v___x_211_, 3, v___f_219_);
lean_ctor_set(v___x_211_, 2, v___f_220_);
lean_ctor_set(v___x_211_, 1, v___f_213_);
lean_ctor_set(v___x_211_, 0, v___x_217_);
v___x_222_ = v___x_211_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v___x_217_);
lean_ctor_set(v_reuseFailAlloc_230_, 1, v___f_213_);
lean_ctor_set(v_reuseFailAlloc_230_, 2, v___f_220_);
lean_ctor_set(v_reuseFailAlloc_230_, 3, v___f_219_);
lean_ctor_set(v_reuseFailAlloc_230_, 4, v___f_218_);
v___x_222_ = v_reuseFailAlloc_230_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
lean_object* v___x_224_; 
if (v_isShared_205_ == 0)
{
lean_ctor_set(v___x_204_, 1, v___f_214_);
lean_ctor_set(v___x_204_, 0, v___x_222_);
v___x_224_ = v___x_204_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v___x_222_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v___f_214_);
v___x_224_ = v_reuseFailAlloc_229_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_8992__overap_227_; lean_object* v___x_228_; 
v___x_225_ = lean_box(0);
v___x_226_ = l_instInhabitedOfMonad___redArg(v___x_224_, v___x_225_);
v___x_8992__overap_227_ = lean_panic_fn_borrowed(v___x_226_, v_msg_170_);
lean_dec(v___x_226_);
lean_inc(v___y_174_);
lean_inc_ref(v___y_173_);
lean_inc(v___y_172_);
lean_inc_ref(v___y_171_);
v___x_228_ = lean_apply_5(v___x_8992__overap_227_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, lean_box(0));
return v___x_228_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0___boxed(lean_object* v_msg_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0(v_msg_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
return v_res_247_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__1(void){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__0));
v___x_250_ = l_Lean_stringToMessageData(v___x_249_);
return v___x_250_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__3(void){
_start:
{
lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_252_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__2));
v___x_253_ = l_Lean_stringToMessageData(v___x_252_);
return v___x_253_;
}
}
static lean_object* _init_lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__7(void){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_257_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__6));
v___x_258_ = lean_unsigned_to_nat(11u);
v___x_259_ = lean_unsigned_to_nat(122u);
v___x_260_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__5));
v___x_261_ = ((lean_object*)(lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__4));
v___x_262_ = l_mkPanicMessageWithDecl(v___x_261_, v___x_260_, v___x_259_, v___x_258_, v___x_257_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0(lean_object* v_constName_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v___x_277_; lean_object* v_env_278_; uint8_t v___x_279_; lean_object* v___x_280_; 
v___x_277_ = lean_st_ref_get(v___y_267_);
v_env_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc_ref(v_env_278_);
lean_dec(v___x_277_);
v___x_279_ = 0;
lean_inc(v_constName_263_);
v___x_280_ = l_Lean_Environment_findAsync_x3f(v_env_278_, v_constName_263_, v___x_279_);
if (lean_obj_tag(v___x_280_) == 1)
{
lean_object* v_val_281_; uint8_t v_kind_282_; 
v_val_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_val_281_);
lean_dec_ref_known(v___x_280_, 1);
v_kind_282_ = lean_ctor_get_uint8(v_val_281_, sizeof(void*)*3);
if (v_kind_282_ == 6)
{
lean_object* v___x_283_; 
v___x_283_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_281_);
if (lean_obj_tag(v___x_283_) == 6)
{
lean_object* v_val_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_291_; 
lean_dec(v_constName_263_);
v_val_284_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_291_ == 0)
{
v___x_286_ = v___x_283_;
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_val_284_);
lean_dec(v___x_283_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_291_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_289_; 
if (v_isShared_287_ == 0)
{
lean_ctor_set_tag(v___x_286_, 0);
v___x_289_ = v___x_286_;
goto v_reusejp_288_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v_val_284_);
v___x_289_ = v_reuseFailAlloc_290_;
goto v_reusejp_288_;
}
v_reusejp_288_:
{
return v___x_289_;
}
}
}
else
{
lean_object* v___x_292_; lean_object* v___x_293_; 
lean_dec_ref(v___x_283_);
v___x_292_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__7, &lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__7_once, _init_lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__7);
v___x_293_ = lp_mathlib_panic___at___00Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0_spec__0(v___x_292_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
if (lean_obj_tag(v___x_293_) == 0)
{
lean_object* v_a_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_302_; 
v_a_294_ = lean_ctor_get(v___x_293_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_293_);
if (v_isSharedCheck_302_ == 0)
{
v___x_296_ = v___x_293_;
v_isShared_297_ = v_isSharedCheck_302_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_a_294_);
lean_dec(v___x_293_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_302_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
if (lean_obj_tag(v_a_294_) == 0)
{
lean_del_object(v___x_296_);
goto v___jp_269_;
}
else
{
lean_object* v_val_298_; lean_object* v___x_300_; 
lean_dec(v_constName_263_);
v_val_298_ = lean_ctor_get(v_a_294_, 0);
lean_inc(v_val_298_);
lean_dec_ref_known(v_a_294_, 1);
if (v_isShared_297_ == 0)
{
lean_ctor_set(v___x_296_, 0, v_val_298_);
v___x_300_ = v___x_296_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v_val_298_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
lean_dec(v_constName_263_);
v_a_303_ = lean_ctor_get(v___x_293_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_293_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_293_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_293_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
}
else
{
lean_dec(v_val_281_);
goto v___jp_269_;
}
}
else
{
lean_dec(v___x_280_);
goto v___jp_269_;
}
v___jp_269_:
{
lean_object* v___x_270_; uint8_t v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_270_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__1, &lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__1_once, _init_lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__1);
v___x_271_ = 0;
v___x_272_ = l_Lean_MessageData_ofConstName(v_constName_263_, v___x_271_);
v___x_273_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_270_);
lean_ctor_set(v___x_273_, 1, v___x_272_);
v___x_274_ = lean_obj_once(&lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__3, &lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__3_once, _init_lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___closed__3);
v___x_275_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_273_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg(v___x_275_, v___y_264_, v___y_265_, v___y_266_, v___y_267_);
return v___x_276_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0___boxed(lean_object* v_constName_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0(v_constName_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_);
lean_dec(v___y_315_);
lean_dec_ref(v___y_314_);
lean_dec(v___y_313_);
lean_dec_ref(v___y_312_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg(lean_object* v_a_318_, lean_object* v_as_319_, size_t v_sz_320_, size_t v_i_321_, lean_object* v_b_322_){
_start:
{
lean_object* v_a_325_; uint8_t v___x_329_; 
v___x_329_ = lean_usize_dec_lt(v_i_321_, v_sz_320_);
if (v___x_329_ == 0)
{
lean_object* v___x_330_; 
v___x_330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_330_, 0, v_b_322_);
return v___x_330_;
}
else
{
lean_object* v_snd_331_; lean_object* v_snd_332_; lean_object* v_snd_333_; lean_object* v_snd_334_; lean_object* v_fst_335_; lean_object* v___x_337_; uint8_t v_isShared_338_; uint8_t v_isSharedCheck_456_; 
v_snd_331_ = lean_ctor_get(v_b_322_, 1);
lean_inc(v_snd_331_);
v_snd_332_ = lean_ctor_get(v_snd_331_, 1);
lean_inc(v_snd_332_);
v_snd_333_ = lean_ctor_get(v_snd_332_, 1);
lean_inc(v_snd_333_);
v_snd_334_ = lean_ctor_get(v_snd_333_, 1);
lean_inc(v_snd_334_);
v_fst_335_ = lean_ctor_get(v_b_322_, 0);
v_isSharedCheck_456_ = !lean_is_exclusive(v_b_322_);
if (v_isSharedCheck_456_ == 0)
{
lean_object* v_unused_457_; 
v_unused_457_ = lean_ctor_get(v_b_322_, 1);
lean_dec(v_unused_457_);
v___x_337_ = v_b_322_;
v_isShared_338_ = v_isSharedCheck_456_;
goto v_resetjp_336_;
}
else
{
lean_inc(v_fst_335_);
lean_dec(v_b_322_);
v___x_337_ = lean_box(0);
v_isShared_338_ = v_isSharedCheck_456_;
goto v_resetjp_336_;
}
v_resetjp_336_:
{
lean_object* v_fst_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_454_; 
v_fst_339_ = lean_ctor_get(v_snd_331_, 0);
v_isSharedCheck_454_ = !lean_is_exclusive(v_snd_331_);
if (v_isSharedCheck_454_ == 0)
{
lean_object* v_unused_455_; 
v_unused_455_ = lean_ctor_get(v_snd_331_, 1);
lean_dec(v_unused_455_);
v___x_341_ = v_snd_331_;
v_isShared_342_ = v_isSharedCheck_454_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_fst_339_);
lean_dec(v_snd_331_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_454_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v_fst_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_452_; 
v_fst_343_ = lean_ctor_get(v_snd_332_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v_snd_332_);
if (v_isSharedCheck_452_ == 0)
{
lean_object* v_unused_453_; 
v_unused_453_ = lean_ctor_get(v_snd_332_, 1);
lean_dec(v_unused_453_);
v___x_345_ = v_snd_332_;
v_isShared_346_ = v_isSharedCheck_452_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_fst_343_);
lean_dec(v_snd_332_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_452_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v_fst_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_450_; 
v_fst_347_ = lean_ctor_get(v_snd_333_, 0);
v_isSharedCheck_450_ = !lean_is_exclusive(v_snd_333_);
if (v_isSharedCheck_450_ == 0)
{
lean_object* v_unused_451_; 
v_unused_451_ = lean_ctor_get(v_snd_333_, 1);
lean_dec(v_unused_451_);
v___x_349_ = v_snd_333_;
v_isShared_350_ = v_isSharedCheck_450_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_fst_347_);
lean_dec(v_snd_333_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_450_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v_start_351_; lean_object* v_stop_352_; lean_object* v_step_353_; uint8_t v___x_354_; 
v_start_351_ = lean_ctor_get(v_snd_334_, 0);
v_stop_352_ = lean_ctor_get(v_snd_334_, 1);
v_step_353_ = lean_ctor_get(v_snd_334_, 2);
v___x_354_ = lean_nat_dec_lt(v_start_351_, v_stop_352_);
if (v___x_354_ == 0)
{
lean_object* v___x_356_; 
if (v_isShared_350_ == 0)
{
v___x_356_ = v___x_349_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_fst_347_);
lean_ctor_set(v_reuseFailAlloc_367_, 1, v_snd_334_);
v___x_356_ = v_reuseFailAlloc_367_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
lean_object* v___x_358_; 
if (v_isShared_346_ == 0)
{
lean_ctor_set(v___x_345_, 1, v___x_356_);
v___x_358_ = v___x_345_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_fst_343_);
lean_ctor_set(v_reuseFailAlloc_366_, 1, v___x_356_);
v___x_358_ = v_reuseFailAlloc_366_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
lean_object* v___x_360_; 
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 1, v___x_358_);
v___x_360_ = v___x_341_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_365_, 1, v___x_358_);
v___x_360_ = v_reuseFailAlloc_365_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
lean_object* v___x_362_; 
if (v_isShared_338_ == 0)
{
lean_ctor_set(v___x_337_, 1, v___x_360_);
v___x_362_ = v___x_337_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v_fst_335_);
lean_ctor_set(v_reuseFailAlloc_364_, 1, v___x_360_);
v___x_362_ = v_reuseFailAlloc_364_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_363_; 
v___x_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
return v___x_363_;
}
}
}
}
}
else
{
lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_446_; 
lean_inc(v_step_353_);
lean_inc(v_stop_352_);
lean_inc(v_start_351_);
v_isSharedCheck_446_ = !lean_is_exclusive(v_snd_334_);
if (v_isSharedCheck_446_ == 0)
{
lean_object* v_unused_447_; lean_object* v_unused_448_; lean_object* v_unused_449_; 
v_unused_447_ = lean_ctor_get(v_snd_334_, 2);
lean_dec(v_unused_447_);
v_unused_448_ = lean_ctor_get(v_snd_334_, 1);
lean_dec(v_unused_448_);
v_unused_449_ = lean_ctor_get(v_snd_334_, 0);
lean_dec(v_unused_449_);
v___x_369_ = v_snd_334_;
v_isShared_370_ = v_isSharedCheck_446_;
goto v_resetjp_368_;
}
else
{
lean_dec(v_snd_334_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_446_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v_array_371_; lean_object* v_start_372_; lean_object* v_stop_373_; lean_object* v___x_374_; lean_object* v___x_376_; 
v_array_371_ = lean_ctor_get(v_fst_347_, 0);
v_start_372_ = lean_ctor_get(v_fst_347_, 1);
v_stop_373_ = lean_ctor_get(v_fst_347_, 2);
v___x_374_ = lean_nat_add(v_start_351_, v_step_353_);
if (v_isShared_370_ == 0)
{
lean_ctor_set(v___x_369_, 0, v___x_374_);
v___x_376_ = v___x_369_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v___x_374_);
lean_ctor_set(v_reuseFailAlloc_445_, 1, v_stop_352_);
lean_ctor_set(v_reuseFailAlloc_445_, 2, v_step_353_);
v___x_376_ = v_reuseFailAlloc_445_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
uint8_t v___x_377_; 
v___x_377_ = lean_nat_dec_lt(v_start_372_, v_stop_373_);
if (v___x_377_ == 0)
{
lean_object* v___x_379_; 
lean_dec(v_start_351_);
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 1, v___x_376_);
v___x_379_ = v___x_349_;
goto v_reusejp_378_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v_fst_347_);
lean_ctor_set(v_reuseFailAlloc_390_, 1, v___x_376_);
v___x_379_ = v_reuseFailAlloc_390_;
goto v_reusejp_378_;
}
v_reusejp_378_:
{
lean_object* v___x_381_; 
if (v_isShared_346_ == 0)
{
lean_ctor_set(v___x_345_, 1, v___x_379_);
v___x_381_ = v___x_345_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_389_; 
v_reuseFailAlloc_389_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_389_, 0, v_fst_343_);
lean_ctor_set(v_reuseFailAlloc_389_, 1, v___x_379_);
v___x_381_ = v_reuseFailAlloc_389_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_383_; 
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 1, v___x_381_);
v___x_383_ = v___x_341_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_388_, 1, v___x_381_);
v___x_383_ = v_reuseFailAlloc_388_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
lean_object* v___x_385_; 
if (v_isShared_338_ == 0)
{
lean_ctor_set(v___x_337_, 1, v___x_383_);
v___x_385_ = v___x_337_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v_fst_335_);
lean_ctor_set(v_reuseFailAlloc_387_, 1, v___x_383_);
v___x_385_ = v_reuseFailAlloc_387_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
lean_object* v___x_386_; 
v___x_386_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
return v___x_386_;
}
}
}
}
}
else
{
lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_441_; 
lean_inc(v_stop_373_);
lean_inc(v_start_372_);
lean_inc_ref(v_array_371_);
v_isSharedCheck_441_ = !lean_is_exclusive(v_fst_347_);
if (v_isSharedCheck_441_ == 0)
{
lean_object* v_unused_442_; lean_object* v_unused_443_; lean_object* v_unused_444_; 
v_unused_442_ = lean_ctor_get(v_fst_347_, 2);
lean_dec(v_unused_442_);
v_unused_443_ = lean_ctor_get(v_fst_347_, 1);
lean_dec(v_unused_443_);
v_unused_444_ = lean_ctor_get(v_fst_347_, 0);
lean_dec(v_unused_444_);
v___x_392_ = v_fst_347_;
v_isShared_393_ = v_isSharedCheck_441_;
goto v_resetjp_391_;
}
else
{
lean_dec(v_fst_347_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_441_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
lean_object* v_numParams_394_; lean_object* v___x_395_; lean_object* v_a_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_400_; 
v_numParams_394_ = lean_ctor_get(v_a_318_, 3);
v___x_395_ = lean_array_fget(v_array_371_, v_start_372_);
v_a_396_ = lean_array_uget_borrowed(v_as_319_, v_i_321_);
v___x_397_ = lean_unsigned_to_nat(1u);
v___x_398_ = lean_nat_add(v_start_372_, v___x_397_);
lean_dec(v_start_372_);
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v___x_398_);
v___x_400_ = v___x_392_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v_array_371_);
lean_ctor_set(v_reuseFailAlloc_440_, 1, v___x_398_);
lean_ctor_set(v_reuseFailAlloc_440_, 2, v_stop_373_);
v___x_400_ = v_reuseFailAlloc_440_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
uint8_t v___x_431_; 
v___x_431_ = lean_nat_dec_le(v_numParams_394_, v_start_351_);
lean_dec(v_start_351_);
if (v___x_431_ == 0)
{
goto v___jp_401_;
}
else
{
uint8_t v___x_432_; uint8_t v___x_433_; 
v___x_432_ = lean_unbox(v___x_395_);
v___x_433_ = l_Lean_BinderInfo_isExplicit(v___x_432_);
if (v___x_433_ == 0)
{
goto v___jp_401_;
}
else
{
lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; 
lean_dec(v___x_395_);
lean_del_object(v___x_349_);
lean_del_object(v___x_345_);
lean_del_object(v___x_341_);
lean_del_object(v___x_337_);
v___x_434_ = l_Lean_Expr_mvarId_x21(v_a_396_);
v___x_435_ = lean_array_push(v_fst_335_, v___x_434_);
v___x_436_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_400_);
lean_ctor_set(v___x_436_, 1, v___x_376_);
v___x_437_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_437_, 0, v_fst_343_);
lean_ctor_set(v___x_437_, 1, v___x_436_);
v___x_438_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_438_, 0, v_fst_339_);
lean_ctor_set(v___x_438_, 1, v___x_437_);
v___x_439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_439_, 0, v___x_435_);
lean_ctor_set(v___x_439_, 1, v___x_438_);
v_a_325_ = v___x_439_;
goto v___jp_324_;
}
}
v___jp_401_:
{
lean_object* v___x_402_; lean_object* v___x_403_; uint8_t v___x_404_; uint8_t v___x_405_; 
v___x_402_ = l_Lean_Expr_mvarId_x21(v_a_396_);
lean_inc(v___x_402_);
v___x_403_ = lean_array_push(v_fst_339_, v___x_402_);
v___x_404_ = lean_unbox(v___x_395_);
lean_dec(v___x_395_);
v___x_405_ = l_Lean_BinderInfo_isInstImplicit(v___x_404_);
if (v___x_405_ == 0)
{
lean_object* v___x_407_; 
lean_dec(v___x_402_);
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 1, v___x_376_);
lean_ctor_set(v___x_349_, 0, v___x_400_);
v___x_407_ = v___x_349_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v___x_400_);
lean_ctor_set(v_reuseFailAlloc_417_, 1, v___x_376_);
v___x_407_ = v_reuseFailAlloc_417_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
lean_object* v___x_409_; 
if (v_isShared_346_ == 0)
{
lean_ctor_set(v___x_345_, 1, v___x_407_);
v___x_409_ = v___x_345_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_fst_343_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v___x_407_);
v___x_409_ = v_reuseFailAlloc_416_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
lean_object* v___x_411_; 
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 1, v___x_409_);
lean_ctor_set(v___x_341_, 0, v___x_403_);
v___x_411_ = v___x_341_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v___x_403_);
lean_ctor_set(v_reuseFailAlloc_415_, 1, v___x_409_);
v___x_411_ = v_reuseFailAlloc_415_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
lean_object* v___x_413_; 
if (v_isShared_338_ == 0)
{
lean_ctor_set(v___x_337_, 1, v___x_411_);
v___x_413_ = v___x_337_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_fst_335_);
lean_ctor_set(v_reuseFailAlloc_414_, 1, v___x_411_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
v_a_325_ = v___x_413_;
goto v___jp_324_;
}
}
}
}
}
else
{
lean_object* v___x_418_; lean_object* v___x_420_; 
v___x_418_ = lean_array_push(v_fst_343_, v___x_402_);
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 1, v___x_376_);
lean_ctor_set(v___x_349_, 0, v___x_400_);
v___x_420_ = v___x_349_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_400_);
lean_ctor_set(v_reuseFailAlloc_430_, 1, v___x_376_);
v___x_420_ = v_reuseFailAlloc_430_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
lean_object* v___x_422_; 
if (v_isShared_346_ == 0)
{
lean_ctor_set(v___x_345_, 1, v___x_420_);
lean_ctor_set(v___x_345_, 0, v___x_418_);
v___x_422_ = v___x_345_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v___x_418_);
lean_ctor_set(v_reuseFailAlloc_429_, 1, v___x_420_);
v___x_422_ = v_reuseFailAlloc_429_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
lean_object* v___x_424_; 
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 1, v___x_422_);
lean_ctor_set(v___x_341_, 0, v___x_403_);
v___x_424_ = v___x_341_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_403_);
lean_ctor_set(v_reuseFailAlloc_428_, 1, v___x_422_);
v___x_424_ = v_reuseFailAlloc_428_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
lean_object* v___x_426_; 
if (v_isShared_338_ == 0)
{
lean_ctor_set(v___x_337_, 1, v___x_424_);
v___x_426_ = v___x_337_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v_fst_335_);
lean_ctor_set(v_reuseFailAlloc_427_, 1, v___x_424_);
v___x_426_ = v_reuseFailAlloc_427_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
v_a_325_ = v___x_426_;
goto v___jp_324_;
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
}
}
}
}
}
}
}
v___jp_324_:
{
size_t v___x_326_; size_t v___x_327_; 
v___x_326_ = ((size_t)1ULL);
v___x_327_ = lean_usize_add(v_i_321_, v___x_326_);
v_i_321_ = v___x_327_;
v_b_322_ = v_a_325_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg___boxed(lean_object* v_a_458_, lean_object* v_as_459_, lean_object* v_sz_460_, lean_object* v_i_461_, lean_object* v_b_462_, lean_object* v___y_463_){
_start:
{
size_t v_sz_boxed_464_; size_t v_i_boxed_465_; lean_object* v_res_466_; 
v_sz_boxed_464_ = lean_unbox_usize(v_sz_460_);
lean_dec(v_sz_460_);
v_i_boxed_465_ = lean_unbox_usize(v_i_461_);
lean_dec(v_i_461_);
v_res_466_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg(v_a_458_, v_as_459_, v_sz_boxed_464_, v_i_boxed_465_, v_b_462_);
lean_dec_ref(v_as_459_);
lean_dec_ref(v_a_458_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8_spec__9___redArg(lean_object* v_x_467_, lean_object* v_x_468_, lean_object* v_x_469_, lean_object* v_x_470_){
_start:
{
lean_object* v_ks_471_; lean_object* v_vs_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_496_; 
v_ks_471_ = lean_ctor_get(v_x_467_, 0);
v_vs_472_ = lean_ctor_get(v_x_467_, 1);
v_isSharedCheck_496_ = !lean_is_exclusive(v_x_467_);
if (v_isSharedCheck_496_ == 0)
{
v___x_474_ = v_x_467_;
v_isShared_475_ = v_isSharedCheck_496_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_vs_472_);
lean_inc(v_ks_471_);
lean_dec(v_x_467_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_496_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_476_; uint8_t v___x_477_; 
v___x_476_ = lean_array_get_size(v_ks_471_);
v___x_477_ = lean_nat_dec_lt(v_x_468_, v___x_476_);
if (v___x_477_ == 0)
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_481_; 
lean_dec(v_x_468_);
v___x_478_ = lean_array_push(v_ks_471_, v_x_469_);
v___x_479_ = lean_array_push(v_vs_472_, v_x_470_);
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 1, v___x_479_);
lean_ctor_set(v___x_474_, 0, v___x_478_);
v___x_481_ = v___x_474_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v___x_478_);
lean_ctor_set(v_reuseFailAlloc_482_, 1, v___x_479_);
v___x_481_ = v_reuseFailAlloc_482_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
return v___x_481_;
}
}
else
{
lean_object* v_k_x27_483_; uint8_t v___x_484_; 
v_k_x27_483_ = lean_array_fget_borrowed(v_ks_471_, v_x_468_);
v___x_484_ = l_Lean_instBEqMVarId_beq(v_x_469_, v_k_x27_483_);
if (v___x_484_ == 0)
{
lean_object* v___x_486_; 
if (v_isShared_475_ == 0)
{
v___x_486_ = v___x_474_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v_ks_471_);
lean_ctor_set(v_reuseFailAlloc_490_, 1, v_vs_472_);
v___x_486_ = v_reuseFailAlloc_490_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_487_ = lean_unsigned_to_nat(1u);
v___x_488_ = lean_nat_add(v_x_468_, v___x_487_);
lean_dec(v_x_468_);
v_x_467_ = v___x_486_;
v_x_468_ = v___x_488_;
goto _start;
}
}
else
{
lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_494_; 
v___x_491_ = lean_array_fset(v_ks_471_, v_x_468_, v_x_469_);
v___x_492_ = lean_array_fset(v_vs_472_, v_x_468_, v_x_470_);
lean_dec(v_x_468_);
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 1, v___x_492_);
lean_ctor_set(v___x_474_, 0, v___x_491_);
v___x_494_ = v___x_474_;
goto v_reusejp_493_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v___x_491_);
lean_ctor_set(v_reuseFailAlloc_495_, 1, v___x_492_);
v___x_494_ = v_reuseFailAlloc_495_;
goto v_reusejp_493_;
}
v_reusejp_493_:
{
return v___x_494_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8___redArg(lean_object* v_n_497_, lean_object* v_k_498_, lean_object* v_v_499_){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = lean_unsigned_to_nat(0u);
v___x_501_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8_spec__9___redArg(v_n_497_, v___x_500_, v_k_498_, v_v_499_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(lean_object* v_x_503_, size_t v_x_504_, size_t v_x_505_, lean_object* v_x_506_, lean_object* v_x_507_){
_start:
{
if (lean_obj_tag(v_x_503_) == 0)
{
lean_object* v_es_508_; size_t v___x_509_; size_t v___x_510_; lean_object* v_j_511_; lean_object* v___x_512_; uint8_t v___x_513_; 
v_es_508_ = lean_ctor_get(v_x_503_, 0);
v___x_509_ = ((size_t)31ULL);
v___x_510_ = lean_usize_land(v_x_504_, v___x_509_);
v_j_511_ = lean_usize_to_nat(v___x_510_);
v___x_512_ = lean_array_get_size(v_es_508_);
v___x_513_ = lean_nat_dec_lt(v_j_511_, v___x_512_);
if (v___x_513_ == 0)
{
lean_dec(v_j_511_);
lean_dec(v_x_507_);
lean_dec(v_x_506_);
return v_x_503_;
}
else
{
lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_552_; 
lean_inc_ref(v_es_508_);
v_isSharedCheck_552_ = !lean_is_exclusive(v_x_503_);
if (v_isSharedCheck_552_ == 0)
{
lean_object* v_unused_553_; 
v_unused_553_ = lean_ctor_get(v_x_503_, 0);
lean_dec(v_unused_553_);
v___x_515_ = v_x_503_;
v_isShared_516_ = v_isSharedCheck_552_;
goto v_resetjp_514_;
}
else
{
lean_dec(v_x_503_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_552_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v_v_517_; lean_object* v___x_518_; lean_object* v_xs_x27_519_; lean_object* v___y_521_; 
v_v_517_ = lean_array_fget(v_es_508_, v_j_511_);
v___x_518_ = lean_box(0);
v_xs_x27_519_ = lean_array_fset(v_es_508_, v_j_511_, v___x_518_);
switch(lean_obj_tag(v_v_517_))
{
case 0:
{
lean_object* v_key_526_; lean_object* v_val_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_537_; 
v_key_526_ = lean_ctor_get(v_v_517_, 0);
v_val_527_ = lean_ctor_get(v_v_517_, 1);
v_isSharedCheck_537_ = !lean_is_exclusive(v_v_517_);
if (v_isSharedCheck_537_ == 0)
{
v___x_529_ = v_v_517_;
v_isShared_530_ = v_isSharedCheck_537_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_val_527_);
lean_inc(v_key_526_);
lean_dec(v_v_517_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_537_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
uint8_t v___x_531_; 
v___x_531_ = l_Lean_instBEqMVarId_beq(v_x_506_, v_key_526_);
if (v___x_531_ == 0)
{
lean_object* v___x_532_; lean_object* v___x_533_; 
lean_del_object(v___x_529_);
v___x_532_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_526_, v_val_527_, v_x_506_, v_x_507_);
v___x_533_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_533_, 0, v___x_532_);
v___y_521_ = v___x_533_;
goto v___jp_520_;
}
else
{
lean_object* v___x_535_; 
lean_dec(v_val_527_);
lean_dec(v_key_526_);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 1, v_x_507_);
lean_ctor_set(v___x_529_, 0, v_x_506_);
v___x_535_ = v___x_529_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_536_; 
v_reuseFailAlloc_536_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_536_, 0, v_x_506_);
lean_ctor_set(v_reuseFailAlloc_536_, 1, v_x_507_);
v___x_535_ = v_reuseFailAlloc_536_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
v___y_521_ = v___x_535_;
goto v___jp_520_;
}
}
}
}
case 1:
{
lean_object* v_node_538_; lean_object* v___x_540_; uint8_t v_isShared_541_; uint8_t v_isSharedCheck_550_; 
v_node_538_ = lean_ctor_get(v_v_517_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v_v_517_);
if (v_isSharedCheck_550_ == 0)
{
v___x_540_ = v_v_517_;
v_isShared_541_ = v_isSharedCheck_550_;
goto v_resetjp_539_;
}
else
{
lean_inc(v_node_538_);
lean_dec(v_v_517_);
v___x_540_ = lean_box(0);
v_isShared_541_ = v_isSharedCheck_550_;
goto v_resetjp_539_;
}
v_resetjp_539_:
{
size_t v___x_542_; size_t v___x_543_; size_t v___x_544_; size_t v___x_545_; lean_object* v___x_546_; lean_object* v___x_548_; 
v___x_542_ = ((size_t)5ULL);
v___x_543_ = lean_usize_shift_right(v_x_504_, v___x_542_);
v___x_544_ = ((size_t)1ULL);
v___x_545_ = lean_usize_add(v_x_505_, v___x_544_);
v___x_546_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(v_node_538_, v___x_543_, v___x_545_, v_x_506_, v_x_507_);
if (v_isShared_541_ == 0)
{
lean_ctor_set(v___x_540_, 0, v___x_546_);
v___x_548_ = v___x_540_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v___x_546_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
v___y_521_ = v___x_548_;
goto v___jp_520_;
}
}
}
default: 
{
lean_object* v___x_551_; 
v___x_551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_551_, 0, v_x_506_);
lean_ctor_set(v___x_551_, 1, v_x_507_);
v___y_521_ = v___x_551_;
goto v___jp_520_;
}
}
v___jp_520_:
{
lean_object* v___x_522_; lean_object* v___x_524_; 
v___x_522_ = lean_array_fset(v_xs_x27_519_, v_j_511_, v___y_521_);
lean_dec(v_j_511_);
if (v_isShared_516_ == 0)
{
lean_ctor_set(v___x_515_, 0, v___x_522_);
v___x_524_ = v___x_515_;
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
}
}
}
else
{
lean_object* v_ks_554_; lean_object* v_vs_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_575_; 
v_ks_554_ = lean_ctor_get(v_x_503_, 0);
v_vs_555_ = lean_ctor_get(v_x_503_, 1);
v_isSharedCheck_575_ = !lean_is_exclusive(v_x_503_);
if (v_isSharedCheck_575_ == 0)
{
v___x_557_ = v_x_503_;
v_isShared_558_ = v_isSharedCheck_575_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_vs_555_);
lean_inc(v_ks_554_);
lean_dec(v_x_503_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_575_;
goto v_resetjp_556_;
}
v_resetjp_556_:
{
lean_object* v___x_560_; 
if (v_isShared_558_ == 0)
{
v___x_560_ = v___x_557_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v_ks_554_);
lean_ctor_set(v_reuseFailAlloc_574_, 1, v_vs_555_);
v___x_560_ = v_reuseFailAlloc_574_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
lean_object* v_newNode_561_; uint8_t v___y_563_; size_t v___x_569_; uint8_t v___x_570_; 
v_newNode_561_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8___redArg(v___x_560_, v_x_506_, v_x_507_);
v___x_569_ = ((size_t)7ULL);
v___x_570_ = lean_usize_dec_le(v___x_569_, v_x_505_);
if (v___x_570_ == 0)
{
lean_object* v___x_571_; lean_object* v___x_572_; uint8_t v___x_573_; 
v___x_571_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_561_);
v___x_572_ = lean_unsigned_to_nat(4u);
v___x_573_ = lean_nat_dec_lt(v___x_571_, v___x_572_);
lean_dec(v___x_571_);
v___y_563_ = v___x_573_;
goto v___jp_562_;
}
else
{
v___y_563_ = v___x_570_;
goto v___jp_562_;
}
v___jp_562_:
{
if (v___y_563_ == 0)
{
lean_object* v_ks_564_; lean_object* v_vs_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
v_ks_564_ = lean_ctor_get(v_newNode_561_, 0);
lean_inc_ref(v_ks_564_);
v_vs_565_ = lean_ctor_get(v_newNode_561_, 1);
lean_inc_ref(v_vs_565_);
lean_dec_ref(v_newNode_561_);
v___x_566_ = lean_unsigned_to_nat(0u);
v___x_567_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___closed__0);
v___x_568_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg(v_x_505_, v_ks_564_, v_vs_565_, v___x_566_, v___x_567_);
lean_dec_ref(v_vs_565_);
lean_dec_ref(v_ks_564_);
return v___x_568_;
}
else
{
return v_newNode_561_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg(size_t v_depth_576_, lean_object* v_keys_577_, lean_object* v_vals_578_, lean_object* v_i_579_, lean_object* v_entries_580_){
_start:
{
lean_object* v___x_581_; uint8_t v___x_582_; 
v___x_581_ = lean_array_get_size(v_keys_577_);
v___x_582_ = lean_nat_dec_lt(v_i_579_, v___x_581_);
if (v___x_582_ == 0)
{
lean_dec(v_i_579_);
return v_entries_580_;
}
else
{
lean_object* v_k_583_; lean_object* v_v_584_; uint64_t v___x_585_; size_t v_h_586_; size_t v___x_587_; lean_object* v___x_588_; size_t v___x_589_; size_t v___x_590_; size_t v___x_591_; size_t v_h_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v_k_583_ = lean_array_fget_borrowed(v_keys_577_, v_i_579_);
v_v_584_ = lean_array_fget_borrowed(v_vals_578_, v_i_579_);
v___x_585_ = l_Lean_instHashableMVarId_hash(v_k_583_);
v_h_586_ = lean_uint64_to_usize(v___x_585_);
v___x_587_ = ((size_t)5ULL);
v___x_588_ = lean_unsigned_to_nat(1u);
v___x_589_ = ((size_t)1ULL);
v___x_590_ = lean_usize_sub(v_depth_576_, v___x_589_);
v___x_591_ = lean_usize_mul(v___x_587_, v___x_590_);
v_h_592_ = lean_usize_shift_right(v_h_586_, v___x_591_);
v___x_593_ = lean_nat_add(v_i_579_, v___x_588_);
lean_dec(v_i_579_);
lean_inc(v_v_584_);
lean_inc(v_k_583_);
v___x_594_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(v_entries_580_, v_h_592_, v_depth_576_, v_k_583_, v_v_584_);
v_i_579_ = v___x_593_;
v_entries_580_ = v___x_594_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg___boxed(lean_object* v_depth_596_, lean_object* v_keys_597_, lean_object* v_vals_598_, lean_object* v_i_599_, lean_object* v_entries_600_){
_start:
{
size_t v_depth_boxed_601_; lean_object* v_res_602_; 
v_depth_boxed_601_ = lean_unbox_usize(v_depth_596_);
lean_dec(v_depth_596_);
v_res_602_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg(v_depth_boxed_601_, v_keys_597_, v_vals_598_, v_i_599_, v_entries_600_);
lean_dec_ref(v_vals_598_);
lean_dec_ref(v_keys_597_);
return v_res_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_x_603_, lean_object* v_x_604_, lean_object* v_x_605_, lean_object* v_x_606_, lean_object* v_x_607_){
_start:
{
size_t v_x_10771__boxed_608_; size_t v_x_10772__boxed_609_; lean_object* v_res_610_; 
v_x_10771__boxed_608_ = lean_unbox_usize(v_x_604_);
lean_dec(v_x_604_);
v_x_10772__boxed_609_ = lean_unbox_usize(v_x_605_);
lean_dec(v_x_605_);
v_res_610_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(v_x_603_, v_x_10771__boxed_608_, v_x_10772__boxed_609_, v_x_606_, v_x_607_);
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3___redArg(lean_object* v_x_611_, lean_object* v_x_612_, lean_object* v_x_613_){
_start:
{
uint64_t v___x_614_; size_t v___x_615_; size_t v___x_616_; lean_object* v___x_617_; 
v___x_614_ = l_Lean_instHashableMVarId_hash(v_x_612_);
v___x_615_ = lean_uint64_to_usize(v___x_614_);
v___x_616_ = ((size_t)1ULL);
v___x_617_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(v_x_611_, v___x_615_, v___x_616_, v_x_612_, v_x_613_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg(lean_object* v_mvarId_618_, lean_object* v_val_619_, lean_object* v___y_620_){
_start:
{
lean_object* v___x_622_; lean_object* v_mctx_623_; lean_object* v_cache_624_; lean_object* v_zetaDeltaFVarIds_625_; lean_object* v_postponed_626_; lean_object* v_diag_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_655_; 
v___x_622_ = lean_st_ref_take(v___y_620_);
v_mctx_623_ = lean_ctor_get(v___x_622_, 0);
v_cache_624_ = lean_ctor_get(v___x_622_, 1);
v_zetaDeltaFVarIds_625_ = lean_ctor_get(v___x_622_, 2);
v_postponed_626_ = lean_ctor_get(v___x_622_, 3);
v_diag_627_ = lean_ctor_get(v___x_622_, 4);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_655_ == 0)
{
v___x_629_ = v___x_622_;
v_isShared_630_ = v_isSharedCheck_655_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_diag_627_);
lean_inc(v_postponed_626_);
lean_inc(v_zetaDeltaFVarIds_625_);
lean_inc(v_cache_624_);
lean_inc(v_mctx_623_);
lean_dec(v___x_622_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_655_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v_depth_631_; lean_object* v_levelAssignDepth_632_; lean_object* v_lmvarCounter_633_; lean_object* v_mvarCounter_634_; lean_object* v_lDecls_635_; lean_object* v_decls_636_; lean_object* v_userNames_637_; lean_object* v_lAssignment_638_; lean_object* v_eAssignment_639_; lean_object* v_dAssignment_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_654_; 
v_depth_631_ = lean_ctor_get(v_mctx_623_, 0);
v_levelAssignDepth_632_ = lean_ctor_get(v_mctx_623_, 1);
v_lmvarCounter_633_ = lean_ctor_get(v_mctx_623_, 2);
v_mvarCounter_634_ = lean_ctor_get(v_mctx_623_, 3);
v_lDecls_635_ = lean_ctor_get(v_mctx_623_, 4);
v_decls_636_ = lean_ctor_get(v_mctx_623_, 5);
v_userNames_637_ = lean_ctor_get(v_mctx_623_, 6);
v_lAssignment_638_ = lean_ctor_get(v_mctx_623_, 7);
v_eAssignment_639_ = lean_ctor_get(v_mctx_623_, 8);
v_dAssignment_640_ = lean_ctor_get(v_mctx_623_, 9);
v_isSharedCheck_654_ = !lean_is_exclusive(v_mctx_623_);
if (v_isSharedCheck_654_ == 0)
{
v___x_642_ = v_mctx_623_;
v_isShared_643_ = v_isSharedCheck_654_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_dAssignment_640_);
lean_inc(v_eAssignment_639_);
lean_inc(v_lAssignment_638_);
lean_inc(v_userNames_637_);
lean_inc(v_decls_636_);
lean_inc(v_lDecls_635_);
lean_inc(v_mvarCounter_634_);
lean_inc(v_lmvarCounter_633_);
lean_inc(v_levelAssignDepth_632_);
lean_inc(v_depth_631_);
lean_dec(v_mctx_623_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_654_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_644_; lean_object* v___x_646_; 
v___x_644_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3___redArg(v_eAssignment_639_, v_mvarId_618_, v_val_619_);
if (v_isShared_643_ == 0)
{
lean_ctor_set(v___x_642_, 8, v___x_644_);
v___x_646_ = v___x_642_;
goto v_reusejp_645_;
}
else
{
lean_object* v_reuseFailAlloc_653_; 
v_reuseFailAlloc_653_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_653_, 0, v_depth_631_);
lean_ctor_set(v_reuseFailAlloc_653_, 1, v_levelAssignDepth_632_);
lean_ctor_set(v_reuseFailAlloc_653_, 2, v_lmvarCounter_633_);
lean_ctor_set(v_reuseFailAlloc_653_, 3, v_mvarCounter_634_);
lean_ctor_set(v_reuseFailAlloc_653_, 4, v_lDecls_635_);
lean_ctor_set(v_reuseFailAlloc_653_, 5, v_decls_636_);
lean_ctor_set(v_reuseFailAlloc_653_, 6, v_userNames_637_);
lean_ctor_set(v_reuseFailAlloc_653_, 7, v_lAssignment_638_);
lean_ctor_set(v_reuseFailAlloc_653_, 8, v___x_644_);
lean_ctor_set(v_reuseFailAlloc_653_, 9, v_dAssignment_640_);
v___x_646_ = v_reuseFailAlloc_653_;
goto v_reusejp_645_;
}
v_reusejp_645_:
{
lean_object* v___x_648_; 
if (v_isShared_630_ == 0)
{
lean_ctor_set(v___x_629_, 0, v___x_646_);
v___x_648_ = v___x_629_;
goto v_reusejp_647_;
}
else
{
lean_object* v_reuseFailAlloc_652_; 
v_reuseFailAlloc_652_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_652_, 0, v___x_646_);
lean_ctor_set(v_reuseFailAlloc_652_, 1, v_cache_624_);
lean_ctor_set(v_reuseFailAlloc_652_, 2, v_zetaDeltaFVarIds_625_);
lean_ctor_set(v_reuseFailAlloc_652_, 3, v_postponed_626_);
lean_ctor_set(v_reuseFailAlloc_652_, 4, v_diag_627_);
v___x_648_ = v_reuseFailAlloc_652_;
goto v_reusejp_647_;
}
v_reusejp_647_:
{
lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_649_ = lean_st_ref_set(v___y_620_, v___x_648_);
v___x_650_ = lean_box(0);
v___x_651_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_651_, 0, v___x_650_);
return v___x_651_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg___boxed(lean_object* v_mvarId_656_, lean_object* v_val_657_, lean_object* v___y_658_, lean_object* v___y_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg(v_mvarId_656_, v_val_657_, v___y_658_);
lean_dec(v___y_658_);
return v_res_660_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__1(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__0));
v___x_663_ = l_Lean_stringToMessageData(v___x_662_);
return v___x_663_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__3(void){
_start:
{
lean_object* v___x_665_; lean_object* v___x_666_; 
v___x_665_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__2));
v___x_666_ = l_Lean_stringToMessageData(v___x_665_);
return v___x_666_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__6(void){
_start:
{
lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__5));
v___x_671_ = l_Lean_stringToMessageData(v___x_670_);
return v___x_671_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__8(void){
_start:
{
lean_object* v___x_673_; lean_object* v___x_674_; 
v___x_673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__7));
v___x_674_ = l_Lean_stringToMessageData(v___x_673_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0(lean_object* v_mvarId_675_, lean_object* v___x_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v___x_682_; 
lean_inc(v___x_676_);
lean_inc(v_mvarId_675_);
v___x_682_ = l_Lean_MVarId_checkNotAssigned(v_mvarId_675_, v___x_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_682_) == 0)
{
lean_object* v___x_683_; 
lean_dec_ref_known(v___x_682_, 1);
lean_inc(v_mvarId_675_);
v___x_683_ = l_Lean_MVarId_getType_x27(v_mvarId_675_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_683_) == 0)
{
lean_object* v_a_684_; lean_object* v___y_686_; lean_object* v___y_687_; lean_object* v___y_688_; lean_object* v___y_689_; lean_object* v___y_696_; lean_object* v___y_697_; lean_object* v___y_698_; lean_object* v___y_699_; lean_object* v___x_705_; 
v_a_684_ = lean_ctor_get(v___x_683_, 0);
lean_inc(v_a_684_);
lean_dec_ref_known(v___x_683_, 1);
v___x_705_ = l_Lean_Expr_getAppFn(v_a_684_);
if (lean_obj_tag(v___x_705_) == 4)
{
lean_object* v_declName_706_; lean_object* v_us_707_; lean_object* v___x_708_; lean_object* v_env_709_; uint8_t v___x_710_; lean_object* v___x_711_; 
v_declName_706_ = lean_ctor_get(v___x_705_, 0);
lean_inc(v_declName_706_);
v_us_707_ = lean_ctor_get(v___x_705_, 1);
lean_inc(v_us_707_);
lean_dec_ref_known(v___x_705_, 2);
v___x_708_ = lean_st_ref_get(v___y_680_);
v_env_709_ = lean_ctor_get(v___x_708_, 0);
lean_inc_ref(v_env_709_);
lean_dec(v___x_708_);
v___x_710_ = 0;
v___x_711_ = l_Lean_Environment_find_x3f(v_env_709_, v_declName_706_, v___x_710_);
if (lean_obj_tag(v___x_711_) == 0)
{
lean_dec(v_us_707_);
v___y_686_ = v___y_677_;
v___y_687_ = v___y_678_;
v___y_688_ = v___y_679_;
v___y_689_ = v___y_680_;
goto v___jp_685_;
}
else
{
lean_object* v_val_712_; 
v_val_712_ = lean_ctor_get(v___x_711_, 0);
lean_inc(v_val_712_);
lean_dec_ref_known(v___x_711_, 1);
if (lean_obj_tag(v_val_712_) == 5)
{
lean_object* v_val_713_; lean_object* v_ctors_714_; 
v_val_713_ = lean_ctor_get(v_val_712_, 0);
lean_inc_ref(v_val_713_);
lean_dec_ref_known(v_val_712_, 1);
v_ctors_714_ = lean_ctor_get(v_val_713_, 4);
lean_inc(v_ctors_714_);
lean_dec_ref(v_val_713_);
if (lean_obj_tag(v_ctors_714_) == 1)
{
lean_object* v_tail_715_; 
v_tail_715_ = lean_ctor_get(v_ctors_714_, 1);
if (lean_obj_tag(v_tail_715_) == 0)
{
lean_object* v_head_716_; lean_object* v___x_718_; uint8_t v_isShared_719_; uint8_t v_isSharedCheck_927_; 
lean_dec(v___x_676_);
v_head_716_ = lean_ctor_get(v_ctors_714_, 0);
v_isSharedCheck_927_ = !lean_is_exclusive(v_ctors_714_);
if (v_isSharedCheck_927_ == 0)
{
lean_object* v_unused_928_; 
v_unused_928_ = lean_ctor_get(v_ctors_714_, 1);
lean_dec(v_unused_928_);
v___x_718_ = v_ctors_714_;
v_isShared_719_ = v_isSharedCheck_927_;
goto v_resetjp_717_;
}
else
{
lean_inc(v_head_716_);
lean_dec(v_ctors_714_);
v___x_718_ = lean_box(0);
v_isShared_719_ = v_isSharedCheck_927_;
goto v_resetjp_717_;
}
v_resetjp_717_:
{
lean_object* v___x_720_; 
lean_inc(v_head_716_);
v___x_720_ = lp_mathlib_Lean_getConstInfoCtor___at___00Mathlib_Tactic_applyTheConstructor_spec__0(v_head_716_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_720_) == 0)
{
lean_object* v_a_721_; lean_object* v___x_722_; lean_object* v___x_723_; 
v_a_721_ = lean_ctor_get(v___x_720_, 0);
lean_inc(v_a_721_);
lean_dec_ref_known(v___x_720_, 1);
v___x_722_ = l_Lean_mkConst(v_head_716_, v_us_707_);
lean_inc(v___y_680_);
lean_inc_ref(v___y_679_);
lean_inc(v___y_678_);
lean_inc_ref(v___y_677_);
lean_inc_ref(v___x_722_);
v___x_723_ = lean_infer_type(v___x_722_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_723_) == 0)
{
lean_object* v_a_724_; lean_object* v___x_725_; uint8_t v___x_726_; lean_object* v___x_727_; 
v_a_724_ = lean_ctor_get(v___x_723_, 0);
lean_inc(v_a_724_);
lean_dec_ref_known(v___x_723_, 1);
v___x_725_ = lean_box(0);
v___x_726_ = 0;
v___x_727_ = l_Lean_Meta_forallMetaTelescopeReducing(v_a_724_, v___x_725_, v___x_726_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_727_) == 0)
{
lean_object* v_a_728_; lean_object* v_snd_729_; lean_object* v_fst_730_; lean_object* v___x_732_; uint8_t v_isShared_733_; uint8_t v_isSharedCheck_902_; 
v_a_728_ = lean_ctor_get(v___x_727_, 0);
lean_inc(v_a_728_);
lean_dec_ref_known(v___x_727_, 1);
v_snd_729_ = lean_ctor_get(v_a_728_, 1);
v_fst_730_ = lean_ctor_get(v_a_728_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v_a_728_);
if (v_isSharedCheck_902_ == 0)
{
v___x_732_ = v_a_728_;
v_isShared_733_ = v_isSharedCheck_902_;
goto v_resetjp_731_;
}
else
{
lean_inc(v_snd_729_);
lean_inc(v_fst_730_);
lean_dec(v_a_728_);
v___x_732_ = lean_box(0);
v_isShared_733_ = v_isSharedCheck_902_;
goto v_resetjp_731_;
}
v_resetjp_731_:
{
lean_object* v_fst_734_; lean_object* v___x_736_; uint8_t v_isShared_737_; uint8_t v_isSharedCheck_900_; 
v_fst_734_ = lean_ctor_get(v_snd_729_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v_snd_729_);
if (v_isSharedCheck_900_ == 0)
{
lean_object* v_unused_901_; 
v_unused_901_ = lean_ctor_get(v_snd_729_, 1);
lean_dec(v_unused_901_);
v___x_736_ = v_snd_729_;
v_isShared_737_ = v_isSharedCheck_900_;
goto v_resetjp_735_;
}
else
{
lean_inc(v_fst_734_);
lean_dec(v_snd_729_);
v___x_736_ = lean_box(0);
v_isShared_737_ = v_isSharedCheck_900_;
goto v_resetjp_735_;
}
v_resetjp_735_:
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_746_; 
v___x_738_ = lean_unsigned_to_nat(0u);
v___x_739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__4));
v___x_740_ = lean_array_get_size(v_fst_734_);
v___x_741_ = l_Array_toSubarray___redArg(v_fst_734_, v___x_738_, v___x_740_);
v___x_742_ = lean_array_get_size(v_fst_730_);
v___x_743_ = lean_unsigned_to_nat(1u);
v___x_744_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_744_, 0, v___x_738_);
lean_ctor_set(v___x_744_, 1, v___x_742_);
lean_ctor_set(v___x_744_, 2, v___x_743_);
if (v_isShared_737_ == 0)
{
lean_ctor_set(v___x_736_, 1, v___x_744_);
lean_ctor_set(v___x_736_, 0, v___x_741_);
v___x_746_ = v___x_736_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v___x_741_);
lean_ctor_set(v_reuseFailAlloc_899_, 1, v___x_744_);
v___x_746_ = v_reuseFailAlloc_899_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
lean_object* v___x_748_; 
if (v_isShared_733_ == 0)
{
lean_ctor_set(v___x_732_, 1, v___x_746_);
lean_ctor_set(v___x_732_, 0, v___x_739_);
v___x_748_ = v___x_732_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v___x_739_);
lean_ctor_set(v_reuseFailAlloc_898_, 1, v___x_746_);
v___x_748_ = v_reuseFailAlloc_898_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
lean_object* v___x_750_; 
if (v_isShared_719_ == 0)
{
lean_ctor_set_tag(v___x_718_, 0);
lean_ctor_set(v___x_718_, 1, v___x_748_);
lean_ctor_set(v___x_718_, 0, v___x_739_);
v___x_750_ = v___x_718_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_897_; 
v_reuseFailAlloc_897_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_897_, 0, v___x_739_);
lean_ctor_set(v_reuseFailAlloc_897_, 1, v___x_748_);
v___x_750_ = v_reuseFailAlloc_897_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
lean_object* v___x_751_; size_t v_sz_752_; size_t v___x_753_; lean_object* v___x_754_; 
v___x_751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_751_, 0, v___x_739_);
lean_ctor_set(v___x_751_, 1, v___x_750_);
v_sz_752_ = lean_array_size(v_fst_730_);
v___x_753_ = ((size_t)0ULL);
v___x_754_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg(v_a_721_, v_fst_730_, v_sz_752_, v___x_753_, v___x_751_);
lean_dec(v_a_721_);
if (lean_obj_tag(v___x_754_) == 0)
{
lean_object* v_a_755_; lean_object* v___x_756_; lean_object* v___x_757_; 
v_a_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_a_755_);
lean_dec_ref_known(v___x_754_, 1);
v___x_756_ = l_Lean_mkAppN(v___x_722_, v_fst_730_);
lean_dec(v_fst_730_);
lean_inc(v___y_680_);
lean_inc_ref(v___y_679_);
lean_inc(v___y_678_);
lean_inc_ref(v___y_677_);
lean_inc_ref(v___x_756_);
v___x_757_ = lean_infer_type(v___x_756_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_757_) == 0)
{
lean_object* v_a_758_; lean_object* v___x_759_; uint8_t v_foApprox_760_; uint8_t v_ctxApprox_761_; uint8_t v_quasiPatternApprox_762_; uint8_t v_constApprox_763_; uint8_t v_isDefEqStuckEx_764_; uint8_t v_unificationHints_765_; uint8_t v_proofIrrelevance_766_; uint8_t v_offsetCnstrs_767_; uint8_t v_transparency_768_; uint8_t v_etaStruct_769_; uint8_t v_univApprox_770_; uint8_t v_iota_771_; uint8_t v_beta_772_; uint8_t v_proj_773_; uint8_t v_zeta_774_; uint8_t v_zetaDelta_775_; uint8_t v_zetaUnused_776_; uint8_t v_zetaHave_777_; uint8_t v_canUnfoldPredicateConfig_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_880_; 
v_a_758_ = lean_ctor_get(v___x_757_, 0);
lean_inc(v_a_758_);
lean_dec_ref_known(v___x_757_, 1);
v___x_759_ = l_Lean_Meta_Context_config(v___y_677_);
v_foApprox_760_ = lean_ctor_get_uint8(v___x_759_, 0);
v_ctxApprox_761_ = lean_ctor_get_uint8(v___x_759_, 1);
v_quasiPatternApprox_762_ = lean_ctor_get_uint8(v___x_759_, 2);
v_constApprox_763_ = lean_ctor_get_uint8(v___x_759_, 3);
v_isDefEqStuckEx_764_ = lean_ctor_get_uint8(v___x_759_, 4);
v_unificationHints_765_ = lean_ctor_get_uint8(v___x_759_, 5);
v_proofIrrelevance_766_ = lean_ctor_get_uint8(v___x_759_, 6);
v_offsetCnstrs_767_ = lean_ctor_get_uint8(v___x_759_, 8);
v_transparency_768_ = lean_ctor_get_uint8(v___x_759_, 9);
v_etaStruct_769_ = lean_ctor_get_uint8(v___x_759_, 10);
v_univApprox_770_ = lean_ctor_get_uint8(v___x_759_, 11);
v_iota_771_ = lean_ctor_get_uint8(v___x_759_, 12);
v_beta_772_ = lean_ctor_get_uint8(v___x_759_, 13);
v_proj_773_ = lean_ctor_get_uint8(v___x_759_, 14);
v_zeta_774_ = lean_ctor_get_uint8(v___x_759_, 15);
v_zetaDelta_775_ = lean_ctor_get_uint8(v___x_759_, 16);
v_zetaUnused_776_ = lean_ctor_get_uint8(v___x_759_, 17);
v_zetaHave_777_ = lean_ctor_get_uint8(v___x_759_, 18);
v_canUnfoldPredicateConfig_778_ = lean_ctor_get_uint8(v___x_759_, 19);
v_isSharedCheck_880_ = !lean_is_exclusive(v___x_759_);
if (v_isSharedCheck_880_ == 0)
{
v___x_780_ = v___x_759_;
v_isShared_781_ = v_isSharedCheck_880_;
goto v_resetjp_779_;
}
else
{
lean_dec(v___x_759_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_880_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
uint8_t v_trackZetaDelta_782_; lean_object* v_zetaDeltaSet_783_; lean_object* v_lctx_784_; lean_object* v_localInstances_785_; lean_object* v_defEqCtx_x3f_786_; lean_object* v_synthPendingDepth_787_; lean_object* v_customCanUnfoldPredicate_x3f_788_; uint8_t v_univApprox_789_; uint8_t v_inTypeClassResolution_790_; uint8_t v_cacheInferType_791_; uint8_t v___x_792_; lean_object* v___x_794_; 
v_trackZetaDelta_782_ = lean_ctor_get_uint8(v___y_677_, sizeof(void*)*7);
v_zetaDeltaSet_783_ = lean_ctor_get(v___y_677_, 1);
v_lctx_784_ = lean_ctor_get(v___y_677_, 2);
v_localInstances_785_ = lean_ctor_get(v___y_677_, 3);
v_defEqCtx_x3f_786_ = lean_ctor_get(v___y_677_, 4);
v_synthPendingDepth_787_ = lean_ctor_get(v___y_677_, 5);
v_customCanUnfoldPredicate_x3f_788_ = lean_ctor_get(v___y_677_, 6);
v_univApprox_789_ = lean_ctor_get_uint8(v___y_677_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_790_ = lean_ctor_get_uint8(v___y_677_, sizeof(void*)*7 + 2);
v_cacheInferType_791_ = lean_ctor_get_uint8(v___y_677_, sizeof(void*)*7 + 3);
v___x_792_ = 1;
if (v_isShared_781_ == 0)
{
v___x_794_ = v___x_780_;
goto v_reusejp_793_;
}
else
{
lean_object* v_reuseFailAlloc_879_; 
v_reuseFailAlloc_879_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 0, v_foApprox_760_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 1, v_ctxApprox_761_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 2, v_quasiPatternApprox_762_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 3, v_constApprox_763_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 4, v_isDefEqStuckEx_764_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 5, v_unificationHints_765_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 6, v_proofIrrelevance_766_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 8, v_offsetCnstrs_767_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 9, v_transparency_768_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 10, v_etaStruct_769_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 11, v_univApprox_770_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 12, v_iota_771_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 13, v_beta_772_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 14, v_proj_773_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 15, v_zeta_774_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 16, v_zetaDelta_775_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 17, v_zetaUnused_776_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 18, v_zetaHave_777_);
lean_ctor_set_uint8(v_reuseFailAlloc_879_, 19, v_canUnfoldPredicateConfig_778_);
v___x_794_ = v_reuseFailAlloc_879_;
goto v_reusejp_793_;
}
v_reusejp_793_:
{
uint64_t v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
lean_ctor_set_uint8(v___x_794_, 7, v___x_792_);
v___x_795_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_794_);
v___x_796_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_796_, 0, v___x_794_);
lean_ctor_set_uint64(v___x_796_, sizeof(void*)*1, v___x_795_);
lean_inc(v_customCanUnfoldPredicate_x3f_788_);
lean_inc(v_synthPendingDepth_787_);
lean_inc(v_defEqCtx_x3f_786_);
lean_inc_ref(v_localInstances_785_);
lean_inc_ref(v_lctx_784_);
lean_inc(v_zetaDeltaSet_783_);
v___x_797_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_797_, 0, v___x_796_);
lean_ctor_set(v___x_797_, 1, v_zetaDeltaSet_783_);
lean_ctor_set(v___x_797_, 2, v_lctx_784_);
lean_ctor_set(v___x_797_, 3, v_localInstances_785_);
lean_ctor_set(v___x_797_, 4, v_defEqCtx_x3f_786_);
lean_ctor_set(v___x_797_, 5, v_synthPendingDepth_787_);
lean_ctor_set(v___x_797_, 6, v_customCanUnfoldPredicate_x3f_788_);
lean_ctor_set_uint8(v___x_797_, sizeof(void*)*7, v_trackZetaDelta_782_);
lean_ctor_set_uint8(v___x_797_, sizeof(void*)*7 + 1, v_univApprox_789_);
lean_ctor_set_uint8(v___x_797_, sizeof(void*)*7 + 2, v_inTypeClassResolution_790_);
lean_ctor_set_uint8(v___x_797_, sizeof(void*)*7 + 3, v_cacheInferType_791_);
lean_inc(v_a_684_);
lean_inc(v_a_758_);
v___x_798_ = l_Lean_Meta_isExprDefEq(v_a_758_, v_a_684_, v___x_797_, v___y_678_, v___y_679_, v___y_680_);
lean_dec_ref_known(v___x_797_, 7);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_object* v_snd_799_; lean_object* v_snd_800_; lean_object* v_a_801_; lean_object* v_fst_802_; lean_object* v___x_804_; uint8_t v_isShared_805_; uint8_t v_isSharedCheck_869_; 
v_snd_799_ = lean_ctor_get(v_a_755_, 1);
lean_inc(v_snd_799_);
v_snd_800_ = lean_ctor_get(v_snd_799_, 1);
lean_inc(v_snd_800_);
v_a_801_ = lean_ctor_get(v___x_798_, 0);
lean_inc(v_a_801_);
lean_dec_ref_known(v___x_798_, 1);
v_fst_802_ = lean_ctor_get(v_a_755_, 0);
v_isSharedCheck_869_ = !lean_is_exclusive(v_a_755_);
if (v_isSharedCheck_869_ == 0)
{
lean_object* v_unused_870_; 
v_unused_870_ = lean_ctor_get(v_a_755_, 1);
lean_dec(v_unused_870_);
v___x_804_ = v_a_755_;
v_isShared_805_ = v_isSharedCheck_869_;
goto v_resetjp_803_;
}
else
{
lean_inc(v_fst_802_);
lean_dec(v_a_755_);
v___x_804_ = lean_box(0);
v_isShared_805_ = v_isSharedCheck_869_;
goto v_resetjp_803_;
}
v_resetjp_803_:
{
lean_object* v_fst_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_867_; 
v_fst_806_ = lean_ctor_get(v_snd_799_, 0);
v_isSharedCheck_867_ = !lean_is_exclusive(v_snd_799_);
if (v_isSharedCheck_867_ == 0)
{
lean_object* v_unused_868_; 
v_unused_868_ = lean_ctor_get(v_snd_799_, 1);
lean_dec(v_unused_868_);
v___x_808_ = v_snd_799_;
v_isShared_809_ = v_isSharedCheck_867_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_fst_806_);
lean_dec(v_snd_799_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_867_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v_fst_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_865_; 
v_fst_810_ = lean_ctor_get(v_snd_800_, 0);
v_isSharedCheck_865_ = !lean_is_exclusive(v_snd_800_);
if (v_isSharedCheck_865_ == 0)
{
lean_object* v_unused_866_; 
v_unused_866_ = lean_ctor_get(v_snd_800_, 1);
lean_dec(v_unused_866_);
v___x_812_ = v_snd_800_;
v_isShared_813_ = v_isSharedCheck_865_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_fst_810_);
lean_dec(v_snd_800_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_865_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___y_815_; lean_object* v___y_816_; lean_object* v___y_817_; lean_object* v___y_818_; uint8_t v___x_837_; 
v___x_837_ = lean_unbox(v_a_801_);
lean_dec(v_a_801_);
if (v___x_837_ == 0)
{
lean_object* v___x_838_; 
lean_del_object(v___x_812_);
lean_dec(v_fst_810_);
lean_del_object(v___x_808_);
lean_dec(v_fst_806_);
lean_dec(v_fst_802_);
lean_dec(v_mvarId_675_);
v___x_838_ = l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(v_a_758_, v_a_684_, v___x_725_, v___x_739_);
if (lean_obj_tag(v___x_838_) == 0)
{
lean_object* v_a_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_843_; 
v_a_839_ = lean_ctor_get(v___x_838_, 0);
lean_inc(v_a_839_);
lean_dec_ref_known(v___x_838_, 1);
v___x_840_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__6, &lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__6);
v___x_841_ = l_Lean_indentExpr(v___x_756_);
if (v_isShared_805_ == 0)
{
lean_ctor_set_tag(v___x_804_, 7);
lean_ctor_set(v___x_804_, 1, v___x_841_);
lean_ctor_set(v___x_804_, 0, v___x_840_);
v___x_843_ = v___x_804_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v___x_840_);
lean_ctor_set(v_reuseFailAlloc_856_, 1, v___x_841_);
v___x_843_ = v_reuseFailAlloc_856_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v_a_848_; lean_object* v___x_850_; uint8_t v_isShared_851_; uint8_t v_isSharedCheck_855_; 
v___x_844_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__8, &lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__8);
v___x_845_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_845_, 0, v___x_843_);
lean_ctor_set(v___x_845_, 1, v___x_844_);
v___x_846_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_846_, 0, v___x_845_);
lean_ctor_set(v___x_846_, 1, v_a_839_);
v___x_847_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg(v___x_846_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
v_a_848_ = lean_ctor_get(v___x_847_, 0);
v_isSharedCheck_855_ = !lean_is_exclusive(v___x_847_);
if (v_isSharedCheck_855_ == 0)
{
v___x_850_ = v___x_847_;
v_isShared_851_ = v_isSharedCheck_855_;
goto v_resetjp_849_;
}
else
{
lean_inc(v_a_848_);
lean_dec(v___x_847_);
v___x_850_ = lean_box(0);
v_isShared_851_ = v_isSharedCheck_855_;
goto v_resetjp_849_;
}
v_resetjp_849_:
{
lean_object* v___x_853_; 
if (v_isShared_851_ == 0)
{
v___x_853_ = v___x_850_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_854_; 
v_reuseFailAlloc_854_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_854_, 0, v_a_848_);
v___x_853_ = v_reuseFailAlloc_854_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
return v___x_853_;
}
}
}
}
else
{
lean_object* v_a_857_; lean_object* v___x_859_; uint8_t v_isShared_860_; uint8_t v_isSharedCheck_864_; 
lean_del_object(v___x_804_);
lean_dec_ref(v___x_756_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
v_a_857_ = lean_ctor_get(v___x_838_, 0);
v_isSharedCheck_864_ = !lean_is_exclusive(v___x_838_);
if (v_isSharedCheck_864_ == 0)
{
v___x_859_ = v___x_838_;
v_isShared_860_ = v_isSharedCheck_864_;
goto v_resetjp_858_;
}
else
{
lean_inc(v_a_857_);
lean_dec(v___x_838_);
v___x_859_ = lean_box(0);
v_isShared_860_ = v_isSharedCheck_864_;
goto v_resetjp_858_;
}
v_resetjp_858_:
{
lean_object* v___x_862_; 
if (v_isShared_860_ == 0)
{
v___x_862_ = v___x_859_;
goto v_reusejp_861_;
}
else
{
lean_object* v_reuseFailAlloc_863_; 
v_reuseFailAlloc_863_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_863_, 0, v_a_857_);
v___x_862_ = v_reuseFailAlloc_863_;
goto v_reusejp_861_;
}
v_reusejp_861_:
{
return v___x_862_;
}
}
}
}
else
{
lean_del_object(v___x_804_);
lean_dec(v_a_758_);
lean_dec(v_a_684_);
v___y_815_ = v___y_677_;
v___y_816_ = v___y_678_;
v___y_817_ = v___y_679_;
v___y_818_ = v___y_680_;
goto v___jp_814_;
}
v___jp_814_:
{
lean_object* v___x_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_835_; 
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
lean_dec_ref(v___y_815_);
v___x_819_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg(v_mvarId_675_, v___x_756_, v___y_816_);
lean_dec(v___y_816_);
v_isSharedCheck_835_ = !lean_is_exclusive(v___x_819_);
if (v_isSharedCheck_835_ == 0)
{
lean_object* v_unused_836_; 
v_unused_836_ = lean_ctor_get(v___x_819_, 0);
lean_dec(v_unused_836_);
v___x_821_ = v___x_819_;
v_isShared_822_ = v_isSharedCheck_835_;
goto v_resetjp_820_;
}
else
{
lean_dec(v___x_819_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_835_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_827_; 
v___x_823_ = lean_array_to_list(v_fst_802_);
v___x_824_ = lean_array_to_list(v_fst_806_);
v___x_825_ = lean_array_to_list(v_fst_810_);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 1, v___x_825_);
lean_ctor_set(v___x_812_, 0, v___x_824_);
v___x_827_ = v___x_812_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_824_);
lean_ctor_set(v_reuseFailAlloc_834_, 1, v___x_825_);
v___x_827_ = v_reuseFailAlloc_834_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
lean_object* v___x_829_; 
if (v_isShared_809_ == 0)
{
lean_ctor_set(v___x_808_, 1, v___x_827_);
lean_ctor_set(v___x_808_, 0, v___x_823_);
v___x_829_ = v___x_808_;
goto v_reusejp_828_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v___x_823_);
lean_ctor_set(v_reuseFailAlloc_833_, 1, v___x_827_);
v___x_829_ = v_reuseFailAlloc_833_;
goto v_reusejp_828_;
}
v_reusejp_828_:
{
lean_object* v___x_831_; 
if (v_isShared_822_ == 0)
{
lean_ctor_set(v___x_821_, 0, v___x_829_);
v___x_831_ = v___x_821_;
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
}
else
{
lean_object* v_a_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_878_; 
lean_dec(v_a_758_);
lean_dec_ref(v___x_756_);
lean_dec(v_a_755_);
lean_dec(v_a_684_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v_mvarId_675_);
v_a_871_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_878_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_878_ == 0)
{
v___x_873_ = v___x_798_;
v_isShared_874_ = v_isSharedCheck_878_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_a_871_);
lean_dec(v___x_798_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_878_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_876_; 
if (v_isShared_874_ == 0)
{
v___x_876_ = v___x_873_;
goto v_reusejp_875_;
}
else
{
lean_object* v_reuseFailAlloc_877_; 
v_reuseFailAlloc_877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_877_, 0, v_a_871_);
v___x_876_ = v_reuseFailAlloc_877_;
goto v_reusejp_875_;
}
v_reusejp_875_:
{
return v___x_876_;
}
}
}
}
}
}
else
{
lean_object* v_a_881_; lean_object* v___x_883_; uint8_t v_isShared_884_; uint8_t v_isSharedCheck_888_; 
lean_dec_ref(v___x_756_);
lean_dec(v_a_755_);
lean_dec(v_a_684_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v_mvarId_675_);
v_a_881_ = lean_ctor_get(v___x_757_, 0);
v_isSharedCheck_888_ = !lean_is_exclusive(v___x_757_);
if (v_isSharedCheck_888_ == 0)
{
v___x_883_ = v___x_757_;
v_isShared_884_ = v_isSharedCheck_888_;
goto v_resetjp_882_;
}
else
{
lean_inc(v_a_881_);
lean_dec(v___x_757_);
v___x_883_ = lean_box(0);
v_isShared_884_ = v_isSharedCheck_888_;
goto v_resetjp_882_;
}
v_resetjp_882_:
{
lean_object* v___x_886_; 
if (v_isShared_884_ == 0)
{
v___x_886_ = v___x_883_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v_a_881_);
v___x_886_ = v_reuseFailAlloc_887_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
return v___x_886_;
}
}
}
}
else
{
lean_object* v_a_889_; lean_object* v___x_891_; uint8_t v_isShared_892_; uint8_t v_isSharedCheck_896_; 
lean_dec(v_fst_730_);
lean_dec_ref(v___x_722_);
lean_dec(v_a_684_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v_mvarId_675_);
v_a_889_ = lean_ctor_get(v___x_754_, 0);
v_isSharedCheck_896_ = !lean_is_exclusive(v___x_754_);
if (v_isSharedCheck_896_ == 0)
{
v___x_891_ = v___x_754_;
v_isShared_892_ = v_isSharedCheck_896_;
goto v_resetjp_890_;
}
else
{
lean_inc(v_a_889_);
lean_dec(v___x_754_);
v___x_891_ = lean_box(0);
v_isShared_892_ = v_isSharedCheck_896_;
goto v_resetjp_890_;
}
v_resetjp_890_:
{
lean_object* v___x_894_; 
if (v_isShared_892_ == 0)
{
v___x_894_ = v___x_891_;
goto v_reusejp_893_;
}
else
{
lean_object* v_reuseFailAlloc_895_; 
v_reuseFailAlloc_895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_895_, 0, v_a_889_);
v___x_894_ = v_reuseFailAlloc_895_;
goto v_reusejp_893_;
}
v_reusejp_893_:
{
return v___x_894_;
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
lean_object* v_a_903_; lean_object* v___x_905_; uint8_t v_isShared_906_; uint8_t v_isSharedCheck_910_; 
lean_dec_ref(v___x_722_);
lean_dec(v_a_721_);
lean_del_object(v___x_718_);
lean_dec(v_a_684_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v_mvarId_675_);
v_a_903_ = lean_ctor_get(v___x_727_, 0);
v_isSharedCheck_910_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_910_ == 0)
{
v___x_905_ = v___x_727_;
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
else
{
lean_inc(v_a_903_);
lean_dec(v___x_727_);
v___x_905_ = lean_box(0);
v_isShared_906_ = v_isSharedCheck_910_;
goto v_resetjp_904_;
}
v_resetjp_904_:
{
lean_object* v___x_908_; 
if (v_isShared_906_ == 0)
{
v___x_908_ = v___x_905_;
goto v_reusejp_907_;
}
else
{
lean_object* v_reuseFailAlloc_909_; 
v_reuseFailAlloc_909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_909_, 0, v_a_903_);
v___x_908_ = v_reuseFailAlloc_909_;
goto v_reusejp_907_;
}
v_reusejp_907_:
{
return v___x_908_;
}
}
}
}
else
{
lean_object* v_a_911_; lean_object* v___x_913_; uint8_t v_isShared_914_; uint8_t v_isSharedCheck_918_; 
lean_dec_ref(v___x_722_);
lean_dec(v_a_721_);
lean_del_object(v___x_718_);
lean_dec(v_a_684_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v_mvarId_675_);
v_a_911_ = lean_ctor_get(v___x_723_, 0);
v_isSharedCheck_918_ = !lean_is_exclusive(v___x_723_);
if (v_isSharedCheck_918_ == 0)
{
v___x_913_ = v___x_723_;
v_isShared_914_ = v_isSharedCheck_918_;
goto v_resetjp_912_;
}
else
{
lean_inc(v_a_911_);
lean_dec(v___x_723_);
v___x_913_ = lean_box(0);
v_isShared_914_ = v_isSharedCheck_918_;
goto v_resetjp_912_;
}
v_resetjp_912_:
{
lean_object* v___x_916_; 
if (v_isShared_914_ == 0)
{
v___x_916_ = v___x_913_;
goto v_reusejp_915_;
}
else
{
lean_object* v_reuseFailAlloc_917_; 
v_reuseFailAlloc_917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_917_, 0, v_a_911_);
v___x_916_ = v_reuseFailAlloc_917_;
goto v_reusejp_915_;
}
v_reusejp_915_:
{
return v___x_916_;
}
}
}
}
else
{
lean_object* v_a_919_; lean_object* v___x_921_; uint8_t v_isShared_922_; uint8_t v_isSharedCheck_926_; 
lean_del_object(v___x_718_);
lean_dec(v_head_716_);
lean_dec(v_us_707_);
lean_dec(v_a_684_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v_mvarId_675_);
v_a_919_ = lean_ctor_get(v___x_720_, 0);
v_isSharedCheck_926_ = !lean_is_exclusive(v___x_720_);
if (v_isSharedCheck_926_ == 0)
{
v___x_921_ = v___x_720_;
v_isShared_922_ = v_isSharedCheck_926_;
goto v_resetjp_920_;
}
else
{
lean_inc(v_a_919_);
lean_dec(v___x_720_);
v___x_921_ = lean_box(0);
v_isShared_922_ = v_isSharedCheck_926_;
goto v_resetjp_920_;
}
v_resetjp_920_:
{
lean_object* v___x_924_; 
if (v_isShared_922_ == 0)
{
v___x_924_ = v___x_921_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, v_a_919_);
v___x_924_ = v_reuseFailAlloc_925_;
goto v_reusejp_923_;
}
v_reusejp_923_:
{
return v___x_924_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_ctors_714_, 2);
lean_dec(v_us_707_);
v___y_696_ = v___y_677_;
v___y_697_ = v___y_678_;
v___y_698_ = v___y_679_;
v___y_699_ = v___y_680_;
goto v___jp_695_;
}
}
else
{
lean_dec(v_ctors_714_);
lean_dec(v_us_707_);
v___y_696_ = v___y_677_;
v___y_697_ = v___y_678_;
v___y_698_ = v___y_679_;
v___y_699_ = v___y_680_;
goto v___jp_695_;
}
}
else
{
lean_dec(v_val_712_);
lean_dec(v_us_707_);
v___y_686_ = v___y_677_;
v___y_687_ = v___y_678_;
v___y_688_ = v___y_679_;
v___y_689_ = v___y_680_;
goto v___jp_685_;
}
}
}
else
{
lean_dec_ref(v___x_705_);
v___y_686_ = v___y_677_;
v___y_687_ = v___y_678_;
v___y_688_ = v___y_679_;
v___y_689_ = v___y_680_;
goto v___jp_685_;
}
v___jp_685_:
{
lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_690_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__1);
v___x_691_ = l_Lean_indentExpr(v_a_684_);
v___x_692_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_692_, 0, v___x_690_);
lean_ctor_set(v___x_692_, 1, v___x_691_);
v___x_693_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_693_, 0, v___x_692_);
v___x_694_ = l_Lean_Meta_throwTacticEx___redArg(v___x_676_, v_mvarId_675_, v___x_693_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
lean_dec(v___y_689_);
lean_dec_ref(v___y_688_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
return v___x_694_;
}
v___jp_695_:
{
lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; 
v___x_700_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___closed__3);
v___x_701_ = l_Lean_indentExpr(v_a_684_);
v___x_702_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_702_, 0, v___x_700_);
lean_ctor_set(v___x_702_, 1, v___x_701_);
v___x_703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_703_, 0, v___x_702_);
v___x_704_ = l_Lean_Meta_throwTacticEx___redArg(v___x_676_, v_mvarId_675_, v___x_703_, v___y_696_, v___y_697_, v___y_698_, v___y_699_);
lean_dec(v___y_699_);
lean_dec_ref(v___y_698_);
lean_dec(v___y_697_);
lean_dec_ref(v___y_696_);
return v___x_704_;
}
}
else
{
lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_936_; 
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v___x_676_);
lean_dec(v_mvarId_675_);
v_a_929_ = lean_ctor_get(v___x_683_, 0);
v_isSharedCheck_936_ = !lean_is_exclusive(v___x_683_);
if (v_isSharedCheck_936_ == 0)
{
v___x_931_ = v___x_683_;
v_isShared_932_ = v_isSharedCheck_936_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_dec(v___x_683_);
v___x_931_ = lean_box(0);
v_isShared_932_ = v_isSharedCheck_936_;
goto v_resetjp_930_;
}
v_resetjp_930_:
{
lean_object* v___x_934_; 
if (v_isShared_932_ == 0)
{
v___x_934_ = v___x_931_;
goto v_reusejp_933_;
}
else
{
lean_object* v_reuseFailAlloc_935_; 
v_reuseFailAlloc_935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_935_, 0, v_a_929_);
v___x_934_ = v_reuseFailAlloc_935_;
goto v_reusejp_933_;
}
v_reusejp_933_:
{
return v___x_934_;
}
}
}
}
else
{
lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_944_; 
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v___x_676_);
lean_dec(v_mvarId_675_);
v_a_937_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_944_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_944_ == 0)
{
v___x_939_ = v___x_682_;
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_682_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v___x_942_; 
if (v_isShared_940_ == 0)
{
v___x_942_ = v___x_939_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_a_937_);
v___x_942_ = v_reuseFailAlloc_943_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
return v___x_942_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___boxed(lean_object* v_mvarId_945_, lean_object* v___x_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_){
_start:
{
lean_object* v_res_952_; 
v_res_952_ = lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0(v_mvarId_945_, v___x_946_, v___y_947_, v___y_948_, v___y_949_, v___y_950_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor(lean_object* v_mvarId_956_, lean_object* v_a_957_, lean_object* v_a_958_, lean_object* v_a_959_, lean_object* v_a_960_){
_start:
{
lean_object* v___x_962_; lean_object* v___f_963_; lean_object* v___x_964_; 
v___x_962_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___closed__1));
lean_inc(v_mvarId_956_);
v___f_963_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_applyTheConstructor___lam__0___boxed), 7, 2);
lean_closure_set(v___f_963_, 0, v_mvarId_956_);
lean_closure_set(v___f_963_, 1, v___x_962_);
v___x_964_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_applyTheConstructor_spec__4___redArg(v_mvarId_956_, v___f_963_, v_a_957_, v_a_958_, v_a_959_, v_a_960_);
return v___x_964_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyTheConstructor___boxed(lean_object* v_mvarId_965_, lean_object* v_a_966_, lean_object* v_a_967_, lean_object* v_a_968_, lean_object* v_a_969_, lean_object* v_a_970_){
_start:
{
lean_object* v_res_971_; 
v_res_971_ = lp_mathlib_Mathlib_Tactic_applyTheConstructor(v_mvarId_965_, v_a_966_, v_a_967_, v_a_968_, v_a_969_);
lean_dec(v_a_969_);
lean_dec_ref(v_a_968_);
lean_dec(v_a_967_);
lean_dec_ref(v_a_966_);
return v_res_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1(lean_object* v_a_972_, lean_object* v_as_973_, size_t v_sz_974_, size_t v_i_975_, lean_object* v_b_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_){
_start:
{
lean_object* v___x_982_; 
v___x_982_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___redArg(v_a_972_, v_as_973_, v_sz_974_, v_i_975_, v_b_976_);
return v___x_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1___boxed(lean_object* v_a_983_, lean_object* v_as_984_, lean_object* v_sz_985_, lean_object* v_i_986_, lean_object* v_b_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_){
_start:
{
size_t v_sz_boxed_993_; size_t v_i_boxed_994_; lean_object* v_res_995_; 
v_sz_boxed_993_ = lean_unbox_usize(v_sz_985_);
lean_dec(v_sz_985_);
v_i_boxed_994_ = lean_unbox_usize(v_i_986_);
lean_dec(v_i_986_);
v_res_995_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_applyTheConstructor_spec__1(v_a_983_, v_as_984_, v_sz_boxed_993_, v_i_boxed_994_, v_b_987_, v___y_988_, v___y_989_, v___y_990_, v___y_991_);
lean_dec(v___y_991_);
lean_dec_ref(v___y_990_);
lean_dec(v___y_989_);
lean_dec_ref(v___y_988_);
lean_dec_ref(v_as_984_);
lean_dec_ref(v_a_983_);
return v_res_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2(lean_object* v_mvarId_996_, lean_object* v_val_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_){
_start:
{
lean_object* v___x_1003_; 
v___x_1003_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___redArg(v_mvarId_996_, v_val_997_, v___y_999_);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2___boxed(lean_object* v_mvarId_1004_, lean_object* v_val_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
lean_object* v_res_1011_; 
v_res_1011_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2(v_mvarId_1004_, v_val_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_);
lean_dec(v___y_1009_);
lean_dec_ref(v___y_1008_);
lean_dec(v___y_1007_);
lean_dec_ref(v___y_1006_);
return v_res_1011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3(lean_object* v_00_u03b1_1012_, lean_object* v_msg_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_){
_start:
{
lean_object* v___x_1019_; 
v___x_1019_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___redArg(v_msg_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_);
return v___x_1019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3___boxed(lean_object* v_00_u03b1_1020_, lean_object* v_msg_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_){
_start:
{
lean_object* v_res_1027_; 
v_res_1027_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3(v_00_u03b1_1020_, v_msg_1021_, v___y_1022_, v___y_1023_, v___y_1024_, v___y_1025_);
lean_dec(v___y_1025_);
lean_dec_ref(v___y_1024_);
lean_dec(v___y_1023_);
lean_dec_ref(v___y_1022_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3(lean_object* v_00_u03b2_1028_, lean_object* v_x_1029_, lean_object* v_x_1030_, lean_object* v_x_1031_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3___redArg(v_x_1029_, v_x_1030_, v_x_1031_);
return v___x_1032_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_1033_, lean_object* v_x_1034_, size_t v_x_1035_, size_t v_x_1036_, lean_object* v_x_1037_, lean_object* v_x_1038_){
_start:
{
lean_object* v___x_1039_; 
v___x_1039_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___redArg(v_x_1034_, v_x_1035_, v_x_1036_, v_x_1037_, v_x_1038_);
return v___x_1039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1040_, lean_object* v_x_1041_, lean_object* v_x_1042_, lean_object* v_x_1043_, lean_object* v_x_1044_, lean_object* v_x_1045_){
_start:
{
size_t v_x_11582__boxed_1046_; size_t v_x_11583__boxed_1047_; lean_object* v_res_1048_; 
v_x_11582__boxed_1046_ = lean_unbox_usize(v_x_1042_);
lean_dec(v_x_1042_);
v_x_11583__boxed_1047_ = lean_unbox_usize(v_x_1043_);
lean_dec(v_x_1043_);
v_res_1048_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5(v_00_u03b2_1040_, v_x_1041_, v_x_11582__boxed_1046_, v_x_11583__boxed_1047_, v_x_1044_, v_x_1045_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8(lean_object* v_00_u03b2_1049_, lean_object* v_n_1050_, lean_object* v_k_1051_, lean_object* v_v_1052_){
_start:
{
lean_object* v___x_1053_; 
v___x_1053_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8___redArg(v_n_1050_, v_k_1051_, v_v_1052_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9(lean_object* v_00_u03b2_1054_, size_t v_depth_1055_, lean_object* v_keys_1056_, lean_object* v_vals_1057_, lean_object* v_heq_1058_, lean_object* v_i_1059_, lean_object* v_entries_1060_){
_start:
{
lean_object* v___x_1061_; 
v___x_1061_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___redArg(v_depth_1055_, v_keys_1056_, v_vals_1057_, v_i_1059_, v_entries_1060_);
return v___x_1061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9___boxed(lean_object* v_00_u03b2_1062_, lean_object* v_depth_1063_, lean_object* v_keys_1064_, lean_object* v_vals_1065_, lean_object* v_heq_1066_, lean_object* v_i_1067_, lean_object* v_entries_1068_){
_start:
{
size_t v_depth_boxed_1069_; lean_object* v_res_1070_; 
v_depth_boxed_1069_ = lean_unbox_usize(v_depth_1063_);
lean_dec(v_depth_1063_);
v_res_1070_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__9(v_00_u03b2_1062_, v_depth_boxed_1069_, v_keys_1064_, v_vals_1065_, v_heq_1066_, v_i_1067_, v_entries_1068_);
lean_dec_ref(v_vals_1065_);
lean_dec_ref(v_keys_1064_);
return v_res_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8_spec__9(lean_object* v_00_u03b2_1071_, lean_object* v_x_1072_, lean_object* v_x_1073_, lean_object* v_x_1074_, lean_object* v_x_1075_){
_start:
{
lean_object* v___x_1076_; 
v___x_1076_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3_spec__5_spec__8_spec__9___redArg(v_x_1072_, v_x_1073_, v_x_1074_, v_x_1075_);
return v___x_1076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg(lean_object* v_x_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_){
_start:
{
lean_object* v___x_1085_; 
v___x_1085_ = l_Lean_Elab_Term_saveState___redArg(v___y_1079_, v___y_1081_, v___y_1083_);
if (lean_obj_tag(v___x_1085_) == 0)
{
lean_object* v_a_1086_; lean_object* v___x_1087_; 
v_a_1086_ = lean_ctor_get(v___x_1085_, 0);
lean_inc(v_a_1086_);
lean_dec_ref_known(v___x_1085_, 1);
lean_inc(v___y_1083_);
lean_inc_ref(v___y_1082_);
lean_inc(v___y_1081_);
lean_inc_ref(v___y_1080_);
lean_inc(v___y_1079_);
lean_inc_ref(v___y_1078_);
v___x_1087_ = lean_apply_7(v_x_1077_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_, lean_box(0));
if (lean_obj_tag(v___x_1087_) == 0)
{
lean_object* v_a_1088_; lean_object* v___x_1090_; uint8_t v_isShared_1091_; uint8_t v_isSharedCheck_1096_; 
lean_dec(v_a_1086_);
v_a_1088_ = lean_ctor_get(v___x_1087_, 0);
v_isSharedCheck_1096_ = !lean_is_exclusive(v___x_1087_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1090_ = v___x_1087_;
v_isShared_1091_ = v_isSharedCheck_1096_;
goto v_resetjp_1089_;
}
else
{
lean_inc(v_a_1088_);
lean_dec(v___x_1087_);
v___x_1090_ = lean_box(0);
v_isShared_1091_ = v_isSharedCheck_1096_;
goto v_resetjp_1089_;
}
v_resetjp_1089_:
{
lean_object* v___x_1092_; lean_object* v___x_1094_; 
v___x_1092_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1092_, 0, v_a_1088_);
if (v_isShared_1091_ == 0)
{
lean_ctor_set(v___x_1090_, 0, v___x_1092_);
v___x_1094_ = v___x_1090_;
goto v_reusejp_1093_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v___x_1092_);
v___x_1094_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1093_;
}
v_reusejp_1093_:
{
return v___x_1094_;
}
}
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1126_; 
v_a_1097_ = lean_ctor_get(v___x_1087_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v___x_1087_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1099_ = v___x_1087_;
v_isShared_1100_ = v_isSharedCheck_1126_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1087_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1126_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
uint8_t v___y_1102_; uint8_t v___x_1124_; 
v___x_1124_ = l_Lean_Exception_isInterrupt(v_a_1097_);
if (v___x_1124_ == 0)
{
uint8_t v___x_1125_; 
lean_inc(v_a_1097_);
v___x_1125_ = l_Lean_Exception_isRuntime(v_a_1097_);
v___y_1102_ = v___x_1125_;
goto v___jp_1101_;
}
else
{
v___y_1102_ = v___x_1124_;
goto v___jp_1101_;
}
v___jp_1101_:
{
if (v___y_1102_ == 0)
{
lean_object* v___x_1103_; 
lean_del_object(v___x_1099_);
lean_dec(v_a_1097_);
v___x_1103_ = l_Lean_Elab_Term_SavedState_restore(v_a_1086_, v___y_1102_, v___y_1078_, v___y_1079_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
if (lean_obj_tag(v___x_1103_) == 0)
{
lean_object* v___x_1105_; uint8_t v_isShared_1106_; uint8_t v_isSharedCheck_1111_; 
v_isSharedCheck_1111_ = !lean_is_exclusive(v___x_1103_);
if (v_isSharedCheck_1111_ == 0)
{
lean_object* v_unused_1112_; 
v_unused_1112_ = lean_ctor_get(v___x_1103_, 0);
lean_dec(v_unused_1112_);
v___x_1105_ = v___x_1103_;
v_isShared_1106_ = v_isSharedCheck_1111_;
goto v_resetjp_1104_;
}
else
{
lean_dec(v___x_1103_);
v___x_1105_ = lean_box(0);
v_isShared_1106_ = v_isSharedCheck_1111_;
goto v_resetjp_1104_;
}
v_resetjp_1104_:
{
lean_object* v___x_1107_; lean_object* v___x_1109_; 
v___x_1107_ = lean_box(0);
if (v_isShared_1106_ == 0)
{
lean_ctor_set(v___x_1105_, 0, v___x_1107_);
v___x_1109_ = v___x_1105_;
goto v_reusejp_1108_;
}
else
{
lean_object* v_reuseFailAlloc_1110_; 
v_reuseFailAlloc_1110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1110_, 0, v___x_1107_);
v___x_1109_ = v_reuseFailAlloc_1110_;
goto v_reusejp_1108_;
}
v_reusejp_1108_:
{
return v___x_1109_;
}
}
}
else
{
lean_object* v_a_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1120_; 
v_a_1113_ = lean_ctor_get(v___x_1103_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_1103_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1115_ = v___x_1103_;
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_a_1113_);
lean_dec(v___x_1103_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1118_; 
if (v_isShared_1116_ == 0)
{
v___x_1118_ = v___x_1115_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v_a_1113_);
v___x_1118_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
return v___x_1118_;
}
}
}
}
else
{
lean_object* v___x_1122_; 
lean_dec(v_a_1086_);
if (v_isShared_1100_ == 0)
{
v___x_1122_ = v___x_1099_;
goto v_reusejp_1121_;
}
else
{
lean_object* v_reuseFailAlloc_1123_; 
v_reuseFailAlloc_1123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1123_, 0, v_a_1097_);
v___x_1122_ = v_reuseFailAlloc_1123_;
goto v_reusejp_1121_;
}
v_reusejp_1121_:
{
return v___x_1122_;
}
}
}
}
}
}
else
{
lean_object* v_a_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1134_; 
lean_dec_ref(v_x_1077_);
v_a_1127_ = lean_ctor_get(v___x_1085_, 0);
v_isSharedCheck_1134_ = !lean_is_exclusive(v___x_1085_);
if (v_isSharedCheck_1134_ == 0)
{
v___x_1129_ = v___x_1085_;
v_isShared_1130_ = v_isSharedCheck_1134_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_a_1127_);
lean_dec(v___x_1085_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1134_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1132_; 
if (v_isShared_1130_ == 0)
{
v___x_1132_ = v___x_1129_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1133_; 
v_reuseFailAlloc_1133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1133_, 0, v_a_1127_);
v___x_1132_ = v_reuseFailAlloc_1133_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
return v___x_1132_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg___boxed(lean_object* v_x_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_){
_start:
{
lean_object* v_res_1143_; 
v_res_1143_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg(v_x_1135_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
lean_dec(v___y_1139_);
lean_dec_ref(v___y_1138_);
lean_dec(v___y_1137_);
lean_dec_ref(v___y_1136_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3(lean_object* v_00_u03b1_1144_, lean_object* v_x_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_){
_start:
{
lean_object* v___x_1153_; 
v___x_1153_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg(v_x_1145_, v___y_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_);
return v___x_1153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___boxed(lean_object* v_00_u03b1_1154_, lean_object* v_x_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_){
_start:
{
lean_object* v_res_1163_; 
v_res_1163_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3(v_00_u03b1_1154_, v_x_1155_, v___y_1156_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_);
lean_dec(v___y_1161_);
lean_dec_ref(v___y_1160_);
lean_dec(v___y_1159_);
lean_dec_ref(v___y_1158_);
lean_dec(v___y_1157_);
lean_dec_ref(v___y_1156_);
return v_res_1163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___lam__0(lean_object* v_x_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_){
_start:
{
lean_object* v___x_1172_; 
lean_inc(v___y_1166_);
lean_inc_ref(v___y_1165_);
v___x_1172_ = lean_apply_7(v_x_1164_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_, lean_box(0));
return v___x_1172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___lam__0___boxed(lean_object* v_x_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_){
_start:
{
lean_object* v_res_1181_; 
v_res_1181_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___lam__0(v_x_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_);
lean_dec(v___y_1175_);
lean_dec_ref(v___y_1174_);
return v_res_1181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg(lean_object* v_mvarId_1182_, lean_object* v_x_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_){
_start:
{
lean_object* v___f_1191_; lean_object* v___x_1192_; 
lean_inc(v___y_1185_);
lean_inc_ref(v___y_1184_);
v___f_1191_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_1191_, 0, v_x_1183_);
lean_closure_set(v___f_1191_, 1, v___y_1184_);
lean_closure_set(v___f_1191_, 2, v___y_1185_);
v___x_1192_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1182_, v___f_1191_, v___y_1186_, v___y_1187_, v___y_1188_, v___y_1189_);
if (lean_obj_tag(v___x_1192_) == 0)
{
return v___x_1192_;
}
else
{
lean_object* v_a_1193_; lean_object* v___x_1195_; uint8_t v_isShared_1196_; uint8_t v_isSharedCheck_1200_; 
v_a_1193_ = lean_ctor_get(v___x_1192_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1195_ = v___x_1192_;
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
else
{
lean_inc(v_a_1193_);
lean_dec(v___x_1192_);
v___x_1195_ = lean_box(0);
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
v_resetjp_1194_:
{
lean_object* v___x_1198_; 
if (v_isShared_1196_ == 0)
{
v___x_1198_ = v___x_1195_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_a_1193_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg___boxed(lean_object* v_mvarId_1201_, lean_object* v_x_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_){
_start:
{
lean_object* v_res_1210_; 
v_res_1210_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg(v_mvarId_1201_, v_x_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_, v___y_1208_);
lean_dec(v___y_1208_);
lean_dec_ref(v___y_1207_);
lean_dec(v___y_1206_);
lean_dec_ref(v___y_1205_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
return v_res_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4(lean_object* v_00_u03b1_1211_, lean_object* v_mvarId_1212_, lean_object* v_x_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_){
_start:
{
lean_object* v___x_1221_; 
v___x_1221_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg(v_mvarId_1212_, v_x_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_, v___y_1218_, v___y_1219_);
return v___x_1221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___boxed(lean_object* v_00_u03b1_1222_, lean_object* v_mvarId_1223_, lean_object* v_x_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_){
_start:
{
lean_object* v_res_1232_; 
v_res_1232_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4(v_00_u03b1_1222_, v_mvarId_1223_, v_x_1224_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_, v___y_1230_);
lean_dec(v___y_1230_);
lean_dec_ref(v___y_1229_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
return v_res_1232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__0(lean_object* v_cls_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_){
_start:
{
lean_object* v_options_1244_; uint8_t v_hasTrace_1245_; 
v_options_1244_ = lean_ctor_get(v___y_1241_, 2);
v_hasTrace_1245_ = lean_ctor_get_uint8(v_options_1244_, sizeof(void*)*1);
if (v_hasTrace_1245_ == 0)
{
lean_object* v___x_1246_; lean_object* v___x_1247_; 
lean_dec(v_cls_1236_);
v___x_1246_ = lean_box(v_hasTrace_1245_);
v___x_1247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1247_, 0, v___x_1246_);
return v___x_1247_;
}
else
{
lean_object* v_inheritedTraceOptions_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; uint8_t v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; 
v_inheritedTraceOptions_1248_ = lean_ctor_get(v___y_1241_, 13);
v___x_1249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__1));
v___x_1250_ = l_Lean_Name_append(v___x_1249_, v_cls_1236_);
v___x_1251_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1248_, v_options_1244_, v___x_1250_);
lean_dec(v___x_1250_);
v___x_1252_ = lean_box(v___x_1251_);
v___x_1253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1253_, 0, v___x_1252_);
return v___x_1253_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__0___boxed(lean_object* v_cls_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_){
_start:
{
lean_object* v_res_1262_; 
v_res_1262_ = lp_mathlib_Mathlib_Tactic_useLoop___lam__0(v_cls_1254_, v___y_1255_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, v___y_1260_);
lean_dec(v___y_1260_);
lean_dec_ref(v___y_1259_);
lean_dec(v___y_1258_);
lean_dec_ref(v___y_1257_);
lean_dec(v___y_1256_);
lean_dec_ref(v___y_1255_);
return v_res_1262_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1263_; double v___x_1264_; 
v___x_1263_ = lean_unsigned_to_nat(0u);
v___x_1264_ = lean_float_of_nat(v___x_1263_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(lean_object* v_cls_1268_, lean_object* v_msg_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v_ref_1275_; lean_object* v___x_1276_; lean_object* v_a_1277_; lean_object* v___x_1279_; uint8_t v_isShared_1280_; uint8_t v_isSharedCheck_1321_; 
v_ref_1275_ = lean_ctor_get(v___y_1272_, 5);
v___x_1276_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(v_msg_1269_, v___y_1270_, v___y_1271_, v___y_1272_, v___y_1273_);
v_a_1277_ = lean_ctor_get(v___x_1276_, 0);
v_isSharedCheck_1321_ = !lean_is_exclusive(v___x_1276_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1279_ = v___x_1276_;
v_isShared_1280_ = v_isSharedCheck_1321_;
goto v_resetjp_1278_;
}
else
{
lean_inc(v_a_1277_);
lean_dec(v___x_1276_);
v___x_1279_ = lean_box(0);
v_isShared_1280_ = v_isSharedCheck_1321_;
goto v_resetjp_1278_;
}
v_resetjp_1278_:
{
lean_object* v___x_1281_; lean_object* v_traceState_1282_; lean_object* v_env_1283_; lean_object* v_nextMacroScope_1284_; lean_object* v_ngen_1285_; lean_object* v_auxDeclNGen_1286_; lean_object* v_cache_1287_; lean_object* v_messages_1288_; lean_object* v_infoState_1289_; lean_object* v_snapshotTasks_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1320_; 
v___x_1281_ = lean_st_ref_take(v___y_1273_);
v_traceState_1282_ = lean_ctor_get(v___x_1281_, 4);
v_env_1283_ = lean_ctor_get(v___x_1281_, 0);
v_nextMacroScope_1284_ = lean_ctor_get(v___x_1281_, 1);
v_ngen_1285_ = lean_ctor_get(v___x_1281_, 2);
v_auxDeclNGen_1286_ = lean_ctor_get(v___x_1281_, 3);
v_cache_1287_ = lean_ctor_get(v___x_1281_, 5);
v_messages_1288_ = lean_ctor_get(v___x_1281_, 6);
v_infoState_1289_ = lean_ctor_get(v___x_1281_, 7);
v_snapshotTasks_1290_ = lean_ctor_get(v___x_1281_, 8);
v_isSharedCheck_1320_ = !lean_is_exclusive(v___x_1281_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1292_ = v___x_1281_;
v_isShared_1293_ = v_isSharedCheck_1320_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_snapshotTasks_1290_);
lean_inc(v_infoState_1289_);
lean_inc(v_messages_1288_);
lean_inc(v_cache_1287_);
lean_inc(v_traceState_1282_);
lean_inc(v_auxDeclNGen_1286_);
lean_inc(v_ngen_1285_);
lean_inc(v_nextMacroScope_1284_);
lean_inc(v_env_1283_);
lean_dec(v___x_1281_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1320_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
uint64_t v_tid_1294_; lean_object* v_traces_1295_; lean_object* v___x_1297_; uint8_t v_isShared_1298_; uint8_t v_isSharedCheck_1319_; 
v_tid_1294_ = lean_ctor_get_uint64(v_traceState_1282_, sizeof(void*)*1);
v_traces_1295_ = lean_ctor_get(v_traceState_1282_, 0);
v_isSharedCheck_1319_ = !lean_is_exclusive(v_traceState_1282_);
if (v_isSharedCheck_1319_ == 0)
{
v___x_1297_ = v_traceState_1282_;
v_isShared_1298_ = v_isSharedCheck_1319_;
goto v_resetjp_1296_;
}
else
{
lean_inc(v_traces_1295_);
lean_dec(v_traceState_1282_);
v___x_1297_ = lean_box(0);
v_isShared_1298_ = v_isSharedCheck_1319_;
goto v_resetjp_1296_;
}
v_resetjp_1296_:
{
lean_object* v___x_1299_; double v___x_1300_; uint8_t v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1309_; 
v___x_1299_ = lean_box(0);
v___x_1300_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0);
v___x_1301_ = 0;
v___x_1302_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__1));
v___x_1303_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1303_, 0, v_cls_1268_);
lean_ctor_set(v___x_1303_, 1, v___x_1299_);
lean_ctor_set(v___x_1303_, 2, v___x_1302_);
lean_ctor_set_float(v___x_1303_, sizeof(void*)*3, v___x_1300_);
lean_ctor_set_float(v___x_1303_, sizeof(void*)*3 + 8, v___x_1300_);
lean_ctor_set_uint8(v___x_1303_, sizeof(void*)*3 + 16, v___x_1301_);
v___x_1304_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__2));
v___x_1305_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1305_, 0, v___x_1303_);
lean_ctor_set(v___x_1305_, 1, v_a_1277_);
lean_ctor_set(v___x_1305_, 2, v___x_1304_);
lean_inc(v_ref_1275_);
v___x_1306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1306_, 0, v_ref_1275_);
lean_ctor_set(v___x_1306_, 1, v___x_1305_);
v___x_1307_ = l_Lean_PersistentArray_push___redArg(v_traces_1295_, v___x_1306_);
if (v_isShared_1298_ == 0)
{
lean_ctor_set(v___x_1297_, 0, v___x_1307_);
v___x_1309_ = v___x_1297_;
goto v_reusejp_1308_;
}
else
{
lean_object* v_reuseFailAlloc_1318_; 
v_reuseFailAlloc_1318_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1318_, 0, v___x_1307_);
lean_ctor_set_uint64(v_reuseFailAlloc_1318_, sizeof(void*)*1, v_tid_1294_);
v___x_1309_ = v_reuseFailAlloc_1318_;
goto v_reusejp_1308_;
}
v_reusejp_1308_:
{
lean_object* v___x_1311_; 
if (v_isShared_1293_ == 0)
{
lean_ctor_set(v___x_1292_, 4, v___x_1309_);
v___x_1311_ = v___x_1292_;
goto v_reusejp_1310_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_env_1283_);
lean_ctor_set(v_reuseFailAlloc_1317_, 1, v_nextMacroScope_1284_);
lean_ctor_set(v_reuseFailAlloc_1317_, 2, v_ngen_1285_);
lean_ctor_set(v_reuseFailAlloc_1317_, 3, v_auxDeclNGen_1286_);
lean_ctor_set(v_reuseFailAlloc_1317_, 4, v___x_1309_);
lean_ctor_set(v_reuseFailAlloc_1317_, 5, v_cache_1287_);
lean_ctor_set(v_reuseFailAlloc_1317_, 6, v_messages_1288_);
lean_ctor_set(v_reuseFailAlloc_1317_, 7, v_infoState_1289_);
lean_ctor_set(v_reuseFailAlloc_1317_, 8, v_snapshotTasks_1290_);
v___x_1311_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1310_;
}
v_reusejp_1310_:
{
lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1315_; 
v___x_1312_ = lean_st_ref_set(v___y_1273_, v___x_1311_);
v___x_1313_ = lean_box(0);
if (v_isShared_1280_ == 0)
{
lean_ctor_set(v___x_1279_, 0, v___x_1313_);
v___x_1315_ = v___x_1279_;
goto v_reusejp_1314_;
}
else
{
lean_object* v_reuseFailAlloc_1316_; 
v_reuseFailAlloc_1316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1316_, 0, v___x_1313_);
v___x_1315_ = v_reuseFailAlloc_1316_;
goto v_reusejp_1314_;
}
v_reusejp_1314_:
{
return v___x_1315_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___boxed(lean_object* v_cls_1322_, lean_object* v_msg_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_, lean_object* v___y_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_){
_start:
{
lean_object* v_res_1329_; 
v_res_1329_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(v_cls_1322_, v_msg_1323_, v___y_1324_, v___y_1325_, v___y_1326_, v___y_1327_);
lean_dec(v___y_1327_);
lean_dec_ref(v___y_1326_);
lean_dec(v___y_1325_);
lean_dec_ref(v___y_1324_);
return v_res_1329_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1331_; lean_object* v___x_1332_; 
v___x_1331_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__0));
v___x_1332_ = l_Lean_stringToMessageData(v___x_1331_);
return v___x_1332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__1(lean_object* v_head_1333_, lean_object* v_cls_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_){
_start:
{
lean_object* v___x_1342_; 
v___x_1342_ = lp_mathlib_Mathlib_Tactic_applyTheConstructor(v_head_1333_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_);
if (lean_obj_tag(v___x_1342_) == 0)
{
lean_dec(v_cls_1334_);
return v___x_1342_;
}
else
{
lean_object* v_a_1343_; uint8_t v___y_1345_; uint8_t v___x_1372_; 
v_a_1343_ = lean_ctor_get(v___x_1342_, 0);
lean_inc(v_a_1343_);
v___x_1372_ = l_Lean_Exception_isInterrupt(v_a_1343_);
if (v___x_1372_ == 0)
{
uint8_t v___x_1373_; 
lean_inc(v_a_1343_);
v___x_1373_ = l_Lean_Exception_isRuntime(v_a_1343_);
v___y_1345_ = v___x_1373_;
goto v___jp_1344_;
}
else
{
v___y_1345_ = v___x_1372_;
goto v___jp_1344_;
}
v___jp_1344_:
{
if (v___y_1345_ == 0)
{
lean_object* v_options_1346_; uint8_t v_hasTrace_1347_; 
v_options_1346_ = lean_ctor_get(v___y_1339_, 2);
v_hasTrace_1347_ = lean_ctor_get_uint8(v_options_1346_, sizeof(void*)*1);
if (v_hasTrace_1347_ == 0)
{
lean_dec(v_a_1343_);
lean_dec(v_cls_1334_);
return v___x_1342_;
}
else
{
lean_object* v_inheritedTraceOptions_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; uint8_t v___x_1351_; 
v_inheritedTraceOptions_1348_ = lean_ctor_get(v___y_1339_, 13);
v___x_1349_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__1));
lean_inc(v_cls_1334_);
v___x_1350_ = l_Lean_Name_append(v___x_1349_, v_cls_1334_);
v___x_1351_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1348_, v_options_1346_, v___x_1350_);
lean_dec(v___x_1350_);
if (v___x_1351_ == 0)
{
lean_dec(v_a_1343_);
lean_dec(v_cls_1334_);
return v___x_1342_;
}
else
{
lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; 
lean_dec_ref_known(v___x_1342_, 1);
v___x_1352_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__1___closed__1);
lean_inc(v_a_1343_);
v___x_1353_ = l_Lean_Exception_toMessageData(v_a_1343_);
v___x_1354_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1354_, 0, v___x_1352_);
lean_ctor_set(v___x_1354_, 1, v___x_1353_);
v___x_1355_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(v_cls_1334_, v___x_1354_, v___y_1337_, v___y_1338_, v___y_1339_, v___y_1340_);
if (lean_obj_tag(v___x_1355_) == 0)
{
lean_object* v___x_1357_; uint8_t v_isShared_1358_; uint8_t v_isSharedCheck_1362_; 
v_isSharedCheck_1362_ = !lean_is_exclusive(v___x_1355_);
if (v_isSharedCheck_1362_ == 0)
{
lean_object* v_unused_1363_; 
v_unused_1363_ = lean_ctor_get(v___x_1355_, 0);
lean_dec(v_unused_1363_);
v___x_1357_ = v___x_1355_;
v_isShared_1358_ = v_isSharedCheck_1362_;
goto v_resetjp_1356_;
}
else
{
lean_dec(v___x_1355_);
v___x_1357_ = lean_box(0);
v_isShared_1358_ = v_isSharedCheck_1362_;
goto v_resetjp_1356_;
}
v_resetjp_1356_:
{
lean_object* v___x_1360_; 
if (v_isShared_1358_ == 0)
{
lean_ctor_set_tag(v___x_1357_, 1);
lean_ctor_set(v___x_1357_, 0, v_a_1343_);
v___x_1360_ = v___x_1357_;
goto v_reusejp_1359_;
}
else
{
lean_object* v_reuseFailAlloc_1361_; 
v_reuseFailAlloc_1361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1361_, 0, v_a_1343_);
v___x_1360_ = v_reuseFailAlloc_1361_;
goto v_reusejp_1359_;
}
v_reusejp_1359_:
{
return v___x_1360_;
}
}
}
else
{
lean_object* v_a_1364_; lean_object* v___x_1366_; uint8_t v_isShared_1367_; uint8_t v_isSharedCheck_1371_; 
lean_dec(v_a_1343_);
v_a_1364_ = lean_ctor_get(v___x_1355_, 0);
v_isSharedCheck_1371_ = !lean_is_exclusive(v___x_1355_);
if (v_isSharedCheck_1371_ == 0)
{
v___x_1366_ = v___x_1355_;
v_isShared_1367_ = v_isSharedCheck_1371_;
goto v_resetjp_1365_;
}
else
{
lean_inc(v_a_1364_);
lean_dec(v___x_1355_);
v___x_1366_ = lean_box(0);
v_isShared_1367_ = v_isSharedCheck_1371_;
goto v_resetjp_1365_;
}
v_resetjp_1365_:
{
lean_object* v___x_1369_; 
if (v_isShared_1367_ == 0)
{
v___x_1369_ = v___x_1366_;
goto v_reusejp_1368_;
}
else
{
lean_object* v_reuseFailAlloc_1370_; 
v_reuseFailAlloc_1370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1370_, 0, v_a_1364_);
v___x_1369_ = v_reuseFailAlloc_1370_;
goto v_reusejp_1368_;
}
v_reusejp_1368_:
{
return v___x_1369_;
}
}
}
}
}
}
else
{
lean_dec(v_a_1343_);
lean_dec(v_cls_1334_);
return v___x_1342_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__1___boxed(lean_object* v_head_1374_, lean_object* v_cls_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_){
_start:
{
lean_object* v_res_1383_; 
v_res_1383_ = lp_mathlib_Mathlib_Tactic_useLoop___lam__1(v_head_1374_, v_cls_1375_, v___y_1376_, v___y_1377_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_);
lean_dec(v___y_1381_);
lean_dec_ref(v___y_1380_);
lean_dec(v___y_1379_);
lean_dec_ref(v___y_1378_);
lean_dec(v___y_1377_);
lean_dec_ref(v___y_1376_);
return v_res_1383_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg(lean_object* v_keys_1384_, lean_object* v_i_1385_, lean_object* v_k_1386_){
_start:
{
lean_object* v___x_1387_; uint8_t v___x_1388_; 
v___x_1387_ = lean_array_get_size(v_keys_1384_);
v___x_1388_ = lean_nat_dec_lt(v_i_1385_, v___x_1387_);
if (v___x_1388_ == 0)
{
lean_dec(v_i_1385_);
return v___x_1388_;
}
else
{
lean_object* v_k_x27_1389_; uint8_t v___x_1390_; 
v_k_x27_1389_ = lean_array_fget_borrowed(v_keys_1384_, v_i_1385_);
v___x_1390_ = l_Lean_instBEqMVarId_beq(v_k_1386_, v_k_x27_1389_);
if (v___x_1390_ == 0)
{
lean_object* v___x_1391_; lean_object* v___x_1392_; 
v___x_1391_ = lean_unsigned_to_nat(1u);
v___x_1392_ = lean_nat_add(v_i_1385_, v___x_1391_);
lean_dec(v_i_1385_);
v_i_1385_ = v___x_1392_;
goto _start;
}
else
{
lean_dec(v_i_1385_);
return v___x_1390_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg___boxed(lean_object* v_keys_1394_, lean_object* v_i_1395_, lean_object* v_k_1396_){
_start:
{
uint8_t v_res_1397_; lean_object* v_r_1398_; 
v_res_1397_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg(v_keys_1394_, v_i_1395_, v_k_1396_);
lean_dec(v_k_1396_);
lean_dec_ref(v_keys_1394_);
v_r_1398_ = lean_box(v_res_1397_);
return v_r_1398_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg(lean_object* v_x_1399_, size_t v_x_1400_, lean_object* v_x_1401_){
_start:
{
if (lean_obj_tag(v_x_1399_) == 0)
{
lean_object* v_es_1402_; lean_object* v___x_1403_; size_t v___x_1404_; size_t v___x_1405_; lean_object* v_j_1406_; lean_object* v___x_1407_; 
v_es_1402_ = lean_ctor_get(v_x_1399_, 0);
v___x_1403_ = lean_box(2);
v___x_1404_ = ((size_t)31ULL);
v___x_1405_ = lean_usize_land(v_x_1400_, v___x_1404_);
v_j_1406_ = lean_usize_to_nat(v___x_1405_);
v___x_1407_ = lean_array_get_borrowed(v___x_1403_, v_es_1402_, v_j_1406_);
lean_dec(v_j_1406_);
switch(lean_obj_tag(v___x_1407_))
{
case 0:
{
lean_object* v_key_1408_; uint8_t v___x_1409_; 
v_key_1408_ = lean_ctor_get(v___x_1407_, 0);
v___x_1409_ = l_Lean_instBEqMVarId_beq(v_x_1401_, v_key_1408_);
return v___x_1409_;
}
case 1:
{
lean_object* v_node_1410_; size_t v___x_1411_; size_t v___x_1412_; 
v_node_1410_ = lean_ctor_get(v___x_1407_, 0);
v___x_1411_ = ((size_t)5ULL);
v___x_1412_ = lean_usize_shift_right(v_x_1400_, v___x_1411_);
v_x_1399_ = v_node_1410_;
v_x_1400_ = v___x_1412_;
goto _start;
}
default: 
{
uint8_t v___x_1414_; 
v___x_1414_ = 0;
return v___x_1414_;
}
}
}
else
{
lean_object* v_ks_1415_; lean_object* v___x_1416_; uint8_t v___x_1417_; 
v_ks_1415_ = lean_ctor_get(v_x_1399_, 0);
v___x_1416_ = lean_unsigned_to_nat(0u);
v___x_1417_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg(v_ks_1415_, v___x_1416_, v_x_1401_);
return v___x_1417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg___boxed(lean_object* v_x_1418_, lean_object* v_x_1419_, lean_object* v_x_1420_){
_start:
{
size_t v_x_17503__boxed_1421_; uint8_t v_res_1422_; lean_object* v_r_1423_; 
v_x_17503__boxed_1421_ = lean_unbox_usize(v_x_1419_);
lean_dec(v_x_1419_);
v_res_1422_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg(v_x_1418_, v_x_17503__boxed_1421_, v_x_1420_);
lean_dec(v_x_1420_);
lean_dec_ref(v_x_1418_);
v_r_1423_ = lean_box(v_res_1422_);
return v_r_1423_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg(lean_object* v_x_1424_, lean_object* v_x_1425_){
_start:
{
uint64_t v___x_1426_; size_t v___x_1427_; uint8_t v___x_1428_; 
v___x_1426_ = l_Lean_instHashableMVarId_hash(v_x_1425_);
v___x_1427_ = lean_uint64_to_usize(v___x_1426_);
v___x_1428_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg(v_x_1424_, v___x_1427_, v_x_1425_);
return v___x_1428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg___boxed(lean_object* v_x_1429_, lean_object* v_x_1430_){
_start:
{
uint8_t v_res_1431_; lean_object* v_r_1432_; 
v_res_1431_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg(v_x_1429_, v_x_1430_);
lean_dec(v_x_1430_);
lean_dec_ref(v_x_1429_);
v_r_1432_ = lean_box(v_res_1431_);
return v_r_1432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg(lean_object* v_mvarId_1433_, lean_object* v___y_1434_){
_start:
{
lean_object* v___x_1436_; lean_object* v_mctx_1437_; lean_object* v_eAssignment_1438_; uint8_t v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; 
v___x_1436_ = lean_st_ref_get(v___y_1434_);
v_mctx_1437_ = lean_ctor_get(v___x_1436_, 0);
lean_inc_ref(v_mctx_1437_);
lean_dec(v___x_1436_);
v_eAssignment_1438_ = lean_ctor_get(v_mctx_1437_, 8);
lean_inc_ref(v_eAssignment_1438_);
lean_dec_ref(v_mctx_1437_);
v___x_1439_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg(v_eAssignment_1438_, v_mvarId_1433_);
lean_dec_ref(v_eAssignment_1438_);
v___x_1440_ = lean_box(v___x_1439_);
v___x_1441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1441_, 0, v___x_1440_);
return v___x_1441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg___boxed(lean_object* v_mvarId_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_){
_start:
{
lean_object* v_res_1445_; 
v_res_1445_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg(v_mvarId_1442_, v___y_1443_);
lean_dec(v___y_1443_);
lean_dec(v_mvarId_1442_);
return v_res_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__6(lean_object* v_a_1446_, lean_object* v_a_1447_){
_start:
{
if (lean_obj_tag(v_a_1446_) == 0)
{
lean_object* v___x_1448_; 
v___x_1448_ = l_List_reverse___redArg(v_a_1447_);
return v___x_1448_;
}
else
{
lean_object* v_head_1449_; lean_object* v_tail_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1459_; 
v_head_1449_ = lean_ctor_get(v_a_1446_, 0);
v_tail_1450_ = lean_ctor_get(v_a_1446_, 1);
v_isSharedCheck_1459_ = !lean_is_exclusive(v_a_1446_);
if (v_isSharedCheck_1459_ == 0)
{
v___x_1452_ = v_a_1446_;
v_isShared_1453_ = v_isSharedCheck_1459_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_tail_1450_);
lean_inc(v_head_1449_);
lean_dec(v_a_1446_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1459_;
goto v_resetjp_1451_;
}
v_resetjp_1451_:
{
lean_object* v___x_1454_; lean_object* v___x_1456_; 
v___x_1454_ = l_Lean_MessageData_ofSyntax(v_head_1449_);
if (v_isShared_1453_ == 0)
{
lean_ctor_set(v___x_1452_, 1, v_a_1447_);
lean_ctor_set(v___x_1452_, 0, v___x_1454_);
v___x_1456_ = v___x_1452_;
goto v_reusejp_1455_;
}
else
{
lean_object* v_reuseFailAlloc_1458_; 
v_reuseFailAlloc_1458_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1458_, 0, v___x_1454_);
lean_ctor_set(v_reuseFailAlloc_1458_, 1, v_a_1447_);
v___x_1456_ = v_reuseFailAlloc_1458_;
goto v_reusejp_1455_;
}
v_reusejp_1455_:
{
v_a_1446_ = v_tail_1450_;
v_a_1447_ = v___x_1456_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__5(lean_object* v_a_1460_, lean_object* v_a_1461_){
_start:
{
if (lean_obj_tag(v_a_1460_) == 0)
{
lean_object* v___x_1462_; 
v___x_1462_ = l_List_reverse___redArg(v_a_1461_);
return v___x_1462_;
}
else
{
lean_object* v_head_1463_; lean_object* v_tail_1464_; lean_object* v___x_1466_; uint8_t v_isShared_1467_; uint8_t v_isSharedCheck_1473_; 
v_head_1463_ = lean_ctor_get(v_a_1460_, 0);
v_tail_1464_ = lean_ctor_get(v_a_1460_, 1);
v_isSharedCheck_1473_ = !lean_is_exclusive(v_a_1460_);
if (v_isSharedCheck_1473_ == 0)
{
v___x_1466_ = v_a_1460_;
v_isShared_1467_ = v_isSharedCheck_1473_;
goto v_resetjp_1465_;
}
else
{
lean_inc(v_tail_1464_);
lean_inc(v_head_1463_);
lean_dec(v_a_1460_);
v___x_1466_ = lean_box(0);
v_isShared_1467_ = v_isSharedCheck_1473_;
goto v_resetjp_1465_;
}
v_resetjp_1465_:
{
lean_object* v___x_1468_; lean_object* v___x_1470_; 
v___x_1468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1468_, 0, v_head_1463_);
if (v_isShared_1467_ == 0)
{
lean_ctor_set(v___x_1466_, 1, v_a_1461_);
lean_ctor_set(v___x_1466_, 0, v___x_1468_);
v___x_1470_ = v___x_1466_;
goto v_reusejp_1469_;
}
else
{
lean_object* v_reuseFailAlloc_1472_; 
v_reuseFailAlloc_1472_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1472_, 0, v___x_1468_);
lean_ctor_set(v_reuseFailAlloc_1472_, 1, v_a_1461_);
v___x_1470_ = v_reuseFailAlloc_1472_;
goto v_reusejp_1469_;
}
v_reusejp_1469_:
{
v_a_1460_ = v_tail_1464_;
v_a_1461_ = v___x_1470_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0(void){
_start:
{
lean_object* v___x_1474_; lean_object* v___x_1475_; 
v___x_1474_ = lean_box(1);
v___x_1475_ = l_Lean_MessageData_ofFormat(v___x_1474_);
return v___x_1475_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__3(void){
_start:
{
lean_object* v___x_1479_; lean_object* v___x_1480_; 
v___x_1479_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__2));
v___x_1480_ = l_Lean_MessageData_ofFormat(v___x_1479_);
return v___x_1480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9(lean_object* v_x_1481_, lean_object* v_x_1482_){
_start:
{
if (lean_obj_tag(v_x_1482_) == 0)
{
return v_x_1481_;
}
else
{
lean_object* v_head_1483_; lean_object* v_tail_1484_; lean_object* v___x_1486_; uint8_t v_isShared_1487_; uint8_t v_isSharedCheck_1506_; 
v_head_1483_ = lean_ctor_get(v_x_1482_, 0);
v_tail_1484_ = lean_ctor_get(v_x_1482_, 1);
v_isSharedCheck_1506_ = !lean_is_exclusive(v_x_1482_);
if (v_isSharedCheck_1506_ == 0)
{
v___x_1486_ = v_x_1482_;
v_isShared_1487_ = v_isSharedCheck_1506_;
goto v_resetjp_1485_;
}
else
{
lean_inc(v_tail_1484_);
lean_inc(v_head_1483_);
lean_dec(v_x_1482_);
v___x_1486_ = lean_box(0);
v_isShared_1487_ = v_isSharedCheck_1506_;
goto v_resetjp_1485_;
}
v_resetjp_1485_:
{
lean_object* v_before_1488_; lean_object* v___x_1490_; uint8_t v_isShared_1491_; uint8_t v_isSharedCheck_1504_; 
v_before_1488_ = lean_ctor_get(v_head_1483_, 0);
v_isSharedCheck_1504_ = !lean_is_exclusive(v_head_1483_);
if (v_isSharedCheck_1504_ == 0)
{
lean_object* v_unused_1505_; 
v_unused_1505_ = lean_ctor_get(v_head_1483_, 1);
lean_dec(v_unused_1505_);
v___x_1490_ = v_head_1483_;
v_isShared_1491_ = v_isSharedCheck_1504_;
goto v_resetjp_1489_;
}
else
{
lean_inc(v_before_1488_);
lean_dec(v_head_1483_);
v___x_1490_ = lean_box(0);
v_isShared_1491_ = v_isSharedCheck_1504_;
goto v_resetjp_1489_;
}
v_resetjp_1489_:
{
lean_object* v___x_1492_; lean_object* v___x_1494_; 
v___x_1492_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0);
if (v_isShared_1491_ == 0)
{
lean_ctor_set_tag(v___x_1490_, 7);
lean_ctor_set(v___x_1490_, 1, v___x_1492_);
lean_ctor_set(v___x_1490_, 0, v_x_1481_);
v___x_1494_ = v___x_1490_;
goto v_reusejp_1493_;
}
else
{
lean_object* v_reuseFailAlloc_1503_; 
v_reuseFailAlloc_1503_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1503_, 0, v_x_1481_);
lean_ctor_set(v_reuseFailAlloc_1503_, 1, v___x_1492_);
v___x_1494_ = v_reuseFailAlloc_1503_;
goto v_reusejp_1493_;
}
v_reusejp_1493_:
{
lean_object* v___x_1495_; lean_object* v___x_1497_; 
v___x_1495_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__3);
if (v_isShared_1487_ == 0)
{
lean_ctor_set_tag(v___x_1486_, 7);
lean_ctor_set(v___x_1486_, 1, v___x_1495_);
lean_ctor_set(v___x_1486_, 0, v___x_1494_);
v___x_1497_ = v___x_1486_;
goto v_reusejp_1496_;
}
else
{
lean_object* v_reuseFailAlloc_1502_; 
v_reuseFailAlloc_1502_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1502_, 0, v___x_1494_);
lean_ctor_set(v_reuseFailAlloc_1502_, 1, v___x_1495_);
v___x_1497_ = v_reuseFailAlloc_1502_;
goto v_reusejp_1496_;
}
v_reusejp_1496_:
{
lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; 
v___x_1498_ = l_Lean_MessageData_ofSyntax(v_before_1488_);
v___x_1499_ = l_Lean_indentD(v___x_1498_);
v___x_1500_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1500_, 0, v___x_1497_);
lean_ctor_set(v___x_1500_, 1, v___x_1499_);
v_x_1481_ = v___x_1500_;
v_x_1482_ = v_tail_1484_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__8(lean_object* v_opts_1507_, lean_object* v_opt_1508_){
_start:
{
lean_object* v_name_1509_; lean_object* v_defValue_1510_; lean_object* v_map_1511_; lean_object* v___x_1512_; 
v_name_1509_ = lean_ctor_get(v_opt_1508_, 0);
v_defValue_1510_ = lean_ctor_get(v_opt_1508_, 1);
v_map_1511_ = lean_ctor_get(v_opts_1507_, 0);
v___x_1512_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1511_, v_name_1509_);
if (lean_obj_tag(v___x_1512_) == 0)
{
uint8_t v___x_1513_; 
v___x_1513_ = lean_unbox(v_defValue_1510_);
return v___x_1513_;
}
else
{
lean_object* v_val_1514_; 
v_val_1514_ = lean_ctor_get(v___x_1512_, 0);
lean_inc(v_val_1514_);
lean_dec_ref_known(v___x_1512_, 1);
if (lean_obj_tag(v_val_1514_) == 1)
{
uint8_t v_v_1515_; 
v_v_1515_ = lean_ctor_get_uint8(v_val_1514_, 0);
lean_dec_ref_known(v_val_1514_, 0);
return v_v_1515_;
}
else
{
uint8_t v___x_1516_; 
lean_dec(v_val_1514_);
v___x_1516_ = lean_unbox(v_defValue_1510_);
return v___x_1516_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__8___boxed(lean_object* v_opts_1517_, lean_object* v_opt_1518_){
_start:
{
uint8_t v_res_1519_; lean_object* v_r_1520_; 
v_res_1519_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__8(v_opts_1517_, v_opt_1518_);
lean_dec_ref(v_opt_1518_);
lean_dec_ref(v_opts_1517_);
v_r_1520_ = lean_box(v_res_1519_);
return v_r_1520_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; 
v___x_1524_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__1));
v___x_1525_ = l_Lean_MessageData_ofFormat(v___x_1524_);
return v___x_1525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg(lean_object* v_msgData_1526_, lean_object* v_macroStack_1527_, lean_object* v___y_1528_){
_start:
{
lean_object* v_options_1530_; lean_object* v___x_1531_; uint8_t v___x_1532_; 
v_options_1530_ = lean_ctor_get(v___y_1528_, 2);
v___x_1531_ = l_Lean_Elab_pp_macroStack;
v___x_1532_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__8(v_options_1530_, v___x_1531_);
if (v___x_1532_ == 0)
{
lean_object* v___x_1533_; 
lean_dec(v_macroStack_1527_);
v___x_1533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1533_, 0, v_msgData_1526_);
return v___x_1533_;
}
else
{
if (lean_obj_tag(v_macroStack_1527_) == 0)
{
lean_object* v___x_1534_; 
v___x_1534_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1534_, 0, v_msgData_1526_);
return v___x_1534_;
}
else
{
lean_object* v_head_1535_; lean_object* v_after_1536_; lean_object* v___x_1538_; uint8_t v_isShared_1539_; uint8_t v_isSharedCheck_1551_; 
v_head_1535_ = lean_ctor_get(v_macroStack_1527_, 0);
lean_inc(v_head_1535_);
v_after_1536_ = lean_ctor_get(v_head_1535_, 1);
v_isSharedCheck_1551_ = !lean_is_exclusive(v_head_1535_);
if (v_isSharedCheck_1551_ == 0)
{
lean_object* v_unused_1552_; 
v_unused_1552_ = lean_ctor_get(v_head_1535_, 0);
lean_dec(v_unused_1552_);
v___x_1538_ = v_head_1535_;
v_isShared_1539_ = v_isSharedCheck_1551_;
goto v_resetjp_1537_;
}
else
{
lean_inc(v_after_1536_);
lean_dec(v_head_1535_);
v___x_1538_ = lean_box(0);
v_isShared_1539_ = v_isSharedCheck_1551_;
goto v_resetjp_1537_;
}
v_resetjp_1537_:
{
lean_object* v___x_1540_; lean_object* v___x_1542_; 
v___x_1540_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9___closed__0);
if (v_isShared_1539_ == 0)
{
lean_ctor_set_tag(v___x_1538_, 7);
lean_ctor_set(v___x_1538_, 1, v___x_1540_);
lean_ctor_set(v___x_1538_, 0, v_msgData_1526_);
v___x_1542_ = v___x_1538_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v_msgData_1526_);
lean_ctor_set(v_reuseFailAlloc_1550_, 1, v___x_1540_);
v___x_1542_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v_msgData_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; 
v___x_1543_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___closed__2);
v___x_1544_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1544_, 0, v___x_1542_);
lean_ctor_set(v___x_1544_, 1, v___x_1543_);
v___x_1545_ = l_Lean_MessageData_ofSyntax(v_after_1536_);
v___x_1546_ = l_Lean_indentD(v___x_1545_);
v_msgData_1547_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_1547_, 0, v___x_1544_);
lean_ctor_set(v_msgData_1547_, 1, v___x_1546_);
v___x_1548_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3_spec__9(v_msgData_1547_, v_macroStack_1527_);
v___x_1549_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1549_, 0, v___x_1548_);
return v___x_1549_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_msgData_1553_, lean_object* v_macroStack_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_){
_start:
{
lean_object* v_res_1557_; 
v_res_1557_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg(v_msgData_1553_, v_macroStack_1554_, v___y_1555_);
lean_dec_ref(v___y_1555_);
return v_res_1557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg(lean_object* v_msg_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_){
_start:
{
lean_object* v_ref_1566_; lean_object* v___x_1567_; lean_object* v_a_1568_; lean_object* v_macroStack_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v_a_1572_; lean_object* v___x_1574_; uint8_t v_isShared_1575_; uint8_t v_isSharedCheck_1580_; 
v_ref_1566_ = lean_ctor_get(v___y_1563_, 5);
v___x_1567_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(v_msg_1558_, v___y_1561_, v___y_1562_, v___y_1563_, v___y_1564_);
v_a_1568_ = lean_ctor_get(v___x_1567_, 0);
lean_inc(v_a_1568_);
lean_dec_ref(v___x_1567_);
v_macroStack_1569_ = lean_ctor_get(v___y_1559_, 1);
v___x_1570_ = l_Lean_Elab_getBetterRef(v_ref_1566_, v_macroStack_1569_);
lean_inc(v_macroStack_1569_);
v___x_1571_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg(v_a_1568_, v_macroStack_1569_, v___y_1563_);
v_a_1572_ = lean_ctor_get(v___x_1571_, 0);
v_isSharedCheck_1580_ = !lean_is_exclusive(v___x_1571_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1574_ = v___x_1571_;
v_isShared_1575_ = v_isSharedCheck_1580_;
goto v_resetjp_1573_;
}
else
{
lean_inc(v_a_1572_);
lean_dec(v___x_1571_);
v___x_1574_ = lean_box(0);
v_isShared_1575_ = v_isSharedCheck_1580_;
goto v_resetjp_1573_;
}
v_resetjp_1573_:
{
lean_object* v___x_1576_; lean_object* v___x_1578_; 
v___x_1576_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1570_);
lean_ctor_set(v___x_1576_, 1, v_a_1572_);
if (v_isShared_1575_ == 0)
{
lean_ctor_set_tag(v___x_1574_, 1);
lean_ctor_set(v___x_1574_, 0, v___x_1576_);
v___x_1578_ = v___x_1574_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1579_; 
v_reuseFailAlloc_1579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1579_, 0, v___x_1576_);
v___x_1578_ = v_reuseFailAlloc_1579_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
return v___x_1578_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg___boxed(lean_object* v_msg_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_){
_start:
{
lean_object* v_res_1589_; 
v_res_1589_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg(v_msg_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_, v___y_1587_);
lean_dec(v___y_1587_);
lean_dec_ref(v___y_1586_);
lean_dec(v___y_1585_);
lean_dec_ref(v___y_1584_);
lean_dec(v___y_1583_);
lean_dec_ref(v___y_1582_);
return v_res_1589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg(lean_object* v_ref_1590_, lean_object* v_msg_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_){
_start:
{
lean_object* v_fileName_1599_; lean_object* v_fileMap_1600_; lean_object* v_options_1601_; lean_object* v_currRecDepth_1602_; lean_object* v_maxRecDepth_1603_; lean_object* v_ref_1604_; lean_object* v_currNamespace_1605_; lean_object* v_openDecls_1606_; lean_object* v_initHeartbeats_1607_; lean_object* v_maxHeartbeats_1608_; lean_object* v_quotContext_1609_; lean_object* v_currMacroScope_1610_; uint8_t v_diag_1611_; lean_object* v_cancelTk_x3f_1612_; uint8_t v_suppressElabErrors_1613_; lean_object* v_inheritedTraceOptions_1614_; lean_object* v_ref_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; 
v_fileName_1599_ = lean_ctor_get(v___y_1596_, 0);
v_fileMap_1600_ = lean_ctor_get(v___y_1596_, 1);
v_options_1601_ = lean_ctor_get(v___y_1596_, 2);
v_currRecDepth_1602_ = lean_ctor_get(v___y_1596_, 3);
v_maxRecDepth_1603_ = lean_ctor_get(v___y_1596_, 4);
v_ref_1604_ = lean_ctor_get(v___y_1596_, 5);
v_currNamespace_1605_ = lean_ctor_get(v___y_1596_, 6);
v_openDecls_1606_ = lean_ctor_get(v___y_1596_, 7);
v_initHeartbeats_1607_ = lean_ctor_get(v___y_1596_, 8);
v_maxHeartbeats_1608_ = lean_ctor_get(v___y_1596_, 9);
v_quotContext_1609_ = lean_ctor_get(v___y_1596_, 10);
v_currMacroScope_1610_ = lean_ctor_get(v___y_1596_, 11);
v_diag_1611_ = lean_ctor_get_uint8(v___y_1596_, sizeof(void*)*14);
v_cancelTk_x3f_1612_ = lean_ctor_get(v___y_1596_, 12);
v_suppressElabErrors_1613_ = lean_ctor_get_uint8(v___y_1596_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1614_ = lean_ctor_get(v___y_1596_, 13);
v_ref_1615_ = l_Lean_replaceRef(v_ref_1590_, v_ref_1604_);
lean_inc_ref(v_inheritedTraceOptions_1614_);
lean_inc(v_cancelTk_x3f_1612_);
lean_inc(v_currMacroScope_1610_);
lean_inc(v_quotContext_1609_);
lean_inc(v_maxHeartbeats_1608_);
lean_inc(v_initHeartbeats_1607_);
lean_inc(v_openDecls_1606_);
lean_inc(v_currNamespace_1605_);
lean_inc(v_maxRecDepth_1603_);
lean_inc(v_currRecDepth_1602_);
lean_inc_ref(v_options_1601_);
lean_inc_ref(v_fileMap_1600_);
lean_inc_ref(v_fileName_1599_);
v___x_1616_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1616_, 0, v_fileName_1599_);
lean_ctor_set(v___x_1616_, 1, v_fileMap_1600_);
lean_ctor_set(v___x_1616_, 2, v_options_1601_);
lean_ctor_set(v___x_1616_, 3, v_currRecDepth_1602_);
lean_ctor_set(v___x_1616_, 4, v_maxRecDepth_1603_);
lean_ctor_set(v___x_1616_, 5, v_ref_1615_);
lean_ctor_set(v___x_1616_, 6, v_currNamespace_1605_);
lean_ctor_set(v___x_1616_, 7, v_openDecls_1606_);
lean_ctor_set(v___x_1616_, 8, v_initHeartbeats_1607_);
lean_ctor_set(v___x_1616_, 9, v_maxHeartbeats_1608_);
lean_ctor_set(v___x_1616_, 10, v_quotContext_1609_);
lean_ctor_set(v___x_1616_, 11, v_currMacroScope_1610_);
lean_ctor_set(v___x_1616_, 12, v_cancelTk_x3f_1612_);
lean_ctor_set(v___x_1616_, 13, v_inheritedTraceOptions_1614_);
lean_ctor_set_uint8(v___x_1616_, sizeof(void*)*14, v_diag_1611_);
lean_ctor_set_uint8(v___x_1616_, sizeof(void*)*14 + 1, v_suppressElabErrors_1613_);
v___x_1617_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg(v_msg_1591_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_, v___x_1616_, v___y_1597_);
lean_dec_ref_known(v___x_1616_, 14);
return v___x_1617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg___boxed(lean_object* v_ref_1618_, lean_object* v_msg_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_, lean_object* v___y_1626_){
_start:
{
lean_object* v_res_1627_; 
v_res_1627_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg(v_ref_1618_, v_msg_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
lean_dec(v___y_1625_);
lean_dec_ref(v___y_1624_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
lean_dec(v___y_1621_);
lean_dec_ref(v___y_1620_);
lean_dec(v_ref_1618_);
return v_res_1627_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__2(void){
_start:
{
lean_object* v___x_1631_; lean_object* v___x_1632_; 
v___x_1631_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___closed__1));
v___x_1632_ = l_Lean_stringToMessageData(v___x_1631_);
return v___x_1632_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__4(void){
_start:
{
lean_object* v___x_1634_; lean_object* v___x_1635_; 
v___x_1634_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___closed__3));
v___x_1635_ = l_Lean_stringToMessageData(v___x_1634_);
return v___x_1635_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__6(void){
_start:
{
lean_object* v___x_1637_; lean_object* v___x_1638_; 
v___x_1637_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___closed__5));
v___x_1638_ = l_Lean_stringToMessageData(v___x_1637_);
return v___x_1638_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__8(void){
_start:
{
lean_object* v___x_1640_; lean_object* v___x_1641_; 
v___x_1640_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___closed__7));
v___x_1641_ = l_Lean_stringToMessageData(v___x_1640_);
return v___x_1641_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__12(void){
_start:
{
lean_object* v___x_1667_; lean_object* v___x_1668_; 
v___x_1667_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__1));
v___x_1668_ = l_String_toRawSubstring_x27(v___x_1667_);
return v___x_1668_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__39(void){
_start:
{
lean_object* v___x_1729_; lean_object* v___x_1730_; 
v___x_1729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__38));
v___x_1730_ = l_Lean_stringToMessageData(v___x_1729_);
return v___x_1730_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__41(void){
_start:
{
lean_object* v___x_1732_; lean_object* v___x_1733_; 
v___x_1732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__40));
v___x_1733_ = l_Lean_stringToMessageData(v___x_1732_);
return v___x_1733_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__43(void){
_start:
{
lean_object* v___x_1735_; lean_object* v___x_1736_; 
v___x_1735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__42));
v___x_1736_ = l_Lean_stringToMessageData(v___x_1735_);
return v___x_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2(lean_object* v_head_1737_, lean_object* v_tail_1738_, lean_object* v_acc_1739_, lean_object* v_insts_1740_, uint8_t v_eager_1741_, lean_object* v_args_1742_, lean_object* v_head_1743_, lean_object* v_tail_1744_, lean_object* v___f_1745_, lean_object* v___f_1746_, lean_object* v_cls_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_){
_start:
{
lean_object* v___y_1756_; lean_object* v___y_1757_; lean_object* v___y_1758_; lean_object* v___y_1759_; lean_object* v___y_1760_; lean_object* v___y_1761_; lean_object* v___y_1762_; lean_object* v___y_1763_; lean_object* v___y_1764_; lean_object* v___x_1769_; 
v___x_1769_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg(v_head_1737_, v___y_1751_);
if (lean_obj_tag(v___x_1769_) == 0)
{
lean_object* v_a_1770_; uint8_t v___x_1771_; 
v_a_1770_ = lean_ctor_get(v___x_1769_, 0);
lean_inc(v_a_1770_);
lean_dec_ref_known(v___x_1769_, 1);
v___x_1771_ = lean_unbox(v_a_1770_);
if (v___x_1771_ == 0)
{
lean_object* v___x_1772_; 
lean_inc(v_head_1737_);
v___x_1772_ = l_Lean_MVarId_getType(v_head_1737_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
if (lean_obj_tag(v___x_1772_) == 0)
{
lean_object* v_a_1773_; lean_object* v___x_1774_; 
v_a_1773_ = lean_ctor_get(v___x_1772_, 0);
lean_inc(v_a_1773_);
lean_dec_ref_known(v___x_1772_, 1);
v___x_1774_ = l_Lean_Elab_Term_exprToSyntax(v_a_1773_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
if (lean_obj_tag(v___x_1774_) == 0)
{
lean_object* v_a_1775_; lean_object* v_ref_1776_; lean_object* v_quotContext_1777_; lean_object* v_currMacroScope_1778_; uint8_t v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___y_1805_; lean_object* v___y_1806_; lean_object* v___y_1807_; lean_object* v___y_1808_; lean_object* v___y_1809_; lean_object* v___y_1810_; lean_object* v___y_1825_; lean_object* v___y_1826_; lean_object* v___y_1827_; lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v___y_1830_; lean_object* v___y_1898_; lean_object* v___y_1899_; lean_object* v___y_1900_; lean_object* v___y_1901_; lean_object* v___y_1902_; lean_object* v___y_1903_; 
v_a_1775_ = lean_ctor_get(v___x_1774_, 0);
lean_inc(v_a_1775_);
lean_dec_ref_known(v___x_1774_, 1);
v_ref_1776_ = lean_ctor_get(v___y_1752_, 5);
v_quotContext_1777_ = lean_ctor_get(v___y_1752_, 10);
v_currMacroScope_1778_ = lean_ctor_get(v___y_1752_, 11);
v___x_1779_ = lean_unbox(v_a_1770_);
lean_dec(v_a_1770_);
v___x_1780_ = l_Lean_SourceInfo_fromRef(v_ref_1776_, v___x_1779_);
v___x_1781_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__2));
v___x_1782_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__3));
lean_inc_n(v___x_1780_, 9);
v___x_1783_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1783_, 0, v___x_1780_);
lean_ctor_set(v___x_1783_, 1, v___x_1781_);
v___x_1784_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__6));
v___x_1785_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__8));
v___x_1786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__9));
v___x_1787_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1787_, 0, v___x_1780_);
lean_ctor_set(v___x_1787_, 1, v___x_1786_);
v___x_1788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__11));
v___x_1789_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__12, &lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__12);
v___x_1790_ = lean_box(0);
lean_inc(v_currMacroScope_1778_);
lean_inc(v_quotContext_1777_);
v___x_1791_ = l_Lean_addMacroScope(v_quotContext_1777_, v___x_1790_, v_currMacroScope_1778_);
v___x_1792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__33));
v___x_1793_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1793_, 0, v___x_1780_);
lean_ctor_set(v___x_1793_, 1, v___x_1789_);
lean_ctor_set(v___x_1793_, 2, v___x_1791_);
lean_ctor_set(v___x_1793_, 3, v___x_1792_);
v___x_1794_ = l_Lean_Syntax_node1(v___x_1780_, v___x_1788_, v___x_1793_);
v___x_1795_ = l_Lean_Syntax_node2(v___x_1780_, v___x_1785_, v___x_1787_, v___x_1794_);
v___x_1796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__34));
v___x_1797_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1797_, 0, v___x_1780_);
lean_ctor_set(v___x_1797_, 1, v___x_1796_);
v___x_1798_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__36));
v___x_1799_ = l_Lean_Syntax_node1(v___x_1780_, v___x_1798_, v_a_1775_);
v___x_1800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__37));
v___x_1801_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1801_, 0, v___x_1780_);
lean_ctor_set(v___x_1801_, 1, v___x_1800_);
v___x_1802_ = l_Lean_Syntax_node5(v___x_1780_, v___x_1784_, v___x_1795_, v_head_1743_, v___x_1797_, v___x_1799_, v___x_1801_);
v___x_1803_ = l_Lean_Syntax_node2(v___x_1780_, v___x_1782_, v___x_1783_, v___x_1802_);
if (v_eager_1741_ == 0)
{
v___y_1898_ = v___y_1748_;
v___y_1899_ = v___y_1749_;
v___y_1900_ = v___y_1750_;
v___y_1901_ = v___y_1751_;
v___y_1902_ = v___y_1752_;
v___y_1903_ = v___y_1753_;
goto v___jp_1897_;
}
else
{
lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; 
lean_inc(v___x_1803_);
v___x_1905_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_1905_, 0, v___x_1803_);
v___x_1906_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_1906_, 0, lean_box(0));
lean_closure_set(v___x_1906_, 1, v___x_1905_);
lean_inc(v_head_1737_);
v___x_1907_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_1907_, 0, v_head_1737_);
lean_closure_set(v___x_1907_, 1, v___x_1906_);
v___x_1908_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg(v___x_1907_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
if (lean_obj_tag(v___x_1908_) == 0)
{
lean_object* v_a_1909_; 
v_a_1909_ = lean_ctor_get(v___x_1908_, 0);
lean_inc(v_a_1909_);
lean_dec_ref_known(v___x_1908_, 1);
if (lean_obj_tag(v_a_1909_) == 1)
{
lean_object* v_val_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; 
lean_dec(v___x_1803_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_args_1742_);
lean_dec(v_head_1737_);
v_val_1910_ = lean_ctor_get(v_a_1909_, 0);
lean_inc(v_val_1910_);
lean_dec_ref_known(v_a_1909_, 1);
v___x_1911_ = l_List_appendTR___redArg(v_acc_1739_, v_val_1910_);
v___x_1912_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_1741_, v_tail_1738_, v_tail_1744_, v___x_1911_, v_insts_1740_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
return v___x_1912_;
}
else
{
lean_dec(v_a_1909_);
v___y_1898_ = v___y_1748_;
v___y_1899_ = v___y_1749_;
v___y_1900_ = v___y_1750_;
v___y_1901_ = v___y_1751_;
v___y_1902_ = v___y_1752_;
v___y_1903_ = v___y_1753_;
goto v___jp_1897_;
}
}
else
{
lean_object* v_a_1913_; lean_object* v___x_1915_; uint8_t v_isShared_1916_; uint8_t v_isSharedCheck_1920_; 
lean_dec(v___x_1803_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_tail_1744_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1913_ = lean_ctor_get(v___x_1908_, 0);
v_isSharedCheck_1920_ = !lean_is_exclusive(v___x_1908_);
if (v_isSharedCheck_1920_ == 0)
{
v___x_1915_ = v___x_1908_;
v_isShared_1916_ = v_isSharedCheck_1920_;
goto v_resetjp_1914_;
}
else
{
lean_inc(v_a_1913_);
lean_dec(v___x_1908_);
v___x_1915_ = lean_box(0);
v_isShared_1916_ = v_isSharedCheck_1920_;
goto v_resetjp_1914_;
}
v_resetjp_1914_:
{
lean_object* v___x_1918_; 
if (v_isShared_1916_ == 0)
{
v___x_1918_ = v___x_1915_;
goto v_reusejp_1917_;
}
else
{
lean_object* v_reuseFailAlloc_1919_; 
v_reuseFailAlloc_1919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1919_, 0, v_a_1913_);
v___x_1918_ = v_reuseFailAlloc_1919_;
goto v_reusejp_1917_;
}
v_reusejp_1917_:
{
return v___x_1918_;
}
}
}
}
v___jp_1804_:
{
lean_object* v___x_1811_; lean_object* v___x_1812_; 
v___x_1811_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_1811_, 0, v___x_1803_);
v___x_1812_ = l_Lean_Elab_Tactic_run(v_head_1737_, v___x_1811_, v___y_1805_, v___y_1806_, v___y_1807_, v___y_1808_, v___y_1809_, v___y_1810_);
if (lean_obj_tag(v___x_1812_) == 0)
{
lean_object* v_a_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; 
v_a_1813_ = lean_ctor_get(v___x_1812_, 0);
lean_inc(v_a_1813_);
lean_dec_ref_known(v___x_1812_, 1);
v___x_1814_ = l_List_appendTR___redArg(v_acc_1739_, v_a_1813_);
v___x_1815_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_1741_, v_tail_1738_, v_tail_1744_, v___x_1814_, v_insts_1740_, v___y_1805_, v___y_1806_, v___y_1807_, v___y_1808_, v___y_1809_, v___y_1810_);
lean_dec(v___y_1810_);
lean_dec_ref(v___y_1809_);
lean_dec(v___y_1808_);
lean_dec_ref(v___y_1807_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
return v___x_1815_;
}
else
{
lean_object* v_a_1816_; lean_object* v___x_1818_; uint8_t v_isShared_1819_; uint8_t v_isSharedCheck_1823_; 
lean_dec(v___y_1810_);
lean_dec_ref(v___y_1809_);
lean_dec(v___y_1808_);
lean_dec_ref(v___y_1807_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
lean_dec(v_tail_1744_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
v_a_1816_ = lean_ctor_get(v___x_1812_, 0);
v_isSharedCheck_1823_ = !lean_is_exclusive(v___x_1812_);
if (v_isSharedCheck_1823_ == 0)
{
v___x_1818_ = v___x_1812_;
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
else
{
lean_inc(v_a_1816_);
lean_dec(v___x_1812_);
v___x_1818_ = lean_box(0);
v_isShared_1819_ = v_isSharedCheck_1823_;
goto v_resetjp_1817_;
}
v_resetjp_1817_:
{
lean_object* v___x_1821_; 
if (v_isShared_1819_ == 0)
{
v___x_1821_ = v___x_1818_;
goto v_reusejp_1820_;
}
else
{
lean_object* v_reuseFailAlloc_1822_; 
v_reuseFailAlloc_1822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1822_, 0, v_a_1816_);
v___x_1821_ = v_reuseFailAlloc_1822_;
goto v_reusejp_1820_;
}
v_reusejp_1820_:
{
return v___x_1821_;
}
}
}
}
v___jp_1824_:
{
lean_object* v___x_1831_; 
v___x_1831_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_useLoop_spec__3___redArg(v___f_1745_, v___y_1829_, v___y_1828_, v___y_1825_, v___y_1830_, v___y_1827_, v___y_1826_);
if (lean_obj_tag(v___x_1831_) == 0)
{
lean_object* v_a_1832_; 
v_a_1832_ = lean_ctor_get(v___x_1831_, 0);
lean_inc(v_a_1832_);
lean_dec_ref_known(v___x_1831_, 1);
if (lean_obj_tag(v_a_1832_) == 1)
{
lean_object* v_val_1833_; lean_object* v___x_1835_; uint8_t v_isShared_1836_; uint8_t v_isSharedCheck_1888_; 
lean_dec(v___x_1803_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1737_);
v_val_1833_ = lean_ctor_get(v_a_1832_, 0);
v_isSharedCheck_1888_ = !lean_is_exclusive(v_a_1832_);
if (v_isSharedCheck_1888_ == 0)
{
v___x_1835_ = v_a_1832_;
v_isShared_1836_ = v_isSharedCheck_1888_;
goto v_resetjp_1834_;
}
else
{
lean_inc(v_val_1833_);
lean_dec(v_a_1832_);
v___x_1835_ = lean_box(0);
v_isShared_1836_ = v_isSharedCheck_1888_;
goto v_resetjp_1834_;
}
v_resetjp_1834_:
{
lean_object* v_snd_1837_; lean_object* v_fst_1838_; lean_object* v___x_1840_; uint8_t v_isShared_1841_; uint8_t v_isSharedCheck_1887_; 
v_snd_1837_ = lean_ctor_get(v_val_1833_, 1);
v_fst_1838_ = lean_ctor_get(v_val_1833_, 0);
v_isSharedCheck_1887_ = !lean_is_exclusive(v_val_1833_);
if (v_isSharedCheck_1887_ == 0)
{
v___x_1840_ = v_val_1833_;
v_isShared_1841_ = v_isSharedCheck_1887_;
goto v_resetjp_1839_;
}
else
{
lean_inc(v_snd_1837_);
lean_inc(v_fst_1838_);
lean_dec(v_val_1833_);
v___x_1840_ = lean_box(0);
v_isShared_1841_ = v_isSharedCheck_1887_;
goto v_resetjp_1839_;
}
v_resetjp_1839_:
{
lean_object* v_fst_1842_; lean_object* v_snd_1843_; lean_object* v___x_1845_; uint8_t v_isShared_1846_; uint8_t v_isSharedCheck_1886_; 
v_fst_1842_ = lean_ctor_get(v_snd_1837_, 0);
v_snd_1843_ = lean_ctor_get(v_snd_1837_, 1);
v_isSharedCheck_1886_ = !lean_is_exclusive(v_snd_1837_);
if (v_isSharedCheck_1886_ == 0)
{
v___x_1845_ = v_snd_1837_;
v_isShared_1846_ = v_isSharedCheck_1886_;
goto v_resetjp_1844_;
}
else
{
lean_inc(v_snd_1843_);
lean_inc(v_fst_1842_);
lean_dec(v_snd_1837_);
v___x_1845_ = lean_box(0);
v_isShared_1846_ = v_isSharedCheck_1886_;
goto v_resetjp_1844_;
}
v_resetjp_1844_:
{
lean_object* v___x_1847_; 
lean_inc(v___y_1826_);
lean_inc_ref(v___y_1827_);
lean_inc(v___y_1830_);
lean_inc_ref(v___y_1825_);
lean_inc(v___y_1828_);
lean_inc_ref(v___y_1829_);
v___x_1847_ = lean_apply_7(v___f_1746_, v___y_1829_, v___y_1828_, v___y_1825_, v___y_1830_, v___y_1827_, v___y_1826_, lean_box(0));
if (lean_obj_tag(v___x_1847_) == 0)
{
lean_object* v_a_1848_; uint8_t v___x_1849_; 
v_a_1848_ = lean_ctor_get(v___x_1847_, 0);
lean_inc(v_a_1848_);
lean_dec_ref_known(v___x_1847_, 1);
v___x_1849_ = lean_unbox(v_a_1848_);
lean_dec(v_a_1848_);
if (v___x_1849_ == 0)
{
lean_del_object(v___x_1845_);
lean_del_object(v___x_1840_);
lean_del_object(v___x_1835_);
lean_dec(v_cls_1747_);
v___y_1756_ = v_fst_1838_;
v___y_1757_ = v_fst_1842_;
v___y_1758_ = v_snd_1843_;
v___y_1759_ = v___y_1829_;
v___y_1760_ = v___y_1828_;
v___y_1761_ = v___y_1825_;
v___y_1762_ = v___y_1830_;
v___y_1763_ = v___y_1827_;
v___y_1764_ = v___y_1826_;
goto v___jp_1755_;
}
else
{
lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1854_; 
v___x_1850_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__39, &lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__39_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__39);
v___x_1851_ = l_List_lengthTR___redArg(v_fst_1838_);
v___x_1852_ = l_Nat_reprFast(v___x_1851_);
if (v_isShared_1836_ == 0)
{
lean_ctor_set_tag(v___x_1835_, 3);
lean_ctor_set(v___x_1835_, 0, v___x_1852_);
v___x_1854_ = v___x_1835_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1877_; 
v_reuseFailAlloc_1877_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1877_, 0, v___x_1852_);
v___x_1854_ = v_reuseFailAlloc_1877_;
goto v_reusejp_1853_;
}
v_reusejp_1853_:
{
lean_object* v___x_1855_; lean_object* v___x_1857_; 
v___x_1855_ = l_Lean_MessageData_ofFormat(v___x_1854_);
if (v_isShared_1846_ == 0)
{
lean_ctor_set_tag(v___x_1845_, 7);
lean_ctor_set(v___x_1845_, 1, v___x_1855_);
lean_ctor_set(v___x_1845_, 0, v___x_1850_);
v___x_1857_ = v___x_1845_;
goto v_reusejp_1856_;
}
else
{
lean_object* v_reuseFailAlloc_1876_; 
v_reuseFailAlloc_1876_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1876_, 0, v___x_1850_);
lean_ctor_set(v_reuseFailAlloc_1876_, 1, v___x_1855_);
v___x_1857_ = v_reuseFailAlloc_1876_;
goto v_reusejp_1856_;
}
v_reusejp_1856_:
{
lean_object* v___x_1858_; lean_object* v___x_1860_; 
v___x_1858_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__41, &lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__41_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__41);
if (v_isShared_1841_ == 0)
{
lean_ctor_set_tag(v___x_1840_, 7);
lean_ctor_set(v___x_1840_, 1, v___x_1858_);
lean_ctor_set(v___x_1840_, 0, v___x_1857_);
v___x_1860_ = v___x_1840_;
goto v_reusejp_1859_;
}
else
{
lean_object* v_reuseFailAlloc_1875_; 
v_reuseFailAlloc_1875_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1875_, 0, v___x_1857_);
lean_ctor_set(v_reuseFailAlloc_1875_, 1, v___x_1858_);
v___x_1860_ = v_reuseFailAlloc_1875_;
goto v_reusejp_1859_;
}
v_reusejp_1859_:
{
lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; 
v___x_1861_ = l_List_lengthTR___redArg(v_fst_1842_);
v___x_1862_ = l_Nat_reprFast(v___x_1861_);
v___x_1863_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1863_, 0, v___x_1862_);
v___x_1864_ = l_Lean_MessageData_ofFormat(v___x_1863_);
v___x_1865_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1865_, 0, v___x_1860_);
lean_ctor_set(v___x_1865_, 1, v___x_1864_);
v___x_1866_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(v_cls_1747_, v___x_1865_, v___y_1825_, v___y_1830_, v___y_1827_, v___y_1826_);
if (lean_obj_tag(v___x_1866_) == 0)
{
lean_dec_ref_known(v___x_1866_, 1);
v___y_1756_ = v_fst_1838_;
v___y_1757_ = v_fst_1842_;
v___y_1758_ = v_snd_1843_;
v___y_1759_ = v___y_1829_;
v___y_1760_ = v___y_1828_;
v___y_1761_ = v___y_1825_;
v___y_1762_ = v___y_1830_;
v___y_1763_ = v___y_1827_;
v___y_1764_ = v___y_1826_;
goto v___jp_1755_;
}
else
{
lean_object* v_a_1867_; lean_object* v___x_1869_; uint8_t v_isShared_1870_; uint8_t v_isSharedCheck_1874_; 
lean_dec(v_snd_1843_);
lean_dec(v_fst_1842_);
lean_dec(v_fst_1838_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
v_a_1867_ = lean_ctor_get(v___x_1866_, 0);
v_isSharedCheck_1874_ = !lean_is_exclusive(v___x_1866_);
if (v_isSharedCheck_1874_ == 0)
{
v___x_1869_ = v___x_1866_;
v_isShared_1870_ = v_isSharedCheck_1874_;
goto v_resetjp_1868_;
}
else
{
lean_inc(v_a_1867_);
lean_dec(v___x_1866_);
v___x_1869_ = lean_box(0);
v_isShared_1870_ = v_isSharedCheck_1874_;
goto v_resetjp_1868_;
}
v_resetjp_1868_:
{
lean_object* v___x_1872_; 
if (v_isShared_1870_ == 0)
{
v___x_1872_ = v___x_1869_;
goto v_reusejp_1871_;
}
else
{
lean_object* v_reuseFailAlloc_1873_; 
v_reuseFailAlloc_1873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1873_, 0, v_a_1867_);
v___x_1872_ = v_reuseFailAlloc_1873_;
goto v_reusejp_1871_;
}
v_reusejp_1871_:
{
return v___x_1872_;
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
lean_object* v_a_1878_; lean_object* v___x_1880_; uint8_t v_isShared_1881_; uint8_t v_isSharedCheck_1885_; 
lean_del_object(v___x_1845_);
lean_dec(v_snd_1843_);
lean_dec(v_fst_1842_);
lean_del_object(v___x_1840_);
lean_dec(v_fst_1838_);
lean_del_object(v___x_1835_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec(v_cls_1747_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
v_a_1878_ = lean_ctor_get(v___x_1847_, 0);
v_isSharedCheck_1885_ = !lean_is_exclusive(v___x_1847_);
if (v_isSharedCheck_1885_ == 0)
{
v___x_1880_ = v___x_1847_;
v_isShared_1881_ = v_isSharedCheck_1885_;
goto v_resetjp_1879_;
}
else
{
lean_inc(v_a_1878_);
lean_dec(v___x_1847_);
v___x_1880_ = lean_box(0);
v_isShared_1881_ = v_isSharedCheck_1885_;
goto v_resetjp_1879_;
}
v_resetjp_1879_:
{
lean_object* v___x_1883_; 
if (v_isShared_1881_ == 0)
{
v___x_1883_ = v___x_1880_;
goto v_reusejp_1882_;
}
else
{
lean_object* v_reuseFailAlloc_1884_; 
v_reuseFailAlloc_1884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1884_, 0, v_a_1878_);
v___x_1883_ = v_reuseFailAlloc_1884_;
goto v_reusejp_1882_;
}
v_reusejp_1882_:
{
return v___x_1883_;
}
}
}
}
}
}
}
else
{
lean_dec(v_a_1832_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec(v_args_1742_);
v___y_1805_ = v___y_1829_;
v___y_1806_ = v___y_1828_;
v___y_1807_ = v___y_1825_;
v___y_1808_ = v___y_1830_;
v___y_1809_ = v___y_1827_;
v___y_1810_ = v___y_1826_;
goto v___jp_1804_;
}
}
else
{
lean_object* v_a_1889_; lean_object* v___x_1891_; uint8_t v_isShared_1892_; uint8_t v_isSharedCheck_1896_; 
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec(v___x_1803_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec(v_tail_1744_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1889_ = lean_ctor_get(v___x_1831_, 0);
v_isSharedCheck_1896_ = !lean_is_exclusive(v___x_1831_);
if (v_isSharedCheck_1896_ == 0)
{
v___x_1891_ = v___x_1831_;
v_isShared_1892_ = v_isSharedCheck_1896_;
goto v_resetjp_1890_;
}
else
{
lean_inc(v_a_1889_);
lean_dec(v___x_1831_);
v___x_1891_ = lean_box(0);
v_isShared_1892_ = v_isSharedCheck_1896_;
goto v_resetjp_1890_;
}
v_resetjp_1890_:
{
lean_object* v___x_1894_; 
if (v_isShared_1892_ == 0)
{
v___x_1894_ = v___x_1891_;
goto v_reusejp_1893_;
}
else
{
lean_object* v_reuseFailAlloc_1895_; 
v_reuseFailAlloc_1895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1895_, 0, v_a_1889_);
v___x_1894_ = v_reuseFailAlloc_1895_;
goto v_reusejp_1893_;
}
v_reusejp_1893_:
{
return v___x_1894_;
}
}
}
}
v___jp_1897_:
{
if (v_eager_1741_ == 0)
{
uint8_t v___x_1904_; 
v___x_1904_ = l_List_isEmpty___redArg(v_tail_1738_);
if (v___x_1904_ == 0)
{
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_args_1742_);
v___y_1805_ = v___y_1898_;
v___y_1806_ = v___y_1899_;
v___y_1807_ = v___y_1900_;
v___y_1808_ = v___y_1901_;
v___y_1809_ = v___y_1902_;
v___y_1810_ = v___y_1903_;
goto v___jp_1804_;
}
else
{
v___y_1825_ = v___y_1900_;
v___y_1826_ = v___y_1903_;
v___y_1827_ = v___y_1902_;
v___y_1828_ = v___y_1899_;
v___y_1829_ = v___y_1898_;
v___y_1830_ = v___y_1901_;
goto v___jp_1824_;
}
}
else
{
v___y_1825_ = v___y_1900_;
v___y_1826_ = v___y_1903_;
v___y_1827_ = v___y_1902_;
v___y_1828_ = v___y_1899_;
v___y_1829_ = v___y_1898_;
v___y_1830_ = v___y_1901_;
goto v___jp_1824_;
}
}
}
else
{
lean_object* v_a_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_1928_; 
lean_dec(v_a_1770_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1743_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1921_ = lean_ctor_get(v___x_1774_, 0);
v_isSharedCheck_1928_ = !lean_is_exclusive(v___x_1774_);
if (v_isSharedCheck_1928_ == 0)
{
v___x_1923_ = v___x_1774_;
v_isShared_1924_ = v_isSharedCheck_1928_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_a_1921_);
lean_dec(v___x_1774_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_1928_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
lean_object* v___x_1926_; 
if (v_isShared_1924_ == 0)
{
v___x_1926_ = v___x_1923_;
goto v_reusejp_1925_;
}
else
{
lean_object* v_reuseFailAlloc_1927_; 
v_reuseFailAlloc_1927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1927_, 0, v_a_1921_);
v___x_1926_ = v_reuseFailAlloc_1927_;
goto v_reusejp_1925_;
}
v_reusejp_1925_:
{
return v___x_1926_;
}
}
}
}
else
{
lean_object* v_a_1929_; lean_object* v___x_1931_; uint8_t v_isShared_1932_; uint8_t v_isSharedCheck_1936_; 
lean_dec(v_a_1770_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1743_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1929_ = lean_ctor_get(v___x_1772_, 0);
v_isSharedCheck_1936_ = !lean_is_exclusive(v___x_1772_);
if (v_isSharedCheck_1936_ == 0)
{
v___x_1931_ = v___x_1772_;
v_isShared_1932_ = v_isSharedCheck_1936_;
goto v_resetjp_1930_;
}
else
{
lean_inc(v_a_1929_);
lean_dec(v___x_1772_);
v___x_1931_ = lean_box(0);
v_isShared_1932_ = v_isSharedCheck_1936_;
goto v_resetjp_1930_;
}
v_resetjp_1930_:
{
lean_object* v___x_1934_; 
if (v_isShared_1932_ == 0)
{
v___x_1934_ = v___x_1931_;
goto v_reusejp_1933_;
}
else
{
lean_object* v_reuseFailAlloc_1935_; 
v_reuseFailAlloc_1935_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1935_, 0, v_a_1929_);
v___x_1934_ = v_reuseFailAlloc_1935_;
goto v_reusejp_1933_;
}
v_reusejp_1933_:
{
return v___x_1934_;
}
}
}
}
else
{
lean_object* v___x_1937_; 
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_args_1742_);
lean_inc(v_head_1737_);
v___x_1937_ = l_Lean_MVarId_getType(v_head_1737_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
if (lean_obj_tag(v___x_1937_) == 0)
{
lean_object* v_a_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; uint8_t v___x_1941_; uint8_t v___x_1942_; lean_object* v___x_1943_; 
v_a_1938_ = lean_ctor_get(v___x_1937_, 0);
lean_inc(v_a_1938_);
lean_dec_ref_known(v___x_1937_, 1);
v___x_1939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1939_, 0, v_a_1938_);
v___x_1940_ = lean_box(0);
v___x_1941_ = lean_unbox(v_a_1770_);
v___x_1942_ = lean_unbox(v_a_1770_);
lean_dec(v_a_1770_);
lean_inc(v_head_1743_);
v___x_1943_ = l_Lean_Elab_Term_elabTermEnsuringType(v_head_1743_, v___x_1939_, v___x_1941_, v___x_1942_, v___x_1940_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
if (lean_obj_tag(v___x_1943_) == 0)
{
lean_object* v_a_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; 
v_a_1944_ = lean_ctor_get(v___x_1943_, 0);
lean_inc(v_a_1944_);
lean_dec_ref_known(v___x_1943_, 1);
v___x_1945_ = l_Lean_Expr_mvar___override(v_head_1737_);
lean_inc_ref(v___x_1945_);
v___x_1946_ = l_Lean_Meta_isExprDefEq(v_a_1944_, v___x_1945_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
if (lean_obj_tag(v___x_1946_) == 0)
{
lean_object* v_a_1947_; uint8_t v___x_1948_; 
v_a_1947_ = lean_ctor_get(v___x_1946_, 0);
lean_inc(v_a_1947_);
lean_dec_ref_known(v___x_1946_, 1);
v___x_1948_ = lean_unbox(v_a_1947_);
lean_dec(v_a_1947_);
if (v___x_1948_ == 0)
{
lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; 
v___x_1949_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__43, &lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__43_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__43);
v___x_1950_ = l_Lean_indentExpr(v___x_1945_);
v___x_1951_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1951_, 0, v___x_1949_);
lean_ctor_set(v___x_1951_, 1, v___x_1950_);
v___x_1952_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg(v_head_1743_, v___x_1951_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
lean_dec(v_head_1743_);
if (lean_obj_tag(v___x_1952_) == 0)
{
lean_object* v___x_1953_; 
lean_dec_ref_known(v___x_1952_, 1);
v___x_1953_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_1741_, v_tail_1738_, v_tail_1744_, v_acc_1739_, v_insts_1740_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
return v___x_1953_;
}
else
{
lean_object* v_a_1954_; lean_object* v___x_1956_; uint8_t v_isShared_1957_; uint8_t v_isSharedCheck_1961_; 
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_tail_1744_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
v_a_1954_ = lean_ctor_get(v___x_1952_, 0);
v_isSharedCheck_1961_ = !lean_is_exclusive(v___x_1952_);
if (v_isSharedCheck_1961_ == 0)
{
v___x_1956_ = v___x_1952_;
v_isShared_1957_ = v_isSharedCheck_1961_;
goto v_resetjp_1955_;
}
else
{
lean_inc(v_a_1954_);
lean_dec(v___x_1952_);
v___x_1956_ = lean_box(0);
v_isShared_1957_ = v_isSharedCheck_1961_;
goto v_resetjp_1955_;
}
v_resetjp_1955_:
{
lean_object* v___x_1959_; 
if (v_isShared_1957_ == 0)
{
v___x_1959_ = v___x_1956_;
goto v_reusejp_1958_;
}
else
{
lean_object* v_reuseFailAlloc_1960_; 
v_reuseFailAlloc_1960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1960_, 0, v_a_1954_);
v___x_1959_ = v_reuseFailAlloc_1960_;
goto v_reusejp_1958_;
}
v_reusejp_1958_:
{
return v___x_1959_;
}
}
}
}
else
{
lean_object* v___x_1962_; 
lean_dec_ref(v___x_1945_);
lean_dec(v_head_1743_);
v___x_1962_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_1741_, v_tail_1738_, v_tail_1744_, v_acc_1739_, v_insts_1740_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
return v___x_1962_;
}
}
else
{
lean_object* v_a_1963_; lean_object* v___x_1965_; uint8_t v_isShared_1966_; uint8_t v_isSharedCheck_1970_; 
lean_dec_ref(v___x_1945_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1743_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
v_a_1963_ = lean_ctor_get(v___x_1946_, 0);
v_isSharedCheck_1970_ = !lean_is_exclusive(v___x_1946_);
if (v_isSharedCheck_1970_ == 0)
{
v___x_1965_ = v___x_1946_;
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
else
{
lean_inc(v_a_1963_);
lean_dec(v___x_1946_);
v___x_1965_ = lean_box(0);
v_isShared_1966_ = v_isSharedCheck_1970_;
goto v_resetjp_1964_;
}
v_resetjp_1964_:
{
lean_object* v___x_1968_; 
if (v_isShared_1966_ == 0)
{
v___x_1968_ = v___x_1965_;
goto v_reusejp_1967_;
}
else
{
lean_object* v_reuseFailAlloc_1969_; 
v_reuseFailAlloc_1969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1969_, 0, v_a_1963_);
v___x_1968_ = v_reuseFailAlloc_1969_;
goto v_reusejp_1967_;
}
v_reusejp_1967_:
{
return v___x_1968_;
}
}
}
}
else
{
lean_object* v_a_1971_; lean_object* v___x_1973_; uint8_t v_isShared_1974_; uint8_t v_isSharedCheck_1978_; 
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1743_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1971_ = lean_ctor_get(v___x_1943_, 0);
v_isSharedCheck_1978_ = !lean_is_exclusive(v___x_1943_);
if (v_isSharedCheck_1978_ == 0)
{
v___x_1973_ = v___x_1943_;
v_isShared_1974_ = v_isSharedCheck_1978_;
goto v_resetjp_1972_;
}
else
{
lean_inc(v_a_1971_);
lean_dec(v___x_1943_);
v___x_1973_ = lean_box(0);
v_isShared_1974_ = v_isSharedCheck_1978_;
goto v_resetjp_1972_;
}
v_resetjp_1972_:
{
lean_object* v___x_1976_; 
if (v_isShared_1974_ == 0)
{
v___x_1976_ = v___x_1973_;
goto v_reusejp_1975_;
}
else
{
lean_object* v_reuseFailAlloc_1977_; 
v_reuseFailAlloc_1977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1977_, 0, v_a_1971_);
v___x_1976_ = v_reuseFailAlloc_1977_;
goto v_reusejp_1975_;
}
v_reusejp_1975_:
{
return v___x_1976_;
}
}
}
}
else
{
lean_object* v_a_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_1986_; 
lean_dec(v_a_1770_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1743_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1979_ = lean_ctor_get(v___x_1937_, 0);
v_isSharedCheck_1986_ = !lean_is_exclusive(v___x_1937_);
if (v_isSharedCheck_1986_ == 0)
{
v___x_1981_ = v___x_1937_;
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_a_1979_);
lean_dec(v___x_1937_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_1986_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1984_; 
if (v_isShared_1982_ == 0)
{
v___x_1984_ = v___x_1981_;
goto v_reusejp_1983_;
}
else
{
lean_object* v_reuseFailAlloc_1985_; 
v_reuseFailAlloc_1985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1985_, 0, v_a_1979_);
v___x_1984_ = v_reuseFailAlloc_1985_;
goto v_reusejp_1983_;
}
v_reusejp_1983_:
{
return v___x_1984_;
}
}
}
}
}
else
{
lean_object* v_a_1987_; lean_object* v___x_1989_; uint8_t v_isShared_1990_; uint8_t v_isSharedCheck_1994_; 
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec(v_cls_1747_);
lean_dec_ref(v___f_1746_);
lean_dec_ref(v___f_1745_);
lean_dec(v_tail_1744_);
lean_dec(v_head_1743_);
lean_dec(v_args_1742_);
lean_dec(v_insts_1740_);
lean_dec(v_acc_1739_);
lean_dec(v_tail_1738_);
lean_dec(v_head_1737_);
v_a_1987_ = lean_ctor_get(v___x_1769_, 0);
v_isSharedCheck_1994_ = !lean_is_exclusive(v___x_1769_);
if (v_isSharedCheck_1994_ == 0)
{
v___x_1989_ = v___x_1769_;
v_isShared_1990_ = v_isSharedCheck_1994_;
goto v_resetjp_1988_;
}
else
{
lean_inc(v_a_1987_);
lean_dec(v___x_1769_);
v___x_1989_ = lean_box(0);
v_isShared_1990_ = v_isSharedCheck_1994_;
goto v_resetjp_1988_;
}
v_resetjp_1988_:
{
lean_object* v___x_1992_; 
if (v_isShared_1990_ == 0)
{
v___x_1992_ = v___x_1989_;
goto v_reusejp_1991_;
}
else
{
lean_object* v_reuseFailAlloc_1993_; 
v_reuseFailAlloc_1993_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1993_, 0, v_a_1987_);
v___x_1992_ = v_reuseFailAlloc_1993_;
goto v_reusejp_1991_;
}
v_reusejp_1991_:
{
return v___x_1992_;
}
}
}
v___jp_1755_:
{
lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; 
v___x_1765_ = l_List_appendTR___redArg(v___y_1756_, v_tail_1738_);
v___x_1766_ = l_List_appendTR___redArg(v_acc_1739_, v___y_1757_);
v___x_1767_ = l_List_appendTR___redArg(v_insts_1740_, v___y_1758_);
v___x_1768_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_1741_, v___x_1765_, v_args_1742_, v___x_1766_, v___x_1767_, v___y_1759_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_);
lean_dec(v___y_1764_);
lean_dec_ref(v___y_1763_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
return v___x_1768_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___lam__2___boxed(lean_object** _args){
lean_object* v_head_1995_ = _args[0];
lean_object* v_tail_1996_ = _args[1];
lean_object* v_acc_1997_ = _args[2];
lean_object* v_insts_1998_ = _args[3];
lean_object* v_eager_1999_ = _args[4];
lean_object* v_args_2000_ = _args[5];
lean_object* v_head_2001_ = _args[6];
lean_object* v_tail_2002_ = _args[7];
lean_object* v___f_2003_ = _args[8];
lean_object* v___f_2004_ = _args[9];
lean_object* v_cls_2005_ = _args[10];
lean_object* v___y_2006_ = _args[11];
lean_object* v___y_2007_ = _args[12];
lean_object* v___y_2008_ = _args[13];
lean_object* v___y_2009_ = _args[14];
lean_object* v___y_2010_ = _args[15];
lean_object* v___y_2011_ = _args[16];
lean_object* v___y_2012_ = _args[17];
_start:
{
uint8_t v_eager_boxed_2013_; lean_object* v_res_2014_; 
v_eager_boxed_2013_ = lean_unbox(v_eager_1999_);
v_res_2014_ = lp_mathlib_Mathlib_Tactic_useLoop___lam__2(v_head_1995_, v_tail_1996_, v_acc_1997_, v_insts_1998_, v_eager_boxed_2013_, v_args_2000_, v_head_2001_, v_tail_2002_, v___f_2003_, v___f_2004_, v_cls_2005_, v___y_2006_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_, v___y_2011_);
return v_res_2014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop(uint8_t v_eager_2015_, lean_object* v_gs_2016_, lean_object* v_args_2017_, lean_object* v_acc_2018_, lean_object* v_insts_2019_, lean_object* v_a_2020_, lean_object* v_a_2021_, lean_object* v_a_2022_, lean_object* v_a_2023_, lean_object* v_a_2024_, lean_object* v_a_2025_){
_start:
{
lean_object* v_cls_2027_; lean_object* v___f_2028_; lean_object* v___y_2030_; lean_object* v___y_2031_; lean_object* v___y_2032_; lean_object* v___y_2033_; lean_object* v___y_2034_; lean_object* v___y_2035_; lean_object* v___x_2050_; 
v_cls_2027_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_));
v___f_2028_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___closed__0));
v___x_2050_ = lp_mathlib_Mathlib_Tactic_useLoop___lam__0(v_cls_2027_, v_a_2020_, v_a_2021_, v_a_2022_, v_a_2023_, v_a_2024_, v_a_2025_);
if (lean_obj_tag(v___x_2050_) == 0)
{
lean_object* v_a_2051_; uint8_t v___x_2052_; 
v_a_2051_ = lean_ctor_get(v___x_2050_, 0);
lean_inc(v_a_2051_);
lean_dec_ref_known(v___x_2050_, 1);
v___x_2052_ = lean_unbox(v_a_2051_);
lean_dec(v_a_2051_);
if (v___x_2052_ == 0)
{
v___y_2030_ = v_a_2020_;
v___y_2031_ = v_a_2021_;
v___y_2032_ = v_a_2022_;
v___y_2033_ = v_a_2023_;
v___y_2034_ = v_a_2024_;
v___y_2035_ = v_a_2025_;
goto v___jp_2029_;
}
else
{
lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; 
v___x_2053_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___closed__4, &lp_mathlib_Mathlib_Tactic_useLoop___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__4);
v___x_2054_ = lean_box(0);
lean_inc(v_gs_2016_);
v___x_2055_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__5(v_gs_2016_, v___x_2054_);
v___x_2056_ = l_Lean_MessageData_ofList(v___x_2055_);
v___x_2057_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2057_, 0, v___x_2053_);
lean_ctor_set(v___x_2057_, 1, v___x_2056_);
v___x_2058_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___closed__6, &lp_mathlib_Mathlib_Tactic_useLoop___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__6);
v___x_2059_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2059_, 0, v___x_2057_);
lean_ctor_set(v___x_2059_, 1, v___x_2058_);
lean_inc(v_args_2017_);
v___x_2060_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__6(v_args_2017_, v___x_2054_);
v___x_2061_ = l_Lean_MessageData_ofList(v___x_2060_);
v___x_2062_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2062_, 0, v___x_2059_);
lean_ctor_set(v___x_2062_, 1, v___x_2061_);
v___x_2063_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___closed__8, &lp_mathlib_Mathlib_Tactic_useLoop___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__8);
v___x_2064_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2064_, 0, v___x_2062_);
lean_ctor_set(v___x_2064_, 1, v___x_2063_);
lean_inc(v_acc_2018_);
v___x_2065_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_useLoop_spec__5(v_acc_2018_, v___x_2054_);
v___x_2066_ = l_Lean_MessageData_ofList(v___x_2065_);
v___x_2067_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2067_, 0, v___x_2064_);
lean_ctor_set(v___x_2067_, 1, v___x_2066_);
v___x_2068_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(v_cls_2027_, v___x_2067_, v_a_2022_, v_a_2023_, v_a_2024_, v_a_2025_);
if (lean_obj_tag(v___x_2068_) == 0)
{
lean_dec_ref_known(v___x_2068_, 1);
v___y_2030_ = v_a_2020_;
v___y_2031_ = v_a_2021_;
v___y_2032_ = v_a_2022_;
v___y_2033_ = v_a_2023_;
v___y_2034_ = v_a_2024_;
v___y_2035_ = v_a_2025_;
goto v___jp_2029_;
}
else
{
lean_object* v_a_2069_; lean_object* v___x_2071_; uint8_t v_isShared_2072_; uint8_t v_isSharedCheck_2076_; 
lean_dec(v_insts_2019_);
lean_dec(v_acc_2018_);
lean_dec(v_args_2017_);
lean_dec(v_gs_2016_);
v_a_2069_ = lean_ctor_get(v___x_2068_, 0);
v_isSharedCheck_2076_ = !lean_is_exclusive(v___x_2068_);
if (v_isSharedCheck_2076_ == 0)
{
v___x_2071_ = v___x_2068_;
v_isShared_2072_ = v_isSharedCheck_2076_;
goto v_resetjp_2070_;
}
else
{
lean_inc(v_a_2069_);
lean_dec(v___x_2068_);
v___x_2071_ = lean_box(0);
v_isShared_2072_ = v_isSharedCheck_2076_;
goto v_resetjp_2070_;
}
v_resetjp_2070_:
{
lean_object* v___x_2074_; 
if (v_isShared_2072_ == 0)
{
v___x_2074_ = v___x_2071_;
goto v_reusejp_2073_;
}
else
{
lean_object* v_reuseFailAlloc_2075_; 
v_reuseFailAlloc_2075_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2075_, 0, v_a_2069_);
v___x_2074_ = v_reuseFailAlloc_2075_;
goto v_reusejp_2073_;
}
v_reusejp_2073_:
{
return v___x_2074_;
}
}
}
}
}
else
{
lean_object* v_a_2077_; lean_object* v___x_2079_; uint8_t v_isShared_2080_; uint8_t v_isSharedCheck_2084_; 
lean_dec(v_insts_2019_);
lean_dec(v_acc_2018_);
lean_dec(v_args_2017_);
lean_dec(v_gs_2016_);
v_a_2077_ = lean_ctor_get(v___x_2050_, 0);
v_isSharedCheck_2084_ = !lean_is_exclusive(v___x_2050_);
if (v_isSharedCheck_2084_ == 0)
{
v___x_2079_ = v___x_2050_;
v_isShared_2080_ = v_isSharedCheck_2084_;
goto v_resetjp_2078_;
}
else
{
lean_inc(v_a_2077_);
lean_dec(v___x_2050_);
v___x_2079_ = lean_box(0);
v_isShared_2080_ = v_isSharedCheck_2084_;
goto v_resetjp_2078_;
}
v_resetjp_2078_:
{
lean_object* v___x_2082_; 
if (v_isShared_2080_ == 0)
{
v___x_2082_ = v___x_2079_;
goto v_reusejp_2081_;
}
else
{
lean_object* v_reuseFailAlloc_2083_; 
v_reuseFailAlloc_2083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2083_, 0, v_a_2077_);
v___x_2082_ = v_reuseFailAlloc_2083_;
goto v_reusejp_2081_;
}
v_reusejp_2081_:
{
return v___x_2082_;
}
}
}
v___jp_2029_:
{
if (lean_obj_tag(v_args_2017_) == 0)
{
lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; 
v___x_2036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2036_, 0, v_acc_2018_);
lean_ctor_set(v___x_2036_, 1, v_insts_2019_);
v___x_2037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2037_, 0, v_gs_2016_);
lean_ctor_set(v___x_2037_, 1, v___x_2036_);
v___x_2038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2038_, 0, v___x_2037_);
return v___x_2038_;
}
else
{
if (lean_obj_tag(v_gs_2016_) == 0)
{
lean_object* v_head_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; 
lean_dec(v_insts_2019_);
lean_dec(v_acc_2018_);
v_head_2039_ = lean_ctor_get(v_args_2017_, 0);
lean_inc(v_head_2039_);
lean_dec_ref_known(v_args_2017_, 2);
v___x_2040_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useLoop___closed__2, &lp_mathlib_Mathlib_Tactic_useLoop___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_useLoop___closed__2);
v___x_2041_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg(v_head_2039_, v___x_2040_, v___y_2030_, v___y_2031_, v___y_2032_, v___y_2033_, v___y_2034_, v___y_2035_);
lean_dec(v_head_2039_);
return v___x_2041_;
}
else
{
lean_object* v_head_2042_; lean_object* v_tail_2043_; lean_object* v_head_2044_; lean_object* v_tail_2045_; lean_object* v___f_2046_; lean_object* v___x_2047_; lean_object* v___f_2048_; lean_object* v___x_2049_; 
v_head_2042_ = lean_ctor_get(v_args_2017_, 0);
lean_inc(v_head_2042_);
v_tail_2043_ = lean_ctor_get(v_args_2017_, 1);
lean_inc(v_tail_2043_);
v_head_2044_ = lean_ctor_get(v_gs_2016_, 0);
lean_inc_n(v_head_2044_, 3);
v_tail_2045_ = lean_ctor_get(v_gs_2016_, 1);
lean_inc(v_tail_2045_);
lean_dec_ref_known(v_gs_2016_, 2);
v___f_2046_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__1___boxed), 9, 2);
lean_closure_set(v___f_2046_, 0, v_head_2044_);
lean_closure_set(v___f_2046_, 1, v_cls_2027_);
v___x_2047_ = lean_box(v_eager_2015_);
v___f_2048_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___boxed), 18, 11);
lean_closure_set(v___f_2048_, 0, v_head_2044_);
lean_closure_set(v___f_2048_, 1, v_tail_2045_);
lean_closure_set(v___f_2048_, 2, v_acc_2018_);
lean_closure_set(v___f_2048_, 3, v_insts_2019_);
lean_closure_set(v___f_2048_, 4, v___x_2047_);
lean_closure_set(v___f_2048_, 5, v_args_2017_);
lean_closure_set(v___f_2048_, 6, v_head_2042_);
lean_closure_set(v___f_2048_, 7, v_tail_2043_);
lean_closure_set(v___f_2048_, 8, v___f_2046_);
lean_closure_set(v___f_2048_, 9, v___f_2028_);
lean_closure_set(v___f_2048_, 10, v_cls_2027_);
v___x_2049_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_useLoop_spec__4___redArg(v_head_2044_, v___f_2048_, v___y_2030_, v___y_2031_, v___y_2032_, v___y_2033_, v___y_2034_, v___y_2035_);
return v___x_2049_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_useLoop___boxed(lean_object* v_eager_2085_, lean_object* v_gs_2086_, lean_object* v_args_2087_, lean_object* v_acc_2088_, lean_object* v_insts_2089_, lean_object* v_a_2090_, lean_object* v_a_2091_, lean_object* v_a_2092_, lean_object* v_a_2093_, lean_object* v_a_2094_, lean_object* v_a_2095_, lean_object* v_a_2096_){
_start:
{
uint8_t v_eager_boxed_2097_; lean_object* v_res_2098_; 
v_eager_boxed_2097_ = lean_unbox(v_eager_2085_);
v_res_2098_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_boxed_2097_, v_gs_2086_, v_args_2087_, v_acc_2088_, v_insts_2089_, v_a_2090_, v_a_2091_, v_a_2092_, v_a_2093_, v_a_2094_, v_a_2095_);
lean_dec(v_a_2095_);
lean_dec_ref(v_a_2094_);
lean_dec(v_a_2093_);
lean_dec_ref(v_a_2092_);
lean_dec(v_a_2091_);
lean_dec_ref(v_a_2090_);
return v_res_2098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0(lean_object* v_00_u03b1_2099_, lean_object* v_ref_2100_, lean_object* v_msg_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_){
_start:
{
lean_object* v___x_2109_; 
v___x_2109_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___redArg(v_ref_2100_, v_msg_2101_, v___y_2102_, v___y_2103_, v___y_2104_, v___y_2105_, v___y_2106_, v___y_2107_);
return v___x_2109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0___boxed(lean_object* v_00_u03b1_2110_, lean_object* v_ref_2111_, lean_object* v_msg_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_){
_start:
{
lean_object* v_res_2120_; 
v_res_2120_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0(v_00_u03b1_2110_, v_ref_2111_, v_msg_2112_, v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_);
lean_dec(v___y_2118_);
lean_dec_ref(v___y_2117_);
lean_dec(v___y_2116_);
lean_dec_ref(v___y_2115_);
lean_dec(v___y_2114_);
lean_dec_ref(v___y_2113_);
lean_dec(v_ref_2111_);
return v_res_2120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1(lean_object* v_cls_2121_, lean_object* v_msg_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_){
_start:
{
lean_object* v___x_2130_; 
v___x_2130_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg(v_cls_2121_, v_msg_2122_, v___y_2125_, v___y_2126_, v___y_2127_, v___y_2128_);
return v___x_2130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___boxed(lean_object* v_cls_2131_, lean_object* v_msg_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_){
_start:
{
lean_object* v_res_2140_; 
v_res_2140_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1(v_cls_2131_, v_msg_2132_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_);
lean_dec(v___y_2138_);
lean_dec_ref(v___y_2137_);
lean_dec(v___y_2136_);
lean_dec_ref(v___y_2135_);
lean_dec(v___y_2134_);
lean_dec_ref(v___y_2133_);
return v_res_2140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2(lean_object* v_mvarId_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_){
_start:
{
lean_object* v___x_2149_; 
v___x_2149_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___redArg(v_mvarId_2141_, v___y_2145_);
return v___x_2149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2___boxed(lean_object* v_mvarId_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_){
_start:
{
lean_object* v_res_2158_; 
v_res_2158_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2(v_mvarId_2150_, v___y_2151_, v___y_2152_, v___y_2153_, v___y_2154_, v___y_2155_, v___y_2156_);
lean_dec(v___y_2156_);
lean_dec_ref(v___y_2155_);
lean_dec(v___y_2154_);
lean_dec_ref(v___y_2153_);
lean_dec(v___y_2152_);
lean_dec_ref(v___y_2151_);
lean_dec(v_mvarId_2150_);
return v_res_2158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0(lean_object* v_00_u03b1_2159_, lean_object* v_msg_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_){
_start:
{
lean_object* v___x_2168_; 
v___x_2168_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___redArg(v_msg_2160_, v___y_2161_, v___y_2162_, v___y_2163_, v___y_2164_, v___y_2165_, v___y_2166_);
return v___x_2168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0___boxed(lean_object* v_00_u03b1_2169_, lean_object* v_msg_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_){
_start:
{
lean_object* v_res_2178_; 
v_res_2178_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0(v_00_u03b1_2169_, v_msg_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_);
lean_dec(v___y_2176_);
lean_dec_ref(v___y_2175_);
lean_dec(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2171_);
return v_res_2178_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3(lean_object* v_00_u03b2_2179_, lean_object* v_x_2180_, lean_object* v_x_2181_){
_start:
{
uint8_t v___x_2182_; 
v___x_2182_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg(v_x_2180_, v_x_2181_);
return v___x_2182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___boxed(lean_object* v_00_u03b2_2183_, lean_object* v_x_2184_, lean_object* v_x_2185_){
_start:
{
uint8_t v_res_2186_; lean_object* v_r_2187_; 
v_res_2186_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3(v_00_u03b2_2183_, v_x_2184_, v_x_2185_);
lean_dec(v_x_2185_);
lean_dec_ref(v_x_2184_);
v_r_2187_ = lean_box(v_res_2186_);
return v_r_2187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3(lean_object* v_msgData_2188_, lean_object* v_macroStack_2189_, lean_object* v___y_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_){
_start:
{
lean_object* v___x_2197_; 
v___x_2197_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___redArg(v_msgData_2188_, v_macroStack_2189_, v___y_2194_);
return v___x_2197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3___boxed(lean_object* v_msgData_2198_, lean_object* v_macroStack_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_, lean_object* v___y_2206_){
_start:
{
lean_object* v_res_2207_; 
v_res_2207_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_useLoop_spec__0_spec__0_spec__3(v_msgData_2198_, v_macroStack_2199_, v___y_2200_, v___y_2201_, v___y_2202_, v___y_2203_, v___y_2204_, v___y_2205_);
lean_dec(v___y_2205_);
lean_dec_ref(v___y_2204_);
lean_dec(v___y_2203_);
lean_dec_ref(v___y_2202_);
lean_dec(v___y_2201_);
lean_dec_ref(v___y_2200_);
return v_res_2207_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7(lean_object* v_00_u03b2_2208_, lean_object* v_x_2209_, size_t v_x_2210_, lean_object* v_x_2211_){
_start:
{
uint8_t v___x_2212_; 
v___x_2212_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___redArg(v_x_2209_, v_x_2210_, v_x_2211_);
return v___x_2212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7___boxed(lean_object* v_00_u03b2_2213_, lean_object* v_x_2214_, lean_object* v_x_2215_, lean_object* v_x_2216_){
_start:
{
size_t v_x_18940__boxed_2217_; uint8_t v_res_2218_; lean_object* v_r_2219_; 
v_x_18940__boxed_2217_ = lean_unbox_usize(v_x_2215_);
lean_dec(v_x_2215_);
v_res_2218_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7(v_00_u03b2_2213_, v_x_2214_, v_x_18940__boxed_2217_, v_x_2216_);
lean_dec(v_x_2216_);
lean_dec_ref(v_x_2214_);
v_r_2219_ = lean_box(v_res_2218_);
return v_r_2219_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12(lean_object* v_00_u03b2_2220_, lean_object* v_keys_2221_, lean_object* v_vals_2222_, lean_object* v_heq_2223_, lean_object* v_i_2224_, lean_object* v_k_2225_){
_start:
{
uint8_t v___x_2226_; 
v___x_2226_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___redArg(v_keys_2221_, v_i_2224_, v_k_2225_);
return v___x_2226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12___boxed(lean_object* v_00_u03b2_2227_, lean_object* v_keys_2228_, lean_object* v_vals_2229_, lean_object* v_heq_2230_, lean_object* v_i_2231_, lean_object* v_k_2232_){
_start:
{
uint8_t v_res_2233_; lean_object* v_r_2234_; 
v_res_2233_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3_spec__7_spec__12(v_00_u03b2_2227_, v_keys_2228_, v_vals_2229_, v_heq_2230_, v_i_2231_, v_k_2232_);
lean_dec(v_k_2232_);
lean_dec_ref(v_vals_2229_);
lean_dec_ref(v_keys_2228_);
v_r_2234_ = lean_box(v_res_2233_);
return v_r_2234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___redArg(lean_object* v_x_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_){
_start:
{
lean_object* v___x_2245_; 
v___x_2245_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_2237_, v___y_2239_, v___y_2241_, v___y_2243_);
if (lean_obj_tag(v___x_2245_) == 0)
{
lean_object* v_a_2246_; lean_object* v___x_2247_; 
v_a_2246_ = lean_ctor_get(v___x_2245_, 0);
lean_inc(v_a_2246_);
lean_dec_ref_known(v___x_2245_, 1);
v___x_2247_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_2237_, v___y_2239_, v___y_2241_, v___y_2243_);
if (lean_obj_tag(v___x_2247_) == 0)
{
lean_object* v_a_2248_; lean_object* v___x_2249_; 
v_a_2248_ = lean_ctor_get(v___x_2247_, 0);
lean_inc(v_a_2248_);
lean_dec_ref_known(v___x_2247_, 1);
lean_inc(v___y_2243_);
lean_inc_ref(v___y_2242_);
lean_inc(v___y_2241_);
lean_inc_ref(v___y_2240_);
lean_inc(v___y_2239_);
lean_inc_ref(v___y_2238_);
lean_inc(v___y_2237_);
lean_inc_ref(v___y_2236_);
v___x_2249_ = lean_apply_9(v_x_2235_, v___y_2236_, v___y_2237_, v___y_2238_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_, lean_box(0));
if (lean_obj_tag(v___x_2249_) == 0)
{
lean_object* v_a_2250_; lean_object* v___x_2252_; uint8_t v_isShared_2253_; uint8_t v_isSharedCheck_2258_; 
lean_dec(v_a_2248_);
lean_dec(v_a_2246_);
v_a_2250_ = lean_ctor_get(v___x_2249_, 0);
v_isSharedCheck_2258_ = !lean_is_exclusive(v___x_2249_);
if (v_isSharedCheck_2258_ == 0)
{
v___x_2252_ = v___x_2249_;
v_isShared_2253_ = v_isSharedCheck_2258_;
goto v_resetjp_2251_;
}
else
{
lean_inc(v_a_2250_);
lean_dec(v___x_2249_);
v___x_2252_ = lean_box(0);
v_isShared_2253_ = v_isSharedCheck_2258_;
goto v_resetjp_2251_;
}
v_resetjp_2251_:
{
lean_object* v___x_2254_; lean_object* v___x_2256_; 
v___x_2254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2254_, 0, v_a_2250_);
if (v_isShared_2253_ == 0)
{
lean_ctor_set(v___x_2252_, 0, v___x_2254_);
v___x_2256_ = v___x_2252_;
goto v_reusejp_2255_;
}
else
{
lean_object* v_reuseFailAlloc_2257_; 
v_reuseFailAlloc_2257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2257_, 0, v___x_2254_);
v___x_2256_ = v_reuseFailAlloc_2257_;
goto v_reusejp_2255_;
}
v_reusejp_2255_:
{
return v___x_2256_;
}
}
}
else
{
lean_object* v_a_2259_; lean_object* v___x_2261_; uint8_t v_isShared_2262_; uint8_t v_isSharedCheck_2297_; 
v_a_2259_ = lean_ctor_get(v___x_2249_, 0);
v_isSharedCheck_2297_ = !lean_is_exclusive(v___x_2249_);
if (v_isSharedCheck_2297_ == 0)
{
v___x_2261_ = v___x_2249_;
v_isShared_2262_ = v_isSharedCheck_2297_;
goto v_resetjp_2260_;
}
else
{
lean_inc(v_a_2259_);
lean_dec(v___x_2249_);
v___x_2261_ = lean_box(0);
v_isShared_2262_ = v_isSharedCheck_2297_;
goto v_resetjp_2260_;
}
v_resetjp_2260_:
{
uint8_t v___y_2264_; uint8_t v___x_2295_; 
v___x_2295_ = l_Lean_Exception_isInterrupt(v_a_2259_);
if (v___x_2295_ == 0)
{
uint8_t v___x_2296_; 
lean_inc(v_a_2259_);
v___x_2296_ = l_Lean_Exception_isRuntime(v_a_2259_);
v___y_2264_ = v___x_2296_;
goto v___jp_2263_;
}
else
{
v___y_2264_ = v___x_2295_;
goto v___jp_2263_;
}
v___jp_2263_:
{
if (v___y_2264_ == 0)
{
lean_object* v___x_2265_; 
lean_del_object(v___x_2261_);
lean_dec(v_a_2259_);
v___x_2265_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_2248_, v___y_2264_, v___y_2237_, v___y_2238_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_);
if (lean_obj_tag(v___x_2265_) == 0)
{
lean_object* v___x_2266_; 
lean_dec_ref_known(v___x_2265_, 1);
v___x_2266_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_2246_, v___y_2264_, v___y_2237_, v___y_2238_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_, v___y_2243_);
if (lean_obj_tag(v___x_2266_) == 0)
{
lean_object* v___x_2268_; uint8_t v_isShared_2269_; uint8_t v_isSharedCheck_2274_; 
v_isSharedCheck_2274_ = !lean_is_exclusive(v___x_2266_);
if (v_isSharedCheck_2274_ == 0)
{
lean_object* v_unused_2275_; 
v_unused_2275_ = lean_ctor_get(v___x_2266_, 0);
lean_dec(v_unused_2275_);
v___x_2268_ = v___x_2266_;
v_isShared_2269_ = v_isSharedCheck_2274_;
goto v_resetjp_2267_;
}
else
{
lean_dec(v___x_2266_);
v___x_2268_ = lean_box(0);
v_isShared_2269_ = v_isSharedCheck_2274_;
goto v_resetjp_2267_;
}
v_resetjp_2267_:
{
lean_object* v___x_2270_; lean_object* v___x_2272_; 
v___x_2270_ = lean_box(0);
if (v_isShared_2269_ == 0)
{
lean_ctor_set(v___x_2268_, 0, v___x_2270_);
v___x_2272_ = v___x_2268_;
goto v_reusejp_2271_;
}
else
{
lean_object* v_reuseFailAlloc_2273_; 
v_reuseFailAlloc_2273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2273_, 0, v___x_2270_);
v___x_2272_ = v_reuseFailAlloc_2273_;
goto v_reusejp_2271_;
}
v_reusejp_2271_:
{
return v___x_2272_;
}
}
}
else
{
lean_object* v_a_2276_; lean_object* v___x_2278_; uint8_t v_isShared_2279_; uint8_t v_isSharedCheck_2283_; 
v_a_2276_ = lean_ctor_get(v___x_2266_, 0);
v_isSharedCheck_2283_ = !lean_is_exclusive(v___x_2266_);
if (v_isSharedCheck_2283_ == 0)
{
v___x_2278_ = v___x_2266_;
v_isShared_2279_ = v_isSharedCheck_2283_;
goto v_resetjp_2277_;
}
else
{
lean_inc(v_a_2276_);
lean_dec(v___x_2266_);
v___x_2278_ = lean_box(0);
v_isShared_2279_ = v_isSharedCheck_2283_;
goto v_resetjp_2277_;
}
v_resetjp_2277_:
{
lean_object* v___x_2281_; 
if (v_isShared_2279_ == 0)
{
v___x_2281_ = v___x_2278_;
goto v_reusejp_2280_;
}
else
{
lean_object* v_reuseFailAlloc_2282_; 
v_reuseFailAlloc_2282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2282_, 0, v_a_2276_);
v___x_2281_ = v_reuseFailAlloc_2282_;
goto v_reusejp_2280_;
}
v_reusejp_2280_:
{
return v___x_2281_;
}
}
}
}
else
{
lean_object* v_a_2284_; lean_object* v___x_2286_; uint8_t v_isShared_2287_; uint8_t v_isSharedCheck_2291_; 
lean_dec(v_a_2246_);
v_a_2284_ = lean_ctor_get(v___x_2265_, 0);
v_isSharedCheck_2291_ = !lean_is_exclusive(v___x_2265_);
if (v_isSharedCheck_2291_ == 0)
{
v___x_2286_ = v___x_2265_;
v_isShared_2287_ = v_isSharedCheck_2291_;
goto v_resetjp_2285_;
}
else
{
lean_inc(v_a_2284_);
lean_dec(v___x_2265_);
v___x_2286_ = lean_box(0);
v_isShared_2287_ = v_isSharedCheck_2291_;
goto v_resetjp_2285_;
}
v_resetjp_2285_:
{
lean_object* v___x_2289_; 
if (v_isShared_2287_ == 0)
{
v___x_2289_ = v___x_2286_;
goto v_reusejp_2288_;
}
else
{
lean_object* v_reuseFailAlloc_2290_; 
v_reuseFailAlloc_2290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2290_, 0, v_a_2284_);
v___x_2289_ = v_reuseFailAlloc_2290_;
goto v_reusejp_2288_;
}
v_reusejp_2288_:
{
return v___x_2289_;
}
}
}
}
else
{
lean_object* v___x_2293_; 
lean_dec(v_a_2248_);
lean_dec(v_a_2246_);
if (v_isShared_2262_ == 0)
{
v___x_2293_ = v___x_2261_;
goto v_reusejp_2292_;
}
else
{
lean_object* v_reuseFailAlloc_2294_; 
v_reuseFailAlloc_2294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2294_, 0, v_a_2259_);
v___x_2293_ = v_reuseFailAlloc_2294_;
goto v_reusejp_2292_;
}
v_reusejp_2292_:
{
return v___x_2293_;
}
}
}
}
}
}
else
{
lean_object* v_a_2298_; lean_object* v___x_2300_; uint8_t v_isShared_2301_; uint8_t v_isSharedCheck_2305_; 
lean_dec(v_a_2246_);
lean_dec_ref(v_x_2235_);
v_a_2298_ = lean_ctor_get(v___x_2247_, 0);
v_isSharedCheck_2305_ = !lean_is_exclusive(v___x_2247_);
if (v_isSharedCheck_2305_ == 0)
{
v___x_2300_ = v___x_2247_;
v_isShared_2301_ = v_isSharedCheck_2305_;
goto v_resetjp_2299_;
}
else
{
lean_inc(v_a_2298_);
lean_dec(v___x_2247_);
v___x_2300_ = lean_box(0);
v_isShared_2301_ = v_isSharedCheck_2305_;
goto v_resetjp_2299_;
}
v_resetjp_2299_:
{
lean_object* v___x_2303_; 
if (v_isShared_2301_ == 0)
{
v___x_2303_ = v___x_2300_;
goto v_reusejp_2302_;
}
else
{
lean_object* v_reuseFailAlloc_2304_; 
v_reuseFailAlloc_2304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2304_, 0, v_a_2298_);
v___x_2303_ = v_reuseFailAlloc_2304_;
goto v_reusejp_2302_;
}
v_reusejp_2302_:
{
return v___x_2303_;
}
}
}
}
else
{
lean_object* v_a_2306_; lean_object* v___x_2308_; uint8_t v_isShared_2309_; uint8_t v_isSharedCheck_2313_; 
lean_dec_ref(v_x_2235_);
v_a_2306_ = lean_ctor_get(v___x_2245_, 0);
v_isSharedCheck_2313_ = !lean_is_exclusive(v___x_2245_);
if (v_isSharedCheck_2313_ == 0)
{
v___x_2308_ = v___x_2245_;
v_isShared_2309_ = v_isSharedCheck_2313_;
goto v_resetjp_2307_;
}
else
{
lean_inc(v_a_2306_);
lean_dec(v___x_2245_);
v___x_2308_ = lean_box(0);
v_isShared_2309_ = v_isSharedCheck_2313_;
goto v_resetjp_2307_;
}
v_resetjp_2307_:
{
lean_object* v___x_2311_; 
if (v_isShared_2309_ == 0)
{
v___x_2311_ = v___x_2308_;
goto v_reusejp_2310_;
}
else
{
lean_object* v_reuseFailAlloc_2312_; 
v_reuseFailAlloc_2312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2312_, 0, v_a_2306_);
v___x_2311_ = v_reuseFailAlloc_2312_;
goto v_reusejp_2310_;
}
v_reusejp_2310_:
{
return v___x_2311_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___redArg___boxed(lean_object* v_x_2314_, lean_object* v___y_2315_, lean_object* v___y_2316_, lean_object* v___y_2317_, lean_object* v___y_2318_, lean_object* v___y_2319_, lean_object* v___y_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_){
_start:
{
lean_object* v_res_2324_; 
v_res_2324_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___redArg(v_x_2314_, v___y_2315_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_, v___y_2320_, v___y_2321_, v___y_2322_);
lean_dec(v___y_2322_);
lean_dec_ref(v___y_2321_);
lean_dec(v___y_2320_);
lean_dec_ref(v___y_2319_);
lean_dec(v___y_2318_);
lean_dec_ref(v___y_2317_);
lean_dec(v___y_2316_);
lean_dec_ref(v___y_2315_);
return v_res_2324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2(lean_object* v_00_u03b1_2325_, lean_object* v_x_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_, lean_object* v___y_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_){
_start:
{
lean_object* v___x_2336_; 
v___x_2336_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___redArg(v_x_2326_, v___y_2327_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
return v___x_2336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___boxed(lean_object* v_00_u03b1_2337_, lean_object* v_x_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_, lean_object* v___y_2343_, lean_object* v___y_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_){
_start:
{
lean_object* v_res_2348_; 
v_res_2348_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2(v_00_u03b1_2337_, v_x_2338_, v___y_2339_, v___y_2340_, v___y_2341_, v___y_2342_, v___y_2343_, v___y_2344_, v___y_2345_, v___y_2346_);
lean_dec(v___y_2346_);
lean_dec_ref(v___y_2345_);
lean_dec(v___y_2344_);
lean_dec_ref(v___y_2343_);
lean_dec(v___y_2342_);
lean_dec_ref(v___y_2341_);
lean_dec(v___y_2340_);
lean_dec_ref(v___y_2339_);
return v_res_2348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___lam__0(lean_object* v_x_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_){
_start:
{
lean_object* v___x_2359_; 
lean_inc(v___y_2353_);
lean_inc_ref(v___y_2352_);
lean_inc(v___y_2351_);
lean_inc_ref(v___y_2350_);
v___x_2359_ = lean_apply_9(v_x_2349_, v___y_2350_, v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_, v___y_2356_, v___y_2357_, lean_box(0));
return v___x_2359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___lam__0___boxed(lean_object* v_x_2360_, lean_object* v___y_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_){
_start:
{
lean_object* v_res_2370_; 
v_res_2370_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___lam__0(v_x_2360_, v___y_2361_, v___y_2362_, v___y_2363_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_, v___y_2368_);
lean_dec(v___y_2364_);
lean_dec_ref(v___y_2363_);
lean_dec(v___y_2362_);
lean_dec_ref(v___y_2361_);
return v_res_2370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg(lean_object* v_mvarId_2371_, lean_object* v_x_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_){
_start:
{
lean_object* v___f_2382_; lean_object* v___x_2383_; 
lean_inc(v___y_2376_);
lean_inc_ref(v___y_2375_);
lean_inc(v___y_2374_);
lean_inc_ref(v___y_2373_);
v___f_2382_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_2382_, 0, v_x_2372_);
lean_closure_set(v___f_2382_, 1, v___y_2373_);
lean_closure_set(v___f_2382_, 2, v___y_2374_);
lean_closure_set(v___f_2382_, 3, v___y_2375_);
lean_closure_set(v___f_2382_, 4, v___y_2376_);
v___x_2383_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2371_, v___f_2382_, v___y_2377_, v___y_2378_, v___y_2379_, v___y_2380_);
if (lean_obj_tag(v___x_2383_) == 0)
{
return v___x_2383_;
}
else
{
lean_object* v_a_2384_; lean_object* v___x_2386_; uint8_t v_isShared_2387_; uint8_t v_isSharedCheck_2391_; 
v_a_2384_ = lean_ctor_get(v___x_2383_, 0);
v_isSharedCheck_2391_ = !lean_is_exclusive(v___x_2383_);
if (v_isSharedCheck_2391_ == 0)
{
v___x_2386_ = v___x_2383_;
v_isShared_2387_ = v_isSharedCheck_2391_;
goto v_resetjp_2385_;
}
else
{
lean_inc(v_a_2384_);
lean_dec(v___x_2383_);
v___x_2386_ = lean_box(0);
v_isShared_2387_ = v_isSharedCheck_2391_;
goto v_resetjp_2385_;
}
v_resetjp_2385_:
{
lean_object* v___x_2389_; 
if (v_isShared_2387_ == 0)
{
v___x_2389_ = v___x_2386_;
goto v_reusejp_2388_;
}
else
{
lean_object* v_reuseFailAlloc_2390_; 
v_reuseFailAlloc_2390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2390_, 0, v_a_2384_);
v___x_2389_ = v_reuseFailAlloc_2390_;
goto v_reusejp_2388_;
}
v_reusejp_2388_:
{
return v___x_2389_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg___boxed(lean_object* v_mvarId_2392_, lean_object* v_x_2393_, lean_object* v___y_2394_, lean_object* v___y_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_, lean_object* v___y_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_){
_start:
{
lean_object* v_res_2403_; 
v_res_2403_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg(v_mvarId_2392_, v_x_2393_, v___y_2394_, v___y_2395_, v___y_2396_, v___y_2397_, v___y_2398_, v___y_2399_, v___y_2400_, v___y_2401_);
lean_dec(v___y_2401_);
lean_dec_ref(v___y_2400_);
lean_dec(v___y_2399_);
lean_dec_ref(v___y_2398_);
lean_dec(v___y_2397_);
lean_dec_ref(v___y_2396_);
lean_dec(v___y_2395_);
lean_dec_ref(v___y_2394_);
return v_res_2403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3(lean_object* v_00_u03b1_2404_, lean_object* v_mvarId_2405_, lean_object* v_x_2406_, lean_object* v___y_2407_, lean_object* v___y_2408_, lean_object* v___y_2409_, lean_object* v___y_2410_, lean_object* v___y_2411_, lean_object* v___y_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_){
_start:
{
lean_object* v___x_2416_; 
v___x_2416_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg(v_mvarId_2405_, v_x_2406_, v___y_2407_, v___y_2408_, v___y_2409_, v___y_2410_, v___y_2411_, v___y_2412_, v___y_2413_, v___y_2414_);
return v___x_2416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___boxed(lean_object* v_00_u03b1_2417_, lean_object* v_mvarId_2418_, lean_object* v_x_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_, lean_object* v___y_2427_, lean_object* v___y_2428_){
_start:
{
lean_object* v_res_2429_; 
v_res_2429_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3(v_00_u03b1_2417_, v_mvarId_2418_, v_x_2419_, v___y_2420_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_);
lean_dec(v___y_2427_);
lean_dec_ref(v___y_2426_);
lean_dec(v___y_2425_);
lean_dec_ref(v___y_2424_);
lean_dec(v___y_2423_);
lean_dec_ref(v___y_2422_);
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
return v_res_2429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg(lean_object* v_mvarId_2430_, lean_object* v_val_2431_, lean_object* v___y_2432_){
_start:
{
lean_object* v___x_2434_; lean_object* v_mctx_2435_; lean_object* v_cache_2436_; lean_object* v_zetaDeltaFVarIds_2437_; lean_object* v_postponed_2438_; lean_object* v_diag_2439_; lean_object* v___x_2441_; uint8_t v_isShared_2442_; uint8_t v_isSharedCheck_2467_; 
v___x_2434_ = lean_st_ref_take(v___y_2432_);
v_mctx_2435_ = lean_ctor_get(v___x_2434_, 0);
v_cache_2436_ = lean_ctor_get(v___x_2434_, 1);
v_zetaDeltaFVarIds_2437_ = lean_ctor_get(v___x_2434_, 2);
v_postponed_2438_ = lean_ctor_get(v___x_2434_, 3);
v_diag_2439_ = lean_ctor_get(v___x_2434_, 4);
v_isSharedCheck_2467_ = !lean_is_exclusive(v___x_2434_);
if (v_isSharedCheck_2467_ == 0)
{
v___x_2441_ = v___x_2434_;
v_isShared_2442_ = v_isSharedCheck_2467_;
goto v_resetjp_2440_;
}
else
{
lean_inc(v_diag_2439_);
lean_inc(v_postponed_2438_);
lean_inc(v_zetaDeltaFVarIds_2437_);
lean_inc(v_cache_2436_);
lean_inc(v_mctx_2435_);
lean_dec(v___x_2434_);
v___x_2441_ = lean_box(0);
v_isShared_2442_ = v_isSharedCheck_2467_;
goto v_resetjp_2440_;
}
v_resetjp_2440_:
{
lean_object* v_depth_2443_; lean_object* v_levelAssignDepth_2444_; lean_object* v_lmvarCounter_2445_; lean_object* v_mvarCounter_2446_; lean_object* v_lDecls_2447_; lean_object* v_decls_2448_; lean_object* v_userNames_2449_; lean_object* v_lAssignment_2450_; lean_object* v_eAssignment_2451_; lean_object* v_dAssignment_2452_; lean_object* v___x_2454_; uint8_t v_isShared_2455_; uint8_t v_isSharedCheck_2466_; 
v_depth_2443_ = lean_ctor_get(v_mctx_2435_, 0);
v_levelAssignDepth_2444_ = lean_ctor_get(v_mctx_2435_, 1);
v_lmvarCounter_2445_ = lean_ctor_get(v_mctx_2435_, 2);
v_mvarCounter_2446_ = lean_ctor_get(v_mctx_2435_, 3);
v_lDecls_2447_ = lean_ctor_get(v_mctx_2435_, 4);
v_decls_2448_ = lean_ctor_get(v_mctx_2435_, 5);
v_userNames_2449_ = lean_ctor_get(v_mctx_2435_, 6);
v_lAssignment_2450_ = lean_ctor_get(v_mctx_2435_, 7);
v_eAssignment_2451_ = lean_ctor_get(v_mctx_2435_, 8);
v_dAssignment_2452_ = lean_ctor_get(v_mctx_2435_, 9);
v_isSharedCheck_2466_ = !lean_is_exclusive(v_mctx_2435_);
if (v_isSharedCheck_2466_ == 0)
{
v___x_2454_ = v_mctx_2435_;
v_isShared_2455_ = v_isSharedCheck_2466_;
goto v_resetjp_2453_;
}
else
{
lean_inc(v_dAssignment_2452_);
lean_inc(v_eAssignment_2451_);
lean_inc(v_lAssignment_2450_);
lean_inc(v_userNames_2449_);
lean_inc(v_decls_2448_);
lean_inc(v_lDecls_2447_);
lean_inc(v_mvarCounter_2446_);
lean_inc(v_lmvarCounter_2445_);
lean_inc(v_levelAssignDepth_2444_);
lean_inc(v_depth_2443_);
lean_dec(v_mctx_2435_);
v___x_2454_ = lean_box(0);
v_isShared_2455_ = v_isSharedCheck_2466_;
goto v_resetjp_2453_;
}
v_resetjp_2453_:
{
lean_object* v___x_2456_; lean_object* v___x_2458_; 
v___x_2456_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_applyTheConstructor_spec__2_spec__3___redArg(v_eAssignment_2451_, v_mvarId_2430_, v_val_2431_);
if (v_isShared_2455_ == 0)
{
lean_ctor_set(v___x_2454_, 8, v___x_2456_);
v___x_2458_ = v___x_2454_;
goto v_reusejp_2457_;
}
else
{
lean_object* v_reuseFailAlloc_2465_; 
v_reuseFailAlloc_2465_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2465_, 0, v_depth_2443_);
lean_ctor_set(v_reuseFailAlloc_2465_, 1, v_levelAssignDepth_2444_);
lean_ctor_set(v_reuseFailAlloc_2465_, 2, v_lmvarCounter_2445_);
lean_ctor_set(v_reuseFailAlloc_2465_, 3, v_mvarCounter_2446_);
lean_ctor_set(v_reuseFailAlloc_2465_, 4, v_lDecls_2447_);
lean_ctor_set(v_reuseFailAlloc_2465_, 5, v_decls_2448_);
lean_ctor_set(v_reuseFailAlloc_2465_, 6, v_userNames_2449_);
lean_ctor_set(v_reuseFailAlloc_2465_, 7, v_lAssignment_2450_);
lean_ctor_set(v_reuseFailAlloc_2465_, 8, v___x_2456_);
lean_ctor_set(v_reuseFailAlloc_2465_, 9, v_dAssignment_2452_);
v___x_2458_ = v_reuseFailAlloc_2465_;
goto v_reusejp_2457_;
}
v_reusejp_2457_:
{
lean_object* v___x_2460_; 
if (v_isShared_2442_ == 0)
{
lean_ctor_set(v___x_2441_, 0, v___x_2458_);
v___x_2460_ = v___x_2441_;
goto v_reusejp_2459_;
}
else
{
lean_object* v_reuseFailAlloc_2464_; 
v_reuseFailAlloc_2464_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2464_, 0, v___x_2458_);
lean_ctor_set(v_reuseFailAlloc_2464_, 1, v_cache_2436_);
lean_ctor_set(v_reuseFailAlloc_2464_, 2, v_zetaDeltaFVarIds_2437_);
lean_ctor_set(v_reuseFailAlloc_2464_, 3, v_postponed_2438_);
lean_ctor_set(v_reuseFailAlloc_2464_, 4, v_diag_2439_);
v___x_2460_ = v_reuseFailAlloc_2464_;
goto v_reusejp_2459_;
}
v_reusejp_2459_:
{
lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v___x_2463_; 
v___x_2461_ = lean_st_ref_set(v___y_2432_, v___x_2460_);
v___x_2462_ = lean_box(0);
v___x_2463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2463_, 0, v___x_2462_);
return v___x_2463_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg___boxed(lean_object* v_mvarId_2468_, lean_object* v_val_2469_, lean_object* v___y_2470_, lean_object* v___y_2471_){
_start:
{
lean_object* v_res_2472_; 
v_res_2472_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg(v_mvarId_2468_, v_val_2469_, v___y_2470_);
lean_dec(v___y_2470_);
return v_res_2472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___lam__0(lean_object* v_head_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_){
_start:
{
lean_object* v___x_2483_; 
lean_inc(v_head_2473_);
v___x_2483_ = l_Lean_MVarId_getType(v_head_2473_, v___y_2478_, v___y_2479_, v___y_2480_, v___y_2481_);
if (lean_obj_tag(v___x_2483_) == 0)
{
lean_object* v_a_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; 
v_a_2484_ = lean_ctor_get(v___x_2483_, 0);
lean_inc(v_a_2484_);
lean_dec_ref_known(v___x_2483_, 1);
v___x_2485_ = lean_box(0);
v___x_2486_ = l_Lean_Meta_synthInstance(v_a_2484_, v___x_2485_, v___y_2478_, v___y_2479_, v___y_2480_, v___y_2481_);
if (lean_obj_tag(v___x_2486_) == 0)
{
lean_object* v_a_2487_; lean_object* v___x_2488_; 
v_a_2487_ = lean_ctor_get(v___x_2486_, 0);
lean_inc(v_a_2487_);
lean_dec_ref_known(v___x_2486_, 1);
v___x_2488_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg(v_head_2473_, v_a_2487_, v___y_2479_);
return v___x_2488_;
}
else
{
lean_object* v_a_2489_; lean_object* v___x_2491_; uint8_t v_isShared_2492_; uint8_t v_isSharedCheck_2496_; 
lean_dec(v_head_2473_);
v_a_2489_ = lean_ctor_get(v___x_2486_, 0);
v_isSharedCheck_2496_ = !lean_is_exclusive(v___x_2486_);
if (v_isSharedCheck_2496_ == 0)
{
v___x_2491_ = v___x_2486_;
v_isShared_2492_ = v_isSharedCheck_2496_;
goto v_resetjp_2490_;
}
else
{
lean_inc(v_a_2489_);
lean_dec(v___x_2486_);
v___x_2491_ = lean_box(0);
v_isShared_2492_ = v_isSharedCheck_2496_;
goto v_resetjp_2490_;
}
v_resetjp_2490_:
{
lean_object* v___x_2494_; 
if (v_isShared_2492_ == 0)
{
v___x_2494_ = v___x_2491_;
goto v_reusejp_2493_;
}
else
{
lean_object* v_reuseFailAlloc_2495_; 
v_reuseFailAlloc_2495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2495_, 0, v_a_2489_);
v___x_2494_ = v_reuseFailAlloc_2495_;
goto v_reusejp_2493_;
}
v_reusejp_2493_:
{
return v___x_2494_;
}
}
}
}
else
{
lean_object* v_a_2497_; lean_object* v___x_2499_; uint8_t v_isShared_2500_; uint8_t v_isSharedCheck_2504_; 
lean_dec(v_head_2473_);
v_a_2497_ = lean_ctor_get(v___x_2483_, 0);
v_isSharedCheck_2504_ = !lean_is_exclusive(v___x_2483_);
if (v_isSharedCheck_2504_ == 0)
{
v___x_2499_ = v___x_2483_;
v_isShared_2500_ = v_isSharedCheck_2504_;
goto v_resetjp_2498_;
}
else
{
lean_inc(v_a_2497_);
lean_dec(v___x_2483_);
v___x_2499_ = lean_box(0);
v_isShared_2500_ = v_isSharedCheck_2504_;
goto v_resetjp_2498_;
}
v_resetjp_2498_:
{
lean_object* v___x_2502_; 
if (v_isShared_2500_ == 0)
{
v___x_2502_ = v___x_2499_;
goto v_reusejp_2501_;
}
else
{
lean_object* v_reuseFailAlloc_2503_; 
v_reuseFailAlloc_2503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2503_, 0, v_a_2497_);
v___x_2502_ = v_reuseFailAlloc_2503_;
goto v_reusejp_2501_;
}
v_reusejp_2501_:
{
return v___x_2502_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___lam__0___boxed(lean_object* v_head_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_){
_start:
{
lean_object* v_res_2515_; 
v_res_2515_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___lam__0(v_head_2505_, v___y_2506_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_, v___y_2511_, v___y_2512_, v___y_2513_);
lean_dec(v___y_2513_);
lean_dec_ref(v___y_2512_);
lean_dec(v___y_2511_);
lean_dec_ref(v___y_2510_);
lean_dec(v___y_2509_);
lean_dec_ref(v___y_2508_);
lean_dec(v___y_2507_);
lean_dec_ref(v___y_2506_);
return v_res_2515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg(lean_object* v_mvarId_2516_, lean_object* v___y_2517_){
_start:
{
lean_object* v___x_2519_; lean_object* v_mctx_2520_; lean_object* v_eAssignment_2521_; uint8_t v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; 
v___x_2519_ = lean_st_ref_get(v___y_2517_);
v_mctx_2520_ = lean_ctor_get(v___x_2519_, 0);
lean_inc_ref(v_mctx_2520_);
lean_dec(v___x_2519_);
v_eAssignment_2521_ = lean_ctor_get(v_mctx_2520_, 8);
lean_inc_ref(v_eAssignment_2521_);
lean_dec_ref(v_mctx_2520_);
v___x_2522_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssigned___at___00Mathlib_Tactic_useLoop_spec__2_spec__3___redArg(v_eAssignment_2521_, v_mvarId_2516_);
lean_dec_ref(v_eAssignment_2521_);
v___x_2523_ = lean_box(v___x_2522_);
v___x_2524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2524_, 0, v___x_2523_);
return v___x_2524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg___boxed(lean_object* v_mvarId_2525_, lean_object* v___y_2526_, lean_object* v___y_2527_){
_start:
{
lean_object* v_res_2528_; 
v_res_2528_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg(v_mvarId_2525_, v___y_2526_);
lean_dec(v___y_2526_);
lean_dec(v_mvarId_2525_);
return v_res_2528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg(lean_object* v_as_x27_2529_, lean_object* v_b_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_, lean_object* v___y_2533_, lean_object* v___y_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_){
_start:
{
if (lean_obj_tag(v_as_x27_2529_) == 0)
{
lean_object* v___x_2540_; 
v___x_2540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2540_, 0, v_b_2530_);
return v___x_2540_;
}
else
{
lean_object* v_head_2541_; lean_object* v_tail_2542_; lean_object* v___x_2543_; lean_object* v_a_2544_; lean_object* v___x_2545_; uint8_t v___x_2546_; 
v_head_2541_ = lean_ctor_get(v_as_x27_2529_, 0);
v_tail_2542_ = lean_ctor_get(v_as_x27_2529_, 1);
v___x_2543_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg(v_head_2541_, v___y_2536_);
v_a_2544_ = lean_ctor_get(v___x_2543_, 0);
lean_inc(v_a_2544_);
lean_dec_ref(v___x_2543_);
v___x_2545_ = lean_box(0);
v___x_2546_ = lean_unbox(v_a_2544_);
lean_dec(v_a_2544_);
if (v___x_2546_ == 0)
{
lean_object* v___f_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; 
lean_inc_n(v_head_2541_, 2);
v___f_2547_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_2547_, 0, v_head_2541_);
v___x_2548_ = lean_alloc_closure((void*)(lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_runUse_spec__2___boxed), 11, 2);
lean_closure_set(v___x_2548_, 0, lean_box(0));
lean_closure_set(v___x_2548_, 1, v___f_2547_);
v___x_2549_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg(v_head_2541_, v___x_2548_, v___y_2531_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_, v___y_2537_, v___y_2538_);
if (lean_obj_tag(v___x_2549_) == 0)
{
lean_dec_ref_known(v___x_2549_, 1);
v_as_x27_2529_ = v_tail_2542_;
v_b_2530_ = v___x_2545_;
goto _start;
}
else
{
lean_object* v_a_2551_; lean_object* v___x_2553_; uint8_t v_isShared_2554_; uint8_t v_isSharedCheck_2558_; 
v_a_2551_ = lean_ctor_get(v___x_2549_, 0);
v_isSharedCheck_2558_ = !lean_is_exclusive(v___x_2549_);
if (v_isSharedCheck_2558_ == 0)
{
v___x_2553_ = v___x_2549_;
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
else
{
lean_inc(v_a_2551_);
lean_dec(v___x_2549_);
v___x_2553_ = lean_box(0);
v_isShared_2554_ = v_isSharedCheck_2558_;
goto v_resetjp_2552_;
}
v_resetjp_2552_:
{
lean_object* v___x_2556_; 
if (v_isShared_2554_ == 0)
{
v___x_2556_ = v___x_2553_;
goto v_reusejp_2555_;
}
else
{
lean_object* v_reuseFailAlloc_2557_; 
v_reuseFailAlloc_2557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2557_, 0, v_a_2551_);
v___x_2556_ = v_reuseFailAlloc_2557_;
goto v_reusejp_2555_;
}
v_reusejp_2555_:
{
return v___x_2556_;
}
}
}
}
else
{
v_as_x27_2529_ = v_tail_2542_;
v_b_2530_ = v___x_2545_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg___boxed(lean_object* v_as_x27_2560_, lean_object* v_b_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_, lean_object* v___y_2569_, lean_object* v___y_2570_){
_start:
{
lean_object* v_res_2571_; 
v_res_2571_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg(v_as_x27_2560_, v_b_2561_, v___y_2562_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_, v___y_2569_);
lean_dec(v___y_2569_);
lean_dec_ref(v___y_2568_);
lean_dec(v___y_2567_);
lean_dec_ref(v___y_2566_);
lean_dec(v___y_2565_);
lean_dec_ref(v___y_2564_);
lean_dec(v___y_2563_);
lean_dec_ref(v___y_2562_);
lean_dec(v_as_x27_2560_);
return v_res_2571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse___lam__0(uint8_t v_eager_2572_, lean_object* v_args_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_, lean_object* v___y_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_){
_start:
{
lean_object* v___x_2583_; 
v___x_2583_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_2575_);
if (lean_obj_tag(v___x_2583_) == 0)
{
lean_object* v_a_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; 
v_a_2584_ = lean_ctor_get(v___x_2583_, 0);
lean_inc(v_a_2584_);
lean_dec_ref_known(v___x_2583_, 1);
v___x_2585_ = lean_box(0);
v___x_2586_ = lp_mathlib_Mathlib_Tactic_useLoop(v_eager_2572_, v_a_2584_, v_args_2573_, v___x_2585_, v___x_2585_, v___y_2576_, v___y_2577_, v___y_2578_, v___y_2579_, v___y_2580_, v___y_2581_);
if (lean_obj_tag(v___x_2586_) == 0)
{
lean_object* v_a_2587_; lean_object* v_snd_2588_; lean_object* v_fst_2589_; lean_object* v_fst_2590_; lean_object* v_snd_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; 
v_a_2587_ = lean_ctor_get(v___x_2586_, 0);
lean_inc(v_a_2587_);
lean_dec_ref_known(v___x_2586_, 1);
v_snd_2588_ = lean_ctor_get(v_a_2587_, 1);
lean_inc(v_snd_2588_);
v_fst_2589_ = lean_ctor_get(v_a_2587_, 0);
lean_inc(v_fst_2589_);
lean_dec(v_a_2587_);
v_fst_2590_ = lean_ctor_get(v_snd_2588_, 0);
lean_inc(v_fst_2590_);
v_snd_2591_ = lean_ctor_get(v_snd_2588_, 1);
lean_inc(v_snd_2591_);
lean_dec(v_snd_2588_);
v___x_2592_ = lean_box(0);
v___x_2593_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg(v_snd_2591_, v___x_2592_, v___y_2574_, v___y_2575_, v___y_2576_, v___y_2577_, v___y_2578_, v___y_2579_, v___y_2580_, v___y_2581_);
lean_dec(v_snd_2591_);
if (lean_obj_tag(v___x_2593_) == 0)
{
lean_object* v___x_2594_; lean_object* v___x_2595_; 
lean_dec_ref_known(v___x_2593_, 1);
lean_inc(v_fst_2589_);
v___x_2594_ = l_List_appendTR___redArg(v_fst_2589_, v_fst_2590_);
v___x_2595_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_2594_, v___y_2575_);
if (lean_obj_tag(v___x_2595_) == 0)
{
lean_object* v___x_2596_; 
lean_dec_ref_known(v___x_2595_, 1);
v___x_2596_ = l_Lean_Elab_Tactic_pruneSolvedGoals(v___y_2574_, v___y_2575_, v___y_2576_, v___y_2577_, v___y_2578_, v___y_2579_, v___y_2580_, v___y_2581_);
if (lean_obj_tag(v___x_2596_) == 0)
{
lean_object* v___x_2598_; uint8_t v_isShared_2599_; uint8_t v_isSharedCheck_2603_; 
v_isSharedCheck_2603_ = !lean_is_exclusive(v___x_2596_);
if (v_isSharedCheck_2603_ == 0)
{
lean_object* v_unused_2604_; 
v_unused_2604_ = lean_ctor_get(v___x_2596_, 0);
lean_dec(v_unused_2604_);
v___x_2598_ = v___x_2596_;
v_isShared_2599_ = v_isSharedCheck_2603_;
goto v_resetjp_2597_;
}
else
{
lean_dec(v___x_2596_);
v___x_2598_ = lean_box(0);
v_isShared_2599_ = v_isSharedCheck_2603_;
goto v_resetjp_2597_;
}
v_resetjp_2597_:
{
lean_object* v___x_2601_; 
if (v_isShared_2599_ == 0)
{
lean_ctor_set(v___x_2598_, 0, v_fst_2589_);
v___x_2601_ = v___x_2598_;
goto v_reusejp_2600_;
}
else
{
lean_object* v_reuseFailAlloc_2602_; 
v_reuseFailAlloc_2602_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2602_, 0, v_fst_2589_);
v___x_2601_ = v_reuseFailAlloc_2602_;
goto v_reusejp_2600_;
}
v_reusejp_2600_:
{
return v___x_2601_;
}
}
}
else
{
lean_object* v_a_2605_; lean_object* v___x_2607_; uint8_t v_isShared_2608_; uint8_t v_isSharedCheck_2612_; 
lean_dec(v_fst_2589_);
v_a_2605_ = lean_ctor_get(v___x_2596_, 0);
v_isSharedCheck_2612_ = !lean_is_exclusive(v___x_2596_);
if (v_isSharedCheck_2612_ == 0)
{
v___x_2607_ = v___x_2596_;
v_isShared_2608_ = v_isSharedCheck_2612_;
goto v_resetjp_2606_;
}
else
{
lean_inc(v_a_2605_);
lean_dec(v___x_2596_);
v___x_2607_ = lean_box(0);
v_isShared_2608_ = v_isSharedCheck_2612_;
goto v_resetjp_2606_;
}
v_resetjp_2606_:
{
lean_object* v___x_2610_; 
if (v_isShared_2608_ == 0)
{
v___x_2610_ = v___x_2607_;
goto v_reusejp_2609_;
}
else
{
lean_object* v_reuseFailAlloc_2611_; 
v_reuseFailAlloc_2611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2611_, 0, v_a_2605_);
v___x_2610_ = v_reuseFailAlloc_2611_;
goto v_reusejp_2609_;
}
v_reusejp_2609_:
{
return v___x_2610_;
}
}
}
}
else
{
lean_object* v_a_2613_; lean_object* v___x_2615_; uint8_t v_isShared_2616_; uint8_t v_isSharedCheck_2620_; 
lean_dec(v_fst_2589_);
v_a_2613_ = lean_ctor_get(v___x_2595_, 0);
v_isSharedCheck_2620_ = !lean_is_exclusive(v___x_2595_);
if (v_isSharedCheck_2620_ == 0)
{
v___x_2615_ = v___x_2595_;
v_isShared_2616_ = v_isSharedCheck_2620_;
goto v_resetjp_2614_;
}
else
{
lean_inc(v_a_2613_);
lean_dec(v___x_2595_);
v___x_2615_ = lean_box(0);
v_isShared_2616_ = v_isSharedCheck_2620_;
goto v_resetjp_2614_;
}
v_resetjp_2614_:
{
lean_object* v___x_2618_; 
if (v_isShared_2616_ == 0)
{
v___x_2618_ = v___x_2615_;
goto v_reusejp_2617_;
}
else
{
lean_object* v_reuseFailAlloc_2619_; 
v_reuseFailAlloc_2619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2619_, 0, v_a_2613_);
v___x_2618_ = v_reuseFailAlloc_2619_;
goto v_reusejp_2617_;
}
v_reusejp_2617_:
{
return v___x_2618_;
}
}
}
}
else
{
lean_object* v_a_2621_; lean_object* v___x_2623_; uint8_t v_isShared_2624_; uint8_t v_isSharedCheck_2628_; 
lean_dec(v_fst_2590_);
lean_dec(v_fst_2589_);
v_a_2621_ = lean_ctor_get(v___x_2593_, 0);
v_isSharedCheck_2628_ = !lean_is_exclusive(v___x_2593_);
if (v_isSharedCheck_2628_ == 0)
{
v___x_2623_ = v___x_2593_;
v_isShared_2624_ = v_isSharedCheck_2628_;
goto v_resetjp_2622_;
}
else
{
lean_inc(v_a_2621_);
lean_dec(v___x_2593_);
v___x_2623_ = lean_box(0);
v_isShared_2624_ = v_isSharedCheck_2628_;
goto v_resetjp_2622_;
}
v_resetjp_2622_:
{
lean_object* v___x_2626_; 
if (v_isShared_2624_ == 0)
{
v___x_2626_ = v___x_2623_;
goto v_reusejp_2625_;
}
else
{
lean_object* v_reuseFailAlloc_2627_; 
v_reuseFailAlloc_2627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2627_, 0, v_a_2621_);
v___x_2626_ = v_reuseFailAlloc_2627_;
goto v_reusejp_2625_;
}
v_reusejp_2625_:
{
return v___x_2626_;
}
}
}
}
else
{
lean_object* v_a_2629_; lean_object* v___x_2631_; uint8_t v_isShared_2632_; uint8_t v_isSharedCheck_2636_; 
v_a_2629_ = lean_ctor_get(v___x_2586_, 0);
v_isSharedCheck_2636_ = !lean_is_exclusive(v___x_2586_);
if (v_isSharedCheck_2636_ == 0)
{
v___x_2631_ = v___x_2586_;
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
else
{
lean_inc(v_a_2629_);
lean_dec(v___x_2586_);
v___x_2631_ = lean_box(0);
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
v_resetjp_2630_:
{
lean_object* v___x_2634_; 
if (v_isShared_2632_ == 0)
{
v___x_2634_ = v___x_2631_;
goto v_reusejp_2633_;
}
else
{
lean_object* v_reuseFailAlloc_2635_; 
v_reuseFailAlloc_2635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2635_, 0, v_a_2629_);
v___x_2634_ = v_reuseFailAlloc_2635_;
goto v_reusejp_2633_;
}
v_reusejp_2633_:
{
return v___x_2634_;
}
}
}
}
else
{
lean_dec(v_args_2573_);
return v___x_2583_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse___lam__0___boxed(lean_object* v_eager_2637_, lean_object* v_args_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_, lean_object* v___y_2641_, lean_object* v___y_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_, lean_object* v___y_2645_, lean_object* v___y_2646_, lean_object* v___y_2647_){
_start:
{
uint8_t v_eager_boxed_2648_; lean_object* v_res_2649_; 
v_eager_boxed_2648_ = lean_unbox(v_eager_2637_);
v_res_2649_ = lp_mathlib_Mathlib_Tactic_runUse___lam__0(v_eager_boxed_2648_, v_args_2638_, v___y_2639_, v___y_2640_, v___y_2641_, v___y_2642_, v___y_2643_, v___y_2644_, v___y_2645_, v___y_2646_);
lean_dec(v___y_2646_);
lean_dec_ref(v___y_2645_);
lean_dec(v___y_2644_);
lean_dec_ref(v___y_2643_);
lean_dec(v___y_2642_);
lean_dec_ref(v___y_2641_);
lean_dec(v___y_2640_);
lean_dec_ref(v___y_2639_);
return v_res_2649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg(lean_object* v_cls_2650_, lean_object* v_msg_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_, lean_object* v___y_2655_){
_start:
{
lean_object* v_ref_2657_; lean_object* v___x_2658_; lean_object* v_a_2659_; lean_object* v___x_2661_; uint8_t v_isShared_2662_; uint8_t v_isSharedCheck_2703_; 
v_ref_2657_ = lean_ctor_get(v___y_2654_, 5);
v___x_2658_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyTheConstructor_spec__3_spec__5(v_msg_2651_, v___y_2652_, v___y_2653_, v___y_2654_, v___y_2655_);
v_a_2659_ = lean_ctor_get(v___x_2658_, 0);
v_isSharedCheck_2703_ = !lean_is_exclusive(v___x_2658_);
if (v_isSharedCheck_2703_ == 0)
{
v___x_2661_ = v___x_2658_;
v_isShared_2662_ = v_isSharedCheck_2703_;
goto v_resetjp_2660_;
}
else
{
lean_inc(v_a_2659_);
lean_dec(v___x_2658_);
v___x_2661_ = lean_box(0);
v_isShared_2662_ = v_isSharedCheck_2703_;
goto v_resetjp_2660_;
}
v_resetjp_2660_:
{
lean_object* v___x_2663_; lean_object* v_traceState_2664_; lean_object* v_env_2665_; lean_object* v_nextMacroScope_2666_; lean_object* v_ngen_2667_; lean_object* v_auxDeclNGen_2668_; lean_object* v_cache_2669_; lean_object* v_messages_2670_; lean_object* v_infoState_2671_; lean_object* v_snapshotTasks_2672_; lean_object* v___x_2674_; uint8_t v_isShared_2675_; uint8_t v_isSharedCheck_2702_; 
v___x_2663_ = lean_st_ref_take(v___y_2655_);
v_traceState_2664_ = lean_ctor_get(v___x_2663_, 4);
v_env_2665_ = lean_ctor_get(v___x_2663_, 0);
v_nextMacroScope_2666_ = lean_ctor_get(v___x_2663_, 1);
v_ngen_2667_ = lean_ctor_get(v___x_2663_, 2);
v_auxDeclNGen_2668_ = lean_ctor_get(v___x_2663_, 3);
v_cache_2669_ = lean_ctor_get(v___x_2663_, 5);
v_messages_2670_ = lean_ctor_get(v___x_2663_, 6);
v_infoState_2671_ = lean_ctor_get(v___x_2663_, 7);
v_snapshotTasks_2672_ = lean_ctor_get(v___x_2663_, 8);
v_isSharedCheck_2702_ = !lean_is_exclusive(v___x_2663_);
if (v_isSharedCheck_2702_ == 0)
{
v___x_2674_ = v___x_2663_;
v_isShared_2675_ = v_isSharedCheck_2702_;
goto v_resetjp_2673_;
}
else
{
lean_inc(v_snapshotTasks_2672_);
lean_inc(v_infoState_2671_);
lean_inc(v_messages_2670_);
lean_inc(v_cache_2669_);
lean_inc(v_traceState_2664_);
lean_inc(v_auxDeclNGen_2668_);
lean_inc(v_ngen_2667_);
lean_inc(v_nextMacroScope_2666_);
lean_inc(v_env_2665_);
lean_dec(v___x_2663_);
v___x_2674_ = lean_box(0);
v_isShared_2675_ = v_isSharedCheck_2702_;
goto v_resetjp_2673_;
}
v_resetjp_2673_:
{
uint64_t v_tid_2676_; lean_object* v_traces_2677_; lean_object* v___x_2679_; uint8_t v_isShared_2680_; uint8_t v_isSharedCheck_2701_; 
v_tid_2676_ = lean_ctor_get_uint64(v_traceState_2664_, sizeof(void*)*1);
v_traces_2677_ = lean_ctor_get(v_traceState_2664_, 0);
v_isSharedCheck_2701_ = !lean_is_exclusive(v_traceState_2664_);
if (v_isSharedCheck_2701_ == 0)
{
v___x_2679_ = v_traceState_2664_;
v_isShared_2680_ = v_isSharedCheck_2701_;
goto v_resetjp_2678_;
}
else
{
lean_inc(v_traces_2677_);
lean_dec(v_traceState_2664_);
v___x_2679_ = lean_box(0);
v_isShared_2680_ = v_isSharedCheck_2701_;
goto v_resetjp_2678_;
}
v_resetjp_2678_:
{
lean_object* v___x_2681_; double v___x_2682_; uint8_t v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2691_; 
v___x_2681_ = lean_box(0);
v___x_2682_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__0);
v___x_2683_ = 0;
v___x_2684_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__1));
v___x_2685_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2685_, 0, v_cls_2650_);
lean_ctor_set(v___x_2685_, 1, v___x_2681_);
lean_ctor_set(v___x_2685_, 2, v___x_2684_);
lean_ctor_set_float(v___x_2685_, sizeof(void*)*3, v___x_2682_);
lean_ctor_set_float(v___x_2685_, sizeof(void*)*3 + 8, v___x_2682_);
lean_ctor_set_uint8(v___x_2685_, sizeof(void*)*3 + 16, v___x_2683_);
v___x_2686_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_useLoop_spec__1___redArg___closed__2));
v___x_2687_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2687_, 0, v___x_2685_);
lean_ctor_set(v___x_2687_, 1, v_a_2659_);
lean_ctor_set(v___x_2687_, 2, v___x_2686_);
lean_inc(v_ref_2657_);
v___x_2688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2688_, 0, v_ref_2657_);
lean_ctor_set(v___x_2688_, 1, v___x_2687_);
v___x_2689_ = l_Lean_PersistentArray_push___redArg(v_traces_2677_, v___x_2688_);
if (v_isShared_2680_ == 0)
{
lean_ctor_set(v___x_2679_, 0, v___x_2689_);
v___x_2691_ = v___x_2679_;
goto v_reusejp_2690_;
}
else
{
lean_object* v_reuseFailAlloc_2700_; 
v_reuseFailAlloc_2700_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2700_, 0, v___x_2689_);
lean_ctor_set_uint64(v_reuseFailAlloc_2700_, sizeof(void*)*1, v_tid_2676_);
v___x_2691_ = v_reuseFailAlloc_2700_;
goto v_reusejp_2690_;
}
v_reusejp_2690_:
{
lean_object* v___x_2693_; 
if (v_isShared_2675_ == 0)
{
lean_ctor_set(v___x_2674_, 4, v___x_2691_);
v___x_2693_ = v___x_2674_;
goto v_reusejp_2692_;
}
else
{
lean_object* v_reuseFailAlloc_2699_; 
v_reuseFailAlloc_2699_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2699_, 0, v_env_2665_);
lean_ctor_set(v_reuseFailAlloc_2699_, 1, v_nextMacroScope_2666_);
lean_ctor_set(v_reuseFailAlloc_2699_, 2, v_ngen_2667_);
lean_ctor_set(v_reuseFailAlloc_2699_, 3, v_auxDeclNGen_2668_);
lean_ctor_set(v_reuseFailAlloc_2699_, 4, v___x_2691_);
lean_ctor_set(v_reuseFailAlloc_2699_, 5, v_cache_2669_);
lean_ctor_set(v_reuseFailAlloc_2699_, 6, v_messages_2670_);
lean_ctor_set(v_reuseFailAlloc_2699_, 7, v_infoState_2671_);
lean_ctor_set(v_reuseFailAlloc_2699_, 8, v_snapshotTasks_2672_);
v___x_2693_ = v_reuseFailAlloc_2699_;
goto v_reusejp_2692_;
}
v_reusejp_2692_:
{
lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2697_; 
v___x_2694_ = lean_st_ref_set(v___y_2655_, v___x_2693_);
v___x_2695_ = lean_box(0);
if (v_isShared_2662_ == 0)
{
lean_ctor_set(v___x_2661_, 0, v___x_2695_);
v___x_2697_ = v___x_2661_;
goto v_reusejp_2696_;
}
else
{
lean_object* v_reuseFailAlloc_2698_; 
v_reuseFailAlloc_2698_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2698_, 0, v___x_2695_);
v___x_2697_ = v_reuseFailAlloc_2698_;
goto v_reusejp_2696_;
}
v_reusejp_2696_:
{
return v___x_2697_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg___boxed(lean_object* v_cls_2704_, lean_object* v_msg_2705_, lean_object* v___y_2706_, lean_object* v___y_2707_, lean_object* v___y_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_){
_start:
{
lean_object* v_res_2711_; 
v_res_2711_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg(v_cls_2704_, v_msg_2705_, v___y_2706_, v___y_2707_, v___y_2708_, v___y_2709_);
lean_dec(v___y_2709_);
lean_dec_ref(v___y_2708_);
lean_dec(v___y_2707_);
lean_dec_ref(v___y_2706_);
return v_res_2711_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; 
v___x_2712_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_));
v___x_2713_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__0___closed__1));
v___x_2714_ = l_Lean_Name_append(v___x_2713_, v___x_2712_);
return v___x_2714_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__2(void){
_start:
{
lean_object* v___x_2716_; lean_object* v___x_2717_; 
v___x_2716_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__1));
v___x_2717_ = l_Lean_stringToMessageData(v___x_2716_);
return v___x_2717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0(lean_object* v_head_2718_, lean_object* v___x_2719_, lean_object* v_discharger_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_){
_start:
{
lean_object* v___y_2731_; lean_object* v___y_2732_; lean_object* v___y_2733_; lean_object* v___y_2734_; lean_object* v___y_2735_; lean_object* v___y_2736_; lean_object* v___x_2754_; 
lean_inc(v_head_2718_);
v___x_2754_ = l_Lean_MVarId_getType(v_head_2718_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_);
if (lean_obj_tag(v___x_2754_) == 0)
{
lean_object* v_a_2755_; lean_object* v___x_2756_; 
v_a_2755_ = lean_ctor_get(v___x_2754_, 0);
lean_inc(v_a_2755_);
lean_dec_ref_known(v___x_2754_, 1);
v___x_2756_ = l_Lean_Meta_isProp(v_a_2755_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_);
if (lean_obj_tag(v___x_2756_) == 0)
{
lean_object* v_a_2757_; lean_object* v___x_2759_; uint8_t v_isShared_2760_; uint8_t v_isSharedCheck_2775_; 
v_a_2757_ = lean_ctor_get(v___x_2756_, 0);
v_isSharedCheck_2775_ = !lean_is_exclusive(v___x_2756_);
if (v_isSharedCheck_2775_ == 0)
{
v___x_2759_ = v___x_2756_;
v_isShared_2760_ = v_isSharedCheck_2775_;
goto v_resetjp_2758_;
}
else
{
lean_inc(v_a_2757_);
lean_dec(v___x_2756_);
v___x_2759_ = lean_box(0);
v_isShared_2760_ = v_isSharedCheck_2775_;
goto v_resetjp_2758_;
}
v_resetjp_2758_:
{
uint8_t v___x_2761_; 
v___x_2761_ = lean_unbox(v_a_2757_);
lean_dec(v_a_2757_);
if (v___x_2761_ == 0)
{
lean_object* v___x_2763_; 
lean_dec_ref(v_discharger_2720_);
lean_dec(v_head_2718_);
if (v_isShared_2760_ == 0)
{
lean_ctor_set(v___x_2759_, 0, v___x_2719_);
v___x_2763_ = v___x_2759_;
goto v_reusejp_2762_;
}
else
{
lean_object* v_reuseFailAlloc_2764_; 
v_reuseFailAlloc_2764_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2764_, 0, v___x_2719_);
v___x_2763_ = v_reuseFailAlloc_2764_;
goto v_reusejp_2762_;
}
v_reusejp_2762_:
{
return v___x_2763_;
}
}
else
{
lean_object* v_options_2765_; uint8_t v_hasTrace_2766_; 
lean_del_object(v___x_2759_);
v_options_2765_ = lean_ctor_get(v___y_2727_, 2);
v_hasTrace_2766_ = lean_ctor_get_uint8(v_options_2765_, sizeof(void*)*1);
if (v_hasTrace_2766_ == 0)
{
v___y_2731_ = v___y_2723_;
v___y_2732_ = v___y_2724_;
v___y_2733_ = v___y_2725_;
v___y_2734_ = v___y_2726_;
v___y_2735_ = v___y_2727_;
v___y_2736_ = v___y_2728_;
goto v___jp_2730_;
}
else
{
lean_object* v_inheritedTraceOptions_2767_; lean_object* v___x_2768_; lean_object* v___x_2769_; uint8_t v___x_2770_; 
v_inheritedTraceOptions_2767_ = lean_ctor_get(v___y_2727_, 13);
v___x_2768_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_));
v___x_2769_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__0, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__0_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__0);
v___x_2770_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2767_, v_options_2765_, v___x_2769_);
if (v___x_2770_ == 0)
{
v___y_2731_ = v___y_2723_;
v___y_2732_ = v___y_2724_;
v___y_2733_ = v___y_2725_;
v___y_2734_ = v___y_2726_;
v___y_2735_ = v___y_2727_;
v___y_2736_ = v___y_2728_;
goto v___jp_2730_;
}
else
{
lean_object* v___x_2771_; lean_object* v___x_2772_; lean_object* v___x_2773_; lean_object* v___x_2774_; 
v___x_2771_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__2, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__2_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___closed__2);
lean_inc(v_head_2718_);
v___x_2772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2772_, 0, v_head_2718_);
v___x_2773_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2773_, 0, v___x_2771_);
lean_ctor_set(v___x_2773_, 1, v___x_2772_);
v___x_2774_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg(v___x_2768_, v___x_2773_, v___y_2725_, v___y_2726_, v___y_2727_, v___y_2728_);
if (lean_obj_tag(v___x_2774_) == 0)
{
lean_dec_ref_known(v___x_2774_, 1);
v___y_2731_ = v___y_2723_;
v___y_2732_ = v___y_2724_;
v___y_2733_ = v___y_2725_;
v___y_2734_ = v___y_2726_;
v___y_2735_ = v___y_2727_;
v___y_2736_ = v___y_2728_;
goto v___jp_2730_;
}
else
{
lean_dec_ref(v_discharger_2720_);
lean_dec(v_head_2718_);
return v___x_2774_;
}
}
}
}
}
}
else
{
lean_object* v_a_2776_; lean_object* v___x_2778_; uint8_t v_isShared_2779_; uint8_t v_isSharedCheck_2783_; 
lean_dec_ref(v_discharger_2720_);
lean_dec(v_head_2718_);
v_a_2776_ = lean_ctor_get(v___x_2756_, 0);
v_isSharedCheck_2783_ = !lean_is_exclusive(v___x_2756_);
if (v_isSharedCheck_2783_ == 0)
{
v___x_2778_ = v___x_2756_;
v_isShared_2779_ = v_isSharedCheck_2783_;
goto v_resetjp_2777_;
}
else
{
lean_inc(v_a_2776_);
lean_dec(v___x_2756_);
v___x_2778_ = lean_box(0);
v_isShared_2779_ = v_isSharedCheck_2783_;
goto v_resetjp_2777_;
}
v_resetjp_2777_:
{
lean_object* v___x_2781_; 
if (v_isShared_2779_ == 0)
{
v___x_2781_ = v___x_2778_;
goto v_reusejp_2780_;
}
else
{
lean_object* v_reuseFailAlloc_2782_; 
v_reuseFailAlloc_2782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2782_, 0, v_a_2776_);
v___x_2781_ = v_reuseFailAlloc_2782_;
goto v_reusejp_2780_;
}
v_reusejp_2780_:
{
return v___x_2781_;
}
}
}
}
else
{
lean_object* v_a_2784_; lean_object* v___x_2786_; uint8_t v_isShared_2787_; uint8_t v_isSharedCheck_2791_; 
lean_dec_ref(v_discharger_2720_);
lean_dec(v_head_2718_);
v_a_2784_ = lean_ctor_get(v___x_2754_, 0);
v_isSharedCheck_2791_ = !lean_is_exclusive(v___x_2754_);
if (v_isSharedCheck_2791_ == 0)
{
v___x_2786_ = v___x_2754_;
v_isShared_2787_ = v_isSharedCheck_2791_;
goto v_resetjp_2785_;
}
else
{
lean_inc(v_a_2784_);
lean_dec(v___x_2754_);
v___x_2786_ = lean_box(0);
v_isShared_2787_ = v_isSharedCheck_2791_;
goto v_resetjp_2785_;
}
v_resetjp_2785_:
{
lean_object* v___x_2789_; 
if (v_isShared_2787_ == 0)
{
v___x_2789_ = v___x_2786_;
goto v_reusejp_2788_;
}
else
{
lean_object* v_reuseFailAlloc_2790_; 
v_reuseFailAlloc_2790_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2790_, 0, v_a_2784_);
v___x_2789_ = v_reuseFailAlloc_2790_;
goto v_reusejp_2788_;
}
v_reusejp_2788_:
{
return v___x_2789_;
}
}
}
v___jp_2730_:
{
lean_object* v___x_2737_; 
v___x_2737_ = l_Lean_Elab_Tactic_run(v_head_2718_, v_discharger_2720_, v___y_2731_, v___y_2732_, v___y_2733_, v___y_2734_, v___y_2735_, v___y_2736_);
if (lean_obj_tag(v___x_2737_) == 0)
{
lean_object* v___x_2739_; uint8_t v_isShared_2740_; uint8_t v_isSharedCheck_2744_; 
v_isSharedCheck_2744_ = !lean_is_exclusive(v___x_2737_);
if (v_isSharedCheck_2744_ == 0)
{
lean_object* v_unused_2745_; 
v_unused_2745_ = lean_ctor_get(v___x_2737_, 0);
lean_dec(v_unused_2745_);
v___x_2739_ = v___x_2737_;
v_isShared_2740_ = v_isSharedCheck_2744_;
goto v_resetjp_2738_;
}
else
{
lean_dec(v___x_2737_);
v___x_2739_ = lean_box(0);
v_isShared_2740_ = v_isSharedCheck_2744_;
goto v_resetjp_2738_;
}
v_resetjp_2738_:
{
lean_object* v___x_2742_; 
if (v_isShared_2740_ == 0)
{
lean_ctor_set(v___x_2739_, 0, v___x_2719_);
v___x_2742_ = v___x_2739_;
goto v_reusejp_2741_;
}
else
{
lean_object* v_reuseFailAlloc_2743_; 
v_reuseFailAlloc_2743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2743_, 0, v___x_2719_);
v___x_2742_ = v_reuseFailAlloc_2743_;
goto v_reusejp_2741_;
}
v_reusejp_2741_:
{
return v___x_2742_;
}
}
}
else
{
lean_object* v_a_2746_; lean_object* v___x_2748_; uint8_t v_isShared_2749_; uint8_t v_isSharedCheck_2753_; 
v_a_2746_ = lean_ctor_get(v___x_2737_, 0);
v_isSharedCheck_2753_ = !lean_is_exclusive(v___x_2737_);
if (v_isSharedCheck_2753_ == 0)
{
v___x_2748_ = v___x_2737_;
v_isShared_2749_ = v_isSharedCheck_2753_;
goto v_resetjp_2747_;
}
else
{
lean_inc(v_a_2746_);
lean_dec(v___x_2737_);
v___x_2748_ = lean_box(0);
v_isShared_2749_ = v_isSharedCheck_2753_;
goto v_resetjp_2747_;
}
v_resetjp_2747_:
{
lean_object* v___x_2751_; 
if (v_isShared_2749_ == 0)
{
v___x_2751_ = v___x_2748_;
goto v_reusejp_2750_;
}
else
{
lean_object* v_reuseFailAlloc_2752_; 
v_reuseFailAlloc_2752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2752_, 0, v_a_2746_);
v___x_2751_ = v_reuseFailAlloc_2752_;
goto v_reusejp_2750_;
}
v_reusejp_2750_:
{
return v___x_2751_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___boxed(lean_object* v_head_2792_, lean_object* v___x_2793_, lean_object* v_discharger_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_, lean_object* v___y_2802_, lean_object* v___y_2803_){
_start:
{
lean_object* v_res_2804_; 
v_res_2804_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0(v_head_2792_, v___x_2793_, v_discharger_2794_, v___y_2795_, v___y_2796_, v___y_2797_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_, v___y_2802_);
lean_dec(v___y_2802_);
lean_dec_ref(v___y_2801_);
lean_dec(v___y_2800_);
lean_dec_ref(v___y_2799_);
lean_dec(v___y_2798_);
lean_dec_ref(v___y_2797_);
lean_dec(v___y_2796_);
lean_dec_ref(v___y_2795_);
return v_res_2804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg(lean_object* v_discharger_2805_, lean_object* v_as_x27_2806_, lean_object* v_b_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_){
_start:
{
if (lean_obj_tag(v_as_x27_2806_) == 0)
{
lean_object* v___x_2817_; 
lean_dec_ref(v_discharger_2805_);
v___x_2817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2817_, 0, v_b_2807_);
return v___x_2817_;
}
else
{
lean_object* v_head_2818_; lean_object* v_tail_2819_; lean_object* v___x_2820_; lean_object* v_a_2821_; lean_object* v___x_2822_; uint8_t v___x_2823_; 
v_head_2818_ = lean_ctor_get(v_as_x27_2806_, 0);
v_tail_2819_ = lean_ctor_get(v_as_x27_2806_, 1);
v___x_2820_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg(v_head_2818_, v___y_2813_);
v_a_2821_ = lean_ctor_get(v___x_2820_, 0);
lean_inc(v_a_2821_);
lean_dec_ref(v___x_2820_);
v___x_2822_ = lean_box(0);
v___x_2823_ = lean_unbox(v_a_2821_);
lean_dec(v_a_2821_);
if (v___x_2823_ == 0)
{
lean_object* v___f_2824_; lean_object* v___x_2825_; 
lean_inc_ref(v_discharger_2805_);
lean_inc_n(v_head_2818_, 2);
v___f_2824_ = lean_alloc_closure((void*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___lam__0___boxed), 12, 3);
lean_closure_set(v___f_2824_, 0, v_head_2818_);
lean_closure_set(v___f_2824_, 1, v___x_2822_);
lean_closure_set(v___f_2824_, 2, v_discharger_2805_);
v___x_2825_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_runUse_spec__3___redArg(v_head_2818_, v___f_2824_, v___y_2808_, v___y_2809_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_, v___y_2814_, v___y_2815_);
if (lean_obj_tag(v___x_2825_) == 0)
{
lean_dec_ref_known(v___x_2825_, 1);
v_as_x27_2806_ = v_tail_2819_;
v_b_2807_ = v___x_2822_;
goto _start;
}
else
{
lean_dec_ref(v_discharger_2805_);
return v___x_2825_;
}
}
else
{
v_as_x27_2806_ = v_tail_2819_;
v_b_2807_ = v___x_2822_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg___boxed(lean_object* v_discharger_2828_, lean_object* v_as_x27_2829_, lean_object* v_b_2830_, lean_object* v___y_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_){
_start:
{
lean_object* v_res_2840_; 
v_res_2840_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg(v_discharger_2828_, v_as_x27_2829_, v_b_2830_, v___y_2831_, v___y_2832_, v___y_2833_, v___y_2834_, v___y_2835_, v___y_2836_, v___y_2837_, v___y_2838_);
lean_dec(v___y_2838_);
lean_dec_ref(v___y_2837_);
lean_dec(v___y_2836_);
lean_dec_ref(v___y_2835_);
lean_dec(v___y_2834_);
lean_dec_ref(v___y_2833_);
lean_dec(v___y_2832_);
lean_dec_ref(v___y_2831_);
lean_dec(v_as_x27_2829_);
return v_res_2840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse(uint8_t v_eager_2841_, lean_object* v_discharger_2842_, lean_object* v_args_2843_, lean_object* v_a_2844_, lean_object* v_a_2845_, lean_object* v_a_2846_, lean_object* v_a_2847_, lean_object* v_a_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_){
_start:
{
lean_object* v___x_2853_; lean_object* v___f_2854_; lean_object* v___x_2855_; 
v___x_2853_ = lean_box(v_eager_2841_);
v___f_2854_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_runUse___lam__0___boxed), 11, 2);
lean_closure_set(v___f_2854_, 0, v___x_2853_);
lean_closure_set(v___f_2854_, 1, v_args_2843_);
v___x_2855_ = l_Lean_Elab_Tactic_focus___redArg(v___f_2854_, v_a_2844_, v_a_2845_, v_a_2846_, v_a_2847_, v_a_2848_, v_a_2849_, v_a_2850_, v_a_2851_);
if (lean_obj_tag(v___x_2855_) == 0)
{
lean_object* v_a_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; 
v_a_2856_ = lean_ctor_get(v___x_2855_, 0);
lean_inc(v_a_2856_);
lean_dec_ref_known(v___x_2855_, 1);
v___x_2857_ = lean_box(0);
v___x_2858_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg(v_discharger_2842_, v_a_2856_, v___x_2857_, v_a_2844_, v_a_2845_, v_a_2846_, v_a_2847_, v_a_2848_, v_a_2849_, v_a_2850_, v_a_2851_);
lean_dec(v_a_2856_);
if (lean_obj_tag(v___x_2858_) == 0)
{
lean_object* v___x_2860_; uint8_t v_isShared_2861_; uint8_t v_isSharedCheck_2865_; 
v_isSharedCheck_2865_ = !lean_is_exclusive(v___x_2858_);
if (v_isSharedCheck_2865_ == 0)
{
lean_object* v_unused_2866_; 
v_unused_2866_ = lean_ctor_get(v___x_2858_, 0);
lean_dec(v_unused_2866_);
v___x_2860_ = v___x_2858_;
v_isShared_2861_ = v_isSharedCheck_2865_;
goto v_resetjp_2859_;
}
else
{
lean_dec(v___x_2858_);
v___x_2860_ = lean_box(0);
v_isShared_2861_ = v_isSharedCheck_2865_;
goto v_resetjp_2859_;
}
v_resetjp_2859_:
{
lean_object* v___x_2863_; 
if (v_isShared_2861_ == 0)
{
lean_ctor_set(v___x_2860_, 0, v___x_2857_);
v___x_2863_ = v___x_2860_;
goto v_reusejp_2862_;
}
else
{
lean_object* v_reuseFailAlloc_2864_; 
v_reuseFailAlloc_2864_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2864_, 0, v___x_2857_);
v___x_2863_ = v_reuseFailAlloc_2864_;
goto v_reusejp_2862_;
}
v_reusejp_2862_:
{
return v___x_2863_;
}
}
}
else
{
return v___x_2858_;
}
}
else
{
lean_object* v_a_2867_; lean_object* v___x_2869_; uint8_t v_isShared_2870_; uint8_t v_isSharedCheck_2874_; 
lean_dec_ref(v_discharger_2842_);
v_a_2867_ = lean_ctor_get(v___x_2855_, 0);
v_isSharedCheck_2874_ = !lean_is_exclusive(v___x_2855_);
if (v_isSharedCheck_2874_ == 0)
{
v___x_2869_ = v___x_2855_;
v_isShared_2870_ = v_isSharedCheck_2874_;
goto v_resetjp_2868_;
}
else
{
lean_inc(v_a_2867_);
lean_dec(v___x_2855_);
v___x_2869_ = lean_box(0);
v_isShared_2870_ = v_isSharedCheck_2874_;
goto v_resetjp_2868_;
}
v_resetjp_2868_:
{
lean_object* v___x_2872_; 
if (v_isShared_2870_ == 0)
{
v___x_2872_ = v___x_2869_;
goto v_reusejp_2871_;
}
else
{
lean_object* v_reuseFailAlloc_2873_; 
v_reuseFailAlloc_2873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2873_, 0, v_a_2867_);
v___x_2872_ = v_reuseFailAlloc_2873_;
goto v_reusejp_2871_;
}
v_reusejp_2871_:
{
return v___x_2872_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_runUse___boxed(lean_object* v_eager_2875_, lean_object* v_discharger_2876_, lean_object* v_args_2877_, lean_object* v_a_2878_, lean_object* v_a_2879_, lean_object* v_a_2880_, lean_object* v_a_2881_, lean_object* v_a_2882_, lean_object* v_a_2883_, lean_object* v_a_2884_, lean_object* v_a_2885_, lean_object* v_a_2886_){
_start:
{
uint8_t v_eager_boxed_2887_; lean_object* v_res_2888_; 
v_eager_boxed_2887_ = lean_unbox(v_eager_2875_);
v_res_2888_ = lp_mathlib_Mathlib_Tactic_runUse(v_eager_boxed_2887_, v_discharger_2876_, v_args_2877_, v_a_2878_, v_a_2879_, v_a_2880_, v_a_2881_, v_a_2882_, v_a_2883_, v_a_2884_, v_a_2885_);
lean_dec(v_a_2885_);
lean_dec_ref(v_a_2884_);
lean_dec(v_a_2883_);
lean_dec_ref(v_a_2882_);
lean_dec(v_a_2881_);
lean_dec_ref(v_a_2880_);
lean_dec(v_a_2879_);
lean_dec_ref(v_a_2878_);
return v_res_2888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0(lean_object* v_mvarId_2889_, lean_object* v_val_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_, lean_object* v___y_2897_, lean_object* v___y_2898_){
_start:
{
lean_object* v___x_2900_; 
v___x_2900_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___redArg(v_mvarId_2889_, v_val_2890_, v___y_2896_);
return v___x_2900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0___boxed(lean_object* v_mvarId_2901_, lean_object* v_val_2902_, lean_object* v___y_2903_, lean_object* v___y_2904_, lean_object* v___y_2905_, lean_object* v___y_2906_, lean_object* v___y_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_){
_start:
{
lean_object* v_res_2912_; 
v_res_2912_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_runUse_spec__0(v_mvarId_2901_, v_val_2902_, v___y_2903_, v___y_2904_, v___y_2905_, v___y_2906_, v___y_2907_, v___y_2908_, v___y_2909_, v___y_2910_);
lean_dec(v___y_2910_);
lean_dec_ref(v___y_2909_);
lean_dec(v___y_2908_);
lean_dec_ref(v___y_2907_);
lean_dec(v___y_2906_);
lean_dec_ref(v___y_2905_);
lean_dec(v___y_2904_);
lean_dec_ref(v___y_2903_);
return v_res_2912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1(lean_object* v_mvarId_2913_, lean_object* v___y_2914_, lean_object* v___y_2915_, lean_object* v___y_2916_, lean_object* v___y_2917_, lean_object* v___y_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_){
_start:
{
lean_object* v___x_2923_; 
v___x_2923_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___redArg(v_mvarId_2913_, v___y_2919_);
return v___x_2923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1___boxed(lean_object* v_mvarId_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_, lean_object* v___y_2928_, lean_object* v___y_2929_, lean_object* v___y_2930_, lean_object* v___y_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_){
_start:
{
lean_object* v_res_2934_; 
v_res_2934_ = lp_mathlib_Lean_MVarId_isAssigned___at___00Mathlib_Tactic_runUse_spec__1(v_mvarId_2924_, v___y_2925_, v___y_2926_, v___y_2927_, v___y_2928_, v___y_2929_, v___y_2930_, v___y_2931_, v___y_2932_);
lean_dec(v___y_2932_);
lean_dec_ref(v___y_2931_);
lean_dec(v___y_2930_);
lean_dec_ref(v___y_2929_);
lean_dec(v___y_2928_);
lean_dec_ref(v___y_2927_);
lean_dec(v___y_2926_);
lean_dec_ref(v___y_2925_);
lean_dec(v_mvarId_2924_);
return v_res_2934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4(lean_object* v_as_2935_, lean_object* v_as_x27_2936_, lean_object* v_b_2937_, lean_object* v_a_2938_, lean_object* v___y_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_, lean_object* v___y_2943_, lean_object* v___y_2944_, lean_object* v___y_2945_, lean_object* v___y_2946_){
_start:
{
lean_object* v___x_2948_; 
v___x_2948_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___redArg(v_as_x27_2936_, v_b_2937_, v___y_2939_, v___y_2940_, v___y_2941_, v___y_2942_, v___y_2943_, v___y_2944_, v___y_2945_, v___y_2946_);
return v___x_2948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4___boxed(lean_object* v_as_2949_, lean_object* v_as_x27_2950_, lean_object* v_b_2951_, lean_object* v_a_2952_, lean_object* v___y_2953_, lean_object* v___y_2954_, lean_object* v___y_2955_, lean_object* v___y_2956_, lean_object* v___y_2957_, lean_object* v___y_2958_, lean_object* v___y_2959_, lean_object* v___y_2960_, lean_object* v___y_2961_){
_start:
{
lean_object* v_res_2962_; 
v_res_2962_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__4(v_as_2949_, v_as_x27_2950_, v_b_2951_, v_a_2952_, v___y_2953_, v___y_2954_, v___y_2955_, v___y_2956_, v___y_2957_, v___y_2958_, v___y_2959_, v___y_2960_);
lean_dec(v___y_2960_);
lean_dec_ref(v___y_2959_);
lean_dec(v___y_2958_);
lean_dec_ref(v___y_2957_);
lean_dec(v___y_2956_);
lean_dec_ref(v___y_2955_);
lean_dec(v___y_2954_);
lean_dec_ref(v___y_2953_);
lean_dec(v_as_x27_2950_);
lean_dec(v_as_2949_);
return v_res_2962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5(lean_object* v_cls_2963_, lean_object* v_msg_2964_, lean_object* v___y_2965_, lean_object* v___y_2966_, lean_object* v___y_2967_, lean_object* v___y_2968_, lean_object* v___y_2969_, lean_object* v___y_2970_, lean_object* v___y_2971_, lean_object* v___y_2972_){
_start:
{
lean_object* v___x_2974_; 
v___x_2974_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___redArg(v_cls_2963_, v_msg_2964_, v___y_2969_, v___y_2970_, v___y_2971_, v___y_2972_);
return v___x_2974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5___boxed(lean_object* v_cls_2975_, lean_object* v_msg_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_, lean_object* v___y_2985_){
_start:
{
lean_object* v_res_2986_; 
v_res_2986_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_runUse_spec__5(v_cls_2975_, v_msg_2976_, v___y_2977_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_, v___y_2983_, v___y_2984_);
lean_dec(v___y_2984_);
lean_dec_ref(v___y_2983_);
lean_dec(v___y_2982_);
lean_dec_ref(v___y_2981_);
lean_dec(v___y_2980_);
lean_dec_ref(v___y_2979_);
lean_dec(v___y_2978_);
lean_dec_ref(v___y_2977_);
return v_res_2986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6(lean_object* v_discharger_2987_, lean_object* v_as_2988_, lean_object* v_as_x27_2989_, lean_object* v_b_2990_, lean_object* v_a_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_){
_start:
{
lean_object* v___x_3001_; 
v___x_3001_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___redArg(v_discharger_2987_, v_as_x27_2989_, v_b_2990_, v___y_2992_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_, v___y_2997_, v___y_2998_, v___y_2999_);
return v___x_3001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6___boxed(lean_object* v_discharger_3002_, lean_object* v_as_3003_, lean_object* v_as_x27_3004_, lean_object* v_b_3005_, lean_object* v_a_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_){
_start:
{
lean_object* v_res_3016_; 
v_res_3016_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_runUse_spec__6(v_discharger_3002_, v_as_3003_, v_as_x27_3004_, v_b_3005_, v_a_3006_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_, v___y_3013_, v___y_3014_);
lean_dec(v___y_3014_);
lean_dec_ref(v___y_3013_);
lean_dec(v___y_3012_);
lean_dec_ref(v___y_3011_);
lean_dec(v___y_3010_);
lean_dec_ref(v___y_3009_);
lean_dec(v___y_3008_);
lean_dec_ref(v___y_3007_);
lean_dec(v_as_x27_3004_);
lean_dec(v_as_3003_);
return v_res_3016_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__5(void){
_start:
{
lean_object* v___x_3044_; lean_object* v___x_3045_; 
v___x_3044_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__4));
v___x_3045_ = l_String_toRawSubstring_x27(v___x_3044_);
return v___x_3045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1(lean_object* v_x_3063_, lean_object* v_a_3064_, lean_object* v_a_3065_){
_start:
{
lean_object* v___x_3066_; uint8_t v___x_3067_; 
v___x_3066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1));
v___x_3067_ = l_Lean_Syntax_isOfKind(v_x_3063_, v___x_3066_);
if (v___x_3067_ == 0)
{
lean_object* v___x_3068_; lean_object* v___x_3069_; 
v___x_3068_ = lean_box(1);
v___x_3069_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3069_, 0, v___x_3068_);
lean_ctor_set(v___x_3069_, 1, v_a_3065_);
return v___x_3069_;
}
else
{
lean_object* v_quotContext_3070_; lean_object* v_currMacroScope_3071_; lean_object* v_ref_3072_; uint8_t v___x_3073_; lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; lean_object* v___x_3077_; lean_object* v___x_3078_; lean_object* v___x_3079_; lean_object* v___x_3080_; lean_object* v___x_3081_; lean_object* v___x_3082_; lean_object* v___x_3083_; lean_object* v___x_3084_; lean_object* v___x_3085_; lean_object* v___x_3086_; lean_object* v___x_3087_; lean_object* v___x_3088_; lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; 
v_quotContext_3070_ = lean_ctor_get(v_a_3064_, 1);
v_currMacroScope_3071_ = lean_ctor_get(v_a_3064_, 2);
v_ref_3072_ = lean_ctor_get(v_a_3064_, 5);
v___x_3073_ = 0;
v___x_3074_ = l_Lean_SourceInfo_fromRef(v_ref_3072_, v___x_3073_);
v___x_3075_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1));
v___x_3076_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2));
v___x_3077_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3));
lean_inc_n(v___x_3074_, 6);
v___x_3078_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3078_, 0, v___x_3074_);
lean_ctor_set(v___x_3078_, 1, v___x_3076_);
v___x_3079_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__5);
v___x_3080_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__8));
lean_inc(v_currMacroScope_3071_);
lean_inc(v_quotContext_3070_);
v___x_3081_ = l_Lean_addMacroScope(v_quotContext_3070_, v___x_3080_, v_currMacroScope_3071_);
v___x_3082_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__12));
v___x_3083_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3083_, 0, v___x_3074_);
lean_ctor_set(v___x_3083_, 1, v___x_3079_);
lean_ctor_set(v___x_3083_, 2, v___x_3081_);
lean_ctor_set(v___x_3083_, 3, v___x_3082_);
v___x_3084_ = l_Lean_Syntax_node2(v___x_3074_, v___x_3077_, v___x_3078_, v___x_3083_);
v___x_3085_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__13));
v___x_3086_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3086_, 0, v___x_3074_);
lean_ctor_set(v___x_3086_, 1, v___x_3085_);
v___x_3087_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2));
v___x_3088_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3088_, 0, v___x_3074_);
lean_ctor_set(v___x_3088_, 1, v___x_3087_);
v___x_3089_ = l_Lean_Syntax_node1(v___x_3074_, v___x_3066_, v___x_3088_);
v___x_3090_ = l_Lean_Syntax_node3(v___x_3074_, v___x_3075_, v___x_3084_, v___x_3086_, v___x_3089_);
v___x_3091_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3091_, 0, v___x_3090_);
lean_ctor_set(v___x_3091_, 1, v_a_3065_);
return v___x_3091_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___boxed(lean_object* v_x_3092_, lean_object* v_a_3093_, lean_object* v_a_3094_){
_start:
{
lean_object* v_res_3095_; 
v_res_3095_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1(v_x_3092_, v_a_3093_, v_a_3094_);
lean_dec_ref(v_a_3093_);
return v_res_3095_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__1(void){
_start:
{
lean_object* v___x_3097_; lean_object* v___x_3098_; 
v___x_3097_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__0));
v___x_3098_ = l_String_toRawSubstring_x27(v___x_3097_);
return v___x_3098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2(lean_object* v_x_3110_, lean_object* v_a_3111_, lean_object* v_a_3112_){
_start:
{
lean_object* v___x_3113_; uint8_t v___x_3114_; 
v___x_3113_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1));
v___x_3114_ = l_Lean_Syntax_isOfKind(v_x_3110_, v___x_3113_);
if (v___x_3114_ == 0)
{
lean_object* v___x_3115_; lean_object* v___x_3116_; 
v___x_3115_ = lean_box(1);
v___x_3116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3116_, 0, v___x_3115_);
lean_ctor_set(v___x_3116_, 1, v_a_3112_);
return v___x_3116_;
}
else
{
lean_object* v_quotContext_3117_; lean_object* v_currMacroScope_3118_; lean_object* v_ref_3119_; uint8_t v___x_3120_; lean_object* v___x_3121_; lean_object* v___x_3122_; lean_object* v___x_3123_; lean_object* v___x_3124_; lean_object* v___x_3125_; lean_object* v___x_3126_; lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v___x_3133_; lean_object* v___x_3134_; lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; lean_object* v___x_3138_; 
v_quotContext_3117_ = lean_ctor_get(v_a_3111_, 1);
v_currMacroScope_3118_ = lean_ctor_get(v_a_3111_, 2);
v_ref_3119_ = lean_ctor_get(v_a_3111_, 5);
v___x_3120_ = 0;
v___x_3121_ = l_Lean_SourceInfo_fromRef(v_ref_3119_, v___x_3120_);
v___x_3122_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__1));
v___x_3123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2));
v___x_3124_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3));
lean_inc_n(v___x_3121_, 6);
v___x_3125_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3125_, 0, v___x_3121_);
lean_ctor_set(v___x_3125_, 1, v___x_3123_);
v___x_3126_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__1);
v___x_3127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__4));
lean_inc(v_currMacroScope_3118_);
lean_inc(v_quotContext_3117_);
v___x_3128_ = l_Lean_addMacroScope(v_quotContext_3117_, v___x_3127_, v_currMacroScope_3118_);
v___x_3129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___closed__6));
v___x_3130_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3130_, 0, v___x_3121_);
lean_ctor_set(v___x_3130_, 1, v___x_3126_);
lean_ctor_set(v___x_3130_, 2, v___x_3128_);
lean_ctor_set(v___x_3130_, 3, v___x_3129_);
v___x_3131_ = l_Lean_Syntax_node2(v___x_3121_, v___x_3124_, v___x_3125_, v___x_3130_);
v___x_3132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__13));
v___x_3133_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3133_, 0, v___x_3121_);
lean_ctor_set(v___x_3133_, 1, v___x_3132_);
v___x_3134_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2));
v___x_3135_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3135_, 0, v___x_3121_);
lean_ctor_set(v___x_3135_, 1, v___x_3134_);
v___x_3136_ = l_Lean_Syntax_node1(v___x_3121_, v___x_3113_, v___x_3135_);
v___x_3137_ = l_Lean_Syntax_node3(v___x_3121_, v___x_3122_, v___x_3131_, v___x_3133_, v___x_3136_);
v___x_3138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3138_, 0, v___x_3137_);
lean_ctor_set(v___x_3138_, 1, v_a_3112_);
return v___x_3138_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2___boxed(lean_object* v_x_3139_, lean_object* v_a_3140_, lean_object* v_a_3141_){
_start:
{
lean_object* v_res_3142_; 
v_res_3142_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__2(v_x_3139_, v_a_3140_, v_a_3141_);
lean_dec_ref(v_a_3140_);
return v_res_3142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3(lean_object* v_x_3150_, lean_object* v_a_3151_, lean_object* v_a_3152_){
_start:
{
lean_object* v___x_3153_; uint8_t v___x_3154_; 
v___x_3153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1));
v___x_3154_ = l_Lean_Syntax_isOfKind(v_x_3150_, v___x_3153_);
if (v___x_3154_ == 0)
{
lean_object* v___x_3155_; lean_object* v___x_3156_; 
v___x_3155_ = lean_box(1);
v___x_3156_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3156_, 0, v___x_3155_);
lean_ctor_set(v___x_3156_, 1, v_a_3152_);
return v___x_3156_;
}
else
{
lean_object* v_ref_3157_; uint8_t v___x_3158_; lean_object* v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; 
v_ref_3157_ = lean_ctor_get(v_a_3151_, 5);
v___x_3158_ = 0;
v___x_3159_ = l_Lean_SourceInfo_fromRef(v_ref_3157_, v___x_3158_);
v___x_3160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__1));
v___x_3161_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___closed__2));
lean_inc(v___x_3159_);
v___x_3162_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3162_, 0, v___x_3159_);
lean_ctor_set(v___x_3162_, 1, v___x_3161_);
v___x_3163_ = l_Lean_Syntax_node1(v___x_3159_, v___x_3160_, v___x_3162_);
v___x_3164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3164_, 0, v___x_3163_);
lean_ctor_set(v___x_3164_, 1, v_a_3152_);
return v___x_3164_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3___boxed(lean_object* v_x_3165_, lean_object* v_a_3166_, lean_object* v_a_3167_){
_start:
{
lean_object* v_res_3168_; 
v_res_3168_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__3(v_x_3165_, v_a_3166_, v_a_3167_);
lean_dec_ref(v_a_3166_);
return v_res_3168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4(lean_object* v_x_3175_, lean_object* v_a_3176_, lean_object* v_a_3177_){
_start:
{
lean_object* v___x_3178_; uint8_t v___x_3179_; 
v___x_3178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1));
v___x_3179_ = l_Lean_Syntax_isOfKind(v_x_3175_, v___x_3178_);
if (v___x_3179_ == 0)
{
lean_object* v___x_3180_; lean_object* v___x_3181_; 
v___x_3180_ = lean_box(1);
v___x_3181_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3181_, 0, v___x_3180_);
lean_ctor_set(v___x_3181_, 1, v_a_3177_);
return v___x_3181_;
}
else
{
lean_object* v_ref_3182_; uint8_t v___x_3183_; lean_object* v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; 
v_ref_3182_ = lean_ctor_get(v_a_3176_, 5);
v___x_3183_ = 0;
v___x_3184_ = l_Lean_SourceInfo_fromRef(v_ref_3182_, v___x_3183_);
v___x_3185_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__0));
v___x_3186_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___closed__1));
lean_inc(v___x_3184_);
v___x_3187_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3187_, 0, v___x_3184_);
lean_ctor_set(v___x_3187_, 1, v___x_3185_);
v___x_3188_ = l_Lean_Syntax_node1(v___x_3184_, v___x_3186_, v___x_3187_);
v___x_3189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3189_, 0, v___x_3188_);
lean_ctor_set(v___x_3189_, 1, v_a_3177_);
return v___x_3189_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4___boxed(lean_object* v_x_3190_, lean_object* v_a_3191_, lean_object* v_a_3192_){
_start:
{
lean_object* v_res_3193_; 
v_res_3193_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__4(v_x_3190_, v_a_3191_, v_a_3192_);
lean_dec_ref(v_a_3191_);
return v_res_3193_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__1(void){
_start:
{
lean_object* v___x_3195_; lean_object* v___x_3196_; 
v___x_3195_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__0));
v___x_3196_ = l_String_toRawSubstring_x27(v___x_3195_);
return v___x_3196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5(lean_object* v_x_3207_, lean_object* v_a_3208_, lean_object* v_a_3209_){
_start:
{
lean_object* v___x_3210_; uint8_t v___x_3211_; 
v___x_3210_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1));
v___x_3211_ = l_Lean_Syntax_isOfKind(v_x_3207_, v___x_3210_);
if (v___x_3211_ == 0)
{
lean_object* v___x_3212_; lean_object* v___x_3213_; 
v___x_3212_ = lean_box(1);
v___x_3213_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3213_, 0, v___x_3212_);
lean_ctor_set(v___x_3213_, 1, v_a_3209_);
return v___x_3213_;
}
else
{
lean_object* v_quotContext_3214_; lean_object* v_currMacroScope_3215_; lean_object* v_ref_3216_; uint8_t v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v___x_3220_; lean_object* v___x_3221_; lean_object* v___x_3222_; lean_object* v___x_3223_; lean_object* v___x_3224_; lean_object* v___x_3225_; lean_object* v___x_3226_; lean_object* v___x_3227_; lean_object* v___x_3228_; 
v_quotContext_3214_ = lean_ctor_get(v_a_3208_, 1);
v_currMacroScope_3215_ = lean_ctor_get(v_a_3208_, 2);
v_ref_3216_ = lean_ctor_get(v_a_3208_, 5);
v___x_3217_ = 0;
v___x_3218_ = l_Lean_SourceInfo_fromRef(v_ref_3216_, v___x_3217_);
v___x_3219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__2));
v___x_3220_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__1___closed__3));
lean_inc_n(v___x_3218_, 2);
v___x_3221_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3221_, 0, v___x_3218_);
lean_ctor_set(v___x_3221_, 1, v___x_3219_);
v___x_3222_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__1);
v___x_3223_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__3));
lean_inc(v_currMacroScope_3215_);
lean_inc(v_quotContext_3214_);
v___x_3224_ = l_Lean_addMacroScope(v_quotContext_3214_, v___x_3223_, v_currMacroScope_3215_);
v___x_3225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___closed__5));
v___x_3226_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3226_, 0, v___x_3218_);
lean_ctor_set(v___x_3226_, 1, v___x_3222_);
lean_ctor_set(v___x_3226_, 2, v___x_3224_);
lean_ctor_set(v___x_3226_, 3, v___x_3225_);
v___x_3227_ = l_Lean_Syntax_node2(v___x_3218_, v___x_3220_, v___x_3221_, v___x_3226_);
v___x_3228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3228_, 0, v___x_3227_);
lean_ctor_set(v___x_3228_, 1, v_a_3209_);
return v___x_3228_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5___boxed(lean_object* v_x_3229_, lean_object* v_a_3230_, lean_object* v_a_3231_){
_start:
{
lean_object* v_res_3232_; 
v_res_3232_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______macroRules__Mathlib__Tactic__tacticUse__discharger__5(v_x_3229_, v_a_3230_, v_a_3231_);
lean_dec_ref(v_a_3230_);
return v_res_3232_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3233_; lean_object* v___x_3234_; lean_object* v___x_3235_; 
v___x_3233_ = lean_box(0);
v___x_3234_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3235_, 0, v___x_3234_);
lean_ctor_set(v___x_3235_, 1, v___x_3233_);
return v___x_3235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg(){
_start:
{
lean_object* v___x_3237_; lean_object* v___x_3238_; 
v___x_3237_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___closed__0);
v___x_3238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3238_, 0, v___x_3237_);
return v___x_3238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg___boxed(lean_object* v___y_3239_){
_start:
{
lean_object* v_res_3240_; 
v_res_3240_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg();
return v_res_3240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0(lean_object* v_00_u03b1_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_, lean_object* v___y_3244_, lean_object* v___y_3245_, lean_object* v___y_3246_, lean_object* v___y_3247_, lean_object* v___y_3248_, lean_object* v___y_3249_){
_start:
{
lean_object* v___x_3251_; 
v___x_3251_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg();
return v___x_3251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___boxed(lean_object* v_00_u03b1_3252_, lean_object* v___y_3253_, lean_object* v___y_3254_, lean_object* v___y_3255_, lean_object* v___y_3256_, lean_object* v___y_3257_, lean_object* v___y_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_){
_start:
{
lean_object* v_res_3262_; 
v_res_3262_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0(v_00_u03b1_3252_, v___y_3253_, v___y_3254_, v___y_3255_, v___y_3256_, v___y_3257_, v___y_3258_, v___y_3259_, v___y_3260_);
lean_dec(v___y_3260_);
lean_dec_ref(v___y_3259_);
lean_dec(v___y_3258_);
lean_dec_ref(v___y_3257_);
lean_dec(v___y_3256_);
lean_dec_ref(v___y_3255_);
lean_dec(v___y_3254_);
lean_dec_ref(v___y_3253_);
return v_res_3262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger(lean_object* v_discharger_x3f_3301_, lean_object* v_a_3302_, lean_object* v_a_3303_, lean_object* v_a_3304_, lean_object* v_a_3305_, lean_object* v_a_3306_, lean_object* v_a_3307_, lean_object* v_a_3308_, lean_object* v_a_3309_){
_start:
{
lean_object* v_discharger_3312_; 
if (lean_obj_tag(v_discharger_x3f_3301_) == 1)
{
lean_object* v_val_3315_; lean_object* v___x_3316_; uint8_t v___x_3317_; 
v_val_3315_ = lean_ctor_get(v_discharger_x3f_3301_, 0);
lean_inc_n(v_val_3315_, 2);
lean_dec_ref_known(v_discharger_x3f_3301_, 1);
v___x_3316_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__1));
v___x_3317_ = l_Lean_Syntax_isOfKind(v_val_3315_, v___x_3316_);
if (v___x_3317_ == 0)
{
lean_object* v___x_3318_; lean_object* v_a_3319_; lean_object* v___x_3321_; uint8_t v_isShared_3322_; uint8_t v_isSharedCheck_3326_; 
lean_dec(v_val_3315_);
v___x_3318_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg();
v_a_3319_ = lean_ctor_get(v___x_3318_, 0);
v_isSharedCheck_3326_ = !lean_is_exclusive(v___x_3318_);
if (v_isSharedCheck_3326_ == 0)
{
v___x_3321_ = v___x_3318_;
v_isShared_3322_ = v_isSharedCheck_3326_;
goto v_resetjp_3320_;
}
else
{
lean_inc(v_a_3319_);
lean_dec(v___x_3318_);
v___x_3321_ = lean_box(0);
v_isShared_3322_ = v_isSharedCheck_3326_;
goto v_resetjp_3320_;
}
v_resetjp_3320_:
{
lean_object* v___x_3324_; 
if (v_isShared_3322_ == 0)
{
v___x_3324_ = v___x_3321_;
goto v_reusejp_3323_;
}
else
{
lean_object* v_reuseFailAlloc_3325_; 
v_reuseFailAlloc_3325_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3325_, 0, v_a_3319_);
v___x_3324_ = v_reuseFailAlloc_3325_;
goto v_reusejp_3323_;
}
v_reusejp_3323_:
{
return v___x_3324_;
}
}
}
else
{
lean_object* v_ref_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; uint8_t v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; lean_object* v___x_3337_; 
v_ref_3327_ = lean_ctor_get(v_a_3308_, 5);
v___x_3328_ = lean_unsigned_to_nat(3u);
v___x_3329_ = l_Lean_Syntax_getArg(v_val_3315_, v___x_3328_);
lean_dec(v_val_3315_);
v___x_3330_ = 0;
v___x_3331_ = l_Lean_SourceInfo_fromRef(v_ref_3327_, v___x_3330_);
v___x_3332_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__3));
v___x_3333_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__9));
lean_inc_n(v___x_3331_, 2);
v___x_3334_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3334_, 0, v___x_3331_);
lean_ctor_set(v___x_3334_, 1, v___x_3333_);
v___x_3335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__37));
v___x_3336_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3336_, 0, v___x_3331_);
lean_ctor_set(v___x_3336_, 1, v___x_3335_);
v___x_3337_ = l_Lean_Syntax_node3(v___x_3331_, v___x_3332_, v___x_3334_, v___x_3329_, v___x_3336_);
v_discharger_3312_ = v___x_3337_;
goto v___jp_3311_;
}
}
else
{
lean_object* v_ref_3338_; uint8_t v___x_3339_; lean_object* v___x_3340_; lean_object* v___x_3341_; lean_object* v___x_3342_; lean_object* v___x_3343_; lean_object* v___x_3344_; lean_object* v___x_3345_; lean_object* v___x_3346_; lean_object* v___x_3347_; lean_object* v___x_3348_; lean_object* v___x_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; lean_object* v___x_3354_; lean_object* v___x_3355_; lean_object* v___x_3356_; lean_object* v___x_3357_; lean_object* v___x_3358_; lean_object* v___x_3359_; lean_object* v___x_3360_; lean_object* v___x_3361_; 
lean_dec(v_discharger_x3f_3301_);
v_ref_3338_ = lean_ctor_get(v_a_3308_, 5);
v___x_3339_ = 0;
v___x_3340_ = l_Lean_SourceInfo_fromRef(v_ref_3338_, v___x_3339_);
v___x_3341_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__5));
v___x_3342_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__6));
lean_inc_n(v___x_3340_, 11);
v___x_3343_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3343_, 0, v___x_3340_);
lean_ctor_set(v___x_3343_, 1, v___x_3342_);
v___x_3344_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__8));
v___x_3345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__10));
v___x_3346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useLoop___lam__2___closed__36));
v___x_3347_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__12));
v___x_3348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkUseDischarger___closed__13));
v___x_3349_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3349_, 0, v___x_3340_);
lean_ctor_set(v___x_3349_, 1, v___x_3348_);
v___x_3350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__1));
v___x_3351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse__discharger___closed__2));
v___x_3352_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3352_, 0, v___x_3340_);
lean_ctor_set(v___x_3352_, 1, v___x_3351_);
v___x_3353_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3350_, v___x_3352_);
v___x_3354_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3346_, v___x_3353_);
v___x_3355_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3345_, v___x_3354_);
v___x_3356_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3344_, v___x_3355_);
v___x_3357_ = l_Lean_Syntax_node2(v___x_3340_, v___x_3347_, v___x_3349_, v___x_3356_);
v___x_3358_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3346_, v___x_3357_);
v___x_3359_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3345_, v___x_3358_);
v___x_3360_ = l_Lean_Syntax_node1(v___x_3340_, v___x_3344_, v___x_3359_);
v___x_3361_ = l_Lean_Syntax_node2(v___x_3340_, v___x_3341_, v___x_3343_, v___x_3360_);
v_discharger_3312_ = v___x_3361_;
goto v___jp_3311_;
}
v___jp_3311_:
{
lean_object* v___x_3313_; lean_object* v___x_3314_; 
v___x_3313_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_3313_, 0, v_discharger_3312_);
v___x_3314_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3314_, 0, v___x_3313_);
return v___x_3314_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkUseDischarger___boxed(lean_object* v_discharger_x3f_3362_, lean_object* v_a_3363_, lean_object* v_a_3364_, lean_object* v_a_3365_, lean_object* v_a_3366_, lean_object* v_a_3367_, lean_object* v_a_3368_, lean_object* v_a_3369_, lean_object* v_a_3370_, lean_object* v_a_3371_){
_start:
{
lean_object* v_res_3372_; 
v_res_3372_ = lp_mathlib_Mathlib_Tactic_mkUseDischarger(v_discharger_x3f_3362_, v_a_3363_, v_a_3364_, v_a_3365_, v_a_3366_, v_a_3367_, v_a_3368_, v_a_3369_, v_a_3370_);
lean_dec(v_a_3370_);
lean_dec_ref(v_a_3369_);
lean_dec(v_a_3368_);
lean_dec_ref(v_a_3367_);
lean_dec(v_a_3366_);
lean_dec_ref(v_a_3365_);
lean_dec(v_a_3364_);
lean_dec_ref(v_a_3363_);
return v_res_3372_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__7(void){
_start:
{
lean_object* v___x_3387_; lean_object* v___x_3388_; lean_object* v___x_3389_; 
v___x_3387_ = l_Lean_Parser_Tactic_discharger;
v___x_3388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__6));
v___x_3389_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3389_, 0, v___x_3388_);
lean_ctor_set(v___x_3389_, 1, v___x_3387_);
return v___x_3389_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__8(void){
_start:
{
lean_object* v___x_3390_; lean_object* v___x_3391_; lean_object* v___x_3392_; lean_object* v___x_3393_; 
v___x_3390_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useSyntax___closed__7, &lp_mathlib_Mathlib_Tactic_useSyntax___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__7);
v___x_3391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__4));
v___x_3392_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__3));
v___x_3393_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3393_, 0, v___x_3392_);
lean_ctor_set(v___x_3393_, 1, v___x_3391_);
lean_ctor_set(v___x_3393_, 2, v___x_3390_);
return v___x_3393_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__15(void){
_start:
{
lean_object* v___x_3405_; lean_object* v___x_3406_; lean_object* v___x_3407_; lean_object* v___x_3408_; 
v___x_3405_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__14));
v___x_3406_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useSyntax___closed__8, &lp_mathlib_Mathlib_Tactic_useSyntax___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__8);
v___x_3407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__3));
v___x_3408_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3408_, 0, v___x_3407_);
lean_ctor_set(v___x_3408_, 1, v___x_3406_);
lean_ctor_set(v___x_3408_, 2, v___x_3405_);
return v___x_3408_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__23(void){
_start:
{
lean_object* v___x_3424_; lean_object* v___x_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; 
v___x_3424_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__22));
v___x_3425_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useSyntax___closed__15, &lp_mathlib_Mathlib_Tactic_useSyntax___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__15);
v___x_3426_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__3));
v___x_3427_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3427_, 0, v___x_3426_);
lean_ctor_set(v___x_3427_, 1, v___x_3425_);
lean_ctor_set(v___x_3427_, 2, v___x_3424_);
return v___x_3427_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__24(void){
_start:
{
lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; 
v___x_3428_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useSyntax___closed__23, &lp_mathlib_Mathlib_Tactic_useSyntax___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__23);
v___x_3429_ = lean_unsigned_to_nat(1022u);
v___x_3430_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__1));
v___x_3431_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3431_, 0, v___x_3430_);
lean_ctor_set(v___x_3431_, 1, v___x_3429_);
lean_ctor_set(v___x_3431_, 2, v___x_3428_);
return v___x_3431_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_useSyntax(void){
_start:
{
lean_object* v___x_3432_; 
v___x_3432_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useSyntax___closed__24, &lp_mathlib_Mathlib_Tactic_useSyntax___closed__24_once, _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__24);
return v___x_3432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__useSyntax__1(lean_object* v_x_3433_, lean_object* v_a_3434_, lean_object* v_a_3435_, lean_object* v_a_3436_, lean_object* v_a_3437_, lean_object* v_a_3438_, lean_object* v_a_3439_, lean_object* v_a_3440_, lean_object* v_a_3441_){
_start:
{
lean_object* v___x_3443_; uint8_t v___x_3444_; 
v___x_3443_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__1));
lean_inc(v_x_3433_);
v___x_3444_ = l_Lean_Syntax_isOfKind(v_x_3433_, v___x_3443_);
if (v___x_3444_ == 0)
{
lean_object* v___x_3445_; 
lean_dec(v_x_3433_);
v___x_3445_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg();
return v___x_3445_;
}
else
{
lean_object* v___x_3446_; lean_object* v___x_3447_; lean_object* v___x_3448_; lean_object* v___x_3449_; lean_object* v_args_3450_; lean_object* v___y_3452_; lean_object* v___x_3467_; 
v___x_3446_ = lean_unsigned_to_nat(1u);
v___x_3447_ = l_Lean_Syntax_getArg(v_x_3433_, v___x_3446_);
v___x_3448_ = lean_unsigned_to_nat(3u);
v___x_3449_ = l_Lean_Syntax_getArg(v_x_3433_, v___x_3448_);
lean_dec(v_x_3433_);
v_args_3450_ = l_Lean_Syntax_getArgs(v___x_3449_);
lean_dec(v___x_3449_);
v___x_3467_ = l_Lean_Syntax_getOptional_x3f(v___x_3447_);
lean_dec(v___x_3447_);
if (lean_obj_tag(v___x_3467_) == 0)
{
lean_object* v___x_3468_; 
v___x_3468_ = lean_box(0);
v___y_3452_ = v___x_3468_;
goto v___jp_3451_;
}
else
{
lean_object* v_val_3469_; lean_object* v___x_3471_; uint8_t v_isShared_3472_; uint8_t v_isSharedCheck_3476_; 
v_val_3469_ = lean_ctor_get(v___x_3467_, 0);
v_isSharedCheck_3476_ = !lean_is_exclusive(v___x_3467_);
if (v_isSharedCheck_3476_ == 0)
{
v___x_3471_ = v___x_3467_;
v_isShared_3472_ = v_isSharedCheck_3476_;
goto v_resetjp_3470_;
}
else
{
lean_inc(v_val_3469_);
lean_dec(v___x_3467_);
v___x_3471_ = lean_box(0);
v_isShared_3472_ = v_isSharedCheck_3476_;
goto v_resetjp_3470_;
}
v_resetjp_3470_:
{
lean_object* v___x_3474_; 
if (v_isShared_3472_ == 0)
{
v___x_3474_ = v___x_3471_;
goto v_reusejp_3473_;
}
else
{
lean_object* v_reuseFailAlloc_3475_; 
v_reuseFailAlloc_3475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3475_, 0, v_val_3469_);
v___x_3474_ = v_reuseFailAlloc_3475_;
goto v_reusejp_3473_;
}
v_reusejp_3473_:
{
v___y_3452_ = v___x_3474_;
goto v___jp_3451_;
}
}
}
v___jp_3451_:
{
lean_object* v___x_3453_; 
v___x_3453_ = lp_mathlib_Mathlib_Tactic_mkUseDischarger(v___y_3452_, v_a_3434_, v_a_3435_, v_a_3436_, v_a_3437_, v_a_3438_, v_a_3439_, v_a_3440_, v_a_3441_);
if (lean_obj_tag(v___x_3453_) == 0)
{
lean_object* v_a_3454_; uint8_t v___x_3455_; lean_object* v___x_3456_; lean_object* v___x_3457_; lean_object* v___x_3458_; 
v_a_3454_ = lean_ctor_get(v___x_3453_, 0);
lean_inc(v_a_3454_);
lean_dec_ref_known(v___x_3453_, 1);
v___x_3455_ = 0;
v___x_3456_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_args_3450_);
lean_dec_ref(v_args_3450_);
v___x_3457_ = lean_array_to_list(v___x_3456_);
v___x_3458_ = lp_mathlib_Mathlib_Tactic_runUse(v___x_3455_, v_a_3454_, v___x_3457_, v_a_3434_, v_a_3435_, v_a_3436_, v_a_3437_, v_a_3438_, v_a_3439_, v_a_3440_, v_a_3441_);
return v___x_3458_;
}
else
{
lean_object* v_a_3459_; lean_object* v___x_3461_; uint8_t v_isShared_3462_; uint8_t v_isSharedCheck_3466_; 
lean_dec_ref(v_args_3450_);
v_a_3459_ = lean_ctor_get(v___x_3453_, 0);
v_isSharedCheck_3466_ = !lean_is_exclusive(v___x_3453_);
if (v_isSharedCheck_3466_ == 0)
{
v___x_3461_ = v___x_3453_;
v_isShared_3462_ = v_isSharedCheck_3466_;
goto v_resetjp_3460_;
}
else
{
lean_inc(v_a_3459_);
lean_dec(v___x_3453_);
v___x_3461_ = lean_box(0);
v_isShared_3462_ = v_isSharedCheck_3466_;
goto v_resetjp_3460_;
}
v_resetjp_3460_:
{
lean_object* v___x_3464_; 
if (v_isShared_3462_ == 0)
{
v___x_3464_ = v___x_3461_;
goto v_reusejp_3463_;
}
else
{
lean_object* v_reuseFailAlloc_3465_; 
v_reuseFailAlloc_3465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3465_, 0, v_a_3459_);
v___x_3464_ = v_reuseFailAlloc_3465_;
goto v_reusejp_3463_;
}
v_reusejp_3463_:
{
return v___x_3464_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__useSyntax__1___boxed(lean_object* v_x_3477_, lean_object* v_a_3478_, lean_object* v_a_3479_, lean_object* v_a_3480_, lean_object* v_a_3481_, lean_object* v_a_3482_, lean_object* v_a_3483_, lean_object* v_a_3484_, lean_object* v_a_3485_, lean_object* v_a_3486_){
_start:
{
lean_object* v_res_3487_; 
v_res_3487_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__useSyntax__1(v_x_3477_, v_a_3478_, v_a_3479_, v_a_3480_, v_a_3481_, v_a_3482_, v_a_3483_, v_a_3484_, v_a_3485_);
lean_dec(v_a_3485_);
lean_dec_ref(v_a_3484_);
lean_dec(v_a_3483_);
lean_dec_ref(v_a_3482_);
lean_dec(v_a_3481_);
lean_dec_ref(v_a_3480_);
lean_dec(v_a_3479_);
lean_dec_ref(v_a_3478_);
return v_res_3487_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__4(void){
_start:
{
lean_object* v___x_3497_; lean_object* v___x_3498_; lean_object* v___x_3499_; lean_object* v___x_3500_; 
v___x_3497_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_useSyntax___closed__7, &lp_mathlib_Mathlib_Tactic_useSyntax___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_useSyntax___closed__7);
v___x_3498_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__3));
v___x_3499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__3));
v___x_3500_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3500_, 0, v___x_3499_);
lean_ctor_set(v___x_3500_, 1, v___x_3498_);
lean_ctor_set(v___x_3500_, 2, v___x_3497_);
return v___x_3500_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__5(void){
_start:
{
lean_object* v___x_3501_; lean_object* v___x_3502_; lean_object* v___x_3503_; lean_object* v___x_3504_; 
v___x_3501_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__14));
v___x_3502_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__4, &lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__4);
v___x_3503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__3));
v___x_3504_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3504_, 0, v___x_3503_);
lean_ctor_set(v___x_3504_, 1, v___x_3502_);
lean_ctor_set(v___x_3504_, 2, v___x_3501_);
return v___x_3504_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__6(void){
_start:
{
lean_object* v___x_3505_; lean_object* v___x_3506_; lean_object* v___x_3507_; lean_object* v___x_3508_; 
v___x_3505_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__22));
v___x_3506_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__5, &lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__5);
v___x_3507_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_useSyntax___closed__3));
v___x_3508_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3508_, 0, v___x_3507_);
lean_ctor_set(v___x_3508_, 1, v___x_3506_);
lean_ctor_set(v___x_3508_, 2, v___x_3505_);
return v___x_3508_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__7(void){
_start:
{
lean_object* v___x_3509_; lean_object* v___x_3510_; lean_object* v___x_3511_; lean_object* v___x_3512_; 
v___x_3509_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__6, &lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__6);
v___x_3510_ = lean_unsigned_to_nat(1022u);
v___x_3511_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1));
v___x_3512_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3512_, 0, v___x_3511_);
lean_ctor_set(v___x_3512_, 1, v___x_3510_);
lean_ctor_set(v___x_3512_, 2, v___x_3509_);
return v___x_3512_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c(void){
_start:
{
lean_object* v___x_3513_; 
v___x_3513_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__7, &lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__7);
return v___x_3513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__tacticUse_x21_______x2c_x2c__1(lean_object* v_x_3514_, lean_object* v_a_3515_, lean_object* v_a_3516_, lean_object* v_a_3517_, lean_object* v_a_3518_, lean_object* v_a_3519_, lean_object* v_a_3520_, lean_object* v_a_3521_, lean_object* v_a_3522_){
_start:
{
lean_object* v___x_3524_; uint8_t v___x_3525_; 
v___x_3524_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c___closed__1));
lean_inc(v_x_3514_);
v___x_3525_ = l_Lean_Syntax_isOfKind(v_x_3514_, v___x_3524_);
if (v___x_3525_ == 0)
{
lean_object* v___x_3526_; 
lean_dec(v_x_3514_);
v___x_3526_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_mkUseDischarger_spec__0___redArg();
return v___x_3526_;
}
else
{
lean_object* v___x_3527_; lean_object* v___x_3528_; lean_object* v___x_3529_; lean_object* v___x_3530_; lean_object* v_args_3531_; lean_object* v___y_3533_; lean_object* v___x_3547_; 
v___x_3527_ = lean_unsigned_to_nat(1u);
v___x_3528_ = l_Lean_Syntax_getArg(v_x_3514_, v___x_3527_);
v___x_3529_ = lean_unsigned_to_nat(3u);
v___x_3530_ = l_Lean_Syntax_getArg(v_x_3514_, v___x_3529_);
lean_dec(v_x_3514_);
v_args_3531_ = l_Lean_Syntax_getArgs(v___x_3530_);
lean_dec(v___x_3530_);
v___x_3547_ = l_Lean_Syntax_getOptional_x3f(v___x_3528_);
lean_dec(v___x_3528_);
if (lean_obj_tag(v___x_3547_) == 0)
{
lean_object* v___x_3548_; 
v___x_3548_ = lean_box(0);
v___y_3533_ = v___x_3548_;
goto v___jp_3532_;
}
else
{
lean_object* v_val_3549_; lean_object* v___x_3551_; uint8_t v_isShared_3552_; uint8_t v_isSharedCheck_3556_; 
v_val_3549_ = lean_ctor_get(v___x_3547_, 0);
v_isSharedCheck_3556_ = !lean_is_exclusive(v___x_3547_);
if (v_isSharedCheck_3556_ == 0)
{
v___x_3551_ = v___x_3547_;
v_isShared_3552_ = v_isSharedCheck_3556_;
goto v_resetjp_3550_;
}
else
{
lean_inc(v_val_3549_);
lean_dec(v___x_3547_);
v___x_3551_ = lean_box(0);
v_isShared_3552_ = v_isSharedCheck_3556_;
goto v_resetjp_3550_;
}
v_resetjp_3550_:
{
lean_object* v___x_3554_; 
if (v_isShared_3552_ == 0)
{
v___x_3554_ = v___x_3551_;
goto v_reusejp_3553_;
}
else
{
lean_object* v_reuseFailAlloc_3555_; 
v_reuseFailAlloc_3555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3555_, 0, v_val_3549_);
v___x_3554_ = v_reuseFailAlloc_3555_;
goto v_reusejp_3553_;
}
v_reusejp_3553_:
{
v___y_3533_ = v___x_3554_;
goto v___jp_3532_;
}
}
}
v___jp_3532_:
{
lean_object* v___x_3534_; 
v___x_3534_ = lp_mathlib_Mathlib_Tactic_mkUseDischarger(v___y_3533_, v_a_3515_, v_a_3516_, v_a_3517_, v_a_3518_, v_a_3519_, v_a_3520_, v_a_3521_, v_a_3522_);
if (lean_obj_tag(v___x_3534_) == 0)
{
lean_object* v_a_3535_; lean_object* v___x_3536_; lean_object* v___x_3537_; lean_object* v___x_3538_; 
v_a_3535_ = lean_ctor_get(v___x_3534_, 0);
lean_inc(v_a_3535_);
lean_dec_ref_known(v___x_3534_, 1);
v___x_3536_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_args_3531_);
lean_dec_ref(v_args_3531_);
v___x_3537_ = lean_array_to_list(v___x_3536_);
v___x_3538_ = lp_mathlib_Mathlib_Tactic_runUse(v___x_3525_, v_a_3535_, v___x_3537_, v_a_3515_, v_a_3516_, v_a_3517_, v_a_3518_, v_a_3519_, v_a_3520_, v_a_3521_, v_a_3522_);
return v___x_3538_;
}
else
{
lean_object* v_a_3539_; lean_object* v___x_3541_; uint8_t v_isShared_3542_; uint8_t v_isSharedCheck_3546_; 
lean_dec_ref(v_args_3531_);
v_a_3539_ = lean_ctor_get(v___x_3534_, 0);
v_isSharedCheck_3546_ = !lean_is_exclusive(v___x_3534_);
if (v_isSharedCheck_3546_ == 0)
{
v___x_3541_ = v___x_3534_;
v_isShared_3542_ = v_isSharedCheck_3546_;
goto v_resetjp_3540_;
}
else
{
lean_inc(v_a_3539_);
lean_dec(v___x_3534_);
v___x_3541_ = lean_box(0);
v_isShared_3542_ = v_isSharedCheck_3546_;
goto v_resetjp_3540_;
}
v_resetjp_3540_:
{
lean_object* v___x_3544_; 
if (v_isShared_3542_ == 0)
{
v___x_3544_ = v___x_3541_;
goto v_reusejp_3543_;
}
else
{
lean_object* v_reuseFailAlloc_3545_; 
v_reuseFailAlloc_3545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3545_, 0, v_a_3539_);
v___x_3544_ = v_reuseFailAlloc_3545_;
goto v_reusejp_3543_;
}
v_reusejp_3543_:
{
return v___x_3544_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__tacticUse_x21_______x2c_x2c__1___boxed(lean_object* v_x_3557_, lean_object* v_a_3558_, lean_object* v_a_3559_, lean_object* v_a_3560_, lean_object* v_a_3561_, lean_object* v_a_3562_, lean_object* v_a_3563_, lean_object* v_a_3564_, lean_object* v_a_3565_, lean_object* v_a_3566_){
_start:
{
lean_object* v_res_3567_; 
v_res_3567_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Use______elabRules__Mathlib__Tactic__tacticUse_x21_______x2c_x2c__1(v_x_3557_, v_a_3558_, v_a_3559_, v_a_3560_, v_a_3561_, v_a_3562_, v_a_3563_, v_a_3564_, v_a_3565_);
lean_dec(v_a_3565_);
lean_dec_ref(v_a_3564_);
lean_dec(v_a_3563_);
lean_dec_ref(v_a_3562_);
lean_dec(v_a_3561_);
lean_dec_ref(v_a_3560_);
lean_dec(v_a_3559_);
lean_dec_ref(v_a_3558_);
return v_res_3567_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Meta_Tactic_Util(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Use_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_Use_721082523____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_useSyntax = _init_lp_mathlib_Mathlib_Tactic_useSyntax();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_useSyntax);
lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c = _init_lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticUse_x21_______x2c_x2c);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Util(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Use(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Use(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Use(builtin);
}
#ifdef __cplusplus
}
#endif
