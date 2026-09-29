// Lean compiler output
// Module: Mathlib.Tactic.ApplyFun
// Imports: public import Init public meta import Init public meta import Mathlib.Lean.Expr.Basic public import Mathlib.Order.Hom.Basic public meta import Mathlib.Tactic.ToDual
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
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabAppArgs(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_ensureHasType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsUsingDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_MVarId_assumptionCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_MVarId_note(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* l_Lean_Meta_mkAppM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermForApply(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_runTermElab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withCollectingNewGoalsFrom(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_congrN_x21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshTypeMVar(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_appendTag(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "apply_fun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(72, 203, 7, 159, 55, 125, 170, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ApplyFun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(210, 164, 88, 44, 106, 199, 177, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(83, 153, 54, 222, 83, 149, 153, 59)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(46, 138, 217, 51, 24, 78, 51, 231)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 18, 153, 4, 156, 139, 199, 132)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(82, 90, 4, 39, 250, 211, 147, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(251, 35, 29, 6, 171, 230, 10, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(182, 27, 161, 62, 75, 152, 213, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 147, 242, 253, 189, 195, 175, 181)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 117, 132, 37, 220, 75, 246, 190)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "In generated equality, right-hand side "};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Injective"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 162, 25, 76, 92, 227, 14, 201)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__2_value),LEAN_SCALAR_PTR_LITERAL(229, 149, 41, 255, 152, 186, 255, 68)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 89, .m_capacity = 89, .m_length = 84, .m_data = "apply_fun can only handle hypotheses of the form `a = b`, `a ≠ b`, `a ≤ b`, `a < b`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "apply_fun can only handle negations of equality."};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 162, 25, 76, 92, 227, 14, 201)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "`apply_fun` could not construct congruence"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Monotone"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__16_value),LEAN_SCALAR_PTR_LITERAL(56, 128, 239, 21, 200, 119, 140, 54)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "StrictMono"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__19_value),LEAN_SCALAR_PTR_LITERAL(151, 223, 70, 197, 21, 143, 251, 23)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "`apply_fun` could not apply `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "` to the main goal."};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Equiv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "injective"};
static const lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 253, 123, 237, 128, 91, 245, 83)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__1_value),LEAN_SCALAR_PTR_LITERAL(248, 47, 30, 209, 86, 213, 148, 195)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Using clause "};
static const lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Function.Injective"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(22, 239, 111, 188, 150, 212, 129, 32)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__0_value),LEAN_SCALAR_PTR_LITERAL(101, 202, 61, 49, 146, 184, 41, 95)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "lt_of_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(22, 239, 111, 188, 150, 212, 129, 32)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__2_value),LEAN_SCALAR_PTR_LITERAL(45, 70, 42, 246, 189, 3, 235, 251)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "inj"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__4_value),LEAN_SCALAR_PTR_LITERAL(38, 11, 58, 56, 192, 58, 162, 195)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ne_of_apply_ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__6_value),LEAN_SCALAR_PTR_LITERAL(185, 14, 131, 53, 226, 232, 125, 114)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "GT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "gt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFun___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "applyFun"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__0_value),LEAN_SCALAR_PTR_LITERAL(57, 142, 37, 208, 65, 166, 6, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFun___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFun___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "apply_fun "};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFun___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFun___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__10_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFun___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFun___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_applyFun___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_applyFun___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_applyFun___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFun___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_applyFun___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_applyFun___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFun;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "apply_fun failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = lean_unsigned_to_nat(3448584465u);
v___x_47_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_48_ = l_Lean_Name_num___override(v___x_47_, v___x_46_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_50_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_51_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_);
v___x_52_ = l_Lean_Name_str___override(v___x_51_, v___x_50_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_54_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_55_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_);
v___x_56_ = l_Lean_Name_str___override(v___x_55_, v___x_54_);
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = lean_unsigned_to_nat(2u);
v___x_58_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_);
v___x_59_ = l_Lean_Name_num___override(v___x_58_, v___x_57_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_61_; uint8_t v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_61_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_62_ = 0;
v___x_63_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_);
v___x_64_ = l_Lean_registerTraceClass(v___x_61_, v___x_62_, v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2____boxed(lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_();
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg(lean_object* v_e_67_, lean_object* v___y_68_){
_start:
{
uint8_t v___x_70_; 
v___x_70_ = l_Lean_Expr_hasMVar(v_e_67_);
if (v___x_70_ == 0)
{
lean_object* v___x_71_; 
v___x_71_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_71_, 0, v_e_67_);
return v___x_71_;
}
else
{
lean_object* v___x_72_; lean_object* v_mctx_73_; lean_object* v___x_74_; lean_object* v_fst_75_; lean_object* v_snd_76_; lean_object* v___x_77_; lean_object* v_cache_78_; lean_object* v_zetaDeltaFVarIds_79_; lean_object* v_postponed_80_; lean_object* v_diag_81_; lean_object* v___x_83_; uint8_t v_isShared_84_; uint8_t v_isSharedCheck_90_; 
v___x_72_ = lean_st_ref_get(v___y_68_);
v_mctx_73_ = lean_ctor_get(v___x_72_, 0);
lean_inc_ref(v_mctx_73_);
lean_dec(v___x_72_);
v___x_74_ = l_Lean_instantiateMVarsCore(v_mctx_73_, v_e_67_);
v_fst_75_ = lean_ctor_get(v___x_74_, 0);
lean_inc(v_fst_75_);
v_snd_76_ = lean_ctor_get(v___x_74_, 1);
lean_inc(v_snd_76_);
lean_dec_ref(v___x_74_);
v___x_77_ = lean_st_ref_take(v___y_68_);
v_cache_78_ = lean_ctor_get(v___x_77_, 1);
v_zetaDeltaFVarIds_79_ = lean_ctor_get(v___x_77_, 2);
v_postponed_80_ = lean_ctor_get(v___x_77_, 3);
v_diag_81_ = lean_ctor_get(v___x_77_, 4);
v_isSharedCheck_90_ = !lean_is_exclusive(v___x_77_);
if (v_isSharedCheck_90_ == 0)
{
lean_object* v_unused_91_; 
v_unused_91_ = lean_ctor_get(v___x_77_, 0);
lean_dec(v_unused_91_);
v___x_83_ = v___x_77_;
v_isShared_84_ = v_isSharedCheck_90_;
goto v_resetjp_82_;
}
else
{
lean_inc(v_diag_81_);
lean_inc(v_postponed_80_);
lean_inc(v_zetaDeltaFVarIds_79_);
lean_inc(v_cache_78_);
lean_dec(v___x_77_);
v___x_83_ = lean_box(0);
v_isShared_84_ = v_isSharedCheck_90_;
goto v_resetjp_82_;
}
v_resetjp_82_:
{
lean_object* v___x_86_; 
if (v_isShared_84_ == 0)
{
lean_ctor_set(v___x_83_, 0, v_snd_76_);
v___x_86_ = v___x_83_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_89_; 
v_reuseFailAlloc_89_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_89_, 0, v_snd_76_);
lean_ctor_set(v_reuseFailAlloc_89_, 1, v_cache_78_);
lean_ctor_set(v_reuseFailAlloc_89_, 2, v_zetaDeltaFVarIds_79_);
lean_ctor_set(v_reuseFailAlloc_89_, 3, v_postponed_80_);
lean_ctor_set(v_reuseFailAlloc_89_, 4, v_diag_81_);
v___x_86_ = v_reuseFailAlloc_89_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = lean_st_ref_set(v___y_68_, v___x_86_);
v___x_88_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_88_, 0, v_fst_75_);
return v___x_88_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg___boxed(lean_object* v_e_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg(v_e_92_, v___y_93_);
lean_dec(v___y_93_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1(lean_object* v_e_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg(v_e_96_, v___y_102_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___boxed(lean_object* v_e_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1(v_e_107_, v___y_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg(lean_object* v_e_118_, lean_object* v___y_119_){
_start:
{
uint8_t v___x_121_; 
v___x_121_ = l_Lean_Expr_hasMVar(v_e_118_);
if (v___x_121_ == 0)
{
lean_object* v___x_122_; 
v___x_122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_122_, 0, v_e_118_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v_mctx_124_; lean_object* v___x_125_; lean_object* v_fst_126_; lean_object* v_snd_127_; lean_object* v___x_128_; lean_object* v_cache_129_; lean_object* v_zetaDeltaFVarIds_130_; lean_object* v_postponed_131_; lean_object* v_diag_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_141_; 
v___x_123_ = lean_st_ref_get(v___y_119_);
v_mctx_124_ = lean_ctor_get(v___x_123_, 0);
lean_inc_ref(v_mctx_124_);
lean_dec(v___x_123_);
v___x_125_ = l_Lean_instantiateMVarsCore(v_mctx_124_, v_e_118_);
v_fst_126_ = lean_ctor_get(v___x_125_, 0);
lean_inc(v_fst_126_);
v_snd_127_ = lean_ctor_get(v___x_125_, 1);
lean_inc(v_snd_127_);
lean_dec_ref(v___x_125_);
v___x_128_ = lean_st_ref_take(v___y_119_);
v_cache_129_ = lean_ctor_get(v___x_128_, 1);
v_zetaDeltaFVarIds_130_ = lean_ctor_get(v___x_128_, 2);
v_postponed_131_ = lean_ctor_get(v___x_128_, 3);
v_diag_132_ = lean_ctor_get(v___x_128_, 4);
v_isSharedCheck_141_ = !lean_is_exclusive(v___x_128_);
if (v_isSharedCheck_141_ == 0)
{
lean_object* v_unused_142_; 
v_unused_142_ = lean_ctor_get(v___x_128_, 0);
lean_dec(v_unused_142_);
v___x_134_ = v___x_128_;
v_isShared_135_ = v_isSharedCheck_141_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_diag_132_);
lean_inc(v_postponed_131_);
lean_inc(v_zetaDeltaFVarIds_130_);
lean_inc(v_cache_129_);
lean_dec(v___x_128_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_141_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_137_; 
if (v_isShared_135_ == 0)
{
lean_ctor_set(v___x_134_, 0, v_snd_127_);
v___x_137_ = v___x_134_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v_snd_127_);
lean_ctor_set(v_reuseFailAlloc_140_, 1, v_cache_129_);
lean_ctor_set(v_reuseFailAlloc_140_, 2, v_zetaDeltaFVarIds_130_);
lean_ctor_set(v_reuseFailAlloc_140_, 3, v_postponed_131_);
lean_ctor_set(v_reuseFailAlloc_140_, 4, v_diag_132_);
v___x_137_ = v_reuseFailAlloc_140_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = lean_st_ref_set(v___y_119_, v___x_137_);
v___x_139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_139_, 0, v_fst_126_);
return v___x_139_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg___boxed(lean_object* v_e_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg(v_e_143_, v___y_144_);
lean_dec(v___y_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2(lean_object* v_e_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg(v_e_147_, v___y_151_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___boxed(lean_object* v_e_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2(v_e_156_, v___y_157_, v___y_158_, v___y_159_, v___y_160_, v___y_161_, v___y_162_);
lean_dec(v___y_162_);
lean_dec_ref(v___y_161_);
lean_dec(v___y_160_);
lean_dec_ref(v___y_159_);
lean_dec(v___y_158_);
lean_dec_ref(v___y_157_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0(lean_object* v_msgData_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v___x_171_; lean_object* v_env_172_; lean_object* v___x_173_; lean_object* v_mctx_174_; lean_object* v_lctx_175_; lean_object* v_options_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_171_ = lean_st_ref_get(v___y_169_);
v_env_172_ = lean_ctor_get(v___x_171_, 0);
lean_inc_ref(v_env_172_);
lean_dec(v___x_171_);
v___x_173_ = lean_st_ref_get(v___y_167_);
v_mctx_174_ = lean_ctor_get(v___x_173_, 0);
lean_inc_ref(v_mctx_174_);
lean_dec(v___x_173_);
v_lctx_175_ = lean_ctor_get(v___y_166_, 2);
v_options_176_ = lean_ctor_get(v___y_168_, 2);
lean_inc_ref(v_options_176_);
lean_inc_ref(v_lctx_175_);
v___x_177_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_177_, 0, v_env_172_);
lean_ctor_set(v___x_177_, 1, v_mctx_174_);
lean_ctor_set(v___x_177_, 2, v_lctx_175_);
lean_ctor_set(v___x_177_, 3, v_options_176_);
v___x_178_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v_msgData_165_);
v___x_179_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0___boxed(lean_object* v_msgData_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0(v_msgData_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
return v_res_186_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__5(lean_object* v_opts_187_, lean_object* v_opt_188_){
_start:
{
lean_object* v_name_189_; lean_object* v_defValue_190_; lean_object* v_map_191_; lean_object* v___x_192_; 
v_name_189_ = lean_ctor_get(v_opt_188_, 0);
v_defValue_190_ = lean_ctor_get(v_opt_188_, 1);
v_map_191_ = lean_ctor_get(v_opts_187_, 0);
v___x_192_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_191_, v_name_189_);
if (lean_obj_tag(v___x_192_) == 0)
{
uint8_t v___x_193_; 
v___x_193_ = lean_unbox(v_defValue_190_);
return v___x_193_;
}
else
{
lean_object* v_val_194_; 
v_val_194_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_val_194_);
lean_dec_ref_known(v___x_192_, 1);
if (lean_obj_tag(v_val_194_) == 1)
{
uint8_t v_v_195_; 
v_v_195_ = lean_ctor_get_uint8(v_val_194_, 0);
lean_dec_ref_known(v_val_194_, 0);
return v_v_195_;
}
else
{
uint8_t v___x_196_; 
lean_dec(v_val_194_);
v___x_196_ = lean_unbox(v_defValue_190_);
return v___x_196_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__5___boxed(lean_object* v_opts_197_, lean_object* v_opt_198_){
_start:
{
uint8_t v_res_199_; lean_object* v_r_200_; 
v_res_199_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__5(v_opts_197_, v_opt_198_);
lean_dec_ref(v_opt_198_);
lean_dec_ref(v_opts_197_);
v_r_200_ = lean_box(v_res_199_);
return v_r_200_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_201_ = lean_box(1);
v___x_202_ = l_Lean_MessageData_ofFormat(v___x_201_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__3(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__2));
v___x_207_ = l_Lean_MessageData_ofFormat(v___x_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6(lean_object* v_x_208_, lean_object* v_x_209_){
_start:
{
if (lean_obj_tag(v_x_209_) == 0)
{
return v_x_208_;
}
else
{
lean_object* v_head_210_; lean_object* v_tail_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_233_; 
v_head_210_ = lean_ctor_get(v_x_209_, 0);
v_tail_211_ = lean_ctor_get(v_x_209_, 1);
v_isSharedCheck_233_ = !lean_is_exclusive(v_x_209_);
if (v_isSharedCheck_233_ == 0)
{
v___x_213_ = v_x_209_;
v_isShared_214_ = v_isSharedCheck_233_;
goto v_resetjp_212_;
}
else
{
lean_inc(v_tail_211_);
lean_inc(v_head_210_);
lean_dec(v_x_209_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_233_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
lean_object* v_before_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_231_; 
v_before_215_ = lean_ctor_get(v_head_210_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v_head_210_);
if (v_isSharedCheck_231_ == 0)
{
lean_object* v_unused_232_; 
v_unused_232_ = lean_ctor_get(v_head_210_, 1);
lean_dec(v_unused_232_);
v___x_217_ = v_head_210_;
v_isShared_218_ = v_isSharedCheck_231_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_before_215_);
lean_dec(v_head_210_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_231_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v___x_219_; lean_object* v___x_221_; 
v___x_219_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0);
if (v_isShared_218_ == 0)
{
lean_ctor_set_tag(v___x_217_, 7);
lean_ctor_set(v___x_217_, 1, v___x_219_);
lean_ctor_set(v___x_217_, 0, v_x_208_);
v___x_221_ = v___x_217_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v_x_208_);
lean_ctor_set(v_reuseFailAlloc_230_, 1, v___x_219_);
v___x_221_ = v_reuseFailAlloc_230_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
lean_object* v___x_222_; lean_object* v___x_224_; 
v___x_222_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__3);
if (v_isShared_214_ == 0)
{
lean_ctor_set_tag(v___x_213_, 7);
lean_ctor_set(v___x_213_, 1, v___x_222_);
lean_ctor_set(v___x_213_, 0, v___x_221_);
v___x_224_ = v___x_213_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v___x_221_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v___x_222_);
v___x_224_ = v_reuseFailAlloc_229_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_225_ = l_Lean_MessageData_ofSyntax(v_before_215_);
v___x_226_ = l_Lean_indentD(v___x_225_);
v___x_227_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_224_);
lean_ctor_set(v___x_227_, 1, v___x_226_);
v_x_208_ = v___x_227_;
v_x_209_ = v_tail_211_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_237_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__1));
v___x_238_ = l_Lean_MessageData_ofFormat(v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg(lean_object* v_msgData_239_, lean_object* v_macroStack_240_, lean_object* v___y_241_){
_start:
{
lean_object* v_options_243_; lean_object* v___x_244_; uint8_t v___x_245_; 
v_options_243_ = lean_ctor_get(v___y_241_, 2);
v___x_244_ = l_Lean_Elab_pp_macroStack;
v___x_245_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__5(v_options_243_, v___x_244_);
if (v___x_245_ == 0)
{
lean_object* v___x_246_; 
lean_dec(v_macroStack_240_);
v___x_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_246_, 0, v_msgData_239_);
return v___x_246_;
}
else
{
if (lean_obj_tag(v_macroStack_240_) == 0)
{
lean_object* v___x_247_; 
v___x_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_247_, 0, v_msgData_239_);
return v___x_247_;
}
else
{
lean_object* v_head_248_; lean_object* v_after_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_264_; 
v_head_248_ = lean_ctor_get(v_macroStack_240_, 0);
lean_inc(v_head_248_);
v_after_249_ = lean_ctor_get(v_head_248_, 1);
v_isSharedCheck_264_ = !lean_is_exclusive(v_head_248_);
if (v_isSharedCheck_264_ == 0)
{
lean_object* v_unused_265_; 
v_unused_265_ = lean_ctor_get(v_head_248_, 0);
lean_dec(v_unused_265_);
v___x_251_ = v_head_248_;
v_isShared_252_ = v_isSharedCheck_264_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_after_249_);
lean_dec(v_head_248_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_264_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_253_; lean_object* v___x_255_; 
v___x_253_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6___closed__0);
if (v_isShared_252_ == 0)
{
lean_ctor_set_tag(v___x_251_, 7);
lean_ctor_set(v___x_251_, 1, v___x_253_);
lean_ctor_set(v___x_251_, 0, v_msgData_239_);
v___x_255_ = v___x_251_;
goto v_reusejp_254_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_msgData_239_);
lean_ctor_set(v_reuseFailAlloc_263_, 1, v___x_253_);
v___x_255_ = v_reuseFailAlloc_263_;
goto v_reusejp_254_;
}
v_reusejp_254_:
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v_msgData_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_256_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___closed__2);
v___x_257_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_255_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
v___x_258_ = l_Lean_MessageData_ofSyntax(v_after_249_);
v___x_259_ = l_Lean_indentD(v___x_258_);
v_msgData_260_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_260_, 0, v___x_257_);
lean_ctor_set(v_msgData_260_, 1, v___x_259_);
v___x_261_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4_spec__6(v_msgData_260_, v_macroStack_240_);
v___x_262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_262_, 0, v___x_261_);
return v___x_262_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_266_, lean_object* v_macroStack_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg(v_msgData_266_, v_macroStack_267_, v___y_268_);
lean_dec_ref(v___y_268_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg(lean_object* v_msg_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_){
_start:
{
lean_object* v_ref_279_; lean_object* v___x_280_; lean_object* v_a_281_; lean_object* v_macroStack_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v_a_285_; lean_object* v___x_287_; uint8_t v_isShared_288_; uint8_t v_isSharedCheck_293_; 
v_ref_279_ = lean_ctor_get(v___y_276_, 5);
v___x_280_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0(v_msg_271_, v___y_274_, v___y_275_, v___y_276_, v___y_277_);
v_a_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_a_281_);
lean_dec_ref(v___x_280_);
v_macroStack_282_ = lean_ctor_get(v___y_272_, 1);
v___x_283_ = l_Lean_Elab_getBetterRef(v_ref_279_, v_macroStack_282_);
lean_inc(v_macroStack_282_);
v___x_284_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg(v_a_281_, v_macroStack_282_, v___y_276_);
v_a_285_ = lean_ctor_get(v___x_284_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_284_);
if (v_isSharedCheck_293_ == 0)
{
v___x_287_ = v___x_284_;
v_isShared_288_ = v_isSharedCheck_293_;
goto v_resetjp_286_;
}
else
{
lean_inc(v_a_285_);
lean_dec(v___x_284_);
v___x_287_ = lean_box(0);
v_isShared_288_ = v_isSharedCheck_293_;
goto v_resetjp_286_;
}
v_resetjp_286_:
{
lean_object* v___x_289_; lean_object* v___x_291_; 
v___x_289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_283_);
lean_ctor_set(v___x_289_, 1, v_a_285_);
if (v_isShared_288_ == 0)
{
lean_ctor_set_tag(v___x_287_, 1);
lean_ctor_set(v___x_287_, 0, v___x_289_);
v___x_291_ = v___x_287_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v___x_289_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg___boxed(lean_object* v_msg_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg(v_msg_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v___y_296_);
lean_dec_ref(v___y_295_);
return v_res_302_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__2(void){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_306_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__1));
v___x_307_ = l_Lean_stringToMessageData(v___x_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0(lean_object* v_f_308_, lean_object* v___x_309_, uint8_t v___x_310_, lean_object* v___x_311_, lean_object* v___x_312_, lean_object* v___x_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v___x_321_; 
lean_inc(v___x_309_);
v___x_321_ = l_Lean_Elab_Term_elabTerm(v_f_308_, v___x_309_, v___x_310_, v___x_310_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v_a_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_407_; 
v_a_322_ = lean_ctor_get(v___x_321_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_321_);
if (v_isSharedCheck_407_ == 0)
{
v___x_324_ = v___x_321_;
v_isShared_325_ = v_isSharedCheck_407_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_a_322_);
lean_dec(v___x_321_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_407_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___x_326_; lean_object* v___x_328_; 
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__0));
if (v_isShared_325_ == 0)
{
lean_ctor_set_tag(v___x_324_, 1);
lean_ctor_set(v___x_324_, 0, v___x_311_);
v___x_328_ = v___x_324_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_406_; 
v_reuseFailAlloc_406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_406_, 0, v___x_311_);
v___x_328_ = v_reuseFailAlloc_406_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
lean_object* v___x_329_; lean_object* v___x_330_; uint8_t v___x_331_; lean_object* v___x_332_; 
v___x_329_ = lean_mk_empty_array_with_capacity(v___x_312_);
lean_inc_ref(v___x_329_);
v___x_330_ = lean_array_push(v___x_329_, v___x_328_);
v___x_331_ = 0;
lean_inc(v___x_309_);
lean_inc(v_a_322_);
v___x_332_ = l_Lean_Elab_Term_elabAppArgs(v_a_322_, v___x_326_, v___x_330_, v___x_309_, v___x_331_, v___x_331_, v___x_310_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_332_) == 0)
{
lean_object* v_a_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_405_; 
v_a_333_ = lean_ctor_get(v___x_332_, 0);
v_isSharedCheck_405_ = !lean_is_exclusive(v___x_332_);
if (v_isSharedCheck_405_ == 0)
{
v___x_335_ = v___x_332_;
v_isShared_336_ = v_isSharedCheck_405_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_a_333_);
lean_dec(v___x_332_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_405_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v___x_338_; 
if (v_isShared_336_ == 0)
{
lean_ctor_set_tag(v___x_335_, 1);
lean_ctor_set(v___x_335_, 0, v___x_313_);
v___x_338_ = v___x_335_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v___x_313_);
v___x_338_ = v_reuseFailAlloc_404_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_339_ = lean_array_push(v___x_329_, v___x_338_);
v___x_340_ = l_Lean_Elab_Term_elabAppArgs(v_a_322_, v___x_326_, v___x_339_, v___x_309_, v___x_331_, v___x_331_, v___x_310_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_340_) == 0)
{
lean_object* v_a_341_; lean_object* v___y_343_; lean_object* v___y_344_; lean_object* v___y_345_; lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___x_363_; 
v_a_341_ = lean_ctor_get(v___x_340_, 0);
lean_inc(v_a_341_);
lean_dec_ref_known(v___x_340_, 1);
lean_inc(v___y_319_);
lean_inc_ref(v___y_318_);
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
lean_inc(v_a_333_);
v___x_363_ = lean_infer_type(v_a_333_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; lean_object* v___x_365_; 
v_a_364_ = lean_ctor_get(v___x_363_, 0);
lean_inc(v_a_364_);
lean_dec_ref_known(v___x_363_, 1);
lean_inc(v___y_319_);
lean_inc_ref(v___y_318_);
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
lean_inc(v_a_341_);
v___x_365_ = lean_infer_type(v_a_341_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_object* v_a_366_; lean_object* v___x_367_; 
v_a_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_a_366_);
lean_dec_ref_known(v___x_365_, 1);
v___x_367_ = l_Lean_Meta_isExprDefEq(v_a_364_, v_a_366_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_367_) == 0)
{
lean_object* v_a_368_; uint8_t v___x_369_; 
v_a_368_ = lean_ctor_get(v___x_367_, 0);
lean_inc(v_a_368_);
lean_dec_ref_known(v___x_367_, 1);
v___x_369_ = lean_unbox(v_a_368_);
lean_dec(v_a_368_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; 
lean_inc(v___y_319_);
lean_inc_ref(v___y_318_);
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
v___x_370_ = lean_infer_type(v_a_341_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_object* v_a_371_; lean_object* v___x_372_; 
v_a_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc(v_a_371_);
lean_dec_ref_known(v___x_370_, 1);
lean_inc(v___y_319_);
lean_inc_ref(v___y_318_);
lean_inc(v___y_317_);
lean_inc_ref(v___y_316_);
v___x_372_ = lean_infer_type(v_a_333_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
if (lean_obj_tag(v___x_372_) == 0)
{
lean_object* v_a_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v_a_373_ = lean_ctor_get(v___x_372_, 0);
lean_inc(v_a_373_);
lean_dec_ref_known(v___x_372_, 1);
v___x_374_ = lean_box(0);
v___x_375_ = l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(v_a_371_, v_a_373_, v___x_374_, v___x_326_);
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v_a_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v_a_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_387_; 
v_a_376_ = lean_ctor_get(v___x_375_, 0);
lean_inc(v_a_376_);
lean_dec_ref_known(v___x_375_, 1);
v___x_377_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__2);
v___x_378_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set(v___x_378_, 1, v_a_376_);
v___x_379_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg(v___x_378_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
v_a_380_ = lean_ctor_get(v___x_379_, 0);
v_isSharedCheck_387_ = !lean_is_exclusive(v___x_379_);
if (v_isSharedCheck_387_ == 0)
{
v___x_382_ = v___x_379_;
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_a_380_);
lean_dec(v___x_379_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_385_; 
if (v_isShared_383_ == 0)
{
v___x_385_ = v___x_382_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_a_380_);
v___x_385_ = v_reuseFailAlloc_386_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
return v___x_385_;
}
}
}
else
{
lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_395_; 
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
v_a_388_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_395_ == 0)
{
v___x_390_ = v___x_375_;
v_isShared_391_ = v_isSharedCheck_395_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_375_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_395_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_393_; 
if (v_isShared_391_ == 0)
{
v___x_393_ = v___x_390_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_a_388_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
}
else
{
lean_dec(v_a_371_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
return v___x_372_;
}
}
else
{
lean_dec(v_a_333_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
return v___x_370_;
}
}
else
{
v___y_343_ = v___y_314_;
v___y_344_ = v___y_315_;
v___y_345_ = v___y_316_;
v___y_346_ = v___y_317_;
v___y_347_ = v___y_318_;
v___y_348_ = v___y_319_;
goto v___jp_342_;
}
}
else
{
lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_403_; 
lean_dec(v_a_341_);
lean_dec(v_a_333_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
v_a_396_ = lean_ctor_get(v___x_367_, 0);
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_367_);
if (v_isSharedCheck_403_ == 0)
{
v___x_398_ = v___x_367_;
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_367_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v___x_401_; 
if (v_isShared_399_ == 0)
{
v___x_401_ = v___x_398_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v_a_396_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
}
else
{
lean_dec(v_a_364_);
lean_dec(v_a_341_);
lean_dec(v_a_333_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
return v___x_365_;
}
}
else
{
lean_dec(v_a_341_);
lean_dec(v_a_333_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
return v___x_363_;
}
v___jp_342_:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_349_ = l_Lean_Expr_headBeta(v_a_333_);
v___x_350_ = l_Lean_Expr_headBeta(v_a_341_);
v___x_351_ = l_Lean_Meta_mkEq(v___x_349_, v___x_350_, v___y_345_, v___y_346_, v___y_347_, v___y_348_);
if (lean_obj_tag(v___x_351_) == 0)
{
lean_object* v_a_352_; lean_object* v___x_353_; 
v_a_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_a_352_);
lean_dec_ref_known(v___x_351_, 1);
v___x_353_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsUsingDefault(v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
lean_dec_ref(v___y_345_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_object* v___x_354_; 
lean_dec_ref_known(v___x_353_, 1);
v___x_354_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__2___redArg(v_a_352_, v___y_346_);
lean_dec(v___y_346_);
return v___x_354_;
}
else
{
lean_object* v_a_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_362_; 
lean_dec(v_a_352_);
lean_dec(v___y_346_);
v_a_355_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_362_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_362_ == 0)
{
v___x_357_ = v___x_353_;
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_a_355_);
lean_dec(v___x_353_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v___x_360_; 
if (v_isShared_358_ == 0)
{
v___x_360_ = v___x_357_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v_a_355_);
v___x_360_ = v_reuseFailAlloc_361_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
return v___x_360_;
}
}
}
}
else
{
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
lean_dec(v___y_346_);
lean_dec_ref(v___y_345_);
return v___x_351_;
}
}
}
else
{
lean_dec(v_a_333_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
return v___x_340_;
}
}
}
}
else
{
lean_dec_ref(v___x_329_);
lean_dec(v_a_322_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
lean_dec_ref(v___x_313_);
lean_dec(v___x_309_);
return v___x_332_;
}
}
}
}
else
{
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
lean_dec_ref(v___x_313_);
lean_dec_ref(v___x_311_);
lean_dec(v___x_309_);
return v___x_321_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___boxed(lean_object* v_f_408_, lean_object* v___x_409_, lean_object* v___x_410_, lean_object* v___x_411_, lean_object* v___x_412_, lean_object* v___x_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_){
_start:
{
uint8_t v___x_20061__boxed_421_; lean_object* v_res_422_; 
v___x_20061__boxed_421_ = lean_unbox(v___x_410_);
v_res_422_ = lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0(v_f_408_, v___x_409_, v___x_20061__boxed_421_, v___x_411_, v___x_412_, v___x_413_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
lean_dec(v___x_412_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(lean_object* v_msg_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v_ref_429_; lean_object* v___x_430_; lean_object* v_a_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_439_; 
v_ref_429_ = lean_ctor_get(v___y_426_, 5);
v___x_430_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0(v_msg_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_);
v_a_431_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_439_ == 0)
{
v___x_433_ = v___x_430_;
v_isShared_434_ = v_isSharedCheck_439_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_a_431_);
lean_dec(v___x_430_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_439_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_435_; lean_object* v___x_437_; 
lean_inc(v_ref_429_);
v___x_435_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_435_, 0, v_ref_429_);
lean_ctor_set(v___x_435_, 1, v_a_431_);
if (v_isShared_434_ == 0)
{
lean_ctor_set_tag(v___x_433_, 1);
lean_ctor_set(v___x_433_, 0, v___x_435_);
v___x_437_ = v___x_433_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v___x_435_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg___boxed(lean_object* v_msg_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(v_msg_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
return v_res_446_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__5(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_455_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__4));
v___x_456_ = l_Lean_stringToMessageData(v___x_455_);
return v___x_456_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__7(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_458_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__6));
v___x_459_ = l_Lean_stringToMessageData(v___x_458_);
return v___x_459_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__12(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; 
v___x_466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__11));
v___x_467_ = l_Lean_stringToMessageData(v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp(lean_object* v_f_478_, lean_object* v_using_x3f_479_, lean_object* v_h_480_, lean_object* v_g_481_, lean_object* v_a_482_, lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v___y_492_; lean_object* v_fst_493_; lean_object* v_snd_494_; lean_object* v___y_495_; lean_object* v___y_496_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_538_; lean_object* v_fst_539_; lean_object* v_snd_540_; lean_object* v___y_541_; lean_object* v___y_542_; lean_object* v___y_543_; lean_object* v___y_544_; lean_object* v___y_572_; lean_object* v_fst_573_; lean_object* v_snd_574_; lean_object* v___y_575_; lean_object* v___y_576_; lean_object* v___y_577_; lean_object* v___y_578_; lean_object* v___y_594_; lean_object* v_fst_595_; lean_object* v_snd_596_; lean_object* v___y_597_; lean_object* v___y_598_; lean_object* v___y_599_; lean_object* v___y_600_; lean_object* v___y_616_; lean_object* v___y_617_; lean_object* v___y_618_; lean_object* v___y_619_; lean_object* v___y_620_; lean_object* v___y_621_; lean_object* v___y_622_; lean_object* v___y_623_; lean_object* v___y_624_; lean_object* v___y_636_; lean_object* v___y_637_; lean_object* v___y_638_; lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v___y_642_; lean_object* v___y_643_; lean_object* v___y_644_; lean_object* v_a_656_; 
if (lean_obj_tag(v_using_x3f_479_) == 0)
{
lean_object* v___x_937_; 
v___x_937_ = lean_box(0);
v_a_656_ = v___x_937_;
goto v___jp_655_;
}
else
{
lean_object* v_val_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_957_; 
v_val_938_ = lean_ctor_get(v_using_x3f_479_, 0);
v_isSharedCheck_957_ = !lean_is_exclusive(v_using_x3f_479_);
if (v_isSharedCheck_957_ == 0)
{
v___x_940_ = v_using_x3f_479_;
v_isShared_941_ = v_isSharedCheck_957_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_val_938_);
lean_dec(v_using_x3f_479_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_957_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v___x_942_; uint8_t v___x_943_; lean_object* v___x_944_; 
v___x_942_ = lean_box(0);
v___x_943_ = 0;
v___x_944_ = l_Lean_Elab_Tactic_elabTerm(v_val_938_, v___x_942_, v___x_943_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_944_) == 0)
{
lean_object* v_a_945_; lean_object* v___x_947_; 
v_a_945_ = lean_ctor_get(v___x_944_, 0);
lean_inc(v_a_945_);
lean_dec_ref_known(v___x_944_, 1);
if (v_isShared_941_ == 0)
{
lean_ctor_set(v___x_940_, 0, v_a_945_);
v___x_947_ = v___x_940_;
goto v_reusejp_946_;
}
else
{
lean_object* v_reuseFailAlloc_948_; 
v_reuseFailAlloc_948_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_948_, 0, v_a_945_);
v___x_947_ = v_reuseFailAlloc_948_;
goto v_reusejp_946_;
}
v_reusejp_946_:
{
v_a_656_ = v___x_947_;
goto v___jp_655_;
}
}
else
{
lean_object* v_a_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_956_; 
lean_del_object(v___x_940_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v_a_949_ = lean_ctor_get(v___x_944_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_944_);
if (v_isSharedCheck_956_ == 0)
{
v___x_951_ = v___x_944_;
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_a_949_);
lean_dec(v___x_944_);
v___x_951_ = lean_box(0);
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
v_resetjp_950_:
{
lean_object* v___x_954_; 
if (v_isShared_952_ == 0)
{
v___x_954_ = v___x_951_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v_a_949_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
}
}
v___jp_491_:
{
lean_object* v___x_499_; 
v___x_499_ = l_Lean_MVarId_clear(v_g_481_, v_h_480_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_object* v_a_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; 
v_a_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_a_500_);
lean_dec_ref_known(v___x_499_, 1);
v___x_501_ = l_Lean_LocalDecl_userName(v___y_492_);
lean_dec_ref(v___y_492_);
v___x_502_ = lean_box(0);
v___x_503_ = l_Lean_MVarId_note(v_a_500_, v___x_501_, v_fst_493_, v___x_502_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
if (lean_obj_tag(v___x_503_) == 0)
{
lean_object* v_a_504_; lean_object* v___x_506_; uint8_t v_isShared_507_; uint8_t v_isSharedCheck_520_; 
v_a_504_ = lean_ctor_get(v___x_503_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_503_);
if (v_isSharedCheck_520_ == 0)
{
v___x_506_ = v___x_503_;
v_isShared_507_ = v_isSharedCheck_520_;
goto v_resetjp_505_;
}
else
{
lean_inc(v_a_504_);
lean_dec(v___x_503_);
v___x_506_ = lean_box(0);
v_isShared_507_ = v_isSharedCheck_520_;
goto v_resetjp_505_;
}
v_resetjp_505_:
{
lean_object* v_snd_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_518_; 
v_snd_508_ = lean_ctor_get(v_a_504_, 1);
v_isSharedCheck_518_ = !lean_is_exclusive(v_a_504_);
if (v_isSharedCheck_518_ == 0)
{
lean_object* v_unused_519_; 
v_unused_519_ = lean_ctor_get(v_a_504_, 0);
lean_dec(v_unused_519_);
v___x_510_ = v_a_504_;
v_isShared_511_ = v_isSharedCheck_518_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_snd_508_);
lean_dec(v_a_504_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_518_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_513_; 
if (v_isShared_511_ == 0)
{
lean_ctor_set_tag(v___x_510_, 1);
lean_ctor_set(v___x_510_, 1, v_snd_494_);
lean_ctor_set(v___x_510_, 0, v_snd_508_);
v___x_513_ = v___x_510_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_snd_508_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v_snd_494_);
v___x_513_ = v_reuseFailAlloc_517_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
lean_object* v___x_515_; 
if (v_isShared_507_ == 0)
{
lean_ctor_set(v___x_506_, 0, v___x_513_);
v___x_515_ = v___x_506_;
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
}
}
else
{
lean_object* v_a_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_528_; 
lean_dec(v_snd_494_);
v_a_521_ = lean_ctor_get(v___x_503_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v___x_503_);
if (v_isSharedCheck_528_ == 0)
{
v___x_523_ = v___x_503_;
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_a_521_);
lean_dec(v___x_503_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_528_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___x_526_; 
if (v_isShared_524_ == 0)
{
v___x_526_ = v___x_523_;
goto v_reusejp_525_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v_a_521_);
v___x_526_ = v_reuseFailAlloc_527_;
goto v_reusejp_525_;
}
v_reusejp_525_:
{
return v___x_526_;
}
}
}
}
else
{
lean_object* v_a_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_536_; 
lean_dec(v_snd_494_);
lean_dec_ref(v_fst_493_);
lean_dec_ref(v___y_492_);
v_a_529_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_536_ == 0)
{
v___x_531_ = v___x_499_;
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_a_529_);
lean_dec(v___x_499_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_536_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_534_; 
if (v_isShared_532_ == 0)
{
v___x_534_ = v___x_531_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v_a_529_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
}
}
}
}
v___jp_537_:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; 
v___x_545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__3));
v___x_546_ = lean_unsigned_to_nat(1u);
v___x_547_ = lean_mk_empty_array_with_capacity(v___x_546_);
lean_inc_ref(v___x_547_);
v___x_548_ = lean_array_push(v___x_547_, v_fst_539_);
v___x_549_ = l_Lean_Meta_mkAppM(v___x_545_, v___x_548_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
if (lean_obj_tag(v___x_549_) == 0)
{
lean_object* v_a_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; 
v_a_550_ = lean_ctor_get(v___x_549_, 0);
lean_inc(v_a_550_);
lean_dec_ref_known(v___x_549_, 1);
lean_inc_ref(v___y_538_);
v___x_551_ = l_Lean_LocalDecl_toExpr(v___y_538_);
v___x_552_ = lean_array_push(v___x_547_, v___x_551_);
v___x_553_ = l_Lean_Meta_mkAppM_x27(v_a_550_, v___x_552_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
if (lean_obj_tag(v___x_553_) == 0)
{
lean_object* v_a_554_; 
v_a_554_ = lean_ctor_get(v___x_553_, 0);
lean_inc(v_a_554_);
lean_dec_ref_known(v___x_553_, 1);
v___y_492_ = v___y_538_;
v_fst_493_ = v_a_554_;
v_snd_494_ = v_snd_540_;
v___y_495_ = v___y_541_;
v___y_496_ = v___y_542_;
v___y_497_ = v___y_543_;
v___y_498_ = v___y_544_;
goto v___jp_491_;
}
else
{
lean_object* v_a_555_; lean_object* v___x_557_; uint8_t v_isShared_558_; uint8_t v_isSharedCheck_562_; 
lean_dec(v_snd_540_);
lean_dec_ref(v___y_538_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_555_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_562_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_562_ == 0)
{
v___x_557_ = v___x_553_;
v_isShared_558_ = v_isSharedCheck_562_;
goto v_resetjp_556_;
}
else
{
lean_inc(v_a_555_);
lean_dec(v___x_553_);
v___x_557_ = lean_box(0);
v_isShared_558_ = v_isSharedCheck_562_;
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
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v_a_555_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
}
else
{
lean_object* v_a_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_570_; 
lean_dec_ref(v___x_547_);
lean_dec(v_snd_540_);
lean_dec_ref(v___y_538_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_563_ = lean_ctor_get(v___x_549_, 0);
v_isSharedCheck_570_ = !lean_is_exclusive(v___x_549_);
if (v_isSharedCheck_570_ == 0)
{
v___x_565_ = v___x_549_;
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
else
{
lean_inc(v_a_563_);
lean_dec(v___x_549_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v___x_568_; 
if (v_isShared_566_ == 0)
{
v___x_568_ = v___x_565_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v_a_563_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
v___jp_571_:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
lean_inc_ref(v___y_572_);
v___x_579_ = l_Lean_LocalDecl_toExpr(v___y_572_);
v___x_580_ = lean_unsigned_to_nat(1u);
v___x_581_ = lean_mk_empty_array_with_capacity(v___x_580_);
v___x_582_ = lean_array_push(v___x_581_, v___x_579_);
v___x_583_ = l_Lean_Meta_mkAppM_x27(v_fst_573_, v___x_582_, v___y_575_, v___y_576_, v___y_577_, v___y_578_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v_a_584_; 
v_a_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_584_);
lean_dec_ref_known(v___x_583_, 1);
v___y_492_ = v___y_572_;
v_fst_493_ = v_a_584_;
v_snd_494_ = v_snd_574_;
v___y_495_ = v___y_575_;
v___y_496_ = v___y_576_;
v___y_497_ = v___y_577_;
v___y_498_ = v___y_578_;
goto v___jp_491_;
}
else
{
lean_object* v_a_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
lean_dec(v_snd_574_);
lean_dec_ref(v___y_572_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_585_ = lean_ctor_get(v___x_583_, 0);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_583_);
if (v_isSharedCheck_592_ == 0)
{
v___x_587_ = v___x_583_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_a_585_);
lean_dec(v___x_583_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_a_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
v___jp_593_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
lean_inc_ref(v___y_594_);
v___x_601_ = l_Lean_LocalDecl_toExpr(v___y_594_);
v___x_602_ = lean_unsigned_to_nat(1u);
v___x_603_ = lean_mk_empty_array_with_capacity(v___x_602_);
v___x_604_ = lean_array_push(v___x_603_, v___x_601_);
v___x_605_ = l_Lean_Meta_mkAppM_x27(v_fst_595_, v___x_604_, v___y_597_, v___y_598_, v___y_599_, v___y_600_);
if (lean_obj_tag(v___x_605_) == 0)
{
lean_object* v_a_606_; 
v_a_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_a_606_);
lean_dec_ref_known(v___x_605_, 1);
v___y_492_ = v___y_594_;
v_fst_493_ = v_a_606_;
v_snd_494_ = v_snd_596_;
v___y_495_ = v___y_597_;
v___y_496_ = v___y_598_;
v___y_497_ = v___y_599_;
v___y_498_ = v___y_600_;
goto v___jp_491_;
}
else
{
lean_object* v_a_607_; lean_object* v___x_609_; uint8_t v_isShared_610_; uint8_t v_isSharedCheck_614_; 
lean_dec(v_snd_596_);
lean_dec_ref(v___y_594_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_607_ = lean_ctor_get(v___x_605_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_605_);
if (v_isSharedCheck_614_ == 0)
{
v___x_609_ = v___x_605_;
v_isShared_610_ = v_isSharedCheck_614_;
goto v_resetjp_608_;
}
else
{
lean_inc(v_a_607_);
lean_dec(v___x_605_);
v___x_609_ = lean_box(0);
v_isShared_610_ = v_isSharedCheck_614_;
goto v_resetjp_608_;
}
v_resetjp_608_:
{
lean_object* v___x_612_; 
if (v_isShared_610_ == 0)
{
v___x_612_ = v___x_609_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_a_607_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
v___jp_615_:
{
lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v_a_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_634_; 
lean_dec_ref(v___y_616_);
v___x_625_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__5, &lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__5);
v___x_626_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(v___x_625_, v___y_621_, v___y_622_, v___y_623_, v___y_624_);
v_a_627_ = lean_ctor_get(v___x_626_, 0);
v_isSharedCheck_634_ = !lean_is_exclusive(v___x_626_);
if (v_isSharedCheck_634_ == 0)
{
v___x_629_ = v___x_626_;
v_isShared_630_ = v_isSharedCheck_634_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_a_627_);
lean_dec(v___x_626_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_634_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___x_632_; 
if (v_isShared_630_ == 0)
{
v___x_632_ = v___x_629_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_633_; 
v_reuseFailAlloc_633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_633_, 0, v_a_627_);
v___x_632_ = v_reuseFailAlloc_633_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
return v___x_632_;
}
}
}
v___jp_635_:
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v_a_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_654_; 
lean_dec_ref(v___y_636_);
v___x_645_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__7, &lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__7);
v___x_646_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(v___x_645_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
v_a_647_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_654_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_654_ == 0)
{
v___x_649_ = v___x_646_;
v_isShared_650_ = v_isSharedCheck_654_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_a_647_);
lean_dec(v___x_646_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_654_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
lean_object* v___x_652_; 
if (v_isShared_650_ == 0)
{
v___x_652_ = v___x_649_;
goto v_reusejp_651_;
}
else
{
lean_object* v_reuseFailAlloc_653_; 
v_reuseFailAlloc_653_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_653_, 0, v_a_647_);
v___x_652_ = v_reuseFailAlloc_653_;
goto v_reusejp_651_;
}
v_reusejp_651_:
{
return v___x_652_;
}
}
}
v___jp_655_:
{
lean_object* v___x_657_; 
lean_inc(v_h_480_);
v___x_657_ = l_Lean_FVarId_getDecl___redArg(v_h_480_, v_a_486_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_657_) == 0)
{
lean_object* v_a_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v_a_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_928_; 
v_a_658_ = lean_ctor_get(v___x_657_, 0);
lean_inc(v_a_658_);
lean_dec_ref_known(v___x_657_, 1);
v___x_659_ = l_Lean_LocalDecl_type(v_a_658_);
v___x_660_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg(v___x_659_, v_a_487_);
v_a_661_ = lean_ctor_get(v___x_660_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_660_);
if (v_isSharedCheck_928_ == 0)
{
v___x_663_ = v___x_660_;
v_isShared_664_ = v_isSharedCheck_928_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_a_661_);
lean_dec(v___x_660_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_928_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v___x_665_; 
v___x_665_ = l_Lean_Meta_whnfR(v_a_661_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v_a_666_; lean_object* v___x_667_; lean_object* v_fst_668_; 
v_a_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v___x_665_, 1);
v___x_667_ = l_Lean_Expr_getAppFnArgs(v_a_666_);
v_fst_668_ = lean_ctor_get(v___x_667_, 0);
lean_inc(v_fst_668_);
if (lean_obj_tag(v_fst_668_) == 1)
{
lean_object* v_pre_669_; 
v_pre_669_ = lean_ctor_get(v_fst_668_, 0);
lean_inc(v_pre_669_);
switch(lean_obj_tag(v_pre_669_))
{
case 0:
{
lean_object* v_snd_670_; lean_object* v_str_671_; lean_object* v___x_672_; uint8_t v___x_673_; 
v_snd_670_ = lean_ctor_get(v___x_667_, 1);
lean_inc(v_snd_670_);
lean_dec_ref(v___x_667_);
v_str_671_ = lean_ctor_get(v_fst_668_, 1);
lean_inc_ref(v_str_671_);
lean_dec_ref_known(v_fst_668_, 2);
v___x_672_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__8));
v___x_673_ = lean_string_dec_eq(v_str_671_, v___x_672_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; uint8_t v___x_675_; 
v___x_674_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__9));
v___x_675_ = lean_string_dec_eq(v_str_671_, v___x_674_);
lean_dec_ref(v_str_671_);
if (v___x_675_ == 0)
{
lean_dec(v_snd_670_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
else
{
lean_object* v___x_676_; lean_object* v___x_677_; uint8_t v___x_678_; 
v___x_676_ = lean_array_get_size(v_snd_670_);
v___x_677_ = lean_unsigned_to_nat(1u);
v___x_678_ = lean_nat_dec_eq(v___x_676_, v___x_677_);
if (v___x_678_ == 0)
{
lean_dec(v_snd_670_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
else
{
lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_679_ = lean_unsigned_to_nat(0u);
v___x_680_ = lean_array_fget(v_snd_670_, v___x_679_);
lean_dec(v_snd_670_);
v___x_681_ = l_Lean_Meta_whnfR(v___x_680_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_681_) == 0)
{
lean_object* v_a_682_; lean_object* v___x_683_; lean_object* v_fst_684_; lean_object* v___x_686_; uint8_t v_isShared_687_; uint8_t v_isSharedCheck_735_; 
v_a_682_ = lean_ctor_get(v___x_681_, 0);
lean_inc(v_a_682_);
lean_dec_ref_known(v___x_681_, 1);
v___x_683_ = l_Lean_Expr_getAppFnArgs(v_a_682_);
v_fst_684_ = lean_ctor_get(v___x_683_, 0);
v_isSharedCheck_735_ = !lean_is_exclusive(v___x_683_);
if (v_isSharedCheck_735_ == 0)
{
lean_object* v_unused_736_; 
v_unused_736_ = lean_ctor_get(v___x_683_, 1);
lean_dec(v_unused_736_);
v___x_686_ = v___x_683_;
v_isShared_687_ = v_isSharedCheck_735_;
goto v_resetjp_685_;
}
else
{
lean_inc(v_fst_684_);
lean_dec(v___x_683_);
v___x_686_ = lean_box(0);
v_isShared_687_ = v_isSharedCheck_735_;
goto v_resetjp_685_;
}
v_resetjp_685_:
{
if (lean_obj_tag(v_fst_684_) == 1)
{
lean_object* v_pre_688_; 
v_pre_688_ = lean_ctor_get(v_fst_684_, 0);
lean_inc(v_pre_688_);
if (lean_obj_tag(v_pre_688_) == 0)
{
lean_object* v_str_689_; uint8_t v___x_690_; 
v_str_689_ = lean_ctor_get(v_fst_684_, 1);
lean_inc_ref(v_str_689_);
lean_dec_ref_known(v_fst_684_, 2);
v___x_690_ = lean_string_dec_eq(v_str_689_, v___x_672_);
lean_dec_ref(v_str_689_);
if (v___x_690_ == 0)
{
lean_del_object(v___x_686_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_636_ = v_a_658_;
v___y_637_ = v_a_482_;
v___y_638_ = v_a_483_;
v___y_639_ = v_a_484_;
v___y_640_ = v_a_485_;
v___y_641_ = v_a_486_;
v___y_642_ = v_a_487_;
v___y_643_ = v_a_488_;
v___y_644_ = v_a_489_;
goto v___jp_635_;
}
else
{
if (lean_obj_tag(v_a_656_) == 0)
{
lean_object* v___x_691_; 
v___x_691_ = l_Lean_Elab_Tactic_elabTermForApply(v_f_478_, v___x_690_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_691_) == 0)
{
lean_object* v_a_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; 
v_a_692_ = lean_ctor_get(v___x_691_, 0);
lean_inc(v_a_692_);
lean_dec_ref_known(v___x_691_, 1);
v___x_693_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10));
v___x_694_ = lean_mk_empty_array_with_capacity(v___x_677_);
v___x_695_ = lean_array_push(v___x_694_, v_a_692_);
v___x_696_ = l_Lean_Meta_mkAppM(v___x_693_, v___x_695_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_696_) == 0)
{
lean_object* v_a_697_; lean_object* v___x_699_; 
v_a_697_ = lean_ctor_get(v___x_696_, 0);
lean_inc(v_a_697_);
lean_dec_ref_known(v___x_696_, 1);
if (v_isShared_664_ == 0)
{
lean_ctor_set_tag(v___x_663_, 1);
lean_ctor_set(v___x_663_, 0, v_a_697_);
v___x_699_ = v___x_663_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v_a_697_);
v___x_699_ = v_reuseFailAlloc_716_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
uint8_t v___x_700_; lean_object* v___x_701_; 
v___x_700_ = 0;
v___x_701_ = l_Lean_Meta_mkFreshExprMVar(v___x_699_, v___x_700_, v_pre_688_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_701_) == 0)
{
lean_object* v_a_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_706_; 
v_a_702_ = lean_ctor_get(v___x_701_, 0);
lean_inc(v_a_702_);
lean_dec_ref_known(v___x_701_, 1);
v___x_703_ = l_Lean_Expr_mvarId_x21(v_a_702_);
v___x_704_ = lean_box(0);
if (v_isShared_687_ == 0)
{
lean_ctor_set_tag(v___x_686_, 1);
lean_ctor_set(v___x_686_, 1, v___x_704_);
lean_ctor_set(v___x_686_, 0, v___x_703_);
v___x_706_ = v___x_686_;
goto v_reusejp_705_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v___x_703_);
lean_ctor_set(v_reuseFailAlloc_707_, 1, v___x_704_);
v___x_706_ = v_reuseFailAlloc_707_;
goto v_reusejp_705_;
}
v_reusejp_705_:
{
v___y_538_ = v_a_658_;
v_fst_539_ = v_a_702_;
v_snd_540_ = v___x_706_;
v___y_541_ = v_a_486_;
v___y_542_ = v_a_487_;
v___y_543_ = v_a_488_;
v___y_544_ = v_a_489_;
goto v___jp_537_;
}
}
else
{
lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_715_; 
lean_del_object(v___x_686_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_708_ = lean_ctor_get(v___x_701_, 0);
v_isSharedCheck_715_ = !lean_is_exclusive(v___x_701_);
if (v_isSharedCheck_715_ == 0)
{
v___x_710_ = v___x_701_;
v_isShared_711_ = v_isSharedCheck_715_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_dec(v___x_701_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_715_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_713_; 
if (v_isShared_711_ == 0)
{
v___x_713_ = v___x_710_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v_a_708_);
v___x_713_ = v_reuseFailAlloc_714_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
return v___x_713_;
}
}
}
}
}
else
{
lean_object* v_a_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_724_; 
lean_del_object(v___x_686_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_717_ = lean_ctor_get(v___x_696_, 0);
v_isSharedCheck_724_ = !lean_is_exclusive(v___x_696_);
if (v_isSharedCheck_724_ == 0)
{
v___x_719_ = v___x_696_;
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_a_717_);
lean_dec(v___x_696_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v___x_722_; 
if (v_isShared_720_ == 0)
{
v___x_722_ = v___x_719_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v_a_717_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
}
}
else
{
lean_object* v_a_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_732_; 
lean_del_object(v___x_686_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_725_ = lean_ctor_get(v___x_691_, 0);
v_isSharedCheck_732_ = !lean_is_exclusive(v___x_691_);
if (v_isSharedCheck_732_ == 0)
{
v___x_727_ = v___x_691_;
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_a_725_);
lean_dec(v___x_691_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v___x_730_; 
if (v_isShared_728_ == 0)
{
v___x_730_ = v___x_727_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v_a_725_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
return v___x_730_;
}
}
}
}
else
{
lean_object* v_val_733_; lean_object* v___x_734_; 
lean_del_object(v___x_686_);
lean_del_object(v___x_663_);
lean_dec(v_f_478_);
v_val_733_ = lean_ctor_get(v_a_656_, 0);
lean_inc(v_val_733_);
lean_dec_ref_known(v_a_656_, 1);
v___x_734_ = lean_box(0);
v___y_538_ = v_a_658_;
v_fst_539_ = v_val_733_;
v_snd_540_ = v___x_734_;
v___y_541_ = v_a_486_;
v___y_542_ = v_a_487_;
v___y_543_ = v_a_488_;
v___y_544_ = v_a_489_;
goto v___jp_537_;
}
}
}
else
{
lean_dec_ref_known(v_fst_684_, 2);
lean_dec(v_pre_688_);
lean_del_object(v___x_686_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_636_ = v_a_658_;
v___y_637_ = v_a_482_;
v___y_638_ = v_a_483_;
v___y_639_ = v_a_484_;
v___y_640_ = v_a_485_;
v___y_641_ = v_a_486_;
v___y_642_ = v_a_487_;
v___y_643_ = v_a_488_;
v___y_644_ = v_a_489_;
goto v___jp_635_;
}
}
else
{
lean_del_object(v___x_686_);
lean_dec(v_fst_684_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_636_ = v_a_658_;
v___y_637_ = v_a_482_;
v___y_638_ = v_a_483_;
v___y_639_ = v_a_484_;
v___y_640_ = v_a_485_;
v___y_641_ = v_a_486_;
v___y_642_ = v_a_487_;
v___y_643_ = v_a_488_;
v___y_644_ = v_a_489_;
goto v___jp_635_;
}
}
}
else
{
lean_object* v_a_737_; lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_744_; 
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v_a_737_ = lean_ctor_get(v___x_681_, 0);
v_isSharedCheck_744_ = !lean_is_exclusive(v___x_681_);
if (v_isSharedCheck_744_ == 0)
{
v___x_739_ = v___x_681_;
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_a_737_);
lean_dec(v___x_681_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v___x_742_; 
if (v_isShared_740_ == 0)
{
v___x_742_ = v___x_739_;
goto v_reusejp_741_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v_a_737_);
v___x_742_ = v_reuseFailAlloc_743_;
goto v_reusejp_741_;
}
v_reusejp_741_:
{
return v___x_742_;
}
}
}
}
}
}
else
{
lean_object* v___x_745_; lean_object* v___x_746_; uint8_t v___x_747_; 
lean_dec_ref(v_str_671_);
lean_dec(v_a_656_);
v___x_745_ = lean_array_get_size(v_snd_670_);
v___x_746_ = lean_unsigned_to_nat(3u);
v___x_747_ = lean_nat_dec_eq(v___x_745_, v___x_746_);
if (v___x_747_ == 0)
{
lean_dec(v_snd_670_);
lean_del_object(v___x_663_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
else
{
lean_object* v___x_748_; 
lean_inc(v_g_481_);
v___x_748_ = l_Lean_MVarId_getTag(v_g_481_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_748_) == 0)
{
lean_object* v_a_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___f_756_; uint8_t v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; 
v_a_749_ = lean_ctor_get(v___x_748_, 0);
lean_inc(v_a_749_);
lean_dec_ref_known(v___x_748_, 1);
v___x_750_ = lean_unsigned_to_nat(1u);
v___x_751_ = lean_array_fget(v_snd_670_, v___x_750_);
v___x_752_ = lean_unsigned_to_nat(2u);
v___x_753_ = lean_array_fget(v_snd_670_, v___x_752_);
lean_dec(v_snd_670_);
v___x_754_ = lean_box(0);
v___x_755_ = lean_box(v___x_747_);
v___f_756_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___boxed), 13, 6);
lean_closure_set(v___f_756_, 0, v_f_478_);
lean_closure_set(v___f_756_, 1, v___x_754_);
lean_closure_set(v___f_756_, 2, v___x_755_);
lean_closure_set(v___f_756_, 3, v___x_751_);
lean_closure_set(v___f_756_, 4, v___x_750_);
lean_closure_set(v___f_756_, 5, v___x_753_);
v___x_757_ = 0;
v___x_758_ = lean_box(v___x_757_);
v___x_759_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_runTermElab___boxed), 12, 3);
lean_closure_set(v___x_759_, 0, lean_box(0));
lean_closure_set(v___x_759_, 1, v___f_756_);
lean_closure_set(v___x_759_, 2, v___x_758_);
v___x_760_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_760_, 0, lean_box(0));
lean_closure_set(v___x_760_, 1, v___x_759_);
v___x_761_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_762_ = l_Lean_Elab_Tactic_withCollectingNewGoalsFrom(v___x_760_, v_a_749_, v___x_761_, v___x_757_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_762_) == 0)
{
lean_object* v_a_763_; lean_object* v_fst_764_; lean_object* v_snd_765_; lean_object* v___x_767_; 
v_a_763_ = lean_ctor_get(v___x_762_, 0);
lean_inc(v_a_763_);
lean_dec_ref_known(v___x_762_, 1);
v_fst_764_ = lean_ctor_get(v_a_763_, 0);
lean_inc(v_fst_764_);
v_snd_765_ = lean_ctor_get(v_a_763_, 1);
lean_inc(v_snd_765_);
lean_dec(v_a_763_);
if (v_isShared_664_ == 0)
{
lean_ctor_set_tag(v___x_663_, 1);
lean_ctor_set(v___x_663_, 0, v_fst_764_);
v___x_767_ = v___x_663_;
goto v_reusejp_766_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v_fst_764_);
v___x_767_ = v_reuseFailAlloc_796_;
goto v_reusejp_766_;
}
v_reusejp_766_:
{
uint8_t v___x_768_; lean_object* v___x_769_; 
v___x_768_ = 0;
v___x_769_ = l_Lean_Meta_mkFreshExprMVar(v___x_767_, v___x_768_, v_pre_669_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_769_) == 0)
{
lean_object* v_a_770_; lean_object* v___x_771_; uint8_t v___x_772_; uint8_t v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; 
v_a_770_ = lean_ctor_get(v___x_769_, 0);
lean_inc(v_a_770_);
lean_dec_ref_known(v___x_769_, 1);
v___x_771_ = l_Lean_Expr_mvarId_x21(v_a_770_);
v___x_772_ = 2;
v___x_773_ = 1;
v___x_774_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v___x_774_, 0, v___x_754_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1, v___x_747_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 1, v___x_747_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 2, v___x_772_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 3, v___x_772_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 4, v___x_773_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 5, v___x_747_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 6, v___x_747_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 7, v___x_757_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 8, v___x_757_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 9, v___x_757_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 10, v___x_757_);
lean_ctor_set_uint8(v___x_774_, sizeof(void*)*1 + 11, v___x_747_);
v___x_775_ = lean_box(0);
v___x_776_ = lp_mathlib_Lean_MVarId_congrN_x21(v___x_771_, v___x_754_, v___x_774_, v___x_775_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_776_) == 0)
{
lean_object* v_a_777_; 
v_a_777_ = lean_ctor_get(v___x_776_, 0);
lean_inc(v_a_777_);
lean_dec_ref_known(v___x_776_, 1);
if (lean_obj_tag(v_a_777_) == 0)
{
v___y_492_ = v_a_658_;
v_fst_493_ = v_a_770_;
v_snd_494_ = v_snd_765_;
v___y_495_ = v_a_486_;
v___y_496_ = v_a_487_;
v___y_497_ = v_a_488_;
v___y_498_ = v_a_489_;
goto v___jp_491_;
}
else
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v_a_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_787_; 
lean_dec(v_a_777_);
lean_dec(v_a_770_);
lean_dec(v_snd_765_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v___x_778_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__12, &lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__12);
v___x_779_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(v___x_778_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
v_a_780_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_787_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_787_ == 0)
{
v___x_782_ = v___x_779_;
v_isShared_783_ = v_isSharedCheck_787_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_a_780_);
lean_dec(v___x_779_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_787_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
lean_object* v___x_785_; 
if (v_isShared_783_ == 0)
{
v___x_785_ = v___x_782_;
goto v_reusejp_784_;
}
else
{
lean_object* v_reuseFailAlloc_786_; 
v_reuseFailAlloc_786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_786_, 0, v_a_780_);
v___x_785_ = v_reuseFailAlloc_786_;
goto v_reusejp_784_;
}
v_reusejp_784_:
{
return v___x_785_;
}
}
}
}
else
{
lean_dec(v_a_770_);
lean_dec(v_snd_765_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
return v___x_776_;
}
}
else
{
lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_795_; 
lean_dec(v_snd_765_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_788_ = lean_ctor_get(v___x_769_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_769_);
if (v_isSharedCheck_795_ == 0)
{
v___x_790_ = v___x_769_;
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_769_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v___x_793_; 
if (v_isShared_791_ == 0)
{
v___x_793_ = v___x_790_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_a_788_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
}
}
else
{
lean_object* v_a_797_; lean_object* v___x_799_; uint8_t v_isShared_800_; uint8_t v_isSharedCheck_804_; 
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_797_ = lean_ctor_get(v___x_762_, 0);
v_isSharedCheck_804_ = !lean_is_exclusive(v___x_762_);
if (v_isSharedCheck_804_ == 0)
{
v___x_799_ = v___x_762_;
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
else
{
lean_inc(v_a_797_);
lean_dec(v___x_762_);
v___x_799_ = lean_box(0);
v_isShared_800_ = v_isSharedCheck_804_;
goto v_resetjp_798_;
}
v_resetjp_798_:
{
lean_object* v___x_802_; 
if (v_isShared_800_ == 0)
{
v___x_802_ = v___x_799_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v_a_797_);
v___x_802_ = v_reuseFailAlloc_803_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
return v___x_802_;
}
}
}
}
else
{
lean_object* v_a_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_812_; 
lean_dec(v_snd_670_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v_a_805_ = lean_ctor_get(v___x_748_, 0);
v_isSharedCheck_812_ = !lean_is_exclusive(v___x_748_);
if (v_isSharedCheck_812_ == 0)
{
v___x_807_ = v___x_748_;
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_a_805_);
lean_dec(v___x_748_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_812_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v___x_810_; 
if (v_isShared_808_ == 0)
{
v___x_810_ = v___x_807_;
goto v_reusejp_809_;
}
else
{
lean_object* v_reuseFailAlloc_811_; 
v_reuseFailAlloc_811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_811_, 0, v_a_805_);
v___x_810_ = v_reuseFailAlloc_811_;
goto v_reusejp_809_;
}
v_reusejp_809_:
{
return v___x_810_;
}
}
}
}
}
}
case 1:
{
lean_object* v___x_814_; uint8_t v_isShared_815_; uint8_t v_isSharedCheck_917_; 
v_isSharedCheck_917_ = !lean_is_exclusive(v___x_667_);
if (v_isSharedCheck_917_ == 0)
{
lean_object* v_unused_918_; lean_object* v_unused_919_; 
v_unused_918_ = lean_ctor_get(v___x_667_, 1);
lean_dec(v_unused_918_);
v_unused_919_ = lean_ctor_get(v___x_667_, 0);
lean_dec(v_unused_919_);
v___x_814_ = v___x_667_;
v_isShared_815_ = v_isSharedCheck_917_;
goto v_resetjp_813_;
}
else
{
lean_dec(v___x_667_);
v___x_814_ = lean_box(0);
v_isShared_815_ = v_isSharedCheck_917_;
goto v_resetjp_813_;
}
v_resetjp_813_:
{
lean_object* v_pre_816_; 
v_pre_816_ = lean_ctor_get(v_pre_669_, 0);
lean_inc(v_pre_816_);
if (lean_obj_tag(v_pre_816_) == 0)
{
lean_object* v_str_817_; lean_object* v_str_818_; lean_object* v___x_819_; uint8_t v___x_820_; 
v_str_817_ = lean_ctor_get(v_fst_668_, 1);
lean_inc_ref(v_str_817_);
lean_dec_ref_known(v_fst_668_, 2);
v_str_818_ = lean_ctor_get(v_pre_669_, 1);
lean_inc_ref(v_str_818_);
lean_dec_ref_known(v_pre_669_, 2);
v___x_819_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__13));
v___x_820_ = lean_string_dec_eq(v_str_818_, v___x_819_);
if (v___x_820_ == 0)
{
lean_object* v___x_821_; uint8_t v___x_822_; 
v___x_821_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__14));
v___x_822_ = lean_string_dec_eq(v_str_818_, v___x_821_);
lean_dec_ref(v_str_818_);
if (v___x_822_ == 0)
{
lean_dec_ref(v_str_817_);
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
else
{
lean_object* v___x_823_; uint8_t v___x_824_; 
v___x_823_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__15));
v___x_824_ = lean_string_dec_eq(v_str_817_, v___x_823_);
lean_dec_ref(v_str_817_);
if (v___x_824_ == 0)
{
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
else
{
if (lean_obj_tag(v_a_656_) == 0)
{
lean_object* v___x_825_; 
v___x_825_ = l_Lean_Elab_Tactic_elabTermForApply(v_f_478_, v___x_824_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_825_) == 0)
{
lean_object* v_a_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v_a_826_ = lean_ctor_get(v___x_825_, 0);
lean_inc(v_a_826_);
lean_dec_ref_known(v___x_825_, 1);
v___x_827_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__17));
v___x_828_ = lean_unsigned_to_nat(1u);
v___x_829_ = lean_mk_empty_array_with_capacity(v___x_828_);
v___x_830_ = lean_array_push(v___x_829_, v_a_826_);
v___x_831_ = l_Lean_Meta_mkAppM(v___x_827_, v___x_830_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_831_) == 0)
{
lean_object* v_a_832_; lean_object* v___x_834_; 
v_a_832_ = lean_ctor_get(v___x_831_, 0);
lean_inc(v_a_832_);
lean_dec_ref_known(v___x_831_, 1);
if (v_isShared_664_ == 0)
{
lean_ctor_set_tag(v___x_663_, 1);
lean_ctor_set(v___x_663_, 0, v_a_832_);
v___x_834_ = v___x_663_;
goto v_reusejp_833_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_a_832_);
v___x_834_ = v_reuseFailAlloc_851_;
goto v_reusejp_833_;
}
v_reusejp_833_:
{
uint8_t v___x_835_; lean_object* v___x_836_; 
v___x_835_ = 0;
v___x_836_ = l_Lean_Meta_mkFreshExprMVar(v___x_834_, v___x_835_, v_pre_816_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_836_) == 0)
{
lean_object* v_a_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_841_; 
v_a_837_ = lean_ctor_get(v___x_836_, 0);
lean_inc(v_a_837_);
lean_dec_ref_known(v___x_836_, 1);
v___x_838_ = l_Lean_Expr_mvarId_x21(v_a_837_);
v___x_839_ = lean_box(0);
if (v_isShared_815_ == 0)
{
lean_ctor_set_tag(v___x_814_, 1);
lean_ctor_set(v___x_814_, 1, v___x_839_);
lean_ctor_set(v___x_814_, 0, v___x_838_);
v___x_841_ = v___x_814_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_842_; 
v_reuseFailAlloc_842_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_842_, 0, v___x_838_);
lean_ctor_set(v_reuseFailAlloc_842_, 1, v___x_839_);
v___x_841_ = v_reuseFailAlloc_842_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
v___y_572_ = v_a_658_;
v_fst_573_ = v_a_837_;
v_snd_574_ = v___x_841_;
v___y_575_ = v_a_486_;
v___y_576_ = v_a_487_;
v___y_577_ = v_a_488_;
v___y_578_ = v_a_489_;
goto v___jp_571_;
}
}
else
{
lean_object* v_a_843_; lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_850_; 
lean_del_object(v___x_814_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_843_ = lean_ctor_get(v___x_836_, 0);
v_isSharedCheck_850_ = !lean_is_exclusive(v___x_836_);
if (v_isSharedCheck_850_ == 0)
{
v___x_845_ = v___x_836_;
v_isShared_846_ = v_isSharedCheck_850_;
goto v_resetjp_844_;
}
else
{
lean_inc(v_a_843_);
lean_dec(v___x_836_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_850_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v___x_848_; 
if (v_isShared_846_ == 0)
{
v___x_848_ = v___x_845_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v_a_843_);
v___x_848_ = v_reuseFailAlloc_849_;
goto v_reusejp_847_;
}
v_reusejp_847_:
{
return v___x_848_;
}
}
}
}
}
else
{
lean_object* v_a_852_; lean_object* v___x_854_; uint8_t v_isShared_855_; uint8_t v_isSharedCheck_859_; 
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_852_ = lean_ctor_get(v___x_831_, 0);
v_isSharedCheck_859_ = !lean_is_exclusive(v___x_831_);
if (v_isSharedCheck_859_ == 0)
{
v___x_854_ = v___x_831_;
v_isShared_855_ = v_isSharedCheck_859_;
goto v_resetjp_853_;
}
else
{
lean_inc(v_a_852_);
lean_dec(v___x_831_);
v___x_854_ = lean_box(0);
v_isShared_855_ = v_isSharedCheck_859_;
goto v_resetjp_853_;
}
v_resetjp_853_:
{
lean_object* v___x_857_; 
if (v_isShared_855_ == 0)
{
v___x_857_ = v___x_854_;
goto v_reusejp_856_;
}
else
{
lean_object* v_reuseFailAlloc_858_; 
v_reuseFailAlloc_858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_858_, 0, v_a_852_);
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
else
{
lean_object* v_a_860_; lean_object* v___x_862_; uint8_t v_isShared_863_; uint8_t v_isSharedCheck_867_; 
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_860_ = lean_ctor_get(v___x_825_, 0);
v_isSharedCheck_867_ = !lean_is_exclusive(v___x_825_);
if (v_isSharedCheck_867_ == 0)
{
v___x_862_ = v___x_825_;
v_isShared_863_ = v_isSharedCheck_867_;
goto v_resetjp_861_;
}
else
{
lean_inc(v_a_860_);
lean_dec(v___x_825_);
v___x_862_ = lean_box(0);
v_isShared_863_ = v_isSharedCheck_867_;
goto v_resetjp_861_;
}
v_resetjp_861_:
{
lean_object* v___x_865_; 
if (v_isShared_863_ == 0)
{
v___x_865_ = v___x_862_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_866_; 
v_reuseFailAlloc_866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_866_, 0, v_a_860_);
v___x_865_ = v_reuseFailAlloc_866_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
return v___x_865_;
}
}
}
}
else
{
lean_object* v_val_868_; lean_object* v___x_869_; 
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_f_478_);
v_val_868_ = lean_ctor_get(v_a_656_, 0);
lean_inc(v_val_868_);
lean_dec_ref_known(v_a_656_, 1);
v___x_869_ = lean_box(0);
v___y_572_ = v_a_658_;
v_fst_573_ = v_val_868_;
v_snd_574_ = v___x_869_;
v___y_575_ = v_a_486_;
v___y_576_ = v_a_487_;
v___y_577_ = v_a_488_;
v___y_578_ = v_a_489_;
goto v___jp_571_;
}
}
}
}
else
{
lean_object* v___x_870_; uint8_t v___x_871_; 
lean_dec_ref(v_str_818_);
v___x_870_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__18));
v___x_871_ = lean_string_dec_eq(v_str_817_, v___x_870_);
lean_dec_ref(v_str_817_);
if (v___x_871_ == 0)
{
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
else
{
if (lean_obj_tag(v_a_656_) == 0)
{
lean_object* v___x_872_; 
v___x_872_ = l_Lean_Elab_Tactic_elabTermForApply(v_f_478_, v___x_871_, v_a_482_, v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_872_) == 0)
{
lean_object* v_a_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; 
v_a_873_ = lean_ctor_get(v___x_872_, 0);
lean_inc(v_a_873_);
lean_dec_ref_known(v___x_872_, 1);
v___x_874_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__20));
v___x_875_ = lean_unsigned_to_nat(1u);
v___x_876_ = lean_mk_empty_array_with_capacity(v___x_875_);
v___x_877_ = lean_array_push(v___x_876_, v_a_873_);
v___x_878_ = l_Lean_Meta_mkAppM(v___x_874_, v___x_877_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_878_) == 0)
{
lean_object* v_a_879_; lean_object* v___x_881_; 
v_a_879_ = lean_ctor_get(v___x_878_, 0);
lean_inc(v_a_879_);
lean_dec_ref_known(v___x_878_, 1);
if (v_isShared_664_ == 0)
{
lean_ctor_set_tag(v___x_663_, 1);
lean_ctor_set(v___x_663_, 0, v_a_879_);
v___x_881_ = v___x_663_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v_a_879_);
v___x_881_ = v_reuseFailAlloc_898_;
goto v_reusejp_880_;
}
v_reusejp_880_:
{
uint8_t v___x_882_; lean_object* v___x_883_; 
v___x_882_ = 0;
v___x_883_ = l_Lean_Meta_mkFreshExprMVar(v___x_881_, v___x_882_, v_pre_816_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v_a_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_888_; 
v_a_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref_known(v___x_883_, 1);
v___x_885_ = l_Lean_Expr_mvarId_x21(v_a_884_);
v___x_886_ = lean_box(0);
if (v_isShared_815_ == 0)
{
lean_ctor_set_tag(v___x_814_, 1);
lean_ctor_set(v___x_814_, 1, v___x_886_);
lean_ctor_set(v___x_814_, 0, v___x_885_);
v___x_888_ = v___x_814_;
goto v_reusejp_887_;
}
else
{
lean_object* v_reuseFailAlloc_889_; 
v_reuseFailAlloc_889_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_889_, 0, v___x_885_);
lean_ctor_set(v_reuseFailAlloc_889_, 1, v___x_886_);
v___x_888_ = v_reuseFailAlloc_889_;
goto v_reusejp_887_;
}
v_reusejp_887_:
{
v___y_594_ = v_a_658_;
v_fst_595_ = v_a_884_;
v_snd_596_ = v___x_888_;
v___y_597_ = v_a_486_;
v___y_598_ = v_a_487_;
v___y_599_ = v_a_488_;
v___y_600_ = v_a_489_;
goto v___jp_593_;
}
}
else
{
lean_object* v_a_890_; lean_object* v___x_892_; uint8_t v_isShared_893_; uint8_t v_isSharedCheck_897_; 
lean_del_object(v___x_814_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_890_ = lean_ctor_get(v___x_883_, 0);
v_isSharedCheck_897_ = !lean_is_exclusive(v___x_883_);
if (v_isSharedCheck_897_ == 0)
{
v___x_892_ = v___x_883_;
v_isShared_893_ = v_isSharedCheck_897_;
goto v_resetjp_891_;
}
else
{
lean_inc(v_a_890_);
lean_dec(v___x_883_);
v___x_892_ = lean_box(0);
v_isShared_893_ = v_isSharedCheck_897_;
goto v_resetjp_891_;
}
v_resetjp_891_:
{
lean_object* v___x_895_; 
if (v_isShared_893_ == 0)
{
v___x_895_ = v___x_892_;
goto v_reusejp_894_;
}
else
{
lean_object* v_reuseFailAlloc_896_; 
v_reuseFailAlloc_896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_896_, 0, v_a_890_);
v___x_895_ = v_reuseFailAlloc_896_;
goto v_reusejp_894_;
}
v_reusejp_894_:
{
return v___x_895_;
}
}
}
}
}
else
{
lean_object* v_a_899_; lean_object* v___x_901_; uint8_t v_isShared_902_; uint8_t v_isSharedCheck_906_; 
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_899_ = lean_ctor_get(v___x_878_, 0);
v_isSharedCheck_906_ = !lean_is_exclusive(v___x_878_);
if (v_isSharedCheck_906_ == 0)
{
v___x_901_ = v___x_878_;
v_isShared_902_ = v_isSharedCheck_906_;
goto v_resetjp_900_;
}
else
{
lean_inc(v_a_899_);
lean_dec(v___x_878_);
v___x_901_ = lean_box(0);
v_isShared_902_ = v_isSharedCheck_906_;
goto v_resetjp_900_;
}
v_resetjp_900_:
{
lean_object* v___x_904_; 
if (v_isShared_902_ == 0)
{
v___x_904_ = v___x_901_;
goto v_reusejp_903_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v_a_899_);
v___x_904_ = v_reuseFailAlloc_905_;
goto v_reusejp_903_;
}
v_reusejp_903_:
{
return v___x_904_;
}
}
}
}
else
{
lean_object* v_a_907_; lean_object* v___x_909_; uint8_t v_isShared_910_; uint8_t v_isSharedCheck_914_; 
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
v_a_907_ = lean_ctor_get(v___x_872_, 0);
v_isSharedCheck_914_ = !lean_is_exclusive(v___x_872_);
if (v_isSharedCheck_914_ == 0)
{
v___x_909_ = v___x_872_;
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
else
{
lean_inc(v_a_907_);
lean_dec(v___x_872_);
v___x_909_ = lean_box(0);
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
v_resetjp_908_:
{
lean_object* v___x_912_; 
if (v_isShared_910_ == 0)
{
v___x_912_ = v___x_909_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v_a_907_);
v___x_912_ = v_reuseFailAlloc_913_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
return v___x_912_;
}
}
}
}
else
{
lean_object* v_val_915_; lean_object* v___x_916_; 
lean_del_object(v___x_814_);
lean_del_object(v___x_663_);
lean_dec(v_f_478_);
v_val_915_ = lean_ctor_get(v_a_656_, 0);
lean_inc(v_val_915_);
lean_dec_ref_known(v_a_656_, 1);
v___x_916_ = lean_box(0);
v___y_594_ = v_a_658_;
v_fst_595_ = v_val_915_;
v_snd_596_ = v___x_916_;
v___y_597_ = v_a_486_;
v___y_598_ = v_a_487_;
v___y_599_ = v_a_488_;
v___y_600_ = v_a_489_;
goto v___jp_593_;
}
}
}
}
else
{
lean_dec(v_pre_816_);
lean_del_object(v___x_814_);
lean_dec_ref_known(v_pre_669_, 2);
lean_dec_ref_known(v_fst_668_, 2);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
}
}
default: 
{
lean_dec_ref_known(v_fst_668_, 2);
lean_dec(v_pre_669_);
lean_dec_ref(v___x_667_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
}
}
else
{
lean_dec(v_fst_668_);
lean_dec_ref(v___x_667_);
lean_del_object(v___x_663_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v___y_616_ = v_a_658_;
v___y_617_ = v_a_482_;
v___y_618_ = v_a_483_;
v___y_619_ = v_a_484_;
v___y_620_ = v_a_485_;
v___y_621_ = v_a_486_;
v___y_622_ = v_a_487_;
v___y_623_ = v_a_488_;
v___y_624_ = v_a_489_;
goto v___jp_615_;
}
}
else
{
lean_object* v_a_920_; lean_object* v___x_922_; uint8_t v_isShared_923_; uint8_t v_isSharedCheck_927_; 
lean_del_object(v___x_663_);
lean_dec(v_a_658_);
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v_a_920_ = lean_ctor_get(v___x_665_, 0);
v_isSharedCheck_927_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_927_ == 0)
{
v___x_922_ = v___x_665_;
v_isShared_923_ = v_isSharedCheck_927_;
goto v_resetjp_921_;
}
else
{
lean_inc(v_a_920_);
lean_dec(v___x_665_);
v___x_922_ = lean_box(0);
v_isShared_923_ = v_isSharedCheck_927_;
goto v_resetjp_921_;
}
v_resetjp_921_:
{
lean_object* v___x_925_; 
if (v_isShared_923_ == 0)
{
v___x_925_ = v___x_922_;
goto v_reusejp_924_;
}
else
{
lean_object* v_reuseFailAlloc_926_; 
v_reuseFailAlloc_926_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_926_, 0, v_a_920_);
v___x_925_ = v_reuseFailAlloc_926_;
goto v_reusejp_924_;
}
v_reusejp_924_:
{
return v___x_925_;
}
}
}
}
}
else
{
lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_936_; 
lean_dec(v_a_656_);
lean_dec(v_g_481_);
lean_dec(v_h_480_);
lean_dec(v_f_478_);
v_a_929_ = lean_ctor_get(v___x_657_, 0);
v_isSharedCheck_936_ = !lean_is_exclusive(v___x_657_);
if (v_isSharedCheck_936_ == 0)
{
v___x_931_ = v___x_657_;
v_isShared_932_ = v_isSharedCheck_936_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_dec(v___x_657_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunHyp___boxed(lean_object* v_f_958_, lean_object* v_using_x3f_959_, lean_object* v_h_960_, lean_object* v_g_961_, lean_object* v_a_962_, lean_object* v_a_963_, lean_object* v_a_964_, lean_object* v_a_965_, lean_object* v_a_966_, lean_object* v_a_967_, lean_object* v_a_968_, lean_object* v_a_969_, lean_object* v_a_970_){
_start:
{
lean_object* v_res_971_; 
v_res_971_ = lp_mathlib_Mathlib_Tactic_applyFunHyp(v_f_958_, v_using_x3f_959_, v_h_960_, v_g_961_, v_a_962_, v_a_963_, v_a_964_, v_a_965_, v_a_966_, v_a_967_, v_a_968_, v_a_969_);
lean_dec(v_a_969_);
lean_dec_ref(v_a_968_);
lean_dec(v_a_967_);
lean_dec_ref(v_a_966_);
lean_dec(v_a_965_);
lean_dec_ref(v_a_964_);
lean_dec(v_a_963_);
lean_dec_ref(v_a_962_);
return v_res_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0(lean_object* v_00_u03b1_972_, lean_object* v_msg_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
lean_object* v___x_983_; 
v___x_983_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(v_msg_973_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
return v___x_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___boxed(lean_object* v_00_u03b1_984_, lean_object* v_msg_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_){
_start:
{
lean_object* v_res_995_; 
v_res_995_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0(v_00_u03b1_984_, v_msg_985_, v___y_986_, v___y_987_, v___y_988_, v___y_989_, v___y_990_, v___y_991_, v___y_992_, v___y_993_);
lean_dec(v___y_993_);
lean_dec_ref(v___y_992_);
lean_dec(v___y_991_);
lean_dec_ref(v___y_990_);
lean_dec(v___y_989_);
lean_dec_ref(v___y_988_);
lean_dec(v___y_987_);
lean_dec_ref(v___y_986_);
return v_res_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3(lean_object* v_00_u03b1_996_, lean_object* v_msg_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_){
_start:
{
lean_object* v___x_1005_; 
v___x_1005_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___redArg(v_msg_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_, v___y_1003_);
return v___x_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3___boxed(lean_object* v_00_u03b1_1006_, lean_object* v_msg_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_){
_start:
{
lean_object* v_res_1015_; 
v_res_1015_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3(v_00_u03b1_1006_, v_msg_1007_, v___y_1008_, v___y_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
lean_dec(v___y_1011_);
lean_dec_ref(v___y_1010_);
lean_dec(v___y_1009_);
lean_dec_ref(v___y_1008_);
return v_res_1015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4(lean_object* v_msgData_1016_, lean_object* v_macroStack_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
lean_object* v___x_1025_; 
v___x_1025_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___redArg(v_msgData_1016_, v_macroStack_1017_, v___y_1022_);
return v___x_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4___boxed(lean_object* v_msgData_1026_, lean_object* v_macroStack_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_){
_start:
{
lean_object* v_res_1035_; 
v_res_1035_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__3_spec__4(v_msgData_1026_, v_macroStack_1027_, v___y_1028_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_);
lean_dec(v___y_1033_);
lean_dec_ref(v___y_1032_);
lean_dec(v___y_1031_);
lean_dec_ref(v___y_1030_);
lean_dec(v___y_1029_);
lean_dec_ref(v___y_1028_);
return v_res_1035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(lean_object* v_msg_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_){
_start:
{
lean_object* v_ref_1042_; lean_object* v___x_1043_; lean_object* v_a_1044_; lean_object* v___x_1046_; uint8_t v_isShared_1047_; uint8_t v_isSharedCheck_1052_; 
v_ref_1042_ = lean_ctor_get(v___y_1039_, 5);
v___x_1043_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0_spec__0(v_msg_1036_, v___y_1037_, v___y_1038_, v___y_1039_, v___y_1040_);
v_a_1044_ = lean_ctor_get(v___x_1043_, 0);
v_isSharedCheck_1052_ = !lean_is_exclusive(v___x_1043_);
if (v_isSharedCheck_1052_ == 0)
{
v___x_1046_ = v___x_1043_;
v_isShared_1047_ = v_isSharedCheck_1052_;
goto v_resetjp_1045_;
}
else
{
lean_inc(v_a_1044_);
lean_dec(v___x_1043_);
v___x_1046_ = lean_box(0);
v_isShared_1047_ = v_isSharedCheck_1052_;
goto v_resetjp_1045_;
}
v_resetjp_1045_:
{
lean_object* v___x_1048_; lean_object* v___x_1050_; 
lean_inc(v_ref_1042_);
v___x_1048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1048_, 0, v_ref_1042_);
lean_ctor_set(v___x_1048_, 1, v_a_1044_);
if (v_isShared_1047_ == 0)
{
lean_ctor_set_tag(v___x_1046_, 1);
lean_ctor_set(v___x_1046_, 0, v___x_1048_);
v___x_1050_ = v___x_1046_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v___x_1048_);
v___x_1050_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
return v___x_1050_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg___boxed(lean_object* v_msg_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_){
_start:
{
lean_object* v_res_1059_; 
v_res_1059_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(v_msg_1053_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_);
lean_dec(v___y_1057_);
lean_dec_ref(v___y_1056_);
lean_dec(v___y_1055_);
lean_dec_ref(v___y_1054_);
return v_res_1059_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__1(void){
_start:
{
lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1061_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__0));
v___x_1062_ = l_Lean_stringToMessageData(v___x_1061_);
return v___x_1062_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__3(void){
_start:
{
lean_object* v___x_1064_; lean_object* v___x_1065_; 
v___x_1064_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__2));
v___x_1065_ = l_Lean_stringToMessageData(v___x_1064_);
return v___x_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(lean_object* v_f_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_, lean_object* v_a_1069_, lean_object* v_a_1070_){
_start:
{
lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; 
v___x_1072_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__1, &lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__1);
v___x_1073_ = l_Lean_MessageData_ofSyntax(v_f_1066_);
v___x_1074_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1074_, 0, v___x_1072_);
lean_ctor_set(v___x_1074_, 1, v___x_1073_);
v___x_1075_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__3, &lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___closed__3);
v___x_1076_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1076_, 0, v___x_1074_);
lean_ctor_set(v___x_1076_, 1, v___x_1075_);
v___x_1077_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(v___x_1076_, v_a_1067_, v_a_1068_, v_a_1069_, v_a_1070_);
return v___x_1077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTargetFailure___boxed(lean_object* v_f_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_){
_start:
{
lean_object* v_res_1084_; 
v_res_1084_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_1078_, v_a_1079_, v_a_1080_, v_a_1081_, v_a_1082_);
lean_dec(v_a_1082_);
lean_dec_ref(v_a_1081_);
lean_dec(v_a_1080_);
lean_dec_ref(v_a_1079_);
return v_res_1084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0(lean_object* v_00_u03b1_1085_, lean_object* v_msg_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
lean_object* v___x_1092_; 
v___x_1092_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(v_msg_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___boxed(lean_object* v_00_u03b1_1093_, lean_object* v_msg_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_){
_start:
{
lean_object* v_res_1100_; 
v_res_1100_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0(v_00_u03b1_1093_, v_msg_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
lean_dec(v___y_1098_);
lean_dec_ref(v___y_1097_);
lean_dec(v___y_1096_);
lean_dec_ref(v___y_1095_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg(lean_object* v_x_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
lean_object* v___x_1107_; 
v___x_1107_ = l_Lean_Meta_saveState___redArg(v___y_1103_, v___y_1105_);
if (lean_obj_tag(v___x_1107_) == 0)
{
lean_object* v_a_1108_; lean_object* v___x_1109_; 
v_a_1108_ = lean_ctor_get(v___x_1107_, 0);
lean_inc(v_a_1108_);
lean_dec_ref_known(v___x_1107_, 1);
lean_inc(v___y_1105_);
lean_inc_ref(v___y_1104_);
lean_inc(v___y_1103_);
lean_inc_ref(v___y_1102_);
v___x_1109_ = lean_apply_5(v_x_1101_, v___y_1102_, v___y_1103_, v___y_1104_, v___y_1105_, lean_box(0));
if (lean_obj_tag(v___x_1109_) == 0)
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1118_; 
lean_dec(v_a_1108_);
v_a_1110_ = lean_ctor_get(v___x_1109_, 0);
v_isSharedCheck_1118_ = !lean_is_exclusive(v___x_1109_);
if (v_isSharedCheck_1118_ == 0)
{
v___x_1112_ = v___x_1109_;
v_isShared_1113_ = v_isSharedCheck_1118_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1109_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1118_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1114_; lean_object* v___x_1116_; 
v___x_1114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1114_, 0, v_a_1110_);
if (v_isShared_1113_ == 0)
{
lean_ctor_set(v___x_1112_, 0, v___x_1114_);
v___x_1116_ = v___x_1112_;
goto v_reusejp_1115_;
}
else
{
lean_object* v_reuseFailAlloc_1117_; 
v_reuseFailAlloc_1117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1117_, 0, v___x_1114_);
v___x_1116_ = v_reuseFailAlloc_1117_;
goto v_reusejp_1115_;
}
v_reusejp_1115_:
{
return v___x_1116_;
}
}
}
else
{
lean_object* v_a_1119_; lean_object* v___x_1121_; uint8_t v_isShared_1122_; uint8_t v_isSharedCheck_1148_; 
v_a_1119_ = lean_ctor_get(v___x_1109_, 0);
v_isSharedCheck_1148_ = !lean_is_exclusive(v___x_1109_);
if (v_isSharedCheck_1148_ == 0)
{
v___x_1121_ = v___x_1109_;
v_isShared_1122_ = v_isSharedCheck_1148_;
goto v_resetjp_1120_;
}
else
{
lean_inc(v_a_1119_);
lean_dec(v___x_1109_);
v___x_1121_ = lean_box(0);
v_isShared_1122_ = v_isSharedCheck_1148_;
goto v_resetjp_1120_;
}
v_resetjp_1120_:
{
uint8_t v___y_1124_; uint8_t v___x_1146_; 
v___x_1146_ = l_Lean_Exception_isInterrupt(v_a_1119_);
if (v___x_1146_ == 0)
{
uint8_t v___x_1147_; 
lean_inc(v_a_1119_);
v___x_1147_ = l_Lean_Exception_isRuntime(v_a_1119_);
v___y_1124_ = v___x_1147_;
goto v___jp_1123_;
}
else
{
v___y_1124_ = v___x_1146_;
goto v___jp_1123_;
}
v___jp_1123_:
{
if (v___y_1124_ == 0)
{
lean_object* v___x_1125_; 
lean_del_object(v___x_1121_);
lean_dec(v_a_1119_);
v___x_1125_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1108_, v___y_1103_, v___y_1105_);
lean_dec(v_a_1108_);
if (lean_obj_tag(v___x_1125_) == 0)
{
lean_object* v___x_1127_; uint8_t v_isShared_1128_; uint8_t v_isSharedCheck_1133_; 
v_isSharedCheck_1133_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1133_ == 0)
{
lean_object* v_unused_1134_; 
v_unused_1134_ = lean_ctor_get(v___x_1125_, 0);
lean_dec(v_unused_1134_);
v___x_1127_ = v___x_1125_;
v_isShared_1128_ = v_isSharedCheck_1133_;
goto v_resetjp_1126_;
}
else
{
lean_dec(v___x_1125_);
v___x_1127_ = lean_box(0);
v_isShared_1128_ = v_isSharedCheck_1133_;
goto v_resetjp_1126_;
}
v_resetjp_1126_:
{
lean_object* v___x_1129_; lean_object* v___x_1131_; 
v___x_1129_ = lean_box(0);
if (v_isShared_1128_ == 0)
{
lean_ctor_set(v___x_1127_, 0, v___x_1129_);
v___x_1131_ = v___x_1127_;
goto v_reusejp_1130_;
}
else
{
lean_object* v_reuseFailAlloc_1132_; 
v_reuseFailAlloc_1132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1132_, 0, v___x_1129_);
v___x_1131_ = v_reuseFailAlloc_1132_;
goto v_reusejp_1130_;
}
v_reusejp_1130_:
{
return v___x_1131_;
}
}
}
else
{
lean_object* v_a_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1142_; 
v_a_1135_ = lean_ctor_get(v___x_1125_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1137_ = v___x_1125_;
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_a_1135_);
lean_dec(v___x_1125_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1140_; 
if (v_isShared_1138_ == 0)
{
v___x_1140_ = v___x_1137_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_a_1135_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
else
{
lean_object* v___x_1144_; 
lean_dec(v_a_1108_);
if (v_isShared_1122_ == 0)
{
v___x_1144_ = v___x_1121_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v_a_1119_);
v___x_1144_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1143_;
}
v_reusejp_1143_:
{
return v___x_1144_;
}
}
}
}
}
}
else
{
lean_object* v_a_1149_; lean_object* v___x_1151_; uint8_t v_isShared_1152_; uint8_t v_isSharedCheck_1156_; 
lean_dec_ref(v_x_1101_);
v_a_1149_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1156_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1156_ == 0)
{
v___x_1151_ = v___x_1107_;
v_isShared_1152_ = v_isSharedCheck_1156_;
goto v_resetjp_1150_;
}
else
{
lean_inc(v_a_1149_);
lean_dec(v___x_1107_);
v___x_1151_ = lean_box(0);
v_isShared_1152_ = v_isSharedCheck_1156_;
goto v_resetjp_1150_;
}
v_resetjp_1150_:
{
lean_object* v___x_1154_; 
if (v_isShared_1152_ == 0)
{
v___x_1154_ = v___x_1151_;
goto v_reusejp_1153_;
}
else
{
lean_object* v_reuseFailAlloc_1155_; 
v_reuseFailAlloc_1155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1155_, 0, v_a_1149_);
v___x_1154_ = v_reuseFailAlloc_1155_;
goto v_reusejp_1153_;
}
v_reusejp_1153_:
{
return v___x_1154_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg___boxed(lean_object* v_x_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_){
_start:
{
lean_object* v_res_1163_; 
v_res_1163_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg(v_x_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_);
lean_dec(v___y_1161_);
lean_dec_ref(v___y_1160_);
lean_dec(v___y_1159_);
lean_dec_ref(v___y_1158_);
return v_res_1163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0(lean_object* v_00_u03b1_1164_, lean_object* v_x_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_){
_start:
{
lean_object* v___x_1171_; 
v___x_1171_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg(v_x_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_);
return v___x_1171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___boxed(lean_object* v_00_u03b1_1172_, lean_object* v_x_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_){
_start:
{
lean_object* v_res_1179_; 
v_res_1179_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0(v_00_u03b1_1172_, v_x_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_);
lean_dec(v___y_1177_);
lean_dec_ref(v___y_1176_);
lean_dec(v___y_1175_);
lean_dec_ref(v___y_1174_);
return v_res_1179_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___x_1181_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__0));
v___x_1182_ = l_Lean_stringToMessageData(v___x_1181_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0(lean_object* v___x_1183_, uint8_t v___x_1184_, uint8_t v_a_1185_, lean_object* v___x_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_){
_start:
{
lean_object* v___x_1192_; 
v___x_1192_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v___x_1183_, v___y_1187_, v___y_1188_, v___y_1189_, v___y_1190_);
if (lean_obj_tag(v___x_1192_) == 0)
{
lean_object* v_a_1193_; uint8_t v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; 
v_a_1193_ = lean_ctor_get(v___x_1192_, 0);
lean_inc(v_a_1193_);
lean_dec_ref_known(v___x_1192_, 1);
v___x_1194_ = 0;
v___x_1195_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_1195_, 0, v___x_1194_);
lean_ctor_set_uint8(v___x_1195_, 1, v___x_1184_);
lean_ctor_set_uint8(v___x_1195_, 2, v_a_1185_);
lean_ctor_set_uint8(v___x_1195_, 3, v___x_1184_);
v___x_1196_ = lean_box(0);
v___x_1197_ = l_Lean_MVarId_apply(v___x_1186_, v_a_1193_, v___x_1195_, v___x_1196_, v___y_1187_, v___y_1188_, v___y_1189_, v___y_1190_);
if (lean_obj_tag(v___x_1197_) == 0)
{
lean_object* v_a_1198_; lean_object* v___x_1200_; uint8_t v_isShared_1201_; uint8_t v_isSharedCheck_1208_; 
v_a_1198_ = lean_ctor_get(v___x_1197_, 0);
v_isSharedCheck_1208_ = !lean_is_exclusive(v___x_1197_);
if (v_isSharedCheck_1208_ == 0)
{
v___x_1200_ = v___x_1197_;
v_isShared_1201_ = v_isSharedCheck_1208_;
goto v_resetjp_1199_;
}
else
{
lean_inc(v_a_1198_);
lean_dec(v___x_1197_);
v___x_1200_ = lean_box(0);
v_isShared_1201_ = v_isSharedCheck_1208_;
goto v_resetjp_1199_;
}
v_resetjp_1199_:
{
if (lean_obj_tag(v_a_1198_) == 0)
{
lean_object* v___x_1202_; lean_object* v___x_1204_; 
v___x_1202_ = lean_box(0);
if (v_isShared_1201_ == 0)
{
lean_ctor_set(v___x_1200_, 0, v___x_1202_);
v___x_1204_ = v___x_1200_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v___x_1202_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
else
{
lean_object* v___x_1206_; lean_object* v___x_1207_; 
lean_del_object(v___x_1200_);
lean_dec(v_a_1198_);
v___x_1206_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___closed__1);
v___x_1207_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(v___x_1206_, v___y_1187_, v___y_1188_, v___y_1189_, v___y_1190_);
return v___x_1207_;
}
}
}
else
{
lean_object* v_a_1209_; lean_object* v___x_1211_; uint8_t v_isShared_1212_; uint8_t v_isSharedCheck_1216_; 
v_a_1209_ = lean_ctor_get(v___x_1197_, 0);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___x_1197_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1211_ = v___x_1197_;
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
else
{
lean_inc(v_a_1209_);
lean_dec(v___x_1197_);
v___x_1211_ = lean_box(0);
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
v_resetjp_1210_:
{
lean_object* v___x_1214_; 
if (v_isShared_1212_ == 0)
{
v___x_1214_ = v___x_1211_;
goto v_reusejp_1213_;
}
else
{
lean_object* v_reuseFailAlloc_1215_; 
v_reuseFailAlloc_1215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1215_, 0, v_a_1209_);
v___x_1214_ = v_reuseFailAlloc_1215_;
goto v_reusejp_1213_;
}
v_reusejp_1213_:
{
return v___x_1214_;
}
}
}
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
lean_dec(v___x_1186_);
v_a_1217_ = lean_ctor_get(v___x_1192_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1192_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1192_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___boxed(lean_object* v___x_1225_, lean_object* v___x_1226_, lean_object* v_a_1227_, lean_object* v___x_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
uint8_t v___x_3923__boxed_1234_; uint8_t v_a_3924__boxed_1235_; lean_object* v_res_1236_; 
v___x_3923__boxed_1234_ = lean_unbox(v___x_1226_);
v_a_3924__boxed_1235_ = lean_unbox(v_a_1227_);
v_res_1236_ = lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0(v___x_1225_, v___x_3923__boxed_1234_, v_a_3924__boxed_1235_, v___x_1228_, v___y_1229_, v___y_1230_, v___y_1231_, v___y_1232_);
lean_dec(v___y_1232_);
lean_dec_ref(v___y_1231_);
lean_dec(v___y_1230_);
lean_dec_ref(v___y_1229_);
return v_res_1236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(lean_object* v_x_1237_, lean_object* v_x_1238_, lean_object* v_x_1239_, lean_object* v_x_1240_){
_start:
{
lean_object* v_ks_1241_; lean_object* v_vs_1242_; lean_object* v___x_1244_; uint8_t v_isShared_1245_; uint8_t v_isSharedCheck_1266_; 
v_ks_1241_ = lean_ctor_get(v_x_1237_, 0);
v_vs_1242_ = lean_ctor_get(v_x_1237_, 1);
v_isSharedCheck_1266_ = !lean_is_exclusive(v_x_1237_);
if (v_isSharedCheck_1266_ == 0)
{
v___x_1244_ = v_x_1237_;
v_isShared_1245_ = v_isSharedCheck_1266_;
goto v_resetjp_1243_;
}
else
{
lean_inc(v_vs_1242_);
lean_inc(v_ks_1241_);
lean_dec(v_x_1237_);
v___x_1244_ = lean_box(0);
v_isShared_1245_ = v_isSharedCheck_1266_;
goto v_resetjp_1243_;
}
v_resetjp_1243_:
{
lean_object* v___x_1246_; uint8_t v___x_1247_; 
v___x_1246_ = lean_array_get_size(v_ks_1241_);
v___x_1247_ = lean_nat_dec_lt(v_x_1238_, v___x_1246_);
if (v___x_1247_ == 0)
{
lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1251_; 
lean_dec(v_x_1238_);
v___x_1248_ = lean_array_push(v_ks_1241_, v_x_1239_);
v___x_1249_ = lean_array_push(v_vs_1242_, v_x_1240_);
if (v_isShared_1245_ == 0)
{
lean_ctor_set(v___x_1244_, 1, v___x_1249_);
lean_ctor_set(v___x_1244_, 0, v___x_1248_);
v___x_1251_ = v___x_1244_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1252_; 
v_reuseFailAlloc_1252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1252_, 0, v___x_1248_);
lean_ctor_set(v_reuseFailAlloc_1252_, 1, v___x_1249_);
v___x_1251_ = v_reuseFailAlloc_1252_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
return v___x_1251_;
}
}
else
{
lean_object* v_k_x27_1253_; uint8_t v___x_1254_; 
v_k_x27_1253_ = lean_array_fget_borrowed(v_ks_1241_, v_x_1238_);
v___x_1254_ = l_Lean_instBEqMVarId_beq(v_x_1239_, v_k_x27_1253_);
if (v___x_1254_ == 0)
{
lean_object* v___x_1256_; 
if (v_isShared_1245_ == 0)
{
v___x_1256_ = v___x_1244_;
goto v_reusejp_1255_;
}
else
{
lean_object* v_reuseFailAlloc_1260_; 
v_reuseFailAlloc_1260_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1260_, 0, v_ks_1241_);
lean_ctor_set(v_reuseFailAlloc_1260_, 1, v_vs_1242_);
v___x_1256_ = v_reuseFailAlloc_1260_;
goto v_reusejp_1255_;
}
v_reusejp_1255_:
{
lean_object* v___x_1257_; lean_object* v___x_1258_; 
v___x_1257_ = lean_unsigned_to_nat(1u);
v___x_1258_ = lean_nat_add(v_x_1238_, v___x_1257_);
lean_dec(v_x_1238_);
v_x_1237_ = v___x_1256_;
v_x_1238_ = v___x_1258_;
goto _start;
}
}
else
{
lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1264_; 
v___x_1261_ = lean_array_fset(v_ks_1241_, v_x_1238_, v_x_1239_);
v___x_1262_ = lean_array_fset(v_vs_1242_, v_x_1238_, v_x_1240_);
lean_dec(v_x_1238_);
if (v_isShared_1245_ == 0)
{
lean_ctor_set(v___x_1244_, 1, v___x_1262_);
lean_ctor_set(v___x_1244_, 0, v___x_1261_);
v___x_1264_ = v___x_1244_;
goto v_reusejp_1263_;
}
else
{
lean_object* v_reuseFailAlloc_1265_; 
v_reuseFailAlloc_1265_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1265_, 0, v___x_1261_);
lean_ctor_set(v_reuseFailAlloc_1265_, 1, v___x_1262_);
v___x_1264_ = v_reuseFailAlloc_1265_;
goto v_reusejp_1263_;
}
v_reusejp_1263_:
{
return v___x_1264_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3___redArg(lean_object* v_n_1267_, lean_object* v_k_1268_, lean_object* v_v_1269_){
_start:
{
lean_object* v___x_1270_; lean_object* v___x_1271_; 
v___x_1270_ = lean_unsigned_to_nat(0u);
v___x_1271_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(v_n_1267_, v___x_1270_, v_k_1268_, v_v_1269_);
return v___x_1271_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_1272_; 
v___x_1272_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(lean_object* v_x_1273_, size_t v_x_1274_, size_t v_x_1275_, lean_object* v_x_1276_, lean_object* v_x_1277_){
_start:
{
if (lean_obj_tag(v_x_1273_) == 0)
{
lean_object* v_es_1278_; size_t v___x_1279_; size_t v___x_1280_; lean_object* v_j_1281_; lean_object* v___x_1282_; uint8_t v___x_1283_; 
v_es_1278_ = lean_ctor_get(v_x_1273_, 0);
v___x_1279_ = ((size_t)31ULL);
v___x_1280_ = lean_usize_land(v_x_1274_, v___x_1279_);
v_j_1281_ = lean_usize_to_nat(v___x_1280_);
v___x_1282_ = lean_array_get_size(v_es_1278_);
v___x_1283_ = lean_nat_dec_lt(v_j_1281_, v___x_1282_);
if (v___x_1283_ == 0)
{
lean_dec(v_j_1281_);
lean_dec(v_x_1277_);
lean_dec(v_x_1276_);
return v_x_1273_;
}
else
{
lean_object* v___x_1285_; uint8_t v_isShared_1286_; uint8_t v_isSharedCheck_1322_; 
lean_inc_ref(v_es_1278_);
v_isSharedCheck_1322_ = !lean_is_exclusive(v_x_1273_);
if (v_isSharedCheck_1322_ == 0)
{
lean_object* v_unused_1323_; 
v_unused_1323_ = lean_ctor_get(v_x_1273_, 0);
lean_dec(v_unused_1323_);
v___x_1285_ = v_x_1273_;
v_isShared_1286_ = v_isSharedCheck_1322_;
goto v_resetjp_1284_;
}
else
{
lean_dec(v_x_1273_);
v___x_1285_ = lean_box(0);
v_isShared_1286_ = v_isSharedCheck_1322_;
goto v_resetjp_1284_;
}
v_resetjp_1284_:
{
lean_object* v_v_1287_; lean_object* v___x_1288_; lean_object* v_xs_x27_1289_; lean_object* v___y_1291_; 
v_v_1287_ = lean_array_fget(v_es_1278_, v_j_1281_);
v___x_1288_ = lean_box(0);
v_xs_x27_1289_ = lean_array_fset(v_es_1278_, v_j_1281_, v___x_1288_);
switch(lean_obj_tag(v_v_1287_))
{
case 0:
{
lean_object* v_key_1296_; lean_object* v_val_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1307_; 
v_key_1296_ = lean_ctor_get(v_v_1287_, 0);
v_val_1297_ = lean_ctor_get(v_v_1287_, 1);
v_isSharedCheck_1307_ = !lean_is_exclusive(v_v_1287_);
if (v_isSharedCheck_1307_ == 0)
{
v___x_1299_ = v_v_1287_;
v_isShared_1300_ = v_isSharedCheck_1307_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_val_1297_);
lean_inc(v_key_1296_);
lean_dec(v_v_1287_);
v___x_1299_ = lean_box(0);
v_isShared_1300_ = v_isSharedCheck_1307_;
goto v_resetjp_1298_;
}
v_resetjp_1298_:
{
uint8_t v___x_1301_; 
v___x_1301_ = l_Lean_instBEqMVarId_beq(v_x_1276_, v_key_1296_);
if (v___x_1301_ == 0)
{
lean_object* v___x_1302_; lean_object* v___x_1303_; 
lean_del_object(v___x_1299_);
v___x_1302_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1296_, v_val_1297_, v_x_1276_, v_x_1277_);
v___x_1303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1303_, 0, v___x_1302_);
v___y_1291_ = v___x_1303_;
goto v___jp_1290_;
}
else
{
lean_object* v___x_1305_; 
lean_dec(v_val_1297_);
lean_dec(v_key_1296_);
if (v_isShared_1300_ == 0)
{
lean_ctor_set(v___x_1299_, 1, v_x_1277_);
lean_ctor_set(v___x_1299_, 0, v_x_1276_);
v___x_1305_ = v___x_1299_;
goto v_reusejp_1304_;
}
else
{
lean_object* v_reuseFailAlloc_1306_; 
v_reuseFailAlloc_1306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1306_, 0, v_x_1276_);
lean_ctor_set(v_reuseFailAlloc_1306_, 1, v_x_1277_);
v___x_1305_ = v_reuseFailAlloc_1306_;
goto v_reusejp_1304_;
}
v_reusejp_1304_:
{
v___y_1291_ = v___x_1305_;
goto v___jp_1290_;
}
}
}
}
case 1:
{
lean_object* v_node_1308_; lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1320_; 
v_node_1308_ = lean_ctor_get(v_v_1287_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v_v_1287_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1310_ = v_v_1287_;
v_isShared_1311_ = v_isSharedCheck_1320_;
goto v_resetjp_1309_;
}
else
{
lean_inc(v_node_1308_);
lean_dec(v_v_1287_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1320_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
size_t v___x_1312_; size_t v___x_1313_; size_t v___x_1314_; size_t v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1318_; 
v___x_1312_ = ((size_t)5ULL);
v___x_1313_ = lean_usize_shift_right(v_x_1274_, v___x_1312_);
v___x_1314_ = ((size_t)1ULL);
v___x_1315_ = lean_usize_add(v_x_1275_, v___x_1314_);
v___x_1316_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(v_node_1308_, v___x_1313_, v___x_1315_, v_x_1276_, v_x_1277_);
if (v_isShared_1311_ == 0)
{
lean_ctor_set(v___x_1310_, 0, v___x_1316_);
v___x_1318_ = v___x_1310_;
goto v_reusejp_1317_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v___x_1316_);
v___x_1318_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1317_;
}
v_reusejp_1317_:
{
v___y_1291_ = v___x_1318_;
goto v___jp_1290_;
}
}
}
default: 
{
lean_object* v___x_1321_; 
v___x_1321_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1321_, 0, v_x_1276_);
lean_ctor_set(v___x_1321_, 1, v_x_1277_);
v___y_1291_ = v___x_1321_;
goto v___jp_1290_;
}
}
v___jp_1290_:
{
lean_object* v___x_1292_; lean_object* v___x_1294_; 
v___x_1292_ = lean_array_fset(v_xs_x27_1289_, v_j_1281_, v___y_1291_);
lean_dec(v_j_1281_);
if (v_isShared_1286_ == 0)
{
lean_ctor_set(v___x_1285_, 0, v___x_1292_);
v___x_1294_ = v___x_1285_;
goto v_reusejp_1293_;
}
else
{
lean_object* v_reuseFailAlloc_1295_; 
v_reuseFailAlloc_1295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1295_, 0, v___x_1292_);
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
else
{
lean_object* v_ks_1324_; lean_object* v_vs_1325_; lean_object* v___x_1327_; uint8_t v_isShared_1328_; uint8_t v_isSharedCheck_1345_; 
v_ks_1324_ = lean_ctor_get(v_x_1273_, 0);
v_vs_1325_ = lean_ctor_get(v_x_1273_, 1);
v_isSharedCheck_1345_ = !lean_is_exclusive(v_x_1273_);
if (v_isSharedCheck_1345_ == 0)
{
v___x_1327_ = v_x_1273_;
v_isShared_1328_ = v_isSharedCheck_1345_;
goto v_resetjp_1326_;
}
else
{
lean_inc(v_vs_1325_);
lean_inc(v_ks_1324_);
lean_dec(v_x_1273_);
v___x_1327_ = lean_box(0);
v_isShared_1328_ = v_isSharedCheck_1345_;
goto v_resetjp_1326_;
}
v_resetjp_1326_:
{
lean_object* v___x_1330_; 
if (v_isShared_1328_ == 0)
{
v___x_1330_ = v___x_1327_;
goto v_reusejp_1329_;
}
else
{
lean_object* v_reuseFailAlloc_1344_; 
v_reuseFailAlloc_1344_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1344_, 0, v_ks_1324_);
lean_ctor_set(v_reuseFailAlloc_1344_, 1, v_vs_1325_);
v___x_1330_ = v_reuseFailAlloc_1344_;
goto v_reusejp_1329_;
}
v_reusejp_1329_:
{
lean_object* v_newNode_1331_; uint8_t v___y_1333_; size_t v___x_1339_; uint8_t v___x_1340_; 
v_newNode_1331_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3___redArg(v___x_1330_, v_x_1276_, v_x_1277_);
v___x_1339_ = ((size_t)7ULL);
v___x_1340_ = lean_usize_dec_le(v___x_1339_, v_x_1275_);
if (v___x_1340_ == 0)
{
lean_object* v___x_1341_; lean_object* v___x_1342_; uint8_t v___x_1343_; 
v___x_1341_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1331_);
v___x_1342_ = lean_unsigned_to_nat(4u);
v___x_1343_ = lean_nat_dec_lt(v___x_1341_, v___x_1342_);
lean_dec(v___x_1341_);
v___y_1333_ = v___x_1343_;
goto v___jp_1332_;
}
else
{
v___y_1333_ = v___x_1340_;
goto v___jp_1332_;
}
v___jp_1332_:
{
if (v___y_1333_ == 0)
{
lean_object* v_ks_1334_; lean_object* v_vs_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; 
v_ks_1334_ = lean_ctor_get(v_newNode_1331_, 0);
lean_inc_ref(v_ks_1334_);
v_vs_1335_ = lean_ctor_get(v_newNode_1331_, 1);
lean_inc_ref(v_vs_1335_);
lean_dec_ref(v_newNode_1331_);
v___x_1336_ = lean_unsigned_to_nat(0u);
v___x_1337_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___closed__0);
v___x_1338_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg(v_x_1275_, v_ks_1334_, v_vs_1335_, v___x_1336_, v___x_1337_);
lean_dec_ref(v_vs_1335_);
lean_dec_ref(v_ks_1334_);
return v___x_1338_;
}
else
{
return v_newNode_1331_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg(size_t v_depth_1346_, lean_object* v_keys_1347_, lean_object* v_vals_1348_, lean_object* v_i_1349_, lean_object* v_entries_1350_){
_start:
{
lean_object* v___x_1351_; uint8_t v___x_1352_; 
v___x_1351_ = lean_array_get_size(v_keys_1347_);
v___x_1352_ = lean_nat_dec_lt(v_i_1349_, v___x_1351_);
if (v___x_1352_ == 0)
{
lean_dec(v_i_1349_);
return v_entries_1350_;
}
else
{
lean_object* v_k_1353_; lean_object* v_v_1354_; uint64_t v___x_1355_; size_t v_h_1356_; size_t v___x_1357_; lean_object* v___x_1358_; size_t v___x_1359_; size_t v___x_1360_; size_t v___x_1361_; size_t v_h_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; 
v_k_1353_ = lean_array_fget_borrowed(v_keys_1347_, v_i_1349_);
v_v_1354_ = lean_array_fget_borrowed(v_vals_1348_, v_i_1349_);
v___x_1355_ = l_Lean_instHashableMVarId_hash(v_k_1353_);
v_h_1356_ = lean_uint64_to_usize(v___x_1355_);
v___x_1357_ = ((size_t)5ULL);
v___x_1358_ = lean_unsigned_to_nat(1u);
v___x_1359_ = ((size_t)1ULL);
v___x_1360_ = lean_usize_sub(v_depth_1346_, v___x_1359_);
v___x_1361_ = lean_usize_mul(v___x_1357_, v___x_1360_);
v_h_1362_ = lean_usize_shift_right(v_h_1356_, v___x_1361_);
v___x_1363_ = lean_nat_add(v_i_1349_, v___x_1358_);
lean_dec(v_i_1349_);
lean_inc(v_v_1354_);
lean_inc(v_k_1353_);
v___x_1364_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(v_entries_1350_, v_h_1362_, v_depth_1346_, v_k_1353_, v_v_1354_);
v_i_1349_ = v___x_1363_;
v_entries_1350_ = v___x_1364_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_depth_1366_, lean_object* v_keys_1367_, lean_object* v_vals_1368_, lean_object* v_i_1369_, lean_object* v_entries_1370_){
_start:
{
size_t v_depth_boxed_1371_; lean_object* v_res_1372_; 
v_depth_boxed_1371_ = lean_unbox_usize(v_depth_1366_);
lean_dec(v_depth_1366_);
v_res_1372_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg(v_depth_boxed_1371_, v_keys_1367_, v_vals_1368_, v_i_1369_, v_entries_1370_);
lean_dec_ref(v_vals_1368_);
lean_dec_ref(v_keys_1367_);
return v_res_1372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg___boxed(lean_object* v_x_1373_, lean_object* v_x_1374_, lean_object* v_x_1375_, lean_object* v_x_1376_, lean_object* v_x_1377_){
_start:
{
size_t v_x_4095__boxed_1378_; size_t v_x_4096__boxed_1379_; lean_object* v_res_1380_; 
v_x_4095__boxed_1378_ = lean_unbox_usize(v_x_1374_);
lean_dec(v_x_1374_);
v_x_4096__boxed_1379_ = lean_unbox_usize(v_x_1375_);
lean_dec(v_x_1375_);
v_res_1380_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(v_x_1373_, v_x_4095__boxed_1378_, v_x_4096__boxed_1379_, v_x_1376_, v_x_1377_);
return v_res_1380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1___redArg(lean_object* v_x_1381_, lean_object* v_x_1382_, lean_object* v_x_1383_){
_start:
{
uint64_t v___x_1384_; size_t v___x_1385_; size_t v___x_1386_; lean_object* v___x_1387_; 
v___x_1384_ = l_Lean_instHashableMVarId_hash(v_x_1382_);
v___x_1385_ = lean_uint64_to_usize(v___x_1384_);
v___x_1386_ = ((size_t)1ULL);
v___x_1387_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(v_x_1381_, v___x_1385_, v___x_1386_, v_x_1382_, v_x_1383_);
return v___x_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg(lean_object* v_mvarId_1388_, lean_object* v_val_1389_, lean_object* v___y_1390_){
_start:
{
lean_object* v___x_1392_; lean_object* v_mctx_1393_; lean_object* v_cache_1394_; lean_object* v_zetaDeltaFVarIds_1395_; lean_object* v_postponed_1396_; lean_object* v_diag_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1425_; 
v___x_1392_ = lean_st_ref_take(v___y_1390_);
v_mctx_1393_ = lean_ctor_get(v___x_1392_, 0);
v_cache_1394_ = lean_ctor_get(v___x_1392_, 1);
v_zetaDeltaFVarIds_1395_ = lean_ctor_get(v___x_1392_, 2);
v_postponed_1396_ = lean_ctor_get(v___x_1392_, 3);
v_diag_1397_ = lean_ctor_get(v___x_1392_, 4);
v_isSharedCheck_1425_ = !lean_is_exclusive(v___x_1392_);
if (v_isSharedCheck_1425_ == 0)
{
v___x_1399_ = v___x_1392_;
v_isShared_1400_ = v_isSharedCheck_1425_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_diag_1397_);
lean_inc(v_postponed_1396_);
lean_inc(v_zetaDeltaFVarIds_1395_);
lean_inc(v_cache_1394_);
lean_inc(v_mctx_1393_);
lean_dec(v___x_1392_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1425_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
lean_object* v_depth_1401_; lean_object* v_levelAssignDepth_1402_; lean_object* v_lmvarCounter_1403_; lean_object* v_mvarCounter_1404_; lean_object* v_lDecls_1405_; lean_object* v_decls_1406_; lean_object* v_userNames_1407_; lean_object* v_lAssignment_1408_; lean_object* v_eAssignment_1409_; lean_object* v_dAssignment_1410_; lean_object* v___x_1412_; uint8_t v_isShared_1413_; uint8_t v_isSharedCheck_1424_; 
v_depth_1401_ = lean_ctor_get(v_mctx_1393_, 0);
v_levelAssignDepth_1402_ = lean_ctor_get(v_mctx_1393_, 1);
v_lmvarCounter_1403_ = lean_ctor_get(v_mctx_1393_, 2);
v_mvarCounter_1404_ = lean_ctor_get(v_mctx_1393_, 3);
v_lDecls_1405_ = lean_ctor_get(v_mctx_1393_, 4);
v_decls_1406_ = lean_ctor_get(v_mctx_1393_, 5);
v_userNames_1407_ = lean_ctor_get(v_mctx_1393_, 6);
v_lAssignment_1408_ = lean_ctor_get(v_mctx_1393_, 7);
v_eAssignment_1409_ = lean_ctor_get(v_mctx_1393_, 8);
v_dAssignment_1410_ = lean_ctor_get(v_mctx_1393_, 9);
v_isSharedCheck_1424_ = !lean_is_exclusive(v_mctx_1393_);
if (v_isSharedCheck_1424_ == 0)
{
v___x_1412_ = v_mctx_1393_;
v_isShared_1413_ = v_isSharedCheck_1424_;
goto v_resetjp_1411_;
}
else
{
lean_inc(v_dAssignment_1410_);
lean_inc(v_eAssignment_1409_);
lean_inc(v_lAssignment_1408_);
lean_inc(v_userNames_1407_);
lean_inc(v_decls_1406_);
lean_inc(v_lDecls_1405_);
lean_inc(v_mvarCounter_1404_);
lean_inc(v_lmvarCounter_1403_);
lean_inc(v_levelAssignDepth_1402_);
lean_inc(v_depth_1401_);
lean_dec(v_mctx_1393_);
v___x_1412_ = lean_box(0);
v_isShared_1413_ = v_isSharedCheck_1424_;
goto v_resetjp_1411_;
}
v_resetjp_1411_:
{
lean_object* v___x_1414_; lean_object* v___x_1416_; 
v___x_1414_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1___redArg(v_eAssignment_1409_, v_mvarId_1388_, v_val_1389_);
if (v_isShared_1413_ == 0)
{
lean_ctor_set(v___x_1412_, 8, v___x_1414_);
v___x_1416_ = v___x_1412_;
goto v_reusejp_1415_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v_depth_1401_);
lean_ctor_set(v_reuseFailAlloc_1423_, 1, v_levelAssignDepth_1402_);
lean_ctor_set(v_reuseFailAlloc_1423_, 2, v_lmvarCounter_1403_);
lean_ctor_set(v_reuseFailAlloc_1423_, 3, v_mvarCounter_1404_);
lean_ctor_set(v_reuseFailAlloc_1423_, 4, v_lDecls_1405_);
lean_ctor_set(v_reuseFailAlloc_1423_, 5, v_decls_1406_);
lean_ctor_set(v_reuseFailAlloc_1423_, 6, v_userNames_1407_);
lean_ctor_set(v_reuseFailAlloc_1423_, 7, v_lAssignment_1408_);
lean_ctor_set(v_reuseFailAlloc_1423_, 8, v___x_1414_);
lean_ctor_set(v_reuseFailAlloc_1423_, 9, v_dAssignment_1410_);
v___x_1416_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1415_;
}
v_reusejp_1415_:
{
lean_object* v___x_1418_; 
if (v_isShared_1400_ == 0)
{
lean_ctor_set(v___x_1399_, 0, v___x_1416_);
v___x_1418_ = v___x_1399_;
goto v_reusejp_1417_;
}
else
{
lean_object* v_reuseFailAlloc_1422_; 
v_reuseFailAlloc_1422_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1422_, 0, v___x_1416_);
lean_ctor_set(v_reuseFailAlloc_1422_, 1, v_cache_1394_);
lean_ctor_set(v_reuseFailAlloc_1422_, 2, v_zetaDeltaFVarIds_1395_);
lean_ctor_set(v_reuseFailAlloc_1422_, 3, v_postponed_1396_);
lean_ctor_set(v_reuseFailAlloc_1422_, 4, v_diag_1397_);
v___x_1418_ = v_reuseFailAlloc_1422_;
goto v_reusejp_1417_;
}
v_reusejp_1417_:
{
lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; 
v___x_1419_ = lean_st_ref_set(v___y_1390_, v___x_1418_);
v___x_1420_ = lean_box(0);
v___x_1421_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1421_, 0, v___x_1420_);
return v___x_1421_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg___boxed(lean_object* v_mvarId_1426_, lean_object* v_val_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_){
_start:
{
lean_object* v_res_1430_; 
v_res_1430_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg(v_mvarId_1426_, v_val_1427_, v___y_1428_);
lean_dec(v___y_1428_);
return v_res_1430_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__5(void){
_start:
{
lean_object* v___x_1439_; lean_object* v___x_1440_; 
v___x_1439_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__4));
v___x_1440_ = l_Lean_stringToMessageData(v___x_1439_);
return v___x_1440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective(lean_object* v_ginj_1441_, lean_object* v_using_x3f_1442_, lean_object* v_a_1443_, lean_object* v_a_1444_, lean_object* v_a_1445_, lean_object* v_a_1446_){
_start:
{
lean_object* v___y_1449_; lean_object* v___y_1450_; lean_object* v___y_1451_; lean_object* v___y_1452_; 
if (lean_obj_tag(v_using_x3f_1442_) == 1)
{
lean_object* v_val_1490_; lean_object* v___x_1491_; 
v_val_1490_ = lean_ctor_get(v_using_x3f_1442_, 0);
lean_inc_n(v_val_1490_, 2);
lean_dec_ref_known(v_using_x3f_1442_, 1);
lean_inc_ref(v_ginj_1441_);
v___x_1491_ = l_Lean_Meta_isExprDefEq(v_ginj_1441_, v_val_1490_, v_a_1443_, v_a_1444_, v_a_1445_, v_a_1446_);
if (lean_obj_tag(v___x_1491_) == 0)
{
lean_object* v_a_1492_; uint8_t v___x_1493_; 
v_a_1492_ = lean_ctor_get(v___x_1491_, 0);
lean_inc(v_a_1492_);
lean_dec_ref_known(v___x_1491_, 1);
v___x_1493_ = lean_unbox(v_a_1492_);
if (v___x_1493_ == 0)
{
lean_object* v___x_1494_; 
lean_dec(v_a_1492_);
lean_inc(v_a_1446_);
lean_inc_ref(v_a_1445_);
lean_inc(v_a_1444_);
lean_inc_ref(v_a_1443_);
v___x_1494_ = lean_infer_type(v_val_1490_, v_a_1443_, v_a_1444_, v_a_1445_, v_a_1446_);
if (lean_obj_tag(v___x_1494_) == 0)
{
lean_object* v_a_1495_; lean_object* v___x_1496_; 
v_a_1495_ = lean_ctor_get(v___x_1494_, 0);
lean_inc(v_a_1495_);
lean_dec_ref_known(v___x_1494_, 1);
lean_inc(v_a_1446_);
lean_inc_ref(v_a_1445_);
lean_inc(v_a_1444_);
lean_inc_ref(v_a_1443_);
v___x_1496_ = lean_infer_type(v_ginj_1441_, v_a_1443_, v_a_1444_, v_a_1445_, v_a_1446_);
if (lean_obj_tag(v___x_1496_) == 0)
{
lean_object* v_a_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; 
v_a_1497_ = lean_ctor_get(v___x_1496_, 0);
lean_inc(v_a_1497_);
lean_dec_ref_known(v___x_1496_, 1);
v___x_1498_ = lean_box(0);
v___x_1499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__3));
v___x_1500_ = l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(v_a_1495_, v_a_1497_, v___x_1498_, v___x_1499_);
if (lean_obj_tag(v___x_1500_) == 0)
{
lean_object* v_a_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v_a_1505_; lean_object* v___x_1507_; uint8_t v_isShared_1508_; uint8_t v_isSharedCheck_1512_; 
v_a_1501_ = lean_ctor_get(v___x_1500_, 0);
lean_inc(v_a_1501_);
lean_dec_ref_known(v___x_1500_, 1);
v___x_1502_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__5, &lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__5);
v___x_1503_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1503_, 0, v___x_1502_);
lean_ctor_set(v___x_1503_, 1, v_a_1501_);
v___x_1504_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunTargetFailure_spec__0___redArg(v___x_1503_, v_a_1443_, v_a_1444_, v_a_1445_, v_a_1446_);
v_a_1505_ = lean_ctor_get(v___x_1504_, 0);
v_isSharedCheck_1512_ = !lean_is_exclusive(v___x_1504_);
if (v_isSharedCheck_1512_ == 0)
{
v___x_1507_ = v___x_1504_;
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
else
{
lean_inc(v_a_1505_);
lean_dec(v___x_1504_);
v___x_1507_ = lean_box(0);
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
v_resetjp_1506_:
{
lean_object* v___x_1510_; 
if (v_isShared_1508_ == 0)
{
v___x_1510_ = v___x_1507_;
goto v_reusejp_1509_;
}
else
{
lean_object* v_reuseFailAlloc_1511_; 
v_reuseFailAlloc_1511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1511_, 0, v_a_1505_);
v___x_1510_ = v_reuseFailAlloc_1511_;
goto v_reusejp_1509_;
}
v_reusejp_1509_:
{
return v___x_1510_;
}
}
}
else
{
lean_object* v_a_1513_; lean_object* v___x_1515_; uint8_t v_isShared_1516_; uint8_t v_isSharedCheck_1520_; 
v_a_1513_ = lean_ctor_get(v___x_1500_, 0);
v_isSharedCheck_1520_ = !lean_is_exclusive(v___x_1500_);
if (v_isSharedCheck_1520_ == 0)
{
v___x_1515_ = v___x_1500_;
v_isShared_1516_ = v_isSharedCheck_1520_;
goto v_resetjp_1514_;
}
else
{
lean_inc(v_a_1513_);
lean_dec(v___x_1500_);
v___x_1515_ = lean_box(0);
v_isShared_1516_ = v_isSharedCheck_1520_;
goto v_resetjp_1514_;
}
v_resetjp_1514_:
{
lean_object* v___x_1518_; 
if (v_isShared_1516_ == 0)
{
v___x_1518_ = v___x_1515_;
goto v_reusejp_1517_;
}
else
{
lean_object* v_reuseFailAlloc_1519_; 
v_reuseFailAlloc_1519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1519_, 0, v_a_1513_);
v___x_1518_ = v_reuseFailAlloc_1519_;
goto v_reusejp_1517_;
}
v_reusejp_1517_:
{
return v___x_1518_;
}
}
}
}
else
{
lean_object* v_a_1521_; lean_object* v___x_1523_; uint8_t v_isShared_1524_; uint8_t v_isSharedCheck_1528_; 
lean_dec(v_a_1495_);
v_a_1521_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1528_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1528_ == 0)
{
v___x_1523_ = v___x_1496_;
v_isShared_1524_ = v_isSharedCheck_1528_;
goto v_resetjp_1522_;
}
else
{
lean_inc(v_a_1521_);
lean_dec(v___x_1496_);
v___x_1523_ = lean_box(0);
v_isShared_1524_ = v_isSharedCheck_1528_;
goto v_resetjp_1522_;
}
v_resetjp_1522_:
{
lean_object* v___x_1526_; 
if (v_isShared_1524_ == 0)
{
v___x_1526_ = v___x_1523_;
goto v_reusejp_1525_;
}
else
{
lean_object* v_reuseFailAlloc_1527_; 
v_reuseFailAlloc_1527_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1527_, 0, v_a_1521_);
v___x_1526_ = v_reuseFailAlloc_1527_;
goto v_reusejp_1525_;
}
v_reusejp_1525_:
{
return v___x_1526_;
}
}
}
}
else
{
lean_object* v_a_1529_; lean_object* v___x_1531_; uint8_t v_isShared_1532_; uint8_t v_isSharedCheck_1536_; 
lean_dec_ref(v_ginj_1441_);
v_a_1529_ = lean_ctor_get(v___x_1494_, 0);
v_isSharedCheck_1536_ = !lean_is_exclusive(v___x_1494_);
if (v_isSharedCheck_1536_ == 0)
{
v___x_1531_ = v___x_1494_;
v_isShared_1532_ = v_isSharedCheck_1536_;
goto v_resetjp_1530_;
}
else
{
lean_inc(v_a_1529_);
lean_dec(v___x_1494_);
v___x_1531_ = lean_box(0);
v_isShared_1532_ = v_isSharedCheck_1536_;
goto v_resetjp_1530_;
}
v_resetjp_1530_:
{
lean_object* v___x_1534_; 
if (v_isShared_1532_ == 0)
{
v___x_1534_ = v___x_1531_;
goto v_reusejp_1533_;
}
else
{
lean_object* v_reuseFailAlloc_1535_; 
v_reuseFailAlloc_1535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1535_, 0, v_a_1529_);
v___x_1534_ = v_reuseFailAlloc_1535_;
goto v_reusejp_1533_;
}
v_reusejp_1533_:
{
return v___x_1534_;
}
}
}
}
else
{
lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1540_; uint8_t v_isShared_1541_; uint8_t v_isSharedCheck_1545_; 
v___x_1537_ = l_Lean_Expr_mvarId_x21(v_ginj_1441_);
lean_dec_ref(v_ginj_1441_);
v___x_1538_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg(v___x_1537_, v_val_1490_, v_a_1444_);
v_isSharedCheck_1545_ = !lean_is_exclusive(v___x_1538_);
if (v_isSharedCheck_1545_ == 0)
{
lean_object* v_unused_1546_; 
v_unused_1546_ = lean_ctor_get(v___x_1538_, 0);
lean_dec(v_unused_1546_);
v___x_1540_ = v___x_1538_;
v_isShared_1541_ = v_isSharedCheck_1545_;
goto v_resetjp_1539_;
}
else
{
lean_dec(v___x_1538_);
v___x_1540_ = lean_box(0);
v_isShared_1541_ = v_isSharedCheck_1545_;
goto v_resetjp_1539_;
}
v_resetjp_1539_:
{
lean_object* v___x_1543_; 
if (v_isShared_1541_ == 0)
{
lean_ctor_set(v___x_1540_, 0, v_a_1492_);
v___x_1543_ = v___x_1540_;
goto v_reusejp_1542_;
}
else
{
lean_object* v_reuseFailAlloc_1544_; 
v_reuseFailAlloc_1544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1544_, 0, v_a_1492_);
v___x_1543_ = v_reuseFailAlloc_1544_;
goto v_reusejp_1542_;
}
v_reusejp_1542_:
{
return v___x_1543_;
}
}
}
}
else
{
lean_dec(v_val_1490_);
lean_dec_ref(v_ginj_1441_);
return v___x_1491_;
}
}
else
{
lean_dec(v_using_x3f_1442_);
v___y_1449_ = v_a_1443_;
v___y_1450_ = v_a_1444_;
v___y_1451_ = v_a_1445_;
v___y_1452_ = v_a_1446_;
goto v___jp_1448_;
}
v___jp_1448_:
{
lean_object* v___x_1453_; lean_object* v___x_1454_; 
v___x_1453_ = l_Lean_Expr_mvarId_x21(v_ginj_1441_);
lean_dec_ref(v_ginj_1441_);
lean_inc(v___x_1453_);
v___x_1454_ = l_Lean_MVarId_assumptionCore(v___x_1453_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_);
if (lean_obj_tag(v___x_1454_) == 0)
{
lean_object* v_a_1455_; lean_object* v___x_1457_; uint8_t v_isShared_1458_; uint8_t v_isSharedCheck_1489_; 
v_a_1455_ = lean_ctor_get(v___x_1454_, 0);
v_isSharedCheck_1489_ = !lean_is_exclusive(v___x_1454_);
if (v_isSharedCheck_1489_ == 0)
{
v___x_1457_ = v___x_1454_;
v_isShared_1458_ = v_isSharedCheck_1489_;
goto v_resetjp_1456_;
}
else
{
lean_inc(v_a_1455_);
lean_dec(v___x_1454_);
v___x_1457_ = lean_box(0);
v_isShared_1458_ = v_isSharedCheck_1489_;
goto v_resetjp_1456_;
}
v_resetjp_1456_:
{
uint8_t v___x_1459_; uint8_t v___x_1460_; 
v___x_1459_ = 1;
v___x_1460_ = lean_unbox(v_a_1455_);
if (v___x_1460_ == 0)
{
lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___f_1463_; lean_object* v___x_1464_; 
lean_del_object(v___x_1457_);
v___x_1461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_maybeProveInjective___closed__2));
v___x_1462_ = lean_box(v___x_1459_);
lean_inc(v_a_1455_);
v___f_1463_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_maybeProveInjective___lam__0___boxed), 9, 4);
lean_closure_set(v___f_1463_, 0, v___x_1461_);
lean_closure_set(v___f_1463_, 1, v___x_1462_);
lean_closure_set(v___f_1463_, 2, v_a_1455_);
lean_closure_set(v___f_1463_, 3, v___x_1453_);
v___x_1464_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_maybeProveInjective_spec__0___redArg(v___f_1463_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_);
if (lean_obj_tag(v___x_1464_) == 0)
{
lean_object* v_a_1465_; lean_object* v___x_1467_; uint8_t v_isShared_1468_; uint8_t v_isSharedCheck_1476_; 
v_a_1465_ = lean_ctor_get(v___x_1464_, 0);
v_isSharedCheck_1476_ = !lean_is_exclusive(v___x_1464_);
if (v_isSharedCheck_1476_ == 0)
{
v___x_1467_ = v___x_1464_;
v_isShared_1468_ = v_isSharedCheck_1476_;
goto v_resetjp_1466_;
}
else
{
lean_inc(v_a_1465_);
lean_dec(v___x_1464_);
v___x_1467_ = lean_box(0);
v_isShared_1468_ = v_isSharedCheck_1476_;
goto v_resetjp_1466_;
}
v_resetjp_1466_:
{
if (lean_obj_tag(v_a_1465_) == 0)
{
lean_object* v___x_1470_; 
if (v_isShared_1468_ == 0)
{
lean_ctor_set(v___x_1467_, 0, v_a_1455_);
v___x_1470_ = v___x_1467_;
goto v_reusejp_1469_;
}
else
{
lean_object* v_reuseFailAlloc_1471_; 
v_reuseFailAlloc_1471_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1471_, 0, v_a_1455_);
v___x_1470_ = v_reuseFailAlloc_1471_;
goto v_reusejp_1469_;
}
v_reusejp_1469_:
{
return v___x_1470_;
}
}
else
{
lean_object* v___x_1472_; lean_object* v___x_1474_; 
lean_dec_ref_known(v_a_1465_, 1);
lean_dec(v_a_1455_);
v___x_1472_ = lean_box(v___x_1459_);
if (v_isShared_1468_ == 0)
{
lean_ctor_set(v___x_1467_, 0, v___x_1472_);
v___x_1474_ = v___x_1467_;
goto v_reusejp_1473_;
}
else
{
lean_object* v_reuseFailAlloc_1475_; 
v_reuseFailAlloc_1475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1475_, 0, v___x_1472_);
v___x_1474_ = v_reuseFailAlloc_1475_;
goto v_reusejp_1473_;
}
v_reusejp_1473_:
{
return v___x_1474_;
}
}
}
}
else
{
lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1484_; 
lean_dec(v_a_1455_);
v_a_1477_ = lean_ctor_get(v___x_1464_, 0);
v_isSharedCheck_1484_ = !lean_is_exclusive(v___x_1464_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1479_ = v___x_1464_;
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1464_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1484_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v___x_1482_; 
if (v_isShared_1480_ == 0)
{
v___x_1482_ = v___x_1479_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v_a_1477_);
v___x_1482_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1481_;
}
v_reusejp_1481_:
{
return v___x_1482_;
}
}
}
}
else
{
lean_object* v___x_1485_; lean_object* v___x_1487_; 
lean_dec(v_a_1455_);
lean_dec(v___x_1453_);
v___x_1485_ = lean_box(v___x_1459_);
if (v_isShared_1458_ == 0)
{
lean_ctor_set(v___x_1457_, 0, v___x_1485_);
v___x_1487_ = v___x_1457_;
goto v_reusejp_1486_;
}
else
{
lean_object* v_reuseFailAlloc_1488_; 
v_reuseFailAlloc_1488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1488_, 0, v___x_1485_);
v___x_1487_ = v_reuseFailAlloc_1488_;
goto v_reusejp_1486_;
}
v_reusejp_1486_:
{
return v___x_1487_;
}
}
}
}
else
{
lean_dec(v___x_1453_);
return v___x_1454_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_maybeProveInjective___boxed(lean_object* v_ginj_1547_, lean_object* v_using_x3f_1548_, lean_object* v_a_1549_, lean_object* v_a_1550_, lean_object* v_a_1551_, lean_object* v_a_1552_, lean_object* v_a_1553_){
_start:
{
lean_object* v_res_1554_; 
v_res_1554_ = lp_mathlib_Mathlib_Tactic_maybeProveInjective(v_ginj_1547_, v_using_x3f_1548_, v_a_1549_, v_a_1550_, v_a_1551_, v_a_1552_);
lean_dec(v_a_1552_);
lean_dec_ref(v_a_1551_);
lean_dec(v_a_1550_);
lean_dec_ref(v_a_1549_);
return v_res_1554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1(lean_object* v_mvarId_1555_, lean_object* v_val_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_){
_start:
{
lean_object* v___x_1562_; 
v___x_1562_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___redArg(v_mvarId_1555_, v_val_1556_, v___y_1558_);
return v___x_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1___boxed(lean_object* v_mvarId_1563_, lean_object* v_val_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_){
_start:
{
lean_object* v_res_1570_; 
v_res_1570_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1(v_mvarId_1563_, v_val_1564_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_);
lean_dec(v___y_1568_);
lean_dec_ref(v___y_1567_);
lean_dec(v___y_1566_);
lean_dec_ref(v___y_1565_);
return v_res_1570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1(lean_object* v_00_u03b2_1571_, lean_object* v_x_1572_, lean_object* v_x_1573_, lean_object* v_x_1574_){
_start:
{
lean_object* v___x_1575_; 
v___x_1575_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1___redArg(v_x_1572_, v_x_1573_, v_x_1574_);
return v___x_1575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2(lean_object* v_00_u03b2_1576_, lean_object* v_x_1577_, size_t v_x_1578_, size_t v_x_1579_, lean_object* v_x_1580_, lean_object* v_x_1581_){
_start:
{
lean_object* v___x_1582_; 
v___x_1582_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___redArg(v_x_1577_, v_x_1578_, v_x_1579_, v_x_1580_, v_x_1581_);
return v___x_1582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1583_, lean_object* v_x_1584_, lean_object* v_x_1585_, lean_object* v_x_1586_, lean_object* v_x_1587_, lean_object* v_x_1588_){
_start:
{
size_t v_x_4571__boxed_1589_; size_t v_x_4572__boxed_1590_; lean_object* v_res_1591_; 
v_x_4571__boxed_1589_ = lean_unbox_usize(v_x_1585_);
lean_dec(v_x_1585_);
v_x_4572__boxed_1590_ = lean_unbox_usize(v_x_1586_);
lean_dec(v_x_1586_);
v_res_1591_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2(v_00_u03b2_1583_, v_x_1584_, v_x_4571__boxed_1589_, v_x_4572__boxed_1590_, v_x_1587_, v_x_1588_);
return v_res_1591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_1592_, lean_object* v_n_1593_, lean_object* v_k_1594_, lean_object* v_v_1595_){
_start:
{
lean_object* v___x_1596_; 
v___x_1596_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3___redArg(v_n_1593_, v_k_1594_, v_v_1595_);
return v___x_1596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1597_, size_t v_depth_1598_, lean_object* v_keys_1599_, lean_object* v_vals_1600_, lean_object* v_heq_1601_, lean_object* v_i_1602_, lean_object* v_entries_1603_){
_start:
{
lean_object* v___x_1604_; 
v___x_1604_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___redArg(v_depth_1598_, v_keys_1599_, v_vals_1600_, v_i_1602_, v_entries_1603_);
return v___x_1604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b2_1605_, lean_object* v_depth_1606_, lean_object* v_keys_1607_, lean_object* v_vals_1608_, lean_object* v_heq_1609_, lean_object* v_i_1610_, lean_object* v_entries_1611_){
_start:
{
size_t v_depth_boxed_1612_; lean_object* v_res_1613_; 
v_depth_boxed_1612_ = lean_unbox_usize(v_depth_1606_);
lean_dec(v_depth_1606_);
v_res_1613_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__4(v_00_u03b2_1605_, v_depth_boxed_1612_, v_keys_1607_, v_vals_1608_, v_heq_1609_, v_i_1610_, v_entries_1611_);
lean_dec_ref(v_vals_1608_);
lean_dec_ref(v_keys_1607_);
return v_res_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_1614_, lean_object* v_x_1615_, lean_object* v_x_1616_, lean_object* v_x_1617_, lean_object* v_x_1618_){
_start:
{
lean_object* v___x_1619_; 
v___x_1619_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1_spec__2_spec__3_spec__4___redArg(v_x_1615_, v_x_1616_, v_x_1617_, v_x_1618_);
return v___x_1619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0(lean_object* v_a_1632_, lean_object* v_g_1633_, lean_object* v_f_1634_, lean_object* v_thm_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_){
_start:
{
lean_object* v___x_1643_; 
v___x_1643_ = l_Lean_Elab_Term_exprToSyntax(v_a_1632_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_);
if (lean_obj_tag(v___x_1643_) == 0)
{
lean_object* v_a_1644_; lean_object* v___x_1645_; 
v_a_1644_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1643_, 1);
v___x_1645_ = l_Lean_MVarId_getType(v_g_1633_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_);
if (lean_obj_tag(v___x_1645_) == 0)
{
lean_object* v_a_1646_; lean_object* v___x_1648_; uint8_t v_isShared_1649_; uint8_t v_isSharedCheck_1682_; 
v_a_1646_ = lean_ctor_get(v___x_1645_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v___x_1645_);
if (v_isSharedCheck_1682_ == 0)
{
v___x_1648_ = v___x_1645_;
v_isShared_1649_ = v_isSharedCheck_1682_;
goto v_resetjp_1647_;
}
else
{
lean_inc(v_a_1646_);
lean_dec(v___x_1645_);
v___x_1648_ = lean_box(0);
v_isShared_1649_ = v_isSharedCheck_1682_;
goto v_resetjp_1647_;
}
v_resetjp_1647_:
{
lean_object* v_ref_1650_; uint8_t v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1659_; 
v_ref_1650_ = lean_ctor_get(v___y_1640_, 5);
v___x_1651_ = 0;
v___x_1652_ = l_Lean_SourceInfo_fromRef(v_ref_1650_, v___x_1651_);
v___x_1653_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__1));
lean_inc(v___x_1652_);
v___x_1654_ = l_Lean_Syntax_node2(v___x_1652_, v___x_1653_, v_f_1634_, v_a_1644_);
v___x_1655_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6));
v___x_1656_ = l_Lean_mkIdent(v_thm_1635_);
v___x_1657_ = l_Lean_Syntax_node2(v___x_1652_, v___x_1655_, v___x_1656_, v___x_1654_);
if (v_isShared_1649_ == 0)
{
lean_ctor_set_tag(v___x_1648_, 1);
v___x_1659_ = v___x_1648_;
goto v_reusejp_1658_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v_a_1646_);
v___x_1659_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1658_;
}
v_reusejp_1658_:
{
uint8_t v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; 
v___x_1660_ = 1;
v___x_1661_ = lean_box(0);
v___x_1662_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_1657_, v___x_1659_, v___x_1660_, v___x_1660_, v___x_1661_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_);
if (lean_obj_tag(v___x_1662_) == 0)
{
lean_object* v_a_1663_; lean_object* v___x_1664_; 
v_a_1663_ = lean_ctor_get(v___x_1662_, 0);
lean_inc(v_a_1663_);
lean_dec_ref_known(v___x_1662_, 1);
v___x_1664_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsUsingDefault(v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_);
if (lean_obj_tag(v___x_1664_) == 0)
{
lean_object* v___x_1666_; uint8_t v_isShared_1667_; uint8_t v_isSharedCheck_1671_; 
v_isSharedCheck_1671_ = !lean_is_exclusive(v___x_1664_);
if (v_isSharedCheck_1671_ == 0)
{
lean_object* v_unused_1672_; 
v_unused_1672_ = lean_ctor_get(v___x_1664_, 0);
lean_dec(v_unused_1672_);
v___x_1666_ = v___x_1664_;
v_isShared_1667_ = v_isSharedCheck_1671_;
goto v_resetjp_1665_;
}
else
{
lean_dec(v___x_1664_);
v___x_1666_ = lean_box(0);
v_isShared_1667_ = v_isSharedCheck_1671_;
goto v_resetjp_1665_;
}
v_resetjp_1665_:
{
lean_object* v___x_1669_; 
if (v_isShared_1667_ == 0)
{
lean_ctor_set(v___x_1666_, 0, v_a_1663_);
v___x_1669_ = v___x_1666_;
goto v_reusejp_1668_;
}
else
{
lean_object* v_reuseFailAlloc_1670_; 
v_reuseFailAlloc_1670_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1670_, 0, v_a_1663_);
v___x_1669_ = v_reuseFailAlloc_1670_;
goto v_reusejp_1668_;
}
v_reusejp_1668_:
{
return v___x_1669_;
}
}
}
else
{
lean_object* v_a_1673_; lean_object* v___x_1675_; uint8_t v_isShared_1676_; uint8_t v_isSharedCheck_1680_; 
lean_dec(v_a_1663_);
v_a_1673_ = lean_ctor_get(v___x_1664_, 0);
v_isSharedCheck_1680_ = !lean_is_exclusive(v___x_1664_);
if (v_isSharedCheck_1680_ == 0)
{
v___x_1675_ = v___x_1664_;
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
else
{
lean_inc(v_a_1673_);
lean_dec(v___x_1664_);
v___x_1675_ = lean_box(0);
v_isShared_1676_ = v_isSharedCheck_1680_;
goto v_resetjp_1674_;
}
v_resetjp_1674_:
{
lean_object* v___x_1678_; 
if (v_isShared_1676_ == 0)
{
v___x_1678_ = v___x_1675_;
goto v_reusejp_1677_;
}
else
{
lean_object* v_reuseFailAlloc_1679_; 
v_reuseFailAlloc_1679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1679_, 0, v_a_1673_);
v___x_1678_ = v_reuseFailAlloc_1679_;
goto v_reusejp_1677_;
}
v_reusejp_1677_:
{
return v___x_1678_;
}
}
}
}
else
{
return v___x_1662_;
}
}
}
}
else
{
lean_dec(v_a_1644_);
lean_dec(v_thm_1635_);
lean_dec(v_f_1634_);
return v___x_1645_;
}
}
else
{
lean_object* v_a_1683_; lean_object* v___x_1685_; uint8_t v_isShared_1686_; uint8_t v_isSharedCheck_1690_; 
lean_dec(v_thm_1635_);
lean_dec(v_f_1634_);
lean_dec(v_g_1633_);
v_a_1683_ = lean_ctor_get(v___x_1643_, 0);
v_isSharedCheck_1690_ = !lean_is_exclusive(v___x_1643_);
if (v_isSharedCheck_1690_ == 0)
{
v___x_1685_ = v___x_1643_;
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
else
{
lean_inc(v_a_1683_);
lean_dec(v___x_1643_);
v___x_1685_ = lean_box(0);
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
v_resetjp_1684_:
{
lean_object* v___x_1688_; 
if (v_isShared_1686_ == 0)
{
v___x_1688_ = v___x_1685_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v_a_1683_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___boxed(lean_object* v_a_1691_, lean_object* v_g_1692_, lean_object* v_f_1693_, lean_object* v_thm_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_){
_start:
{
lean_object* v_res_1702_; 
v_res_1702_ = lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0(v_a_1691_, v_g_1692_, v_f_1693_, v_thm_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_, v___y_1699_, v___y_1700_);
lean_dec(v___y_1700_);
lean_dec_ref(v___y_1699_);
lean_dec(v___y_1698_);
lean_dec_ref(v___y_1697_);
lean_dec(v___y_1696_);
lean_dec_ref(v___y_1695_);
return v_res_1702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg(lean_object* v_mvarId_1703_, lean_object* v_val_1704_, lean_object* v___y_1705_){
_start:
{
lean_object* v___x_1707_; lean_object* v_mctx_1708_; lean_object* v_cache_1709_; lean_object* v_zetaDeltaFVarIds_1710_; lean_object* v_postponed_1711_; lean_object* v_diag_1712_; lean_object* v___x_1714_; uint8_t v_isShared_1715_; uint8_t v_isSharedCheck_1740_; 
v___x_1707_ = lean_st_ref_take(v___y_1705_);
v_mctx_1708_ = lean_ctor_get(v___x_1707_, 0);
v_cache_1709_ = lean_ctor_get(v___x_1707_, 1);
v_zetaDeltaFVarIds_1710_ = lean_ctor_get(v___x_1707_, 2);
v_postponed_1711_ = lean_ctor_get(v___x_1707_, 3);
v_diag_1712_ = lean_ctor_get(v___x_1707_, 4);
v_isSharedCheck_1740_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1740_ == 0)
{
v___x_1714_ = v___x_1707_;
v_isShared_1715_ = v_isSharedCheck_1740_;
goto v_resetjp_1713_;
}
else
{
lean_inc(v_diag_1712_);
lean_inc(v_postponed_1711_);
lean_inc(v_zetaDeltaFVarIds_1710_);
lean_inc(v_cache_1709_);
lean_inc(v_mctx_1708_);
lean_dec(v___x_1707_);
v___x_1714_ = lean_box(0);
v_isShared_1715_ = v_isSharedCheck_1740_;
goto v_resetjp_1713_;
}
v_resetjp_1713_:
{
lean_object* v_depth_1716_; lean_object* v_levelAssignDepth_1717_; lean_object* v_lmvarCounter_1718_; lean_object* v_mvarCounter_1719_; lean_object* v_lDecls_1720_; lean_object* v_decls_1721_; lean_object* v_userNames_1722_; lean_object* v_lAssignment_1723_; lean_object* v_eAssignment_1724_; lean_object* v_dAssignment_1725_; lean_object* v___x_1727_; uint8_t v_isShared_1728_; uint8_t v_isSharedCheck_1739_; 
v_depth_1716_ = lean_ctor_get(v_mctx_1708_, 0);
v_levelAssignDepth_1717_ = lean_ctor_get(v_mctx_1708_, 1);
v_lmvarCounter_1718_ = lean_ctor_get(v_mctx_1708_, 2);
v_mvarCounter_1719_ = lean_ctor_get(v_mctx_1708_, 3);
v_lDecls_1720_ = lean_ctor_get(v_mctx_1708_, 4);
v_decls_1721_ = lean_ctor_get(v_mctx_1708_, 5);
v_userNames_1722_ = lean_ctor_get(v_mctx_1708_, 6);
v_lAssignment_1723_ = lean_ctor_get(v_mctx_1708_, 7);
v_eAssignment_1724_ = lean_ctor_get(v_mctx_1708_, 8);
v_dAssignment_1725_ = lean_ctor_get(v_mctx_1708_, 9);
v_isSharedCheck_1739_ = !lean_is_exclusive(v_mctx_1708_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1727_ = v_mctx_1708_;
v_isShared_1728_ = v_isSharedCheck_1739_;
goto v_resetjp_1726_;
}
else
{
lean_inc(v_dAssignment_1725_);
lean_inc(v_eAssignment_1724_);
lean_inc(v_lAssignment_1723_);
lean_inc(v_userNames_1722_);
lean_inc(v_decls_1721_);
lean_inc(v_lDecls_1720_);
lean_inc(v_mvarCounter_1719_);
lean_inc(v_lmvarCounter_1718_);
lean_inc(v_levelAssignDepth_1717_);
lean_inc(v_depth_1716_);
lean_dec(v_mctx_1708_);
v___x_1727_ = lean_box(0);
v_isShared_1728_ = v_isSharedCheck_1739_;
goto v_resetjp_1726_;
}
v_resetjp_1726_:
{
lean_object* v___x_1729_; lean_object* v___x_1731_; 
v___x_1729_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1___redArg(v_eAssignment_1724_, v_mvarId_1703_, v_val_1704_);
if (v_isShared_1728_ == 0)
{
lean_ctor_set(v___x_1727_, 8, v___x_1729_);
v___x_1731_ = v___x_1727_;
goto v_reusejp_1730_;
}
else
{
lean_object* v_reuseFailAlloc_1738_; 
v_reuseFailAlloc_1738_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1738_, 0, v_depth_1716_);
lean_ctor_set(v_reuseFailAlloc_1738_, 1, v_levelAssignDepth_1717_);
lean_ctor_set(v_reuseFailAlloc_1738_, 2, v_lmvarCounter_1718_);
lean_ctor_set(v_reuseFailAlloc_1738_, 3, v_mvarCounter_1719_);
lean_ctor_set(v_reuseFailAlloc_1738_, 4, v_lDecls_1720_);
lean_ctor_set(v_reuseFailAlloc_1738_, 5, v_decls_1721_);
lean_ctor_set(v_reuseFailAlloc_1738_, 6, v_userNames_1722_);
lean_ctor_set(v_reuseFailAlloc_1738_, 7, v_lAssignment_1723_);
lean_ctor_set(v_reuseFailAlloc_1738_, 8, v___x_1729_);
lean_ctor_set(v_reuseFailAlloc_1738_, 9, v_dAssignment_1725_);
v___x_1731_ = v_reuseFailAlloc_1738_;
goto v_reusejp_1730_;
}
v_reusejp_1730_:
{
lean_object* v___x_1733_; 
if (v_isShared_1715_ == 0)
{
lean_ctor_set(v___x_1714_, 0, v___x_1731_);
v___x_1733_ = v___x_1714_;
goto v_reusejp_1732_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v___x_1731_);
lean_ctor_set(v_reuseFailAlloc_1737_, 1, v_cache_1709_);
lean_ctor_set(v_reuseFailAlloc_1737_, 2, v_zetaDeltaFVarIds_1710_);
lean_ctor_set(v_reuseFailAlloc_1737_, 3, v_postponed_1711_);
lean_ctor_set(v_reuseFailAlloc_1737_, 4, v_diag_1712_);
v___x_1733_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1732_;
}
v_reusejp_1732_:
{
lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; 
v___x_1734_ = lean_st_ref_set(v___y_1705_, v___x_1733_);
v___x_1735_ = lean_box(0);
v___x_1736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1736_, 0, v___x_1735_);
return v___x_1736_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg___boxed(lean_object* v_mvarId_1741_, lean_object* v_val_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_){
_start:
{
lean_object* v_res_1745_; 
v_res_1745_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg(v_mvarId_1741_, v_val_1742_, v___y_1743_);
lean_dec(v___y_1743_);
return v_res_1745_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1747_; lean_object* v___x_1748_; 
v___x_1747_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__0));
v___x_1748_ = l_String_toRawSubstring_x27(v___x_1747_);
return v___x_1748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1(uint8_t v___x_1760_, lean_object* v_f_1761_, uint8_t v___x_1762_, lean_object* v_a_1763_, lean_object* v_g_1764_, lean_object* v_a_1765_, lean_object* v_a_1766_, lean_object* v_using_x3f_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_){
_start:
{
lean_object* v_ref_1775_; lean_object* v_quotContext_1776_; lean_object* v_currMacroScope_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; 
v_ref_1775_ = lean_ctor_get(v___y_1772_, 5);
v_quotContext_1776_ = lean_ctor_get(v___y_1772_, 10);
v_currMacroScope_1777_ = lean_ctor_get(v___y_1772_, 11);
v___x_1778_ = l_Lean_SourceInfo_fromRef(v_ref_1775_, v___x_1760_);
v___x_1779_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__6));
v___x_1780_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__1);
v___x_1781_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__10));
lean_inc(v_currMacroScope_1777_);
lean_inc(v_quotContext_1776_);
v___x_1782_ = l_Lean_addMacroScope(v_quotContext_1776_, v___x_1781_, v_currMacroScope_1777_);
v___x_1783_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___closed__5));
lean_inc_n(v___x_1778_, 2);
v___x_1784_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1784_, 0, v___x_1778_);
lean_ctor_set(v___x_1784_, 1, v___x_1780_);
lean_ctor_set(v___x_1784_, 2, v___x_1782_);
lean_ctor_set(v___x_1784_, 3, v___x_1783_);
v___x_1785_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___closed__1));
v___x_1786_ = l_Lean_Syntax_node1(v___x_1778_, v___x_1785_, v_f_1761_);
v___x_1787_ = l_Lean_Syntax_node2(v___x_1778_, v___x_1779_, v___x_1784_, v___x_1786_);
v___x_1788_ = lean_box(0);
v___x_1789_ = l_Lean_Elab_Term_elabTerm(v___x_1787_, v___x_1788_, v___x_1762_, v___x_1762_, v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1789_) == 0)
{
lean_object* v_a_1790_; lean_object* v___x_1791_; 
v_a_1790_ = lean_ctor_get(v___x_1789_, 0);
lean_inc(v_a_1790_);
lean_dec_ref_known(v___x_1789_, 1);
lean_inc(v___y_1773_);
lean_inc_ref(v___y_1772_);
lean_inc(v___y_1771_);
lean_inc_ref(v___y_1770_);
lean_inc_ref(v_a_1763_);
v___x_1791_ = lean_infer_type(v_a_1763_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1791_) == 0)
{
lean_object* v_a_1792_; lean_object* v___x_1794_; uint8_t v_isShared_1795_; uint8_t v_isSharedCheck_1922_; 
v_a_1792_ = lean_ctor_get(v___x_1791_, 0);
v_isSharedCheck_1922_ = !lean_is_exclusive(v___x_1791_);
if (v_isSharedCheck_1922_ == 0)
{
v___x_1794_ = v___x_1791_;
v_isShared_1795_ = v_isSharedCheck_1922_;
goto v_resetjp_1793_;
}
else
{
lean_inc(v_a_1792_);
lean_dec(v___x_1791_);
v___x_1794_ = lean_box(0);
v_isShared_1795_ = v_isSharedCheck_1922_;
goto v_resetjp_1793_;
}
v_resetjp_1793_:
{
lean_object* v___x_1796_; 
lean_inc(v_a_1790_);
v___x_1796_ = l_Lean_Meta_isExprDefEq(v_a_1792_, v_a_1790_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1796_) == 0)
{
lean_object* v___x_1797_; 
lean_dec_ref_known(v___x_1796_, 1);
lean_inc(v_g_1764_);
v___x_1797_ = l_Lean_MVarId_getType(v_g_1764_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1797_) == 0)
{
lean_object* v_a_1798_; lean_object* v___x_1800_; uint8_t v_isShared_1801_; uint8_t v_isSharedCheck_1913_; 
v_a_1798_ = lean_ctor_get(v___x_1797_, 0);
v_isSharedCheck_1913_ = !lean_is_exclusive(v___x_1797_);
if (v_isSharedCheck_1913_ == 0)
{
v___x_1800_ = v___x_1797_;
v_isShared_1801_ = v_isSharedCheck_1913_;
goto v_resetjp_1799_;
}
else
{
lean_inc(v_a_1798_);
lean_dec(v___x_1797_);
v___x_1800_ = lean_box(0);
v_isShared_1801_ = v_isSharedCheck_1913_;
goto v_resetjp_1799_;
}
v_resetjp_1799_:
{
lean_object* v___x_1802_; lean_object* v___x_1804_; 
v___x_1802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___lam__0___closed__0));
if (v_isShared_1801_ == 0)
{
lean_ctor_set_tag(v___x_1800_, 1);
lean_ctor_set(v___x_1800_, 0, v_a_1765_);
v___x_1804_ = v___x_1800_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1912_; 
v_reuseFailAlloc_1912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1912_, 0, v_a_1765_);
v___x_1804_ = v_reuseFailAlloc_1912_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1809_; 
v___x_1805_ = lean_unsigned_to_nat(1u);
v___x_1806_ = lean_mk_empty_array_with_capacity(v___x_1805_);
v___x_1807_ = lean_array_push(v___x_1806_, v___x_1804_);
if (v_isShared_1795_ == 0)
{
lean_ctor_set_tag(v___x_1794_, 1);
lean_ctor_set(v___x_1794_, 0, v_a_1798_);
v___x_1809_ = v___x_1794_;
goto v_reusejp_1808_;
}
else
{
lean_object* v_reuseFailAlloc_1911_; 
v_reuseFailAlloc_1911_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1911_, 0, v_a_1798_);
v___x_1809_ = v_reuseFailAlloc_1911_;
goto v_reusejp_1808_;
}
v_reusejp_1808_:
{
lean_object* v___x_1810_; 
lean_inc_ref(v_a_1763_);
v___x_1810_ = l_Lean_Elab_Term_elabAppArgs(v_a_1763_, v___x_1802_, v___x_1807_, v___x_1809_, v___x_1760_, v___x_1760_, v___x_1762_, v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1810_) == 0)
{
lean_object* v_a_1811_; lean_object* v___x_1812_; 
v_a_1811_ = lean_ctor_get(v___x_1810_, 0);
lean_inc(v_a_1811_);
lean_dec_ref_known(v___x_1810_, 1);
v___x_1812_ = l_Lean_MVarId_getType(v_g_1764_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1812_) == 0)
{
lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1910_; 
v_a_1813_ = lean_ctor_get(v___x_1812_, 0);
v_isSharedCheck_1910_ = !lean_is_exclusive(v___x_1812_);
if (v_isSharedCheck_1910_ == 0)
{
v___x_1815_ = v___x_1812_;
v_isShared_1816_ = v_isSharedCheck_1910_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1812_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1910_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1818_; 
if (v_isShared_1816_ == 0)
{
lean_ctor_set_tag(v___x_1815_, 1);
v___x_1818_ = v___x_1815_;
goto v_reusejp_1817_;
}
else
{
lean_object* v_reuseFailAlloc_1909_; 
v_reuseFailAlloc_1909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1909_, 0, v_a_1813_);
v___x_1818_ = v_reuseFailAlloc_1909_;
goto v_reusejp_1817_;
}
v_reusejp_1817_:
{
lean_object* v___x_1819_; 
v___x_1819_ = l_Lean_Elab_Term_ensureHasType(v___x_1818_, v_a_1811_, v___x_1788_, v___x_1788_, v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1819_) == 0)
{
lean_object* v_a_1820_; lean_object* v_a_1842_; 
v_a_1820_ = lean_ctor_get(v___x_1819_, 0);
lean_inc(v_a_1820_);
lean_dec_ref_known(v___x_1819_, 1);
if (lean_obj_tag(v_using_x3f_1767_) == 0)
{
v_a_1842_ = v___x_1788_;
goto v___jp_1841_;
}
else
{
lean_object* v_val_1892_; lean_object* v___x_1894_; uint8_t v_isShared_1895_; uint8_t v_isSharedCheck_1908_; 
v_val_1892_ = lean_ctor_get(v_using_x3f_1767_, 0);
v_isSharedCheck_1908_ = !lean_is_exclusive(v_using_x3f_1767_);
if (v_isSharedCheck_1908_ == 0)
{
v___x_1894_ = v_using_x3f_1767_;
v_isShared_1895_ = v_isSharedCheck_1908_;
goto v_resetjp_1893_;
}
else
{
lean_inc(v_val_1892_);
lean_dec(v_using_x3f_1767_);
v___x_1894_ = lean_box(0);
v_isShared_1895_ = v_isSharedCheck_1908_;
goto v_resetjp_1893_;
}
v_resetjp_1893_:
{
lean_object* v___x_1897_; 
lean_inc(v_a_1790_);
if (v_isShared_1895_ == 0)
{
lean_ctor_set(v___x_1894_, 0, v_a_1790_);
v___x_1897_ = v___x_1894_;
goto v_reusejp_1896_;
}
else
{
lean_object* v_reuseFailAlloc_1907_; 
v_reuseFailAlloc_1907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1907_, 0, v_a_1790_);
v___x_1897_ = v_reuseFailAlloc_1907_;
goto v_reusejp_1896_;
}
v_reusejp_1896_:
{
lean_object* v___x_1898_; 
v___x_1898_ = l_Lean_Elab_Term_elabTerm(v_val_1892_, v___x_1897_, v___x_1762_, v___x_1762_, v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
if (lean_obj_tag(v___x_1898_) == 0)
{
lean_object* v_a_1899_; lean_object* v___x_1901_; uint8_t v_isShared_1902_; uint8_t v_isSharedCheck_1906_; 
v_a_1899_ = lean_ctor_get(v___x_1898_, 0);
v_isSharedCheck_1906_ = !lean_is_exclusive(v___x_1898_);
if (v_isSharedCheck_1906_ == 0)
{
v___x_1901_ = v___x_1898_;
v_isShared_1902_ = v_isSharedCheck_1906_;
goto v_resetjp_1900_;
}
else
{
lean_inc(v_a_1899_);
lean_dec(v___x_1898_);
v___x_1901_ = lean_box(0);
v_isShared_1902_ = v_isSharedCheck_1906_;
goto v_resetjp_1900_;
}
v_resetjp_1900_:
{
lean_object* v___x_1904_; 
if (v_isShared_1902_ == 0)
{
lean_ctor_set_tag(v___x_1901_, 1);
v___x_1904_ = v___x_1901_;
goto v_reusejp_1903_;
}
else
{
lean_object* v_reuseFailAlloc_1905_; 
v_reuseFailAlloc_1905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1905_, 0, v_a_1899_);
v___x_1904_ = v_reuseFailAlloc_1905_;
goto v_reusejp_1903_;
}
v_reusejp_1903_:
{
v_a_1842_ = v___x_1904_;
goto v___jp_1841_;
}
}
}
else
{
lean_dec(v_a_1820_);
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec_ref(v_a_1763_);
return v___x_1898_;
}
}
}
}
v___jp_1821_:
{
lean_object* v___x_1822_; 
v___x_1822_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsUsingDefault(v___y_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec_ref(v___y_1770_);
if (lean_obj_tag(v___x_1822_) == 0)
{
lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1826_; uint8_t v_isShared_1827_; uint8_t v_isSharedCheck_1831_; 
lean_dec_ref_known(v___x_1822_, 1);
v___x_1823_ = l_Lean_Expr_mvarId_x21(v_a_1766_);
v___x_1824_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg(v___x_1823_, v_a_1820_, v___y_1771_);
lean_dec(v___y_1771_);
v_isSharedCheck_1831_ = !lean_is_exclusive(v___x_1824_);
if (v_isSharedCheck_1831_ == 0)
{
lean_object* v_unused_1832_; 
v_unused_1832_ = lean_ctor_get(v___x_1824_, 0);
lean_dec(v_unused_1832_);
v___x_1826_ = v___x_1824_;
v_isShared_1827_ = v_isSharedCheck_1831_;
goto v_resetjp_1825_;
}
else
{
lean_dec(v___x_1824_);
v___x_1826_ = lean_box(0);
v_isShared_1827_ = v_isSharedCheck_1831_;
goto v_resetjp_1825_;
}
v_resetjp_1825_:
{
lean_object* v___x_1829_; 
if (v_isShared_1827_ == 0)
{
lean_ctor_set(v___x_1826_, 0, v_a_1790_);
v___x_1829_ = v___x_1826_;
goto v_reusejp_1828_;
}
else
{
lean_object* v_reuseFailAlloc_1830_; 
v_reuseFailAlloc_1830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1830_, 0, v_a_1790_);
v___x_1829_ = v_reuseFailAlloc_1830_;
goto v_reusejp_1828_;
}
v_reusejp_1828_:
{
return v___x_1829_;
}
}
}
else
{
lean_object* v_a_1833_; lean_object* v___x_1835_; uint8_t v_isShared_1836_; uint8_t v_isSharedCheck_1840_; 
lean_dec(v_a_1820_);
lean_dec(v_a_1790_);
lean_dec(v___y_1771_);
v_a_1833_ = lean_ctor_get(v___x_1822_, 0);
v_isSharedCheck_1840_ = !lean_is_exclusive(v___x_1822_);
if (v_isSharedCheck_1840_ == 0)
{
v___x_1835_ = v___x_1822_;
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
else
{
lean_inc(v_a_1833_);
lean_dec(v___x_1822_);
v___x_1835_ = lean_box(0);
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
v_resetjp_1834_:
{
lean_object* v___x_1838_; 
if (v_isShared_1836_ == 0)
{
v___x_1838_ = v___x_1835_;
goto v_reusejp_1837_;
}
else
{
lean_object* v_reuseFailAlloc_1839_; 
v_reuseFailAlloc_1839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1839_, 0, v_a_1833_);
v___x_1838_ = v_reuseFailAlloc_1839_;
goto v_reusejp_1837_;
}
v_reusejp_1837_:
{
return v___x_1838_;
}
}
}
}
v___jp_1841_:
{
lean_object* v___x_1843_; uint8_t v_foApprox_1844_; uint8_t v_ctxApprox_1845_; uint8_t v_quasiPatternApprox_1846_; uint8_t v_constApprox_1847_; uint8_t v_isDefEqStuckEx_1848_; uint8_t v_unificationHints_1849_; uint8_t v_proofIrrelevance_1850_; uint8_t v_offsetCnstrs_1851_; uint8_t v_transparency_1852_; uint8_t v_etaStruct_1853_; uint8_t v_univApprox_1854_; uint8_t v_iota_1855_; uint8_t v_beta_1856_; uint8_t v_proj_1857_; uint8_t v_zeta_1858_; uint8_t v_zetaDelta_1859_; uint8_t v_zetaUnused_1860_; uint8_t v_zetaHave_1861_; uint8_t v_canUnfoldPredicateConfig_1862_; lean_object* v___x_1864_; uint8_t v_isShared_1865_; uint8_t v_isSharedCheck_1891_; 
v___x_1843_ = l_Lean_Meta_Context_config(v___y_1770_);
v_foApprox_1844_ = lean_ctor_get_uint8(v___x_1843_, 0);
v_ctxApprox_1845_ = lean_ctor_get_uint8(v___x_1843_, 1);
v_quasiPatternApprox_1846_ = lean_ctor_get_uint8(v___x_1843_, 2);
v_constApprox_1847_ = lean_ctor_get_uint8(v___x_1843_, 3);
v_isDefEqStuckEx_1848_ = lean_ctor_get_uint8(v___x_1843_, 4);
v_unificationHints_1849_ = lean_ctor_get_uint8(v___x_1843_, 5);
v_proofIrrelevance_1850_ = lean_ctor_get_uint8(v___x_1843_, 6);
v_offsetCnstrs_1851_ = lean_ctor_get_uint8(v___x_1843_, 8);
v_transparency_1852_ = lean_ctor_get_uint8(v___x_1843_, 9);
v_etaStruct_1853_ = lean_ctor_get_uint8(v___x_1843_, 10);
v_univApprox_1854_ = lean_ctor_get_uint8(v___x_1843_, 11);
v_iota_1855_ = lean_ctor_get_uint8(v___x_1843_, 12);
v_beta_1856_ = lean_ctor_get_uint8(v___x_1843_, 13);
v_proj_1857_ = lean_ctor_get_uint8(v___x_1843_, 14);
v_zeta_1858_ = lean_ctor_get_uint8(v___x_1843_, 15);
v_zetaDelta_1859_ = lean_ctor_get_uint8(v___x_1843_, 16);
v_zetaUnused_1860_ = lean_ctor_get_uint8(v___x_1843_, 17);
v_zetaHave_1861_ = lean_ctor_get_uint8(v___x_1843_, 18);
v_canUnfoldPredicateConfig_1862_ = lean_ctor_get_uint8(v___x_1843_, 19);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1843_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1864_ = v___x_1843_;
v_isShared_1865_ = v_isSharedCheck_1891_;
goto v_resetjp_1863_;
}
else
{
lean_dec(v___x_1843_);
v___x_1864_ = lean_box(0);
v_isShared_1865_ = v_isSharedCheck_1891_;
goto v_resetjp_1863_;
}
v_resetjp_1863_:
{
uint8_t v_trackZetaDelta_1866_; lean_object* v_zetaDeltaSet_1867_; lean_object* v_lctx_1868_; lean_object* v_localInstances_1869_; lean_object* v_defEqCtx_x3f_1870_; lean_object* v_synthPendingDepth_1871_; lean_object* v_customCanUnfoldPredicate_x3f_1872_; uint8_t v_univApprox_1873_; uint8_t v_inTypeClassResolution_1874_; uint8_t v_cacheInferType_1875_; lean_object* v___x_1877_; 
v_trackZetaDelta_1866_ = lean_ctor_get_uint8(v___y_1770_, sizeof(void*)*7);
v_zetaDeltaSet_1867_ = lean_ctor_get(v___y_1770_, 1);
v_lctx_1868_ = lean_ctor_get(v___y_1770_, 2);
v_localInstances_1869_ = lean_ctor_get(v___y_1770_, 3);
v_defEqCtx_x3f_1870_ = lean_ctor_get(v___y_1770_, 4);
v_synthPendingDepth_1871_ = lean_ctor_get(v___y_1770_, 5);
v_customCanUnfoldPredicate_x3f_1872_ = lean_ctor_get(v___y_1770_, 6);
v_univApprox_1873_ = lean_ctor_get_uint8(v___y_1770_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1874_ = lean_ctor_get_uint8(v___y_1770_, sizeof(void*)*7 + 2);
v_cacheInferType_1875_ = lean_ctor_get_uint8(v___y_1770_, sizeof(void*)*7 + 3);
if (v_isShared_1865_ == 0)
{
v___x_1877_ = v___x_1864_;
goto v_reusejp_1876_;
}
else
{
lean_object* v_reuseFailAlloc_1890_; 
v_reuseFailAlloc_1890_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 0, v_foApprox_1844_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 1, v_ctxApprox_1845_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 2, v_quasiPatternApprox_1846_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 3, v_constApprox_1847_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 4, v_isDefEqStuckEx_1848_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 5, v_unificationHints_1849_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 6, v_proofIrrelevance_1850_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 8, v_offsetCnstrs_1851_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 9, v_transparency_1852_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 10, v_etaStruct_1853_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 11, v_univApprox_1854_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 12, v_iota_1855_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 13, v_beta_1856_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 14, v_proj_1857_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 15, v_zeta_1858_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 16, v_zetaDelta_1859_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 17, v_zetaUnused_1860_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 18, v_zetaHave_1861_);
lean_ctor_set_uint8(v_reuseFailAlloc_1890_, 19, v_canUnfoldPredicateConfig_1862_);
v___x_1877_ = v_reuseFailAlloc_1890_;
goto v_reusejp_1876_;
}
v_reusejp_1876_:
{
uint64_t v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
lean_ctor_set_uint8(v___x_1877_, 7, v___x_1762_);
v___x_1878_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1877_);
v___x_1879_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1879_, 0, v___x_1877_);
lean_ctor_set_uint64(v___x_1879_, sizeof(void*)*1, v___x_1878_);
lean_inc(v_customCanUnfoldPredicate_x3f_1872_);
lean_inc(v_synthPendingDepth_1871_);
lean_inc(v_defEqCtx_x3f_1870_);
lean_inc_ref(v_localInstances_1869_);
lean_inc_ref(v_lctx_1868_);
lean_inc(v_zetaDeltaSet_1867_);
v___x_1880_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1880_, 0, v___x_1879_);
lean_ctor_set(v___x_1880_, 1, v_zetaDeltaSet_1867_);
lean_ctor_set(v___x_1880_, 2, v_lctx_1868_);
lean_ctor_set(v___x_1880_, 3, v_localInstances_1869_);
lean_ctor_set(v___x_1880_, 4, v_defEqCtx_x3f_1870_);
lean_ctor_set(v___x_1880_, 5, v_synthPendingDepth_1871_);
lean_ctor_set(v___x_1880_, 6, v_customCanUnfoldPredicate_x3f_1872_);
lean_ctor_set_uint8(v___x_1880_, sizeof(void*)*7, v_trackZetaDelta_1866_);
lean_ctor_set_uint8(v___x_1880_, sizeof(void*)*7 + 1, v_univApprox_1873_);
lean_ctor_set_uint8(v___x_1880_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1874_);
lean_ctor_set_uint8(v___x_1880_, sizeof(void*)*7 + 3, v_cacheInferType_1875_);
v___x_1881_ = lp_mathlib_Mathlib_Tactic_maybeProveInjective(v_a_1763_, v_a_1842_, v___x_1880_, v___y_1771_, v___y_1772_, v___y_1773_);
lean_dec_ref_known(v___x_1880_, 7);
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_dec_ref_known(v___x_1881_, 1);
goto v___jp_1821_;
}
else
{
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_dec_ref_known(v___x_1881_, 1);
goto v___jp_1821_;
}
else
{
lean_object* v_a_1882_; lean_object* v___x_1884_; uint8_t v_isShared_1885_; uint8_t v_isSharedCheck_1889_; 
lean_dec(v_a_1820_);
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_1889_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_1889_ == 0)
{
v___x_1884_ = v___x_1881_;
v_isShared_1885_ = v_isSharedCheck_1889_;
goto v_resetjp_1883_;
}
else
{
lean_inc(v_a_1882_);
lean_dec(v___x_1881_);
v___x_1884_ = lean_box(0);
v_isShared_1885_ = v_isSharedCheck_1889_;
goto v_resetjp_1883_;
}
v_resetjp_1883_:
{
lean_object* v___x_1887_; 
if (v_isShared_1885_ == 0)
{
v___x_1887_ = v___x_1884_;
goto v_reusejp_1886_;
}
else
{
lean_object* v_reuseFailAlloc_1888_; 
v_reuseFailAlloc_1888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1888_, 0, v_a_1882_);
v___x_1887_ = v_reuseFailAlloc_1888_;
goto v_reusejp_1886_;
}
v_reusejp_1886_:
{
return v___x_1887_;
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
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec_ref(v_a_1763_);
return v___x_1819_;
}
}
}
}
else
{
lean_dec(v_a_1811_);
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec_ref(v_a_1763_);
return v___x_1812_;
}
}
else
{
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec(v_g_1764_);
lean_dec_ref(v_a_1763_);
return v___x_1810_;
}
}
}
}
}
else
{
lean_del_object(v___x_1794_);
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec_ref(v_a_1765_);
lean_dec(v_g_1764_);
lean_dec_ref(v_a_1763_);
return v___x_1797_;
}
}
else
{
lean_object* v_a_1914_; lean_object* v___x_1916_; uint8_t v_isShared_1917_; uint8_t v_isSharedCheck_1921_; 
lean_del_object(v___x_1794_);
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec_ref(v_a_1765_);
lean_dec(v_g_1764_);
lean_dec_ref(v_a_1763_);
v_a_1914_ = lean_ctor_get(v___x_1796_, 0);
v_isSharedCheck_1921_ = !lean_is_exclusive(v___x_1796_);
if (v_isSharedCheck_1921_ == 0)
{
v___x_1916_ = v___x_1796_;
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
else
{
lean_inc(v_a_1914_);
lean_dec(v___x_1796_);
v___x_1916_ = lean_box(0);
v_isShared_1917_ = v_isSharedCheck_1921_;
goto v_resetjp_1915_;
}
v_resetjp_1915_:
{
lean_object* v___x_1919_; 
if (v_isShared_1917_ == 0)
{
v___x_1919_ = v___x_1916_;
goto v_reusejp_1918_;
}
else
{
lean_object* v_reuseFailAlloc_1920_; 
v_reuseFailAlloc_1920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1920_, 0, v_a_1914_);
v___x_1919_ = v_reuseFailAlloc_1920_;
goto v_reusejp_1918_;
}
v_reusejp_1918_:
{
return v___x_1919_;
}
}
}
}
}
else
{
lean_dec(v_a_1790_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec_ref(v_a_1765_);
lean_dec(v_g_1764_);
lean_dec_ref(v_a_1763_);
return v___x_1791_;
}
}
else
{
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec(v_using_x3f_1767_);
lean_dec_ref(v_a_1765_);
lean_dec(v_g_1764_);
lean_dec_ref(v_a_1763_);
return v___x_1789_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___boxed(lean_object* v___x_1923_, lean_object* v_f_1924_, lean_object* v___x_1925_, lean_object* v_a_1926_, lean_object* v_g_1927_, lean_object* v_a_1928_, lean_object* v_a_1929_, lean_object* v_using_x3f_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_){
_start:
{
uint8_t v___x_20204__boxed_1938_; uint8_t v___x_20205__boxed_1939_; lean_object* v_res_1940_; 
v___x_20204__boxed_1938_ = lean_unbox(v___x_1923_);
v___x_20205__boxed_1939_ = lean_unbox(v___x_1925_);
v_res_1940_ = lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1(v___x_20204__boxed_1938_, v_f_1924_, v___x_20205__boxed_1939_, v_a_1926_, v_g_1927_, v_a_1928_, v_a_1929_, v_using_x3f_1930_, v___y_1931_, v___y_1932_, v___y_1933_, v___y_1934_, v___y_1935_, v___y_1936_);
lean_dec(v___y_1932_);
lean_dec_ref(v___y_1931_);
lean_dec_ref(v_a_1929_);
return v_res_1940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg(lean_object* v_mvarId_1941_, lean_object* v_val_1942_, lean_object* v___y_1943_){
_start:
{
lean_object* v___x_1945_; lean_object* v_mctx_1946_; lean_object* v_cache_1947_; lean_object* v_zetaDeltaFVarIds_1948_; lean_object* v_postponed_1949_; lean_object* v_diag_1950_; lean_object* v___x_1952_; uint8_t v_isShared_1953_; uint8_t v_isSharedCheck_1978_; 
v___x_1945_ = lean_st_ref_take(v___y_1943_);
v_mctx_1946_ = lean_ctor_get(v___x_1945_, 0);
v_cache_1947_ = lean_ctor_get(v___x_1945_, 1);
v_zetaDeltaFVarIds_1948_ = lean_ctor_get(v___x_1945_, 2);
v_postponed_1949_ = lean_ctor_get(v___x_1945_, 3);
v_diag_1950_ = lean_ctor_get(v___x_1945_, 4);
v_isSharedCheck_1978_ = !lean_is_exclusive(v___x_1945_);
if (v_isSharedCheck_1978_ == 0)
{
v___x_1952_ = v___x_1945_;
v_isShared_1953_ = v_isSharedCheck_1978_;
goto v_resetjp_1951_;
}
else
{
lean_inc(v_diag_1950_);
lean_inc(v_postponed_1949_);
lean_inc(v_zetaDeltaFVarIds_1948_);
lean_inc(v_cache_1947_);
lean_inc(v_mctx_1946_);
lean_dec(v___x_1945_);
v___x_1952_ = lean_box(0);
v_isShared_1953_ = v_isSharedCheck_1978_;
goto v_resetjp_1951_;
}
v_resetjp_1951_:
{
lean_object* v_depth_1954_; lean_object* v_levelAssignDepth_1955_; lean_object* v_lmvarCounter_1956_; lean_object* v_mvarCounter_1957_; lean_object* v_lDecls_1958_; lean_object* v_decls_1959_; lean_object* v_userNames_1960_; lean_object* v_lAssignment_1961_; lean_object* v_eAssignment_1962_; lean_object* v_dAssignment_1963_; lean_object* v___x_1965_; uint8_t v_isShared_1966_; uint8_t v_isSharedCheck_1977_; 
v_depth_1954_ = lean_ctor_get(v_mctx_1946_, 0);
v_levelAssignDepth_1955_ = lean_ctor_get(v_mctx_1946_, 1);
v_lmvarCounter_1956_ = lean_ctor_get(v_mctx_1946_, 2);
v_mvarCounter_1957_ = lean_ctor_get(v_mctx_1946_, 3);
v_lDecls_1958_ = lean_ctor_get(v_mctx_1946_, 4);
v_decls_1959_ = lean_ctor_get(v_mctx_1946_, 5);
v_userNames_1960_ = lean_ctor_get(v_mctx_1946_, 6);
v_lAssignment_1961_ = lean_ctor_get(v_mctx_1946_, 7);
v_eAssignment_1962_ = lean_ctor_get(v_mctx_1946_, 8);
v_dAssignment_1963_ = lean_ctor_get(v_mctx_1946_, 9);
v_isSharedCheck_1977_ = !lean_is_exclusive(v_mctx_1946_);
if (v_isSharedCheck_1977_ == 0)
{
v___x_1965_ = v_mctx_1946_;
v_isShared_1966_ = v_isSharedCheck_1977_;
goto v_resetjp_1964_;
}
else
{
lean_inc(v_dAssignment_1963_);
lean_inc(v_eAssignment_1962_);
lean_inc(v_lAssignment_1961_);
lean_inc(v_userNames_1960_);
lean_inc(v_decls_1959_);
lean_inc(v_lDecls_1958_);
lean_inc(v_mvarCounter_1957_);
lean_inc(v_lmvarCounter_1956_);
lean_inc(v_levelAssignDepth_1955_);
lean_inc(v_depth_1954_);
lean_dec(v_mctx_1946_);
v___x_1965_ = lean_box(0);
v_isShared_1966_ = v_isSharedCheck_1977_;
goto v_resetjp_1964_;
}
v_resetjp_1964_:
{
lean_object* v___x_1967_; lean_object* v___x_1969_; 
v___x_1967_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_maybeProveInjective_spec__1_spec__1___redArg(v_eAssignment_1962_, v_mvarId_1941_, v_val_1942_);
if (v_isShared_1966_ == 0)
{
lean_ctor_set(v___x_1965_, 8, v___x_1967_);
v___x_1969_ = v___x_1965_;
goto v_reusejp_1968_;
}
else
{
lean_object* v_reuseFailAlloc_1976_; 
v_reuseFailAlloc_1976_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1976_, 0, v_depth_1954_);
lean_ctor_set(v_reuseFailAlloc_1976_, 1, v_levelAssignDepth_1955_);
lean_ctor_set(v_reuseFailAlloc_1976_, 2, v_lmvarCounter_1956_);
lean_ctor_set(v_reuseFailAlloc_1976_, 3, v_mvarCounter_1957_);
lean_ctor_set(v_reuseFailAlloc_1976_, 4, v_lDecls_1958_);
lean_ctor_set(v_reuseFailAlloc_1976_, 5, v_decls_1959_);
lean_ctor_set(v_reuseFailAlloc_1976_, 6, v_userNames_1960_);
lean_ctor_set(v_reuseFailAlloc_1976_, 7, v_lAssignment_1961_);
lean_ctor_set(v_reuseFailAlloc_1976_, 8, v___x_1967_);
lean_ctor_set(v_reuseFailAlloc_1976_, 9, v_dAssignment_1963_);
v___x_1969_ = v_reuseFailAlloc_1976_;
goto v_reusejp_1968_;
}
v_reusejp_1968_:
{
lean_object* v___x_1971_; 
if (v_isShared_1953_ == 0)
{
lean_ctor_set(v___x_1952_, 0, v___x_1969_);
v___x_1971_ = v___x_1952_;
goto v_reusejp_1970_;
}
else
{
lean_object* v_reuseFailAlloc_1975_; 
v_reuseFailAlloc_1975_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1975_, 0, v___x_1969_);
lean_ctor_set(v_reuseFailAlloc_1975_, 1, v_cache_1947_);
lean_ctor_set(v_reuseFailAlloc_1975_, 2, v_zetaDeltaFVarIds_1948_);
lean_ctor_set(v_reuseFailAlloc_1975_, 3, v_postponed_1949_);
lean_ctor_set(v_reuseFailAlloc_1975_, 4, v_diag_1950_);
v___x_1971_ = v_reuseFailAlloc_1975_;
goto v_reusejp_1970_;
}
v_reusejp_1970_:
{
lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; 
v___x_1972_ = lean_st_ref_set(v___y_1943_, v___x_1971_);
v___x_1973_ = lean_box(0);
v___x_1974_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1974_, 0, v___x_1973_);
return v___x_1974_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg___boxed(lean_object* v_mvarId_1979_, lean_object* v_val_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_){
_start:
{
lean_object* v_res_1983_; 
v_res_1983_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg(v_mvarId_1979_, v_val_1980_, v___y_1981_);
lean_dec(v___y_1981_);
return v_res_1983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget(lean_object* v_f_2006_, lean_object* v_using_x3f_2007_, lean_object* v_g_2008_, lean_object* v_a_2009_, lean_object* v_a_2010_, lean_object* v_a_2011_, lean_object* v_a_2012_, lean_object* v_a_2013_, lean_object* v_a_2014_, lean_object* v_a_2015_, lean_object* v_a_2016_){
_start:
{
lean_object* v_thm_2019_; lean_object* v___y_2020_; lean_object* v___y_2021_; lean_object* v___y_2022_; lean_object* v___y_2023_; lean_object* v___y_2024_; lean_object* v___y_2025_; lean_object* v___y_2026_; lean_object* v___y_2027_; lean_object* v___y_2087_; lean_object* v___y_2088_; lean_object* v___y_2089_; lean_object* v___y_2090_; lean_object* v___y_2091_; lean_object* v___y_2092_; lean_object* v___y_2093_; lean_object* v___y_2094_; lean_object* v___y_2097_; lean_object* v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v___y_2104_; lean_object* v___x_2106_; 
lean_inc(v_g_2008_);
v___x_2106_ = l_Lean_MVarId_getType(v_g_2008_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2106_) == 0)
{
lean_object* v_a_2107_; lean_object* v___x_2108_; lean_object* v_a_2109_; lean_object* v___x_2111_; uint8_t v_isShared_2112_; uint8_t v_isSharedCheck_2328_; 
v_a_2107_ = lean_ctor_get(v___x_2106_, 0);
lean_inc(v_a_2107_);
lean_dec_ref_known(v___x_2106_, 1);
v___x_2108_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_applyFunHyp_spec__1___redArg(v_a_2107_, v_a_2014_);
v_a_2109_ = lean_ctor_get(v___x_2108_, 0);
v_isSharedCheck_2328_ = !lean_is_exclusive(v___x_2108_);
if (v_isSharedCheck_2328_ == 0)
{
v___x_2111_ = v___x_2108_;
v_isShared_2112_ = v_isSharedCheck_2328_;
goto v_resetjp_2110_;
}
else
{
lean_inc(v_a_2109_);
lean_dec(v___x_2108_);
v___x_2111_ = lean_box(0);
v_isShared_2112_ = v_isSharedCheck_2328_;
goto v_resetjp_2110_;
}
v_resetjp_2110_:
{
lean_object* v___x_2113_; 
v___x_2113_ = l_Lean_Meta_whnfR(v_a_2109_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2113_) == 0)
{
lean_object* v_a_2114_; lean_object* v___x_2115_; lean_object* v_fst_2116_; 
v_a_2114_ = lean_ctor_get(v___x_2113_, 0);
lean_inc(v_a_2114_);
lean_dec_ref_known(v___x_2113_, 1);
v___x_2115_ = l_Lean_Expr_getAppFnArgs(v_a_2114_);
v_fst_2116_ = lean_ctor_get(v___x_2115_, 0);
lean_inc(v_fst_2116_);
if (lean_obj_tag(v_fst_2116_) == 1)
{
lean_object* v_pre_2117_; 
v_pre_2117_ = lean_ctor_get(v_fst_2116_, 0);
lean_inc(v_pre_2117_);
switch(lean_obj_tag(v_pre_2117_))
{
case 0:
{
lean_object* v_snd_2118_; lean_object* v___x_2120_; uint8_t v_isShared_2121_; uint8_t v_isSharedCheck_2291_; 
v_snd_2118_ = lean_ctor_get(v___x_2115_, 1);
v_isSharedCheck_2291_ = !lean_is_exclusive(v___x_2115_);
if (v_isSharedCheck_2291_ == 0)
{
lean_object* v_unused_2292_; 
v_unused_2292_ = lean_ctor_get(v___x_2115_, 0);
lean_dec(v_unused_2292_);
v___x_2120_ = v___x_2115_;
v_isShared_2121_ = v_isSharedCheck_2291_;
goto v_resetjp_2119_;
}
else
{
lean_inc(v_snd_2118_);
lean_dec(v___x_2115_);
v___x_2120_ = lean_box(0);
v_isShared_2121_ = v_isSharedCheck_2291_;
goto v_resetjp_2119_;
}
v_resetjp_2119_:
{
lean_object* v_str_2122_; lean_object* v___x_2123_; uint8_t v___x_2124_; 
v_str_2122_ = lean_ctor_get(v_fst_2116_, 1);
lean_inc_ref(v_str_2122_);
lean_dec_ref_known(v_fst_2116_, 2);
v___x_2123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__9));
v___x_2124_ = lean_string_dec_eq(v_str_2122_, v___x_2123_);
if (v___x_2124_ == 0)
{
lean_object* v___x_2125_; uint8_t v___x_2126_; 
v___x_2125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__8));
v___x_2126_ = lean_string_dec_eq(v_str_2122_, v___x_2125_);
lean_dec_ref(v_str_2122_);
if (v___x_2126_ == 0)
{
lean_object* v___x_2127_; 
lean_del_object(v___x_2120_);
lean_dec(v_snd_2118_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
v___x_2127_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2127_;
}
else
{
lean_object* v___x_2128_; lean_object* v___x_2129_; uint8_t v___x_2130_; 
v___x_2128_ = lean_array_get_size(v_snd_2118_);
lean_dec(v_snd_2118_);
v___x_2129_ = lean_unsigned_to_nat(3u);
v___x_2130_ = lean_nat_dec_eq(v___x_2128_, v___x_2129_);
if (v___x_2130_ == 0)
{
lean_object* v___x_2131_; 
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
v___x_2131_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2131_;
}
else
{
uint8_t v___x_2132_; lean_object* v___x_2133_; 
v___x_2132_ = 0;
v___x_2133_ = l_Lean_Meta_mkFreshTypeMVar(v___x_2132_, v_pre_2117_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2133_) == 0)
{
lean_object* v_a_2134_; lean_object* v___x_2135_; 
v_a_2134_ = lean_ctor_get(v___x_2133_, 0);
lean_inc(v_a_2134_);
lean_dec_ref_known(v___x_2133_, 1);
lean_inc(v_g_2008_);
v___x_2135_ = l_Lean_MVarId_getTag(v_g_2008_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2135_) == 0)
{
lean_object* v_a_2136_; lean_object* v___x_2137_; 
v_a_2136_ = lean_ctor_get(v___x_2135_, 0);
lean_inc(v_a_2136_);
lean_dec_ref_known(v___x_2135_, 1);
v___x_2137_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_a_2134_, v_a_2136_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2137_) == 0)
{
lean_object* v_a_2138_; lean_object* v___x_2139_; 
v_a_2138_ = lean_ctor_get(v___x_2137_, 0);
lean_inc(v_a_2138_);
lean_dec_ref_known(v___x_2137_, 1);
v___x_2139_ = l_Lean_Meta_mkFreshTypeMVar(v___x_2132_, v_pre_2117_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2139_) == 0)
{
lean_object* v_a_2140_; lean_object* v___x_2141_; 
v_a_2140_ = lean_ctor_get(v___x_2139_, 0);
lean_inc(v_a_2140_);
lean_dec_ref_known(v___x_2139_, 1);
lean_inc(v_g_2008_);
v___x_2141_ = l_Lean_MVarId_getTag(v_g_2008_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2141_) == 0)
{
lean_object* v_a_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; 
v_a_2142_ = lean_ctor_get(v___x_2141_, 0);
lean_inc(v_a_2142_);
lean_dec_ref_known(v___x_2141_, 1);
v___x_2143_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__5));
v___x_2144_ = l_Lean_Meta_appendTag(v_a_2142_, v___x_2143_);
v___x_2145_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_a_2140_, v___x_2144_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2145_) == 0)
{
lean_object* v_a_2146_; lean_object* v___x_2147_; 
v_a_2146_ = lean_ctor_get(v___x_2145_, 0);
lean_inc(v_a_2146_);
lean_dec_ref_known(v___x_2145_, 1);
lean_inc(v_g_2008_);
v___x_2147_ = l_Lean_MVarId_getType(v_g_2008_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2147_) == 0)
{
lean_object* v_a_2148_; lean_object* v___x_2150_; 
v_a_2148_ = lean_ctor_get(v___x_2147_, 0);
lean_inc(v_a_2148_);
lean_dec_ref_known(v___x_2147_, 1);
if (v_isShared_2112_ == 0)
{
lean_ctor_set_tag(v___x_2111_, 1);
lean_ctor_set(v___x_2111_, 0, v_a_2148_);
v___x_2150_ = v___x_2111_;
goto v_reusejp_2149_;
}
else
{
lean_object* v_reuseFailAlloc_2213_; 
v_reuseFailAlloc_2213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2213_, 0, v_a_2148_);
v___x_2150_ = v_reuseFailAlloc_2213_;
goto v_reusejp_2149_;
}
v_reusejp_2149_:
{
lean_object* v___x_2151_; 
v___x_2151_ = l_Lean_Meta_mkFreshExprMVar(v___x_2150_, v___x_2132_, v_pre_2117_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2151_) == 0)
{
lean_object* v_a_2152_; lean_object* v___x_2153_; 
v_a_2152_ = lean_ctor_get(v___x_2151_, 0);
lean_inc(v_a_2152_);
lean_dec_ref_known(v___x_2151_, 1);
lean_inc(v_g_2008_);
v___x_2153_ = l_Lean_MVarId_getTag(v_g_2008_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2153_) == 0)
{
lean_object* v_a_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___f_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; 
v_a_2154_ = lean_ctor_get(v___x_2153_, 0);
lean_inc(v_a_2154_);
lean_dec_ref_known(v___x_2153_, 1);
v___x_2155_ = lean_box(v___x_2124_);
v___x_2156_ = lean_box(v___x_2130_);
lean_inc(v_a_2152_);
lean_inc(v_a_2138_);
lean_inc(v_g_2008_);
lean_inc(v_a_2146_);
v___f_2157_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__1___boxed), 15, 8);
lean_closure_set(v___f_2157_, 0, v___x_2155_);
lean_closure_set(v___f_2157_, 1, v_f_2006_);
lean_closure_set(v___f_2157_, 2, v___x_2156_);
lean_closure_set(v___f_2157_, 3, v_a_2146_);
lean_closure_set(v___f_2157_, 4, v_g_2008_);
lean_closure_set(v___f_2157_, 5, v_a_2138_);
lean_closure_set(v___f_2157_, 6, v_a_2152_);
lean_closure_set(v___f_2157_, 7, v_using_x3f_2007_);
v___x_2158_ = lean_box(v___x_2124_);
v___x_2159_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_runTermElab___boxed), 12, 3);
lean_closure_set(v___x_2159_, 0, lean_box(0));
lean_closure_set(v___x_2159_, 1, v___f_2157_);
lean_closure_set(v___x_2159_, 2, v___x_2158_);
v___x_2160_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_2160_, 0, lean_box(0));
lean_closure_set(v___x_2160_, 1, v___x_2159_);
v___x_2161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_2162_ = l_Lean_Elab_Tactic_withCollectingNewGoalsFrom(v___x_2160_, v_a_2154_, v___x_2161_, v___x_2124_, v_a_2009_, v_a_2010_, v_a_2011_, v_a_2012_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
if (lean_obj_tag(v___x_2162_) == 0)
{
lean_object* v_a_2163_; lean_object* v_snd_2164_; lean_object* v___x_2166_; uint8_t v_isShared_2167_; uint8_t v_isSharedCheck_2187_; 
v_a_2163_ = lean_ctor_get(v___x_2162_, 0);
lean_inc(v_a_2163_);
lean_dec_ref_known(v___x_2162_, 1);
v_snd_2164_ = lean_ctor_get(v_a_2163_, 1);
v_isSharedCheck_2187_ = !lean_is_exclusive(v_a_2163_);
if (v_isSharedCheck_2187_ == 0)
{
lean_object* v_unused_2188_; 
v_unused_2188_ = lean_ctor_get(v_a_2163_, 0);
lean_dec(v_unused_2188_);
v___x_2166_ = v_a_2163_;
v_isShared_2167_ = v_isSharedCheck_2187_;
goto v_resetjp_2165_;
}
else
{
lean_inc(v_snd_2164_);
lean_dec(v_a_2163_);
v___x_2166_ = lean_box(0);
v_isShared_2167_ = v_isSharedCheck_2187_;
goto v_resetjp_2165_;
}
v_resetjp_2165_:
{
lean_object* v___x_2168_; lean_object* v___x_2170_; uint8_t v_isShared_2171_; uint8_t v_isSharedCheck_2185_; 
v___x_2168_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg(v_g_2008_, v_a_2152_, v_a_2014_);
v_isSharedCheck_2185_ = !lean_is_exclusive(v___x_2168_);
if (v_isSharedCheck_2185_ == 0)
{
lean_object* v_unused_2186_; 
v_unused_2186_ = lean_ctor_get(v___x_2168_, 0);
lean_dec(v_unused_2186_);
v___x_2170_ = v___x_2168_;
v_isShared_2171_ = v_isSharedCheck_2185_;
goto v_resetjp_2169_;
}
else
{
lean_dec(v___x_2168_);
v___x_2170_ = lean_box(0);
v_isShared_2171_ = v_isSharedCheck_2185_;
goto v_resetjp_2169_;
}
v_resetjp_2169_:
{
lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2176_; 
v___x_2172_ = l_Lean_Expr_mvarId_x21(v_a_2138_);
lean_dec(v_a_2138_);
v___x_2173_ = l_Lean_Expr_mvarId_x21(v_a_2146_);
lean_dec(v_a_2146_);
v___x_2174_ = lean_box(0);
if (v_isShared_2167_ == 0)
{
lean_ctor_set_tag(v___x_2166_, 1);
lean_ctor_set(v___x_2166_, 1, v___x_2174_);
lean_ctor_set(v___x_2166_, 0, v___x_2173_);
v___x_2176_ = v___x_2166_;
goto v_reusejp_2175_;
}
else
{
lean_object* v_reuseFailAlloc_2184_; 
v_reuseFailAlloc_2184_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2184_, 0, v___x_2173_);
lean_ctor_set(v_reuseFailAlloc_2184_, 1, v___x_2174_);
v___x_2176_ = v_reuseFailAlloc_2184_;
goto v_reusejp_2175_;
}
v_reusejp_2175_:
{
lean_object* v___x_2178_; 
if (v_isShared_2121_ == 0)
{
lean_ctor_set_tag(v___x_2120_, 1);
lean_ctor_set(v___x_2120_, 1, v___x_2176_);
lean_ctor_set(v___x_2120_, 0, v___x_2172_);
v___x_2178_ = v___x_2120_;
goto v_reusejp_2177_;
}
else
{
lean_object* v_reuseFailAlloc_2183_; 
v_reuseFailAlloc_2183_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2183_, 0, v___x_2172_);
lean_ctor_set(v_reuseFailAlloc_2183_, 1, v___x_2176_);
v___x_2178_ = v_reuseFailAlloc_2183_;
goto v_reusejp_2177_;
}
v_reusejp_2177_:
{
lean_object* v___x_2179_; lean_object* v___x_2181_; 
v___x_2179_ = l_List_appendTR___redArg(v___x_2178_, v_snd_2164_);
if (v_isShared_2171_ == 0)
{
lean_ctor_set(v___x_2170_, 0, v___x_2179_);
v___x_2181_ = v___x_2170_;
goto v_reusejp_2180_;
}
else
{
lean_object* v_reuseFailAlloc_2182_; 
v_reuseFailAlloc_2182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2182_, 0, v___x_2179_);
v___x_2181_ = v_reuseFailAlloc_2182_;
goto v_reusejp_2180_;
}
v_reusejp_2180_:
{
return v___x_2181_;
}
}
}
}
}
}
else
{
lean_object* v_a_2189_; lean_object* v___x_2191_; uint8_t v_isShared_2192_; uint8_t v_isSharedCheck_2196_; 
lean_dec(v_a_2152_);
lean_dec(v_a_2146_);
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_dec(v_g_2008_);
v_a_2189_ = lean_ctor_get(v___x_2162_, 0);
v_isSharedCheck_2196_ = !lean_is_exclusive(v___x_2162_);
if (v_isSharedCheck_2196_ == 0)
{
v___x_2191_ = v___x_2162_;
v_isShared_2192_ = v_isSharedCheck_2196_;
goto v_resetjp_2190_;
}
else
{
lean_inc(v_a_2189_);
lean_dec(v___x_2162_);
v___x_2191_ = lean_box(0);
v_isShared_2192_ = v_isSharedCheck_2196_;
goto v_resetjp_2190_;
}
v_resetjp_2190_:
{
lean_object* v___x_2194_; 
if (v_isShared_2192_ == 0)
{
v___x_2194_ = v___x_2191_;
goto v_reusejp_2193_;
}
else
{
lean_object* v_reuseFailAlloc_2195_; 
v_reuseFailAlloc_2195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2195_, 0, v_a_2189_);
v___x_2194_ = v_reuseFailAlloc_2195_;
goto v_reusejp_2193_;
}
v_reusejp_2193_:
{
return v___x_2194_;
}
}
}
}
else
{
lean_object* v_a_2197_; lean_object* v___x_2199_; uint8_t v_isShared_2200_; uint8_t v_isSharedCheck_2204_; 
lean_dec(v_a_2152_);
lean_dec(v_a_2146_);
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2197_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2204_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2204_ == 0)
{
v___x_2199_ = v___x_2153_;
v_isShared_2200_ = v_isSharedCheck_2204_;
goto v_resetjp_2198_;
}
else
{
lean_inc(v_a_2197_);
lean_dec(v___x_2153_);
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
lean_object* v_a_2205_; lean_object* v___x_2207_; uint8_t v_isShared_2208_; uint8_t v_isSharedCheck_2212_; 
lean_dec(v_a_2146_);
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2205_ = lean_ctor_get(v___x_2151_, 0);
v_isSharedCheck_2212_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2212_ == 0)
{
v___x_2207_ = v___x_2151_;
v_isShared_2208_ = v_isSharedCheck_2212_;
goto v_resetjp_2206_;
}
else
{
lean_inc(v_a_2205_);
lean_dec(v___x_2151_);
v___x_2207_ = lean_box(0);
v_isShared_2208_ = v_isSharedCheck_2212_;
goto v_resetjp_2206_;
}
v_resetjp_2206_:
{
lean_object* v___x_2210_; 
if (v_isShared_2208_ == 0)
{
v___x_2210_ = v___x_2207_;
goto v_reusejp_2209_;
}
else
{
lean_object* v_reuseFailAlloc_2211_; 
v_reuseFailAlloc_2211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2211_, 0, v_a_2205_);
v___x_2210_ = v_reuseFailAlloc_2211_;
goto v_reusejp_2209_;
}
v_reusejp_2209_:
{
return v___x_2210_;
}
}
}
}
}
else
{
lean_object* v_a_2214_; lean_object* v___x_2216_; uint8_t v_isShared_2217_; uint8_t v_isSharedCheck_2221_; 
lean_dec(v_a_2146_);
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2214_ = lean_ctor_get(v___x_2147_, 0);
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2147_);
if (v_isSharedCheck_2221_ == 0)
{
v___x_2216_ = v___x_2147_;
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
else
{
lean_inc(v_a_2214_);
lean_dec(v___x_2147_);
v___x_2216_ = lean_box(0);
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
v_resetjp_2215_:
{
lean_object* v___x_2219_; 
if (v_isShared_2217_ == 0)
{
v___x_2219_ = v___x_2216_;
goto v_reusejp_2218_;
}
else
{
lean_object* v_reuseFailAlloc_2220_; 
v_reuseFailAlloc_2220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2220_, 0, v_a_2214_);
v___x_2219_ = v_reuseFailAlloc_2220_;
goto v_reusejp_2218_;
}
v_reusejp_2218_:
{
return v___x_2219_;
}
}
}
}
else
{
lean_object* v_a_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2229_; 
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2222_ = lean_ctor_get(v___x_2145_, 0);
v_isSharedCheck_2229_ = !lean_is_exclusive(v___x_2145_);
if (v_isSharedCheck_2229_ == 0)
{
v___x_2224_ = v___x_2145_;
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_a_2222_);
lean_dec(v___x_2145_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v___x_2227_; 
if (v_isShared_2225_ == 0)
{
v___x_2227_ = v___x_2224_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2228_; 
v_reuseFailAlloc_2228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2228_, 0, v_a_2222_);
v___x_2227_ = v_reuseFailAlloc_2228_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
return v___x_2227_;
}
}
}
}
else
{
lean_object* v_a_2230_; lean_object* v___x_2232_; uint8_t v_isShared_2233_; uint8_t v_isSharedCheck_2237_; 
lean_dec(v_a_2140_);
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2230_ = lean_ctor_get(v___x_2141_, 0);
v_isSharedCheck_2237_ = !lean_is_exclusive(v___x_2141_);
if (v_isSharedCheck_2237_ == 0)
{
v___x_2232_ = v___x_2141_;
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
else
{
lean_inc(v_a_2230_);
lean_dec(v___x_2141_);
v___x_2232_ = lean_box(0);
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
v_resetjp_2231_:
{
lean_object* v___x_2235_; 
if (v_isShared_2233_ == 0)
{
v___x_2235_ = v___x_2232_;
goto v_reusejp_2234_;
}
else
{
lean_object* v_reuseFailAlloc_2236_; 
v_reuseFailAlloc_2236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2236_, 0, v_a_2230_);
v___x_2235_ = v_reuseFailAlloc_2236_;
goto v_reusejp_2234_;
}
v_reusejp_2234_:
{
return v___x_2235_;
}
}
}
}
else
{
lean_object* v_a_2238_; lean_object* v___x_2240_; uint8_t v_isShared_2241_; uint8_t v_isSharedCheck_2245_; 
lean_dec(v_a_2138_);
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2238_ = lean_ctor_get(v___x_2139_, 0);
v_isSharedCheck_2245_ = !lean_is_exclusive(v___x_2139_);
if (v_isSharedCheck_2245_ == 0)
{
v___x_2240_ = v___x_2139_;
v_isShared_2241_ = v_isSharedCheck_2245_;
goto v_resetjp_2239_;
}
else
{
lean_inc(v_a_2238_);
lean_dec(v___x_2139_);
v___x_2240_ = lean_box(0);
v_isShared_2241_ = v_isSharedCheck_2245_;
goto v_resetjp_2239_;
}
v_resetjp_2239_:
{
lean_object* v___x_2243_; 
if (v_isShared_2241_ == 0)
{
v___x_2243_ = v___x_2240_;
goto v_reusejp_2242_;
}
else
{
lean_object* v_reuseFailAlloc_2244_; 
v_reuseFailAlloc_2244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2244_, 0, v_a_2238_);
v___x_2243_ = v_reuseFailAlloc_2244_;
goto v_reusejp_2242_;
}
v_reusejp_2242_:
{
return v___x_2243_;
}
}
}
}
else
{
lean_object* v_a_2246_; lean_object* v___x_2248_; uint8_t v_isShared_2249_; uint8_t v_isSharedCheck_2253_; 
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2246_ = lean_ctor_get(v___x_2137_, 0);
v_isSharedCheck_2253_ = !lean_is_exclusive(v___x_2137_);
if (v_isSharedCheck_2253_ == 0)
{
v___x_2248_ = v___x_2137_;
v_isShared_2249_ = v_isSharedCheck_2253_;
goto v_resetjp_2247_;
}
else
{
lean_inc(v_a_2246_);
lean_dec(v___x_2137_);
v___x_2248_ = lean_box(0);
v_isShared_2249_ = v_isSharedCheck_2253_;
goto v_resetjp_2247_;
}
v_resetjp_2247_:
{
lean_object* v___x_2251_; 
if (v_isShared_2249_ == 0)
{
v___x_2251_ = v___x_2248_;
goto v_reusejp_2250_;
}
else
{
lean_object* v_reuseFailAlloc_2252_; 
v_reuseFailAlloc_2252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2252_, 0, v_a_2246_);
v___x_2251_ = v_reuseFailAlloc_2252_;
goto v_reusejp_2250_;
}
v_reusejp_2250_:
{
return v___x_2251_;
}
}
}
}
else
{
lean_object* v_a_2254_; lean_object* v___x_2256_; uint8_t v_isShared_2257_; uint8_t v_isSharedCheck_2261_; 
lean_dec(v_a_2134_);
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2254_ = lean_ctor_get(v___x_2135_, 0);
v_isSharedCheck_2261_ = !lean_is_exclusive(v___x_2135_);
if (v_isSharedCheck_2261_ == 0)
{
v___x_2256_ = v___x_2135_;
v_isShared_2257_ = v_isSharedCheck_2261_;
goto v_resetjp_2255_;
}
else
{
lean_inc(v_a_2254_);
lean_dec(v___x_2135_);
v___x_2256_ = lean_box(0);
v_isShared_2257_ = v_isSharedCheck_2261_;
goto v_resetjp_2255_;
}
v_resetjp_2255_:
{
lean_object* v___x_2259_; 
if (v_isShared_2257_ == 0)
{
v___x_2259_ = v___x_2256_;
goto v_reusejp_2258_;
}
else
{
lean_object* v_reuseFailAlloc_2260_; 
v_reuseFailAlloc_2260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2260_, 0, v_a_2254_);
v___x_2259_ = v_reuseFailAlloc_2260_;
goto v_reusejp_2258_;
}
v_reusejp_2258_:
{
return v___x_2259_;
}
}
}
}
else
{
lean_object* v_a_2262_; lean_object* v___x_2264_; uint8_t v_isShared_2265_; uint8_t v_isSharedCheck_2269_; 
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2262_ = lean_ctor_get(v___x_2133_, 0);
v_isSharedCheck_2269_ = !lean_is_exclusive(v___x_2133_);
if (v_isSharedCheck_2269_ == 0)
{
v___x_2264_ = v___x_2133_;
v_isShared_2265_ = v_isSharedCheck_2269_;
goto v_resetjp_2263_;
}
else
{
lean_inc(v_a_2262_);
lean_dec(v___x_2133_);
v___x_2264_ = lean_box(0);
v_isShared_2265_ = v_isSharedCheck_2269_;
goto v_resetjp_2263_;
}
v_resetjp_2263_:
{
lean_object* v___x_2267_; 
if (v_isShared_2265_ == 0)
{
v___x_2267_ = v___x_2264_;
goto v_reusejp_2266_;
}
else
{
lean_object* v_reuseFailAlloc_2268_; 
v_reuseFailAlloc_2268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2268_, 0, v_a_2262_);
v___x_2267_ = v_reuseFailAlloc_2268_;
goto v_reusejp_2266_;
}
v_reusejp_2266_:
{
return v___x_2267_;
}
}
}
}
}
}
else
{
lean_object* v___x_2270_; lean_object* v___x_2271_; uint8_t v___x_2272_; 
lean_dec_ref(v_str_2122_);
lean_del_object(v___x_2120_);
lean_del_object(v___x_2111_);
lean_dec(v_using_x3f_2007_);
v___x_2270_ = lean_array_get_size(v_snd_2118_);
v___x_2271_ = lean_unsigned_to_nat(1u);
v___x_2272_ = lean_nat_dec_eq(v___x_2270_, v___x_2271_);
if (v___x_2272_ == 0)
{
lean_object* v___x_2273_; 
lean_dec(v_snd_2118_);
lean_dec(v_g_2008_);
v___x_2273_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2273_;
}
else
{
lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v_fst_2277_; 
v___x_2274_ = lean_unsigned_to_nat(0u);
v___x_2275_ = lean_array_fget(v_snd_2118_, v___x_2274_);
lean_dec(v_snd_2118_);
v___x_2276_ = l_Lean_Expr_getAppFnArgs(v___x_2275_);
v_fst_2277_ = lean_ctor_get(v___x_2276_, 0);
lean_inc(v_fst_2277_);
if (lean_obj_tag(v_fst_2277_) == 1)
{
lean_object* v_pre_2278_; 
v_pre_2278_ = lean_ctor_get(v_fst_2277_, 0);
if (lean_obj_tag(v_pre_2278_) == 0)
{
lean_object* v_snd_2279_; lean_object* v_str_2280_; lean_object* v___x_2281_; uint8_t v___x_2282_; 
v_snd_2279_ = lean_ctor_get(v___x_2276_, 1);
lean_inc(v_snd_2279_);
lean_dec_ref(v___x_2276_);
v_str_2280_ = lean_ctor_get(v_fst_2277_, 1);
lean_inc_ref(v_str_2280_);
lean_dec_ref_known(v_fst_2277_, 2);
v___x_2281_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__8));
v___x_2282_ = lean_string_dec_eq(v_str_2280_, v___x_2281_);
lean_dec_ref(v_str_2280_);
if (v___x_2282_ == 0)
{
lean_object* v___x_2283_; 
lean_dec(v_snd_2279_);
lean_dec(v_g_2008_);
v___x_2283_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2283_;
}
else
{
lean_object* v___x_2284_; lean_object* v___x_2285_; uint8_t v___x_2286_; 
v___x_2284_ = lean_array_get_size(v_snd_2279_);
lean_dec(v_snd_2279_);
v___x_2285_ = lean_unsigned_to_nat(3u);
v___x_2286_ = lean_nat_dec_eq(v___x_2284_, v___x_2285_);
if (v___x_2286_ == 0)
{
lean_object* v___x_2287_; 
lean_dec(v_g_2008_);
v___x_2287_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2287_;
}
else
{
lean_object* v___x_2288_; 
v___x_2288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__7));
v_thm_2019_ = v___x_2288_;
v___y_2020_ = v_a_2009_;
v___y_2021_ = v_a_2010_;
v___y_2022_ = v_a_2011_;
v___y_2023_ = v_a_2012_;
v___y_2024_ = v_a_2013_;
v___y_2025_ = v_a_2014_;
v___y_2026_ = v_a_2015_;
v___y_2027_ = v_a_2016_;
goto v___jp_2018_;
}
}
}
else
{
lean_object* v___x_2289_; 
lean_dec_ref_known(v_fst_2277_, 2);
lean_dec_ref(v___x_2276_);
lean_dec(v_g_2008_);
v___x_2289_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2289_;
}
}
else
{
lean_object* v___x_2290_; 
lean_dec(v_fst_2277_);
lean_dec_ref(v___x_2276_);
lean_dec(v_g_2008_);
v___x_2290_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2290_;
}
}
}
}
}
case 1:
{
lean_object* v_pre_2293_; 
lean_dec_ref(v___x_2115_);
lean_del_object(v___x_2111_);
lean_dec(v_using_x3f_2007_);
v_pre_2293_ = lean_ctor_get(v_pre_2117_, 0);
if (lean_obj_tag(v_pre_2293_) == 0)
{
lean_object* v_str_2294_; lean_object* v_str_2295_; lean_object* v___x_2296_; uint8_t v___x_2297_; 
v_str_2294_ = lean_ctor_get(v_fst_2116_, 1);
lean_inc_ref(v_str_2294_);
lean_dec_ref_known(v_fst_2116_, 2);
v_str_2295_ = lean_ctor_get(v_pre_2117_, 1);
lean_inc_ref(v_str_2295_);
lean_dec_ref_known(v_pre_2117_, 2);
v___x_2296_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__14));
v___x_2297_ = lean_string_dec_eq(v_str_2295_, v___x_2296_);
if (v___x_2297_ == 0)
{
lean_object* v___x_2298_; uint8_t v___x_2299_; 
v___x_2298_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__8));
v___x_2299_ = lean_string_dec_eq(v_str_2295_, v___x_2298_);
if (v___x_2299_ == 0)
{
lean_object* v___x_2300_; uint8_t v___x_2301_; 
v___x_2300_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__13));
v___x_2301_ = lean_string_dec_eq(v_str_2295_, v___x_2300_);
if (v___x_2301_ == 0)
{
lean_object* v___x_2302_; uint8_t v___x_2303_; 
v___x_2302_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__9));
v___x_2303_ = lean_string_dec_eq(v_str_2295_, v___x_2302_);
lean_dec_ref(v_str_2295_);
if (v___x_2303_ == 0)
{
lean_object* v___x_2304_; 
lean_dec_ref(v_str_2294_);
lean_dec(v_g_2008_);
v___x_2304_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2304_;
}
else
{
lean_object* v___x_2305_; uint8_t v___x_2306_; 
v___x_2305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__10));
v___x_2306_ = lean_string_dec_eq(v_str_2294_, v___x_2305_);
lean_dec_ref(v_str_2294_);
if (v___x_2306_ == 0)
{
lean_object* v___x_2307_; 
lean_dec(v_g_2008_);
v___x_2307_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2307_;
}
else
{
v___y_2097_ = v_a_2009_;
v___y_2098_ = v_a_2010_;
v___y_2099_ = v_a_2011_;
v___y_2100_ = v_a_2012_;
v___y_2101_ = v_a_2013_;
v___y_2102_ = v_a_2014_;
v___y_2103_ = v_a_2015_;
v___y_2104_ = v_a_2016_;
goto v___jp_2096_;
}
}
}
else
{
lean_object* v___x_2308_; uint8_t v___x_2309_; 
lean_dec_ref(v_str_2295_);
v___x_2308_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__18));
v___x_2309_ = lean_string_dec_eq(v_str_2294_, v___x_2308_);
lean_dec_ref(v_str_2294_);
if (v___x_2309_ == 0)
{
lean_object* v___x_2310_; 
lean_dec(v_g_2008_);
v___x_2310_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2310_;
}
else
{
v___y_2097_ = v_a_2009_;
v___y_2098_ = v_a_2010_;
v___y_2099_ = v_a_2011_;
v___y_2100_ = v_a_2012_;
v___y_2101_ = v_a_2013_;
v___y_2102_ = v_a_2014_;
v___y_2103_ = v_a_2015_;
v___y_2104_ = v_a_2016_;
goto v___jp_2096_;
}
}
}
else
{
lean_object* v___x_2311_; uint8_t v___x_2312_; 
lean_dec_ref(v_str_2295_);
v___x_2311_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__11));
v___x_2312_ = lean_string_dec_eq(v_str_2294_, v___x_2311_);
lean_dec_ref(v_str_2294_);
if (v___x_2312_ == 0)
{
lean_object* v___x_2313_; 
lean_dec(v_g_2008_);
v___x_2313_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2313_;
}
else
{
v___y_2087_ = v_a_2009_;
v___y_2088_ = v_a_2010_;
v___y_2089_ = v_a_2011_;
v___y_2090_ = v_a_2012_;
v___y_2091_ = v_a_2013_;
v___y_2092_ = v_a_2014_;
v___y_2093_ = v_a_2015_;
v___y_2094_ = v_a_2016_;
goto v___jp_2086_;
}
}
}
else
{
lean_object* v___x_2314_; uint8_t v___x_2315_; 
lean_dec_ref(v_str_2295_);
v___x_2314_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunHyp___closed__15));
v___x_2315_ = lean_string_dec_eq(v_str_2294_, v___x_2314_);
lean_dec_ref(v_str_2294_);
if (v___x_2315_ == 0)
{
lean_object* v___x_2316_; 
lean_dec(v_g_2008_);
v___x_2316_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2316_;
}
else
{
v___y_2087_ = v_a_2009_;
v___y_2088_ = v_a_2010_;
v___y_2089_ = v_a_2011_;
v___y_2090_ = v_a_2012_;
v___y_2091_ = v_a_2013_;
v___y_2092_ = v_a_2014_;
v___y_2093_ = v_a_2015_;
v___y_2094_ = v_a_2016_;
goto v___jp_2086_;
}
}
}
else
{
lean_object* v___x_2317_; 
lean_dec_ref_known(v_pre_2117_, 2);
lean_dec_ref_known(v_fst_2116_, 2);
lean_dec(v_g_2008_);
v___x_2317_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2317_;
}
}
default: 
{
lean_object* v___x_2318_; 
lean_dec_ref_known(v_fst_2116_, 2);
lean_dec(v_pre_2117_);
lean_dec_ref(v___x_2115_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
v___x_2318_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2318_;
}
}
}
else
{
lean_object* v___x_2319_; 
lean_dec(v_fst_2116_);
lean_dec_ref(v___x_2115_);
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
v___x_2319_ = lp_mathlib_Mathlib_Tactic_applyFunTargetFailure(v_f_2006_, v_a_2013_, v_a_2014_, v_a_2015_, v_a_2016_);
return v___x_2319_;
}
}
else
{
lean_object* v_a_2320_; lean_object* v___x_2322_; uint8_t v_isShared_2323_; uint8_t v_isSharedCheck_2327_; 
lean_del_object(v___x_2111_);
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2320_ = lean_ctor_get(v___x_2113_, 0);
v_isSharedCheck_2327_ = !lean_is_exclusive(v___x_2113_);
if (v_isSharedCheck_2327_ == 0)
{
v___x_2322_ = v___x_2113_;
v_isShared_2323_ = v_isSharedCheck_2327_;
goto v_resetjp_2321_;
}
else
{
lean_inc(v_a_2320_);
lean_dec(v___x_2113_);
v___x_2322_ = lean_box(0);
v_isShared_2323_ = v_isSharedCheck_2327_;
goto v_resetjp_2321_;
}
v_resetjp_2321_:
{
lean_object* v___x_2325_; 
if (v_isShared_2323_ == 0)
{
v___x_2325_ = v___x_2322_;
goto v_reusejp_2324_;
}
else
{
lean_object* v_reuseFailAlloc_2326_; 
v_reuseFailAlloc_2326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2326_, 0, v_a_2320_);
v___x_2325_ = v_reuseFailAlloc_2326_;
goto v_reusejp_2324_;
}
v_reusejp_2324_:
{
return v___x_2325_;
}
}
}
}
}
else
{
lean_object* v_a_2329_; lean_object* v___x_2331_; uint8_t v_isShared_2332_; uint8_t v_isSharedCheck_2336_; 
lean_dec(v_g_2008_);
lean_dec(v_using_x3f_2007_);
lean_dec(v_f_2006_);
v_a_2329_ = lean_ctor_get(v___x_2106_, 0);
v_isSharedCheck_2336_ = !lean_is_exclusive(v___x_2106_);
if (v_isSharedCheck_2336_ == 0)
{
v___x_2331_ = v___x_2106_;
v_isShared_2332_ = v_isSharedCheck_2336_;
goto v_resetjp_2330_;
}
else
{
lean_inc(v_a_2329_);
lean_dec(v___x_2106_);
v___x_2331_ = lean_box(0);
v_isShared_2332_ = v_isSharedCheck_2336_;
goto v_resetjp_2330_;
}
v_resetjp_2330_:
{
lean_object* v___x_2334_; 
if (v_isShared_2332_ == 0)
{
v___x_2334_ = v___x_2331_;
goto v_reusejp_2333_;
}
else
{
lean_object* v_reuseFailAlloc_2335_; 
v_reuseFailAlloc_2335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2335_, 0, v_a_2329_);
v___x_2334_ = v_reuseFailAlloc_2335_;
goto v_reusejp_2333_;
}
v_reusejp_2333_:
{
return v___x_2334_;
}
}
}
v___jp_2018_:
{
lean_object* v___x_2028_; uint8_t v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; 
v___x_2028_ = lean_box(0);
v___x_2029_ = 0;
v___x_2030_ = lean_box(0);
v___x_2031_ = l_Lean_Meta_mkFreshExprMVar(v___x_2028_, v___x_2029_, v___x_2030_, v___y_2024_, v___y_2025_, v___y_2026_, v___y_2027_);
if (lean_obj_tag(v___x_2031_) == 0)
{
lean_object* v_a_2032_; lean_object* v___x_2033_; 
v_a_2032_ = lean_ctor_get(v___x_2031_, 0);
lean_inc(v_a_2032_);
lean_dec_ref_known(v___x_2031_, 1);
lean_inc(v_g_2008_);
v___x_2033_ = l_Lean_MVarId_getTag(v_g_2008_, v___y_2024_, v___y_2025_, v___y_2026_, v___y_2027_);
if (lean_obj_tag(v___x_2033_) == 0)
{
lean_object* v_a_2034_; lean_object* v___f_2035_; uint8_t v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; 
v_a_2034_ = lean_ctor_get(v___x_2033_, 0);
lean_inc(v_a_2034_);
lean_dec_ref_known(v___x_2033_, 1);
lean_inc(v_thm_2019_);
lean_inc(v_g_2008_);
lean_inc(v_a_2032_);
v___f_2035_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___lam__0___boxed), 11, 4);
lean_closure_set(v___f_2035_, 0, v_a_2032_);
lean_closure_set(v___f_2035_, 1, v_g_2008_);
lean_closure_set(v___f_2035_, 2, v_f_2006_);
lean_closure_set(v___f_2035_, 3, v_thm_2019_);
v___x_2036_ = 0;
v___x_2037_ = lean_box(v___x_2036_);
v___x_2038_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_runTermElab___boxed), 12, 3);
lean_closure_set(v___x_2038_, 0, lean_box(0));
lean_closure_set(v___x_2038_, 1, v___f_2035_);
lean_closure_set(v___x_2038_, 2, v___x_2037_);
v___x_2039_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_2039_, 0, lean_box(0));
lean_closure_set(v___x_2039_, 1, v___x_2038_);
v___x_2040_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_));
v___x_2041_ = l_Lean_Elab_Tactic_withCollectingNewGoalsFrom(v___x_2039_, v_a_2034_, v___x_2040_, v___x_2036_, v___y_2020_, v___y_2021_, v___y_2022_, v___y_2023_, v___y_2024_, v___y_2025_, v___y_2026_, v___y_2027_);
if (lean_obj_tag(v___x_2041_) == 0)
{
lean_object* v_a_2042_; lean_object* v_fst_2043_; lean_object* v_snd_2044_; lean_object* v___x_2046_; uint8_t v_isShared_2047_; uint8_t v_isSharedCheck_2061_; 
v_a_2042_ = lean_ctor_get(v___x_2041_, 0);
lean_inc(v_a_2042_);
lean_dec_ref_known(v___x_2041_, 1);
v_fst_2043_ = lean_ctor_get(v_a_2042_, 0);
v_snd_2044_ = lean_ctor_get(v_a_2042_, 1);
v_isSharedCheck_2061_ = !lean_is_exclusive(v_a_2042_);
if (v_isSharedCheck_2061_ == 0)
{
v___x_2046_ = v_a_2042_;
v_isShared_2047_ = v_isSharedCheck_2061_;
goto v_resetjp_2045_;
}
else
{
lean_inc(v_snd_2044_);
lean_inc(v_fst_2043_);
lean_dec(v_a_2042_);
v___x_2046_ = lean_box(0);
v_isShared_2047_ = v_isSharedCheck_2061_;
goto v_resetjp_2045_;
}
v_resetjp_2045_:
{
lean_object* v___x_2048_; lean_object* v___x_2050_; uint8_t v_isShared_2051_; uint8_t v_isSharedCheck_2059_; 
v___x_2048_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg(v_g_2008_, v_fst_2043_, v___y_2025_);
v_isSharedCheck_2059_ = !lean_is_exclusive(v___x_2048_);
if (v_isSharedCheck_2059_ == 0)
{
lean_object* v_unused_2060_; 
v_unused_2060_ = lean_ctor_get(v___x_2048_, 0);
lean_dec(v_unused_2060_);
v___x_2050_ = v___x_2048_;
v_isShared_2051_ = v_isSharedCheck_2059_;
goto v_resetjp_2049_;
}
else
{
lean_dec(v___x_2048_);
v___x_2050_ = lean_box(0);
v_isShared_2051_ = v_isSharedCheck_2059_;
goto v_resetjp_2049_;
}
v_resetjp_2049_:
{
lean_object* v___x_2052_; lean_object* v___x_2054_; 
v___x_2052_ = l_Lean_Expr_mvarId_x21(v_a_2032_);
lean_dec(v_a_2032_);
if (v_isShared_2047_ == 0)
{
lean_ctor_set_tag(v___x_2046_, 1);
lean_ctor_set(v___x_2046_, 0, v___x_2052_);
v___x_2054_ = v___x_2046_;
goto v_reusejp_2053_;
}
else
{
lean_object* v_reuseFailAlloc_2058_; 
v_reuseFailAlloc_2058_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2058_, 0, v___x_2052_);
lean_ctor_set(v_reuseFailAlloc_2058_, 1, v_snd_2044_);
v___x_2054_ = v_reuseFailAlloc_2058_;
goto v_reusejp_2053_;
}
v_reusejp_2053_:
{
lean_object* v___x_2056_; 
if (v_isShared_2051_ == 0)
{
lean_ctor_set(v___x_2050_, 0, v___x_2054_);
v___x_2056_ = v___x_2050_;
goto v_reusejp_2055_;
}
else
{
lean_object* v_reuseFailAlloc_2057_; 
v_reuseFailAlloc_2057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2057_, 0, v___x_2054_);
v___x_2056_ = v_reuseFailAlloc_2057_;
goto v_reusejp_2055_;
}
v_reusejp_2055_:
{
return v___x_2056_;
}
}
}
}
}
else
{
lean_object* v_a_2062_; lean_object* v___x_2064_; uint8_t v_isShared_2065_; uint8_t v_isSharedCheck_2069_; 
lean_dec(v_a_2032_);
lean_dec(v_g_2008_);
v_a_2062_ = lean_ctor_get(v___x_2041_, 0);
v_isSharedCheck_2069_ = !lean_is_exclusive(v___x_2041_);
if (v_isSharedCheck_2069_ == 0)
{
v___x_2064_ = v___x_2041_;
v_isShared_2065_ = v_isSharedCheck_2069_;
goto v_resetjp_2063_;
}
else
{
lean_inc(v_a_2062_);
lean_dec(v___x_2041_);
v___x_2064_ = lean_box(0);
v_isShared_2065_ = v_isSharedCheck_2069_;
goto v_resetjp_2063_;
}
v_resetjp_2063_:
{
lean_object* v___x_2067_; 
if (v_isShared_2065_ == 0)
{
v___x_2067_ = v___x_2064_;
goto v_reusejp_2066_;
}
else
{
lean_object* v_reuseFailAlloc_2068_; 
v_reuseFailAlloc_2068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2068_, 0, v_a_2062_);
v___x_2067_ = v_reuseFailAlloc_2068_;
goto v_reusejp_2066_;
}
v_reusejp_2066_:
{
return v___x_2067_;
}
}
}
}
else
{
lean_object* v_a_2070_; lean_object* v___x_2072_; uint8_t v_isShared_2073_; uint8_t v_isSharedCheck_2077_; 
lean_dec(v_a_2032_);
lean_dec(v_g_2008_);
lean_dec(v_f_2006_);
v_a_2070_ = lean_ctor_get(v___x_2033_, 0);
v_isSharedCheck_2077_ = !lean_is_exclusive(v___x_2033_);
if (v_isSharedCheck_2077_ == 0)
{
v___x_2072_ = v___x_2033_;
v_isShared_2073_ = v_isSharedCheck_2077_;
goto v_resetjp_2071_;
}
else
{
lean_inc(v_a_2070_);
lean_dec(v___x_2033_);
v___x_2072_ = lean_box(0);
v_isShared_2073_ = v_isSharedCheck_2077_;
goto v_resetjp_2071_;
}
v_resetjp_2071_:
{
lean_object* v___x_2075_; 
if (v_isShared_2073_ == 0)
{
v___x_2075_ = v___x_2072_;
goto v_reusejp_2074_;
}
else
{
lean_object* v_reuseFailAlloc_2076_; 
v_reuseFailAlloc_2076_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2076_, 0, v_a_2070_);
v___x_2075_ = v_reuseFailAlloc_2076_;
goto v_reusejp_2074_;
}
v_reusejp_2074_:
{
return v___x_2075_;
}
}
}
}
else
{
lean_object* v_a_2078_; lean_object* v___x_2080_; uint8_t v_isShared_2081_; uint8_t v_isSharedCheck_2085_; 
lean_dec(v_g_2008_);
lean_dec(v_f_2006_);
v_a_2078_ = lean_ctor_get(v___x_2031_, 0);
v_isSharedCheck_2085_ = !lean_is_exclusive(v___x_2031_);
if (v_isSharedCheck_2085_ == 0)
{
v___x_2080_ = v___x_2031_;
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
else
{
lean_inc(v_a_2078_);
lean_dec(v___x_2031_);
v___x_2080_ = lean_box(0);
v_isShared_2081_ = v_isSharedCheck_2085_;
goto v_resetjp_2079_;
}
v_resetjp_2079_:
{
lean_object* v___x_2083_; 
if (v_isShared_2081_ == 0)
{
v___x_2083_ = v___x_2080_;
goto v_reusejp_2082_;
}
else
{
lean_object* v_reuseFailAlloc_2084_; 
v_reuseFailAlloc_2084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2084_, 0, v_a_2078_);
v___x_2083_ = v_reuseFailAlloc_2084_;
goto v_reusejp_2082_;
}
v_reusejp_2082_:
{
return v___x_2083_;
}
}
}
}
v___jp_2086_:
{
lean_object* v___x_2095_; 
v___x_2095_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__1));
v_thm_2019_ = v___x_2095_;
v___y_2020_ = v___y_2087_;
v___y_2021_ = v___y_2088_;
v___y_2022_ = v___y_2089_;
v___y_2023_ = v___y_2090_;
v___y_2024_ = v___y_2091_;
v___y_2025_ = v___y_2092_;
v___y_2026_ = v___y_2093_;
v___y_2027_ = v___y_2094_;
goto v___jp_2018_;
}
v___jp_2096_:
{
lean_object* v___x_2105_; 
v___x_2105_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFunTarget___closed__3));
v_thm_2019_ = v___x_2105_;
v___y_2020_ = v___y_2097_;
v___y_2021_ = v___y_2098_;
v___y_2022_ = v___y_2099_;
v___y_2023_ = v___y_2100_;
v___y_2024_ = v___y_2101_;
v___y_2025_ = v___y_2102_;
v___y_2026_ = v___y_2103_;
v___y_2027_ = v___y_2104_;
goto v___jp_2018_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_applyFunTarget___boxed(lean_object* v_f_2337_, lean_object* v_using_x3f_2338_, lean_object* v_g_2339_, lean_object* v_a_2340_, lean_object* v_a_2341_, lean_object* v_a_2342_, lean_object* v_a_2343_, lean_object* v_a_2344_, lean_object* v_a_2345_, lean_object* v_a_2346_, lean_object* v_a_2347_, lean_object* v_a_2348_){
_start:
{
lean_object* v_res_2349_; 
v_res_2349_ = lp_mathlib_Mathlib_Tactic_applyFunTarget(v_f_2337_, v_using_x3f_2338_, v_g_2339_, v_a_2340_, v_a_2341_, v_a_2342_, v_a_2343_, v_a_2344_, v_a_2345_, v_a_2346_, v_a_2347_);
lean_dec(v_a_2347_);
lean_dec_ref(v_a_2346_);
lean_dec(v_a_2345_);
lean_dec_ref(v_a_2344_);
lean_dec(v_a_2343_);
lean_dec_ref(v_a_2342_);
lean_dec(v_a_2341_);
lean_dec_ref(v_a_2340_);
return v_res_2349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0(lean_object* v_mvarId_2350_, lean_object* v_val_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_){
_start:
{
lean_object* v___x_2361_; 
v___x_2361_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___redArg(v_mvarId_2350_, v_val_2351_, v___y_2357_);
return v___x_2361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0___boxed(lean_object* v_mvarId_2362_, lean_object* v_val_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_, lean_object* v___y_2371_, lean_object* v___y_2372_){
_start:
{
lean_object* v_res_2373_; 
v_res_2373_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__0(v_mvarId_2362_, v_val_2363_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_, v___y_2370_, v___y_2371_);
lean_dec(v___y_2371_);
lean_dec_ref(v___y_2370_);
lean_dec(v___y_2369_);
lean_dec_ref(v___y_2368_);
lean_dec(v___y_2367_);
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
return v_res_2373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1(lean_object* v_mvarId_2374_, lean_object* v_val_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_){
_start:
{
lean_object* v___x_2383_; 
v___x_2383_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___redArg(v_mvarId_2374_, v_val_2375_, v___y_2379_);
return v___x_2383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1___boxed(lean_object* v_mvarId_2384_, lean_object* v_val_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_){
_start:
{
lean_object* v_res_2393_; 
v_res_2393_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_applyFunTarget_spec__1(v_mvarId_2384_, v_val_2385_, v___y_2386_, v___y_2387_, v___y_2388_, v___y_2389_, v___y_2390_, v___y_2391_);
lean_dec(v___y_2391_);
lean_dec_ref(v___y_2390_);
lean_dec(v___y_2389_);
lean_dec_ref(v___y_2388_);
lean_dec(v___y_2387_);
lean_dec_ref(v___y_2386_);
return v_res_2393_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__12(void){
_start:
{
lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; 
v___x_2419_ = l_Lean_Parser_Tactic_location;
v___x_2420_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__11));
v___x_2421_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2421_, 0, v___x_2420_);
lean_ctor_set(v___x_2421_, 1, v___x_2419_);
return v___x_2421_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__13(void){
_start:
{
lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; 
v___x_2422_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFun___closed__12, &lp_mathlib_Mathlib_Tactic_applyFun___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__12);
v___x_2423_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__9));
v___x_2424_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__3));
v___x_2425_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2425_, 0, v___x_2424_);
lean_ctor_set(v___x_2425_, 1, v___x_2423_);
lean_ctor_set(v___x_2425_, 2, v___x_2422_);
return v___x_2425_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__18(void){
_start:
{
lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; 
v___x_2436_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__17));
v___x_2437_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFun___closed__13, &lp_mathlib_Mathlib_Tactic_applyFun___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__13);
v___x_2438_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__3));
v___x_2439_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2439_, 0, v___x_2438_);
lean_ctor_set(v___x_2439_, 1, v___x_2437_);
lean_ctor_set(v___x_2439_, 2, v___x_2436_);
return v___x_2439_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__19(void){
_start:
{
lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; 
v___x_2440_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFun___closed__18, &lp_mathlib_Mathlib_Tactic_applyFun___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__18);
v___x_2441_ = lean_unsigned_to_nat(1022u);
v___x_2442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__1));
v___x_2443_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_2443_, 0, v___x_2442_);
lean_ctor_set(v___x_2443_, 1, v___x_2441_);
lean_ctor_set(v___x_2443_, 2, v___x_2440_);
return v___x_2443_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_applyFun(void){
_start:
{
lean_object* v___x_2444_; 
v___x_2444_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_applyFun___closed__19, &lp_mathlib_Mathlib_Tactic_applyFun___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_applyFun___closed__19);
return v___x_2444_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; 
v___x_2445_ = lean_box(0);
v___x_2446_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2447_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2447_, 0, v___x_2446_);
lean_ctor_set(v___x_2447_, 1, v___x_2445_);
return v___x_2447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2449_; lean_object* v___x_2450_; 
v___x_2449_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___closed__0);
v___x_2450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2450_, 0, v___x_2449_);
return v___x_2450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg___boxed(lean_object* v___y_2451_){
_start:
{
lean_object* v_res_2452_; 
v_res_2452_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg();
return v_res_2452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0(lean_object* v_00_u03b1_2453_, lean_object* v___y_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_){
_start:
{
lean_object* v___x_2463_; 
v___x_2463_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg();
return v___x_2463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___boxed(lean_object* v_00_u03b1_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_, lean_object* v___y_2467_, lean_object* v___y_2468_, lean_object* v___y_2469_, lean_object* v___y_2470_, lean_object* v___y_2471_, lean_object* v___y_2472_, lean_object* v___y_2473_){
_start:
{
lean_object* v_res_2474_; 
v_res_2474_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0(v_00_u03b1_2464_, v___y_2465_, v___y_2466_, v___y_2467_, v___y_2468_, v___y_2469_, v___y_2470_, v___y_2471_, v___y_2472_);
lean_dec(v___y_2472_);
lean_dec_ref(v___y_2471_);
lean_dec(v___y_2470_);
lean_dec_ref(v___y_2469_);
lean_dec(v___y_2468_);
lean_dec_ref(v___y_2467_);
lean_dec(v___y_2466_);
lean_dec_ref(v___y_2465_);
return v_res_2474_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2476_; lean_object* v___x_2477_; 
v___x_2476_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__0));
v___x_2477_ = l_Lean_stringToMessageData(v___x_2476_);
return v___x_2477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0(lean_object* v_x_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_, lean_object* v___y_2483_, lean_object* v___y_2484_, lean_object* v___y_2485_, lean_object* v___y_2486_){
_start:
{
lean_object* v___x_2488_; lean_object* v___x_2489_; 
v___x_2488_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___closed__1);
v___x_2489_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_applyFunHyp_spec__0___redArg(v___x_2488_, v___y_2483_, v___y_2484_, v___y_2485_, v___y_2486_);
return v___x_2489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0___boxed(lean_object* v_x_2490_, lean_object* v___y_2491_, lean_object* v___y_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_, lean_object* v___y_2497_, lean_object* v___y_2498_, lean_object* v___y_2499_){
_start:
{
lean_object* v_res_2500_; 
v_res_2500_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__0(v_x_2490_, v___y_2491_, v___y_2492_, v___y_2493_, v___y_2494_, v___y_2495_, v___y_2496_, v___y_2497_, v___y_2498_);
lean_dec(v___y_2498_);
lean_dec_ref(v___y_2497_);
lean_dec(v___y_2496_);
lean_dec_ref(v___y_2495_);
lean_dec(v___y_2494_);
lean_dec_ref(v___y_2493_);
lean_dec(v___y_2492_);
lean_dec_ref(v___y_2491_);
lean_dec(v_x_2490_);
return v_res_2500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__1(lean_object* v_f_2501_, lean_object* v_P_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_, lean_object* v___y_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_){
_start:
{
lean_object* v___x_2512_; 
v___x_2512_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2504_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_);
if (lean_obj_tag(v___x_2512_) == 0)
{
lean_object* v_a_2513_; lean_object* v___x_2514_; 
v_a_2513_ = lean_ctor_get(v___x_2512_, 0);
lean_inc(v_a_2513_);
lean_dec_ref_known(v___x_2512_, 1);
v___x_2514_ = lp_mathlib_Mathlib_Tactic_applyFunTarget(v_f_2501_, v_P_2502_, v_a_2513_, v___y_2503_, v___y_2504_, v___y_2505_, v___y_2506_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_);
if (lean_obj_tag(v___x_2514_) == 0)
{
lean_object* v_a_2515_; lean_object* v___x_2516_; 
v_a_2515_ = lean_ctor_get(v___x_2514_, 0);
lean_inc(v_a_2515_);
lean_dec_ref_known(v___x_2514_, 1);
v___x_2516_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_2515_, v___y_2504_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_);
return v___x_2516_;
}
else
{
lean_object* v_a_2517_; lean_object* v___x_2519_; uint8_t v_isShared_2520_; uint8_t v_isSharedCheck_2524_; 
v_a_2517_ = lean_ctor_get(v___x_2514_, 0);
v_isSharedCheck_2524_ = !lean_is_exclusive(v___x_2514_);
if (v_isSharedCheck_2524_ == 0)
{
v___x_2519_ = v___x_2514_;
v_isShared_2520_ = v_isSharedCheck_2524_;
goto v_resetjp_2518_;
}
else
{
lean_inc(v_a_2517_);
lean_dec(v___x_2514_);
v___x_2519_ = lean_box(0);
v_isShared_2520_ = v_isSharedCheck_2524_;
goto v_resetjp_2518_;
}
v_resetjp_2518_:
{
lean_object* v___x_2522_; 
if (v_isShared_2520_ == 0)
{
v___x_2522_ = v___x_2519_;
goto v_reusejp_2521_;
}
else
{
lean_object* v_reuseFailAlloc_2523_; 
v_reuseFailAlloc_2523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2523_, 0, v_a_2517_);
v___x_2522_ = v_reuseFailAlloc_2523_;
goto v_reusejp_2521_;
}
v_reusejp_2521_:
{
return v___x_2522_;
}
}
}
}
else
{
lean_object* v_a_2525_; lean_object* v___x_2527_; uint8_t v_isShared_2528_; uint8_t v_isSharedCheck_2532_; 
lean_dec(v_P_2502_);
lean_dec(v_f_2501_);
v_a_2525_ = lean_ctor_get(v___x_2512_, 0);
v_isSharedCheck_2532_ = !lean_is_exclusive(v___x_2512_);
if (v_isSharedCheck_2532_ == 0)
{
v___x_2527_ = v___x_2512_;
v_isShared_2528_ = v_isSharedCheck_2532_;
goto v_resetjp_2526_;
}
else
{
lean_inc(v_a_2525_);
lean_dec(v___x_2512_);
v___x_2527_ = lean_box(0);
v_isShared_2528_ = v_isSharedCheck_2532_;
goto v_resetjp_2526_;
}
v_resetjp_2526_:
{
lean_object* v___x_2530_; 
if (v_isShared_2528_ == 0)
{
v___x_2530_ = v___x_2527_;
goto v_reusejp_2529_;
}
else
{
lean_object* v_reuseFailAlloc_2531_; 
v_reuseFailAlloc_2531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2531_, 0, v_a_2525_);
v___x_2530_ = v_reuseFailAlloc_2531_;
goto v_reusejp_2529_;
}
v_reusejp_2529_:
{
return v___x_2530_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__1___boxed(lean_object* v_f_2533_, lean_object* v_P_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_){
_start:
{
lean_object* v_res_2544_; 
v_res_2544_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__1(v_f_2533_, v_P_2534_, v___y_2535_, v___y_2536_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_, v___y_2542_);
lean_dec(v___y_2542_);
lean_dec_ref(v___y_2541_);
lean_dec(v___y_2540_);
lean_dec_ref(v___y_2539_);
lean_dec(v___y_2538_);
lean_dec_ref(v___y_2537_);
lean_dec(v___y_2536_);
lean_dec_ref(v___y_2535_);
return v_res_2544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__2(lean_object* v_f_2545_, lean_object* v_P_2546_, lean_object* v_h_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_, lean_object* v___y_2550_, lean_object* v___y_2551_, lean_object* v___y_2552_, lean_object* v___y_2553_, lean_object* v___y_2554_, lean_object* v___y_2555_){
_start:
{
lean_object* v___x_2557_; 
v___x_2557_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2549_, v___y_2552_, v___y_2553_, v___y_2554_, v___y_2555_);
if (lean_obj_tag(v___x_2557_) == 0)
{
lean_object* v_a_2558_; lean_object* v___x_2559_; 
v_a_2558_ = lean_ctor_get(v___x_2557_, 0);
lean_inc(v_a_2558_);
lean_dec_ref_known(v___x_2557_, 1);
v___x_2559_ = lp_mathlib_Mathlib_Tactic_applyFunHyp(v_f_2545_, v_P_2546_, v_h_2547_, v_a_2558_, v___y_2548_, v___y_2549_, v___y_2550_, v___y_2551_, v___y_2552_, v___y_2553_, v___y_2554_, v___y_2555_);
if (lean_obj_tag(v___x_2559_) == 0)
{
lean_object* v_a_2560_; lean_object* v___x_2561_; 
v_a_2560_ = lean_ctor_get(v___x_2559_, 0);
lean_inc(v_a_2560_);
lean_dec_ref_known(v___x_2559_, 1);
v___x_2561_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_2560_, v___y_2549_, v___y_2552_, v___y_2553_, v___y_2554_, v___y_2555_);
return v___x_2561_;
}
else
{
lean_object* v_a_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2569_; 
v_a_2562_ = lean_ctor_get(v___x_2559_, 0);
v_isSharedCheck_2569_ = !lean_is_exclusive(v___x_2559_);
if (v_isSharedCheck_2569_ == 0)
{
v___x_2564_ = v___x_2559_;
v_isShared_2565_ = v_isSharedCheck_2569_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_a_2562_);
lean_dec(v___x_2559_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2569_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2567_; 
if (v_isShared_2565_ == 0)
{
v___x_2567_ = v___x_2564_;
goto v_reusejp_2566_;
}
else
{
lean_object* v_reuseFailAlloc_2568_; 
v_reuseFailAlloc_2568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2568_, 0, v_a_2562_);
v___x_2567_ = v_reuseFailAlloc_2568_;
goto v_reusejp_2566_;
}
v_reusejp_2566_:
{
return v___x_2567_;
}
}
}
}
else
{
lean_object* v_a_2570_; lean_object* v___x_2572_; uint8_t v_isShared_2573_; uint8_t v_isSharedCheck_2577_; 
lean_dec(v_h_2547_);
lean_dec(v_P_2546_);
lean_dec(v_f_2545_);
v_a_2570_ = lean_ctor_get(v___x_2557_, 0);
v_isSharedCheck_2577_ = !lean_is_exclusive(v___x_2557_);
if (v_isSharedCheck_2577_ == 0)
{
v___x_2572_ = v___x_2557_;
v_isShared_2573_ = v_isSharedCheck_2577_;
goto v_resetjp_2571_;
}
else
{
lean_inc(v_a_2570_);
lean_dec(v___x_2557_);
v___x_2572_ = lean_box(0);
v_isShared_2573_ = v_isSharedCheck_2577_;
goto v_resetjp_2571_;
}
v_resetjp_2571_:
{
lean_object* v___x_2575_; 
if (v_isShared_2573_ == 0)
{
v___x_2575_ = v___x_2572_;
goto v_reusejp_2574_;
}
else
{
lean_object* v_reuseFailAlloc_2576_; 
v_reuseFailAlloc_2576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2576_, 0, v_a_2570_);
v___x_2575_ = v_reuseFailAlloc_2576_;
goto v_reusejp_2574_;
}
v_reusejp_2574_:
{
return v___x_2575_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__2___boxed(lean_object* v_f_2578_, lean_object* v_P_2579_, lean_object* v_h_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_){
_start:
{
lean_object* v_res_2590_; 
v_res_2590_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__2(v_f_2578_, v_P_2579_, v_h_2580_, v___y_2581_, v___y_2582_, v___y_2583_, v___y_2584_, v___y_2585_, v___y_2586_, v___y_2587_, v___y_2588_);
lean_dec(v___y_2588_);
lean_dec_ref(v___y_2587_);
lean_dec(v___y_2586_);
lean_dec_ref(v___y_2585_);
lean_dec(v___y_2584_);
lean_dec_ref(v___y_2583_);
lean_dec(v___y_2582_);
lean_dec_ref(v___y_2581_);
return v_res_2590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1(lean_object* v_x_2592_, lean_object* v_a_2593_, lean_object* v_a_2594_, lean_object* v_a_2595_, lean_object* v_a_2596_, lean_object* v_a_2597_, lean_object* v_a_2598_, lean_object* v_a_2599_, lean_object* v_a_2600_){
_start:
{
lean_object* v___x_2602_; uint8_t v___x_2603_; 
v___x_2602_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_applyFun___closed__1));
lean_inc(v_x_2592_);
v___x_2603_ = l_Lean_Syntax_isOfKind(v_x_2592_, v___x_2602_);
if (v___x_2603_ == 0)
{
lean_object* v___x_2604_; 
lean_dec(v_x_2592_);
v___x_2604_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg();
return v___x_2604_;
}
else
{
lean_object* v___f_2605_; lean_object* v___y_2607_; lean_object* v___y_2608_; lean_object* v___y_2609_; lean_object* v___y_2610_; lean_object* v___y_2611_; lean_object* v___y_2612_; lean_object* v___y_2613_; lean_object* v___y_2614_; lean_object* v___y_2615_; lean_object* v___y_2616_; lean_object* v___y_2617_; lean_object* v___x_2622_; lean_object* v_f_2623_; lean_object* v___y_2625_; lean_object* v_P_2626_; lean_object* v___y_2627_; lean_object* v___y_2628_; lean_object* v___y_2629_; lean_object* v___y_2630_; lean_object* v___y_2631_; lean_object* v___y_2632_; lean_object* v___y_2633_; lean_object* v___y_2634_; lean_object* v___x_2646_; lean_object* v_loc_2648_; lean_object* v___y_2649_; lean_object* v___y_2650_; lean_object* v___y_2651_; lean_object* v___y_2652_; lean_object* v___y_2653_; lean_object* v___y_2654_; lean_object* v___y_2655_; lean_object* v___y_2656_; lean_object* v___x_2665_; uint8_t v___x_2666_; 
v___f_2605_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___closed__0));
v___x_2622_ = lean_unsigned_to_nat(1u);
v_f_2623_ = l_Lean_Syntax_getArg(v_x_2592_, v___x_2622_);
v___x_2646_ = lean_unsigned_to_nat(2u);
v___x_2665_ = l_Lean_Syntax_getArg(v_x_2592_, v___x_2646_);
v___x_2666_ = l_Lean_Syntax_isNone(v___x_2665_);
if (v___x_2666_ == 0)
{
uint8_t v___x_2667_; 
lean_inc(v___x_2665_);
v___x_2667_ = l_Lean_Syntax_matchesNull(v___x_2665_, v___x_2622_);
if (v___x_2667_ == 0)
{
lean_object* v___x_2668_; 
lean_dec(v___x_2665_);
lean_dec(v_f_2623_);
lean_dec(v_x_2592_);
v___x_2668_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg();
return v___x_2668_;
}
else
{
lean_object* v___x_2669_; lean_object* v_loc_2670_; lean_object* v___x_2671_; 
v___x_2669_ = lean_unsigned_to_nat(0u);
v_loc_2670_ = l_Lean_Syntax_getArg(v___x_2665_, v___x_2669_);
lean_dec(v___x_2665_);
v___x_2671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2671_, 0, v_loc_2670_);
v_loc_2648_ = v___x_2671_;
v___y_2649_ = v_a_2593_;
v___y_2650_ = v_a_2594_;
v___y_2651_ = v_a_2595_;
v___y_2652_ = v_a_2596_;
v___y_2653_ = v_a_2597_;
v___y_2654_ = v_a_2598_;
v___y_2655_ = v_a_2599_;
v___y_2656_ = v_a_2600_;
goto v___jp_2647_;
}
}
else
{
lean_object* v___x_2672_; 
lean_dec(v___x_2665_);
v___x_2672_ = lean_box(0);
v_loc_2648_ = v___x_2672_;
v___y_2649_ = v_a_2593_;
v___y_2650_ = v_a_2594_;
v___y_2651_ = v_a_2595_;
v___y_2652_ = v_a_2596_;
v___y_2653_ = v_a_2597_;
v___y_2654_ = v_a_2598_;
v___y_2655_ = v_a_2599_;
v___y_2656_ = v_a_2600_;
goto v___jp_2647_;
}
v___jp_2606_:
{
lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; 
v___x_2618_ = l_Lean_mkOptionalNode(v___y_2617_);
v___x_2619_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_2618_);
lean_dec(v___x_2618_);
v___x_2620_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withMainContext___boxed), 11, 2);
lean_closure_set(v___x_2620_, 0, lean_box(0));
lean_closure_set(v___x_2620_, 1, v___y_2607_);
v___x_2621_ = l_Lean_Elab_Tactic_withLocation(v___x_2619_, v___y_2615_, v___x_2620_, v___f_2605_, v___y_2609_, v___y_2614_, v___y_2611_, v___y_2616_, v___y_2608_, v___y_2610_, v___y_2613_, v___y_2612_);
lean_dec(v___x_2619_);
return v___x_2621_;
}
v___jp_2624_:
{
lean_object* v___f_2635_; lean_object* v___f_2636_; 
lean_inc(v_P_2626_);
lean_inc(v_f_2623_);
v___f_2635_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__1___boxed), 11, 2);
lean_closure_set(v___f_2635_, 0, v_f_2623_);
lean_closure_set(v___f_2635_, 1, v_P_2626_);
v___f_2636_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___lam__2___boxed), 12, 2);
lean_closure_set(v___f_2636_, 0, v_f_2623_);
lean_closure_set(v___f_2636_, 1, v_P_2626_);
if (lean_obj_tag(v___y_2625_) == 0)
{
lean_object* v___x_2637_; 
v___x_2637_ = lean_box(0);
v___y_2607_ = v___f_2635_;
v___y_2608_ = v___y_2631_;
v___y_2609_ = v___y_2627_;
v___y_2610_ = v___y_2632_;
v___y_2611_ = v___y_2629_;
v___y_2612_ = v___y_2634_;
v___y_2613_ = v___y_2633_;
v___y_2614_ = v___y_2628_;
v___y_2615_ = v___f_2636_;
v___y_2616_ = v___y_2630_;
v___y_2617_ = v___x_2637_;
goto v___jp_2606_;
}
else
{
lean_object* v_val_2638_; lean_object* v___x_2640_; uint8_t v_isShared_2641_; uint8_t v_isSharedCheck_2645_; 
v_val_2638_ = lean_ctor_get(v___y_2625_, 0);
v_isSharedCheck_2645_ = !lean_is_exclusive(v___y_2625_);
if (v_isSharedCheck_2645_ == 0)
{
v___x_2640_ = v___y_2625_;
v_isShared_2641_ = v_isSharedCheck_2645_;
goto v_resetjp_2639_;
}
else
{
lean_inc(v_val_2638_);
lean_dec(v___y_2625_);
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
lean_ctor_set(v_reuseFailAlloc_2644_, 0, v_val_2638_);
v___x_2643_ = v_reuseFailAlloc_2644_;
goto v_reusejp_2642_;
}
v_reusejp_2642_:
{
v___y_2607_ = v___f_2635_;
v___y_2608_ = v___y_2631_;
v___y_2609_ = v___y_2627_;
v___y_2610_ = v___y_2632_;
v___y_2611_ = v___y_2629_;
v___y_2612_ = v___y_2634_;
v___y_2613_ = v___y_2633_;
v___y_2614_ = v___y_2628_;
v___y_2615_ = v___f_2636_;
v___y_2616_ = v___y_2630_;
v___y_2617_ = v___x_2643_;
goto v___jp_2606_;
}
}
}
}
v___jp_2647_:
{
lean_object* v___x_2657_; lean_object* v___x_2658_; uint8_t v___x_2659_; 
v___x_2657_ = lean_unsigned_to_nat(3u);
v___x_2658_ = l_Lean_Syntax_getArg(v_x_2592_, v___x_2657_);
lean_dec(v_x_2592_);
v___x_2659_ = l_Lean_Syntax_isNone(v___x_2658_);
if (v___x_2659_ == 0)
{
uint8_t v___x_2660_; 
lean_inc(v___x_2658_);
v___x_2660_ = l_Lean_Syntax_matchesNull(v___x_2658_, v___x_2646_);
if (v___x_2660_ == 0)
{
lean_object* v___x_2661_; 
lean_dec(v___x_2658_);
lean_dec(v_loc_2648_);
lean_dec(v_f_2623_);
v___x_2661_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1_spec__0___redArg();
return v___x_2661_;
}
else
{
lean_object* v_P_2662_; lean_object* v___x_2663_; 
v_P_2662_ = l_Lean_Syntax_getArg(v___x_2658_, v___x_2622_);
lean_dec(v___x_2658_);
v___x_2663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2663_, 0, v_P_2662_);
v___y_2625_ = v_loc_2648_;
v_P_2626_ = v___x_2663_;
v___y_2627_ = v___y_2649_;
v___y_2628_ = v___y_2650_;
v___y_2629_ = v___y_2651_;
v___y_2630_ = v___y_2652_;
v___y_2631_ = v___y_2653_;
v___y_2632_ = v___y_2654_;
v___y_2633_ = v___y_2655_;
v___y_2634_ = v___y_2656_;
goto v___jp_2624_;
}
}
else
{
lean_object* v___x_2664_; 
lean_dec(v___x_2658_);
v___x_2664_ = lean_box(0);
v___y_2625_ = v_loc_2648_;
v_P_2626_ = v___x_2664_;
v___y_2627_ = v___y_2649_;
v___y_2628_ = v___y_2650_;
v___y_2629_ = v___y_2651_;
v___y_2630_ = v___y_2652_;
v___y_2631_ = v___y_2653_;
v___y_2632_ = v___y_2654_;
v___y_2633_ = v___y_2655_;
v___y_2634_ = v___y_2656_;
goto v___jp_2624_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1___boxed(lean_object* v_x_2673_, lean_object* v_a_2674_, lean_object* v_a_2675_, lean_object* v_a_2676_, lean_object* v_a_2677_, lean_object* v_a_2678_, lean_object* v_a_2679_, lean_object* v_a_2680_, lean_object* v_a_2681_, lean_object* v_a_2682_){
_start:
{
lean_object* v_res_2683_; 
v_res_2683_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ApplyFun______elabRules__Mathlib__Tactic__applyFun__1(v_x_2673_, v_a_2674_, v_a_2675_, v_a_2676_, v_a_2677_, v_a_2678_, v_a_2679_, v_a_2680_, v_a_2681_);
lean_dec(v_a_2681_);
lean_dec_ref(v_a_2680_);
lean_dec(v_a_2679_);
lean_dec_ref(v_a_2678_);
lean_dec(v_a_2677_);
lean_dec_ref(v_a_2676_);
lean_dec(v_a_2675_);
lean_dec_ref(v_a_2674_);
return v_res_2683_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_ApplyFun_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_ApplyFun_3448584465____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_applyFun = _init_lp_mathlib_Mathlib_Tactic_applyFun();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_applyFun);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Expr_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Expr_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
}
#ifdef __cplusplus
}
#endif
