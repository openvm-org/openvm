// Lean compiler output
// Module: Mathlib.Tactic.ComputeDegree
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Polynomial.Degree.Lemmas
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Lean_Name_lastComponentAsString(lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_addParenHeuristic(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_applyConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lp_mathlib_Lean_MVarId_getType_x27_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_numeral_x3f(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_constName(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_focus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "DFunLike"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "coe"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Polynomial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "monomial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__8_value),LEAN_SCALAR_PTR_LITERAL(192, 233, 223, 47, 237, 4, 213, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "C"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__10_value),LEAN_SCALAR_PTR_LITERAL(4, 116, 163, 217, 237, 233, 46, 7)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__14_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__17_value),LEAN_SCALAR_PTR_LITERAL(167, 166, 239, 19, 130, 98, 40, 185)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__20_value),LEAN_SCALAR_PTR_LITERAL(147, 155, 141, 233, 87, 0, 52, 207)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "coeff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23_value),LEAN_SCALAR_PTR_LITERAL(102, 237, 198, 135, 4, 12, 7, 116)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "natDegree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "degree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LE"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "LT"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 78, 141, 85, 50, 255, 216, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "congr lemma: '"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "lt_of_le_of_lt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__5_value),LEAN_SCALAR_PTR_LITERAL(229, 251, 24, 69, 185, 113, 227, 11)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_trans"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__7_value),LEAN_SCALAR_PTR_LITERAL(153, 164, 114, 182, 61, 254, 17, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "ComputeDegree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "degree_eq_of_le_of_coeff_ne_zero'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__12_value),LEAN_SCALAR_PTR_LITERAL(254, 92, 61, 130, 187, 234, 213, 43)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "natDegree_eq_of_le_of_coeff_ne_zero'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__14_value),LEAN_SCALAR_PTR_LITERAL(237, 100, 21, 129, 218, 147, 1, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "coeff_congr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__16_value),LEAN_SCALAR_PTR_LITERAL(110, 8, 143, 148, 27, 239, 35, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__4_value),LEAN_SCALAR_PTR_LITERAL(157, 40, 198, 234, 16, 168, 79, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "coeff_congr_lhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__19_value),LEAN_SCALAR_PTR_LITERAL(165, 92, 95, 83, 97, 143, 85, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__1(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__1___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(77, 42, 253, 71, 61, 132, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "le_rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(10, 105, 2, 11, 100, 134, 60, 21)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "natDegree_intCast_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__3_value),LEAN_SCALAR_PTR_LITERAL(47, 32, 86, 136, 139, 133, 54, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "degree_intCast_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__5_value),LEAN_SCALAR_PTR_LITERAL(198, 249, 196, 97, 2, 173, 242, 109)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "coeff_intCast_ite"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__7_value),LEAN_SCALAR_PTR_LITERAL(137, 166, 176, 217, 3, 63, 168, 11)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "natDegree_natCast_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(140, 38, 80, 247, 6, 90, 65, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "degree_natCast_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(157, 107, 161, 199, 147, 47, 39, 247)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "coeff_natCast_ite"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__13_value),LEAN_SCALAR_PTR_LITERAL(210, 91, 182, 58, 162, 205, 252, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "\ndispatchLemma:\n  "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__16_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instBEqOfDecidableEq___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__2_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "natDegree_one_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__18_value),LEAN_SCALAR_PTR_LITERAL(233, 158, 228, 95, 83, 6, 1, 185)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "degree_one_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__20_value),LEAN_SCALAR_PTR_LITERAL(233, 198, 112, 202, 55, 189, 242, 125)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "coeff_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__22_value),LEAN_SCALAR_PTR_LITERAL(253, 110, 117, 48, 141, 45, 65, 10)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "natDegree_zero_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__24_value),LEAN_SCALAR_PTR_LITERAL(241, 218, 226, 6, 28, 89, 76, 25)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "degree_zero_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__26_value),LEAN_SCALAR_PTR_LITERAL(209, 223, 218, 113, 11, 253, 154, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "coeff_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__29_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__28_value),LEAN_SCALAR_PTR_LITERAL(110, 0, 107, 82, 204, 92, 215, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NatCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "IntCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hSMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "natDegree_smul_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__41_value),LEAN_SCALAR_PTR_LITERAL(192, 0, 13, 136, 220, 12, 6, 88)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "degree_smul_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__43_value),LEAN_SCALAR_PTR_LITERAL(99, 4, 254, 11, 106, 34, 161, 11)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "coeff_smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__45_value),LEAN_SCALAR_PTR_LITERAL(238, 143, 77, 58, 33, 84, 28, 255)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "intCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__48_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "natCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__49_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "X"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "natDegree_C_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__51_value),LEAN_SCALAR_PTR_LITERAL(253, 144, 183, 218, 185, 166, 173, 138)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "degree_C_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__54_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__53_value),LEAN_SCALAR_PTR_LITERAL(135, 180, 180, 166, 172, 66, 182, 253)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__54_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "coeff_C"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__56_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__55_value),LEAN_SCALAR_PTR_LITERAL(248, 144, 98, 0, 24, 4, 119, 109)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__56_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "natDegree_monomial_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__58_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__57_value),LEAN_SCALAR_PTR_LITERAL(68, 215, 148, 120, 137, 72, 237, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__58_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "degree_monomial_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__60_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__59_value),LEAN_SCALAR_PTR_LITERAL(118, 69, 97, 0, 136, 131, 105, 72)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__60_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "coeff_monomial"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__62_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__62_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__61_value),LEAN_SCALAR_PTR_LITERAL(111, 14, 24, 108, 126, 2, 161, 99)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__62_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "natDegree_X_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__64_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__63_value),LEAN_SCALAR_PTR_LITERAL(222, 160, 61, 231, 102, 227, 14, 178)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__64_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "degree_X_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__65_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__66_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__66_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__65_value),LEAN_SCALAR_PTR_LITERAL(23, 3, 119, 17, 24, 55, 150, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__66_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "coeff_X"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__68_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__68_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__67_value),LEAN_SCALAR_PTR_LITERAL(243, 118, 74, 176, 231, 198, 173, 222)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "natDegree_neg_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__71_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__70_value),LEAN_SCALAR_PTR_LITERAL(229, 155, 200, 120, 2, 201, 93, 51)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__71_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "degree_neg_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__73_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__73_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__72_value),LEAN_SCALAR_PTR_LITERAL(11, 81, 137, 149, 124, 134, 156, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__73_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "coeff_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__74_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__75_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__75_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__74_value),LEAN_SCALAR_PTR_LITERAL(143, 219, 82, 178, 121, 8, 86, 182)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__75_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hPow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__76_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "natDegree_pow_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__78_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__78_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__77_value),LEAN_SCALAR_PTR_LITERAL(183, 229, 132, 237, 18, 57, 226, 131)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__78_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "degree_pow_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__80_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__80_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__79_value),LEAN_SCALAR_PTR_LITERAL(7, 19, 102, 125, 14, 225, 62, 2)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__80_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "coeff_pow_of_natDegree_le_of_eq_ite'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__81_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__81_value),LEAN_SCALAR_PTR_LITERAL(76, 88, 50, 43, 246, 189, 24, 228)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hMul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__83_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "natDegree_mul_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__84_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__85_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__85_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__84_value),LEAN_SCALAR_PTR_LITERAL(182, 191, 59, 6, 228, 242, 142, 192)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__85_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "degree_mul_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__86_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__87_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__87_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__86_value),LEAN_SCALAR_PTR_LITERAL(44, 130, 160, 179, 190, 112, 232, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__87_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "coeff_mul_add_of_le_natDegree_of_eq_ite"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__88_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__88_value),LEAN_SCALAR_PTR_LITERAL(13, 101, 138, 238, 209, 40, 83, 38)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hSub"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__90_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "natDegree_sub_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__92_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__92_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__91_value),LEAN_SCALAR_PTR_LITERAL(6, 202, 228, 83, 7, 98, 64, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__92_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "degree_sub_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__93_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__94_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__94_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__93_value),LEAN_SCALAR_PTR_LITERAL(114, 62, 118, 113, 10, 155, 162, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__94 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__94_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "coeff_sub_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__95_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__95_value),LEAN_SCALAR_PTR_LITERAL(168, 25, 236, 205, 13, 38, 106, 159)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hAdd"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__97 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__97_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "natDegree_add_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__98_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__99_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__99_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__98_value),LEAN_SCALAR_PTR_LITERAL(241, 31, 221, 108, 4, 252, 31, 219)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__99_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "degree_add_le_of_le"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__100_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__101_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__101_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__100_value),LEAN_SCALAR_PTR_LITERAL(23, 32, 64, 229, 31, 68, 227, 125)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__101_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "coeff_add_of_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__102_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__102_value),LEAN_SCALAR_PTR_LITERAL(109, 220, 125, 204, 159, 47, 95, 103)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "(inl "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__104 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__104_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "(inr "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__106 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__106_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___closed__0 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__4(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_try__rfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_try__rfl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_splitApply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_splitApply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 38, .m_data = "* there may be a term of naïve degree "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28_value),LEAN_SCALAR_PTR_LITERAL(216, 149, 183, 186, 191, 145, 216, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31_value),LEAN_SCALAR_PTR_LITERAL(109, 14, 90, 172, 72, 170, 136, 101)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 45, .m_data = "* there is at least one term of naïve degree "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "* the coefficient of degree "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " may be zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "computeDegree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 110, 230, 101, 152, 133, 154, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "compute_degree"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__0_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__0_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__0_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4_value),LEAN_SCALAR_PTR_LITERAL(221, 100, 251, 2, 101, 88, 143, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__0_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__0_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__1_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__1_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__1_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__2_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__1_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__2_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__2_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__3_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__2_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__3_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__3_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__4_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__3_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__4_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__4_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__5_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__4_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(40, 191, 184, 136, 91, 160, 227, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__5_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__5_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__6_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__5_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 129, 142, 235, 169, 116, 174, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__6_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__6_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__7_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__6_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(116, 176, 186, 149, 118, 148, 109, 167)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__7_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__7_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__8_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__7_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(97, 130, 95, 231, 116, 178, 167, 66)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__8_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__8_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__9_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__8_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(14, 52, 142, 74, 121, 170, 2, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__9_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__9_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__10_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__10_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__10_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__11_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__9_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__10_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(163, 175, 224, 209, 237, 170, 233, 102)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__11_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__11_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__12_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__12_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__12_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__13_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__11_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__12_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(150, 58, 91, 40, 241, 77, 91, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__13_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__13_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__14_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__13_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(175, 82, 205, 52, 53, 4, 132, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__14_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__14_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__15_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__14_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(190, 22, 87, 107, 122, 91, 75, 34)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__15_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__15_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__16_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__15_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(189, 56, 224, 86, 19, 89, 141, 116)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__16_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__16_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__17_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__16_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1830119225) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(124, 79, 219, 25, 17, 209, 71, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__17_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__17_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__18_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__18_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__18_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__19_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__17_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__18_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(211, 199, 75, 97, 189, 192, 63, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__19_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__19_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__20_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__20_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__20_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__21_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__19_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__20_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(147, 235, 194, 101, 168, 111, 175, 38)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__21_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__21_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__22_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__21_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(150, 131, 9, 215, 49, 20, 123, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__22_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__22_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "tacticCompute_degree!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(96, 135, 249, 20, 222, 124, 114, 122)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "compute_degree!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "anyGoals"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__8 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "any_goals"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__9 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__10 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__10_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "normNum"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__11 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__11_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "norm_num"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__12 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__13 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__13_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__15 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__15_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticNorm_cast__"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__16 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__16_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "norm_cast"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__17 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__17_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__18 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value)} };
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convLHS"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "conv_lhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "convSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "posConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "+"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__11;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(236, 252, 83, 10, 217, 228, 80, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(16, 96, 65, 173, 152, 155, 4, 222)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Nat.cast_withBot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "cast_withBot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__35_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__22_value),LEAN_SCALAR_PTR_LITERAL(52, 198, 149, 182, 216, 41, 148, 171)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "normNumConv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "The given degree is '"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__31;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "'.  However,\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__32_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__33;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__34;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "'compute_degree' inapplicable. The goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__35_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__36;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 35, .m_data = "\nis expected to be '≤', '<' or '='."};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__37_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__38;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 100, .m_capacity = 100, .m_length = 99, .m_data = "'compute_degree' inapplicable. The LHS must be an application of 'natDegree', 'degree', or 'coeff'."};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__40;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__41_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 39, .m_capacity = 39, .m_length = 38, .m_data = "'compute_degree' first applies lemma '"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__43_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__44_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "monicityMacro"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__0_value),LEAN_SCALAR_PTR_LITERAL(43, 59, 3, 121, 98, 2, 63, 61)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "monicity"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value_aux_2),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "monic_of_natDegree_le_of_coeff_eq_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(23, 108, 158, 142, 195, 138, 147, 241)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7_value),LEAN_SCALAR_PTR_LITERAL(81, 20, 154, 203, 94, 248, 164, 115)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(229, 15, 135, 195, 242, 248, 42, 79)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "tacticMonicity!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__11_value),LEAN_SCALAR_PTR_LITERAL(108, 89, 121, 84, 3, 70, 234, 59)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 86, 200, 23, 52, 45, 198, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "monicity!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticMonicity_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticMonicity_x21__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__0 = (const lean_object*)&lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__1 = (const lean_object*)&lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__1_value;
static const lean_string_object lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__2 = (const lean_object*)&lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__2_value;
static const lean_array_object lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__3 = (const lean_object*)&lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs(lean_object* v_e_56_){
_start:
{
lean_object* v___y_60_; lean_object* v___y_66_; lean_object* v___y_67_; lean_object* v___y_68_; lean_object* v___y_69_; lean_object* v___y_74_; lean_object* v___y_75_; lean_object* v___x_81_; lean_object* v_fst_82_; 
v___x_81_ = l_Lean_Expr_getAppFnArgs(v_e_56_);
v_fst_82_ = lean_ctor_get(v___x_81_, 0);
lean_inc(v_fst_82_);
if (lean_obj_tag(v_fst_82_) == 1)
{
lean_object* v_snd_83_; lean_object* v_pre_84_; lean_object* v_str_85_; lean_object* v___x_86_; lean_object* v___y_88_; lean_object* v_fst_89_; lean_object* v_fst_90_; lean_object* v_snd_91_; lean_object* v_fst_134_; lean_object* v_fst_135_; lean_object* v_snd_136_; 
v_snd_83_ = lean_ctor_get(v___x_81_, 1);
lean_inc(v_snd_83_);
lean_dec_ref(v___x_81_);
v_pre_84_ = lean_ctor_get(v_fst_82_, 0);
lean_inc(v_pre_84_);
v_str_85_ = lean_ctor_get(v_fst_82_, 1);
lean_inc_ref(v_str_85_);
lean_dec_ref_known(v_fst_82_, 2);
v___x_86_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__4));
switch(lean_obj_tag(v_pre_84_))
{
case 0:
{
lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27));
v___x_225_ = lean_string_dec_eq(v_str_85_, v___x_224_);
lean_dec_ref(v_str_85_);
if (v___x_225_ == 0)
{
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
v___x_226_ = lean_array_get_size(v_snd_83_);
v___x_227_ = lean_unsigned_to_nat(3u);
v___x_228_ = lean_nat_dec_eq(v___x_226_, v___x_227_);
if (v___x_228_ == 0)
{
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_229_ = l_Lean_Name_str___override(v_pre_84_, v___x_224_);
v___x_230_ = lean_unsigned_to_nat(1u);
v___x_231_ = lean_array_fget(v_snd_83_, v___x_230_);
v___x_232_ = lean_unsigned_to_nat(2u);
v___x_233_ = lean_array_fget(v_snd_83_, v___x_232_);
lean_dec(v_snd_83_);
v_fst_134_ = v___x_229_;
v_fst_135_ = v___x_231_;
v_snd_136_ = v___x_233_;
goto v___jp_133_;
}
}
}
case 1:
{
lean_object* v_pre_234_; 
v_pre_234_ = lean_ctor_get(v_pre_84_, 0);
lean_inc(v_pre_234_);
if (lean_obj_tag(v_pre_234_) == 0)
{
lean_object* v_str_235_; lean_object* v___x_236_; uint8_t v___x_237_; 
v_str_235_ = lean_ctor_get(v_pre_84_, 1);
lean_inc_ref(v_str_235_);
lean_dec_ref_known(v_pre_84_, 2);
v___x_236_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_237_ = lean_string_dec_eq(v_str_235_, v___x_236_);
if (v___x_237_ == 0)
{
lean_object* v___x_238_; uint8_t v___x_239_; 
v___x_238_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__29));
v___x_239_ = lean_string_dec_eq(v_str_235_, v___x_238_);
lean_dec_ref(v_str_235_);
if (v___x_239_ == 0)
{
lean_dec_ref(v_str_85_);
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_240_; uint8_t v___x_241_; 
v___x_240_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__30));
v___x_241_ = lean_string_dec_eq(v_str_85_, v___x_240_);
lean_dec_ref(v_str_85_);
if (v___x_241_ == 0)
{
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_242_; lean_object* v___x_243_; uint8_t v___x_244_; 
v___x_242_ = lean_array_get_size(v_snd_83_);
v___x_243_ = lean_unsigned_to_nat(4u);
v___x_244_ = lean_nat_dec_eq(v___x_242_, v___x_243_);
if (v___x_244_ == 0)
{
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_245_ = l_Lean_Name_str___override(v_pre_234_, v___x_238_);
v___x_246_ = l_Lean_Name_str___override(v___x_245_, v___x_240_);
v___x_247_ = lean_unsigned_to_nat(2u);
v___x_248_ = lean_array_fget(v_snd_83_, v___x_247_);
v___x_249_ = lean_unsigned_to_nat(3u);
v___x_250_ = lean_array_fget(v_snd_83_, v___x_249_);
lean_dec(v_snd_83_);
v_fst_134_ = v___x_246_;
v_fst_135_ = v___x_248_;
v_snd_136_ = v___x_250_;
goto v___jp_133_;
}
}
}
}
else
{
lean_object* v___x_251_; uint8_t v___x_252_; 
lean_dec_ref(v_str_235_);
v___x_251_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_252_ = lean_string_dec_eq(v_str_85_, v___x_251_);
lean_dec_ref(v_str_85_);
if (v___x_252_ == 0)
{
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_253_; lean_object* v___x_254_; uint8_t v___x_255_; 
v___x_253_ = lean_array_get_size(v_snd_83_);
v___x_254_ = lean_unsigned_to_nat(4u);
v___x_255_ = lean_nat_dec_eq(v___x_253_, v___x_254_);
if (v___x_255_ == 0)
{
lean_dec(v_snd_83_);
goto v___jp_57_;
}
else
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_256_ = l_Lean_Name_str___override(v_pre_234_, v___x_236_);
v___x_257_ = l_Lean_Name_str___override(v___x_256_, v___x_251_);
v___x_258_ = lean_unsigned_to_nat(2u);
v___x_259_ = lean_array_fget(v_snd_83_, v___x_258_);
v___x_260_ = lean_unsigned_to_nat(3u);
v___x_261_ = lean_array_fget(v_snd_83_, v___x_260_);
lean_dec(v_snd_83_);
v_fst_134_ = v___x_257_;
v_fst_135_ = v___x_259_;
v_snd_136_ = v___x_261_;
goto v___jp_133_;
}
}
}
}
else
{
lean_dec_ref_known(v_pre_84_, 2);
lean_dec(v_pre_234_);
lean_dec_ref(v_str_85_);
lean_dec(v_snd_83_);
goto v___jp_57_;
}
}
default: 
{
lean_dec_ref(v_str_85_);
lean_dec(v_pre_84_);
lean_dec(v_snd_83_);
goto v___jp_57_;
}
}
v___jp_87_:
{
lean_object* v___x_92_; 
lean_inc_ref(v_fst_90_);
v___x_92_ = lp_mathlib_Lean_Expr_numeral_x3f(v_fst_90_);
if (lean_obj_tag(v___x_92_) == 0)
{
lean_object* v___x_93_; lean_object* v_fst_94_; 
v___x_93_ = l_Lean_Expr_getAppFnArgs(v_fst_90_);
v_fst_94_ = lean_ctor_get(v___x_93_, 0);
lean_inc(v_fst_94_);
if (lean_obj_tag(v_fst_94_) == 1)
{
lean_object* v_pre_95_; 
v_pre_95_ = lean_ctor_get(v_fst_94_, 0);
if (lean_obj_tag(v_pre_95_) == 1)
{
lean_object* v_pre_96_; 
v_pre_96_ = lean_ctor_get(v_pre_95_, 0);
if (lean_obj_tag(v_pre_96_) == 0)
{
lean_object* v_snd_97_; lean_object* v_str_98_; lean_object* v_str_99_; lean_object* v___x_100_; uint8_t v___x_101_; 
v_snd_97_ = lean_ctor_get(v___x_93_, 1);
lean_inc(v_snd_97_);
lean_dec_ref(v___x_93_);
v_str_98_ = lean_ctor_get(v_fst_94_, 1);
v_str_99_ = lean_ctor_get(v_pre_95_, 1);
v___x_100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__5));
v___x_101_ = lean_string_dec_eq(v_str_99_, v___x_100_);
if (v___x_101_ == 0)
{
lean_object* v___x_102_; 
lean_dec(v_snd_97_);
v___x_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_102_, 0, v_fst_94_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_102_;
goto v___jp_65_;
}
else
{
lean_object* v___x_103_; uint8_t v___x_104_; 
lean_inc_ref(v_str_98_);
lean_inc(v_pre_96_);
lean_dec_ref_known(v_fst_94_, 2);
v___x_103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__6));
v___x_104_ = lean_string_dec_eq(v_str_98_, v___x_103_);
if (v___x_104_ == 0)
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec(v_snd_97_);
v___x_105_ = l_Lean_Name_str___override(v_pre_96_, v___x_100_);
v___x_106_ = l_Lean_Name_str___override(v___x_105_, v_str_98_);
v___x_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_107_;
goto v___jp_65_;
}
else
{
lean_object* v___x_108_; lean_object* v___x_109_; uint8_t v___x_110_; 
lean_dec_ref(v_str_98_);
v___x_108_ = lean_array_get_size(v_snd_97_);
v___x_109_ = lean_unsigned_to_nat(6u);
v___x_110_ = lean_nat_dec_eq(v___x_108_, v___x_109_);
if (v___x_110_ == 0)
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v_snd_97_);
v___x_111_ = l_Lean_Name_str___override(v_pre_96_, v___x_100_);
v___x_112_ = l_Lean_Name_str___override(v___x_111_, v___x_103_);
v___x_113_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_113_;
goto v___jp_65_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v_na_117_; lean_object* v___x_118_; uint8_t v___x_119_; 
v___x_114_ = lean_unsigned_to_nat(4u);
v___x_115_ = lean_array_fget(v_snd_97_, v___x_114_);
lean_dec(v_snd_97_);
v___x_116_ = l_Lean_Expr_getAppFn(v___x_115_);
lean_dec(v___x_115_);
v_na_117_ = l_Lean_Expr_constName(v___x_116_);
lean_dec_ref(v___x_116_);
v___x_118_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__13));
lean_inc(v_na_117_);
v___x_119_ = l_List_elem___redArg(v___x_86_, v_na_117_, v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; 
lean_dec(v_na_117_);
v___x_120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_120_, 0, v_pre_96_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_120_;
goto v___jp_65_;
}
else
{
lean_object* v___x_121_; 
v___x_121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_121_, 0, v_na_117_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_121_;
goto v___jp_65_;
}
}
}
}
}
else
{
lean_object* v___x_122_; 
lean_dec_ref(v___x_93_);
v___x_122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_122_, 0, v_fst_94_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_122_;
goto v___jp_65_;
}
}
else
{
lean_object* v___x_123_; 
lean_dec_ref(v___x_93_);
v___x_123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_123_, 0, v_fst_94_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_123_;
goto v___jp_65_;
}
}
else
{
lean_object* v___x_124_; 
lean_dec_ref(v___x_93_);
v___x_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_124_, 0, v_fst_94_);
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_124_;
goto v___jp_65_;
}
}
else
{
lean_object* v_val_125_; lean_object* v___x_126_; uint8_t v___x_127_; 
lean_dec_ref(v_fst_90_);
v_val_125_ = lean_ctor_get(v___x_92_, 0);
lean_inc(v_val_125_);
lean_dec_ref_known(v___x_92_, 1);
v___x_126_ = lean_unsigned_to_nat(0u);
v___x_127_ = lean_nat_dec_eq(v_val_125_, v___x_126_);
if (v___x_127_ == 0)
{
lean_object* v___x_128_; uint8_t v___x_129_; 
v___x_128_ = lean_unsigned_to_nat(1u);
v___x_129_ = lean_nat_dec_eq(v_val_125_, v___x_128_);
lean_dec(v_val_125_);
if (v___x_129_ == 0)
{
lean_object* v___x_130_; 
v___x_130_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__16));
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_130_;
goto v___jp_65_;
}
else
{
lean_object* v___x_131_; 
v___x_131_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__19));
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_131_;
goto v___jp_65_;
}
}
else
{
lean_object* v___x_132_; 
lean_dec(v_val_125_);
v___x_132_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__22));
v___y_66_ = v_snd_91_;
v___y_67_ = v_fst_89_;
v___y_68_ = v___y_88_;
v___y_69_ = v___x_132_;
goto v___jp_65_;
}
}
}
v___jp_133_:
{
lean_object* v___x_137_; lean_object* v_fst_138_; 
v___x_137_ = l_Lean_Expr_getAppFnArgs(v_fst_135_);
v_fst_138_ = lean_ctor_get(v___x_137_, 0);
lean_inc(v_fst_138_);
if (lean_obj_tag(v_fst_138_) == 1)
{
lean_object* v_pre_139_; 
v_pre_139_ = lean_ctor_get(v_fst_138_, 0);
lean_inc(v_pre_139_);
if (lean_obj_tag(v_pre_139_) == 1)
{
lean_object* v_pre_140_; 
v_pre_140_ = lean_ctor_get(v_pre_139_, 0);
lean_inc(v_pre_140_);
if (lean_obj_tag(v_pre_140_) == 0)
{
lean_object* v_snd_141_; lean_object* v___x_143_; uint8_t v_isShared_144_; uint8_t v_isSharedCheck_222_; 
v_snd_141_ = lean_ctor_get(v___x_137_, 1);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_137_);
if (v_isSharedCheck_222_ == 0)
{
lean_object* v_unused_223_; 
v_unused_223_ = lean_ctor_get(v___x_137_, 0);
lean_dec(v_unused_223_);
v___x_143_ = v___x_137_;
v_isShared_144_ = v_isSharedCheck_222_;
goto v_resetjp_142_;
}
else
{
lean_inc(v_snd_141_);
lean_dec(v___x_137_);
v___x_143_ = lean_box(0);
v_isShared_144_ = v_isSharedCheck_222_;
goto v_resetjp_142_;
}
v_resetjp_142_:
{
lean_object* v_str_145_; lean_object* v_str_146_; lean_object* v___x_147_; uint8_t v___x_148_; 
v_str_145_ = lean_ctor_get(v_fst_138_, 1);
lean_inc_ref(v_str_145_);
lean_dec_ref_known(v_fst_138_, 2);
v_str_146_ = lean_ctor_get(v_pre_139_, 1);
lean_inc_ref(v_str_146_);
lean_dec_ref_known(v_pre_139_, 2);
v___x_147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7));
v___x_148_ = lean_string_dec_eq(v_str_146_, v___x_147_);
if (v___x_148_ == 0)
{
lean_object* v___x_149_; uint8_t v___x_150_; 
v___x_149_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__5));
v___x_150_ = lean_string_dec_eq(v_str_146_, v___x_149_);
lean_dec_ref(v_str_146_);
if (v___x_150_ == 0)
{
lean_dec_ref(v_str_145_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
else
{
lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__6));
v___x_152_ = lean_string_dec_eq(v_str_145_, v___x_151_);
lean_dec_ref(v_str_145_);
if (v___x_152_ == 0)
{
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
else
{
lean_object* v___x_153_; lean_object* v___x_154_; uint8_t v___x_155_; 
v___x_153_ = lean_array_get_size(v_snd_141_);
v___x_154_ = lean_unsigned_to_nat(6u);
v___x_155_ = lean_nat_dec_eq(v___x_153_, v___x_154_);
if (v___x_155_ == 0)
{
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
else
{
lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v_fst_159_; 
v___x_156_ = lean_unsigned_to_nat(4u);
v___x_157_ = lean_array_fget_borrowed(v_snd_141_, v___x_156_);
lean_inc(v___x_157_);
v___x_158_ = l_Lean_Expr_getAppFnArgs(v___x_157_);
v_fst_159_ = lean_ctor_get(v___x_158_, 0);
lean_inc(v_fst_159_);
if (lean_obj_tag(v_fst_159_) == 1)
{
lean_object* v_pre_160_; 
v_pre_160_ = lean_ctor_get(v_fst_159_, 0);
lean_inc(v_pre_160_);
if (lean_obj_tag(v_pre_160_) == 1)
{
lean_object* v_pre_161_; 
v_pre_161_ = lean_ctor_get(v_pre_160_, 0);
if (lean_obj_tag(v_pre_161_) == 0)
{
lean_object* v_snd_162_; lean_object* v___x_164_; uint8_t v_isShared_165_; uint8_t v_isSharedCheck_190_; 
v_snd_162_ = lean_ctor_get(v___x_158_, 1);
v_isSharedCheck_190_ = !lean_is_exclusive(v___x_158_);
if (v_isSharedCheck_190_ == 0)
{
lean_object* v_unused_191_; 
v_unused_191_ = lean_ctor_get(v___x_158_, 0);
lean_dec(v_unused_191_);
v___x_164_ = v___x_158_;
v_isShared_165_ = v_isSharedCheck_190_;
goto v_resetjp_163_;
}
else
{
lean_inc(v_snd_162_);
lean_dec(v___x_158_);
v___x_164_ = lean_box(0);
v_isShared_165_ = v_isSharedCheck_190_;
goto v_resetjp_163_;
}
v_resetjp_163_:
{
lean_object* v_str_166_; lean_object* v_str_167_; uint8_t v___x_168_; 
v_str_166_ = lean_ctor_get(v_fst_159_, 1);
lean_inc_ref(v_str_166_);
lean_dec_ref_known(v_fst_159_, 2);
v_str_167_ = lean_ctor_get(v_pre_160_, 1);
lean_inc_ref(v_str_167_);
lean_dec_ref_known(v_pre_160_, 2);
v___x_168_ = lean_string_dec_eq(v_str_167_, v___x_147_);
lean_dec_ref(v_str_167_);
if (v___x_168_ == 0)
{
lean_dec_ref(v_str_166_);
lean_del_object(v___x_164_);
lean_dec(v_snd_162_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_74_ = v_pre_140_;
v___y_75_ = v_fst_134_;
goto v___jp_73_;
}
else
{
lean_object* v___x_169_; uint8_t v___x_170_; 
v___x_169_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23));
v___x_170_ = lean_string_dec_eq(v_str_166_, v___x_169_);
lean_dec_ref(v_str_166_);
if (v___x_170_ == 0)
{
lean_del_object(v___x_164_);
lean_dec(v_snd_162_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_74_ = v_pre_140_;
v___y_75_ = v_fst_134_;
goto v___jp_73_;
}
else
{
lean_object* v___x_171_; lean_object* v___x_172_; uint8_t v___x_173_; 
v___x_171_ = lean_array_get_size(v_snd_162_);
v___x_172_ = lean_unsigned_to_nat(3u);
v___x_173_ = lean_nat_dec_eq(v___x_171_, v___x_172_);
if (v___x_173_ == 0)
{
lean_del_object(v___x_164_);
lean_dec(v_snd_162_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_74_ = v_pre_140_;
v___y_75_ = v_fst_134_;
goto v___jp_73_;
}
else
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; uint8_t v___x_179_; uint8_t v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_184_; 
v___x_174_ = lean_unsigned_to_nat(5u);
v___x_175_ = lean_array_fget(v_snd_141_, v___x_174_);
lean_dec(v_snd_141_);
v___x_176_ = lean_unsigned_to_nat(2u);
v___x_177_ = lean_array_fget(v_snd_162_, v___x_176_);
lean_dec(v_snd_162_);
v___x_178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__24));
v___x_179_ = l_Lean_Expr_isMVar(v_snd_136_);
lean_dec_ref(v_snd_136_);
v___x_180_ = l_Lean_Expr_isMVar(v___x_175_);
lean_dec(v___x_175_);
v___x_181_ = lean_box(0);
v___x_182_ = lean_box(v___x_180_);
if (v_isShared_165_ == 0)
{
lean_ctor_set_tag(v___x_164_, 1);
lean_ctor_set(v___x_164_, 1, v___x_181_);
lean_ctor_set(v___x_164_, 0, v___x_182_);
v___x_184_ = v___x_164_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_189_; 
v_reuseFailAlloc_189_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_189_, 0, v___x_182_);
lean_ctor_set(v_reuseFailAlloc_189_, 1, v___x_181_);
v___x_184_ = v_reuseFailAlloc_189_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
lean_object* v___x_185_; lean_object* v___x_187_; 
v___x_185_ = lean_box(v___x_179_);
if (v_isShared_144_ == 0)
{
lean_ctor_set_tag(v___x_143_, 1);
lean_ctor_set(v___x_143_, 1, v___x_184_);
lean_ctor_set(v___x_143_, 0, v___x_185_);
v___x_187_ = v___x_143_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v___x_185_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v___x_184_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
v___y_88_ = v_fst_134_;
v_fst_89_ = v___x_178_;
v_fst_90_ = v___x_177_;
v_snd_91_ = v___x_187_;
goto v___jp_87_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_160_, 2);
lean_dec_ref_known(v_fst_159_, 2);
lean_dec_ref(v___x_158_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_74_ = v_pre_140_;
v___y_75_ = v_fst_134_;
goto v___jp_73_;
}
}
else
{
lean_dec_ref_known(v_fst_159_, 2);
lean_dec(v_pre_160_);
lean_dec_ref(v___x_158_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_74_ = v_pre_140_;
v___y_75_ = v_fst_134_;
goto v___jp_73_;
}
}
else
{
lean_dec(v_fst_159_);
lean_dec_ref(v___x_158_);
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_74_ = v_pre_140_;
v___y_75_ = v_fst_134_;
goto v___jp_73_;
}
}
}
}
}
else
{
lean_object* v___x_192_; uint8_t v___x_193_; 
lean_dec_ref(v_str_146_);
v___x_192_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__25));
v___x_193_ = lean_string_dec_eq(v_str_145_, v___x_192_);
if (v___x_193_ == 0)
{
lean_object* v___x_194_; uint8_t v___x_195_; 
v___x_194_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__26));
v___x_195_ = lean_string_dec_eq(v_str_145_, v___x_194_);
lean_dec_ref(v_str_145_);
if (v___x_195_ == 0)
{
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
else
{
lean_object* v___x_196_; lean_object* v___x_197_; uint8_t v___x_198_; 
v___x_196_ = lean_array_get_size(v_snd_141_);
v___x_197_ = lean_unsigned_to_nat(3u);
v___x_198_ = lean_nat_dec_eq(v___x_196_, v___x_197_);
if (v___x_198_ == 0)
{
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
else
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; uint8_t v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_207_; 
v___x_199_ = l_Lean_Name_str___override(v_pre_140_, v___x_147_);
v___x_200_ = l_Lean_Name_str___override(v___x_199_, v___x_194_);
v___x_201_ = lean_unsigned_to_nat(2u);
v___x_202_ = lean_array_fget(v_snd_141_, v___x_201_);
lean_dec(v_snd_141_);
v___x_203_ = l_Lean_Expr_isMVar(v_snd_136_);
lean_dec_ref(v_snd_136_);
v___x_204_ = lean_box(0);
v___x_205_ = lean_box(v___x_203_);
if (v_isShared_144_ == 0)
{
lean_ctor_set_tag(v___x_143_, 1);
lean_ctor_set(v___x_143_, 1, v___x_204_);
lean_ctor_set(v___x_143_, 0, v___x_205_);
v___x_207_ = v___x_143_;
goto v_reusejp_206_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v___x_205_);
lean_ctor_set(v_reuseFailAlloc_208_, 1, v___x_204_);
v___x_207_ = v_reuseFailAlloc_208_;
goto v_reusejp_206_;
}
v_reusejp_206_:
{
v___y_88_ = v_fst_134_;
v_fst_89_ = v___x_200_;
v_fst_90_ = v___x_202_;
v_snd_91_ = v___x_207_;
goto v___jp_87_;
}
}
}
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; uint8_t v___x_211_; 
lean_dec_ref(v_str_145_);
v___x_209_ = lean_array_get_size(v_snd_141_);
v___x_210_ = lean_unsigned_to_nat(3u);
v___x_211_ = lean_nat_dec_eq(v___x_209_, v___x_210_);
if (v___x_211_ == 0)
{
lean_del_object(v___x_143_);
lean_dec(v_snd_141_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
else
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; uint8_t v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_220_; 
v___x_212_ = l_Lean_Name_str___override(v_pre_140_, v___x_147_);
v___x_213_ = l_Lean_Name_str___override(v___x_212_, v___x_192_);
v___x_214_ = lean_unsigned_to_nat(2u);
v___x_215_ = lean_array_fget(v_snd_141_, v___x_214_);
lean_dec(v_snd_141_);
v___x_216_ = l_Lean_Expr_isMVar(v_snd_136_);
lean_dec_ref(v_snd_136_);
v___x_217_ = lean_box(0);
v___x_218_ = lean_box(v___x_216_);
if (v_isShared_144_ == 0)
{
lean_ctor_set_tag(v___x_143_, 1);
lean_ctor_set(v___x_143_, 1, v___x_217_);
lean_ctor_set(v___x_143_, 0, v___x_218_);
v___x_220_ = v___x_143_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v___x_218_);
lean_ctor_set(v_reuseFailAlloc_221_, 1, v___x_217_);
v___x_220_ = v_reuseFailAlloc_221_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
v___y_88_ = v_fst_134_;
v_fst_89_ = v___x_213_;
v_fst_90_ = v___x_215_;
v_snd_91_ = v___x_220_;
goto v___jp_87_;
}
}
}
}
}
}
else
{
lean_dec(v_pre_140_);
lean_dec_ref_known(v_pre_139_, 2);
lean_dec_ref_known(v_fst_138_, 2);
lean_dec_ref(v___x_137_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
}
else
{
lean_dec_ref_known(v_fst_138_, 2);
lean_dec(v_pre_139_);
lean_dec_ref(v___x_137_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
}
else
{
lean_dec(v_fst_138_);
lean_dec_ref(v___x_137_);
lean_dec_ref(v_snd_136_);
v___y_60_ = v_fst_134_;
goto v___jp_59_;
}
}
}
else
{
lean_dec(v_fst_82_);
lean_dec_ref(v___x_81_);
goto v___jp_57_;
}
v___jp_57_:
{
lean_object* v___x_58_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__3));
return v___x_58_;
}
v___jp_59_:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; 
v___x_61_ = lean_box(0);
v___x_62_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__1));
v___x_63_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_63_, 0, v___y_60_);
lean_ctor_set(v___x_63_, 1, v___x_62_);
v___x_64_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_64_, 0, v___x_61_);
lean_ctor_set(v___x_64_, 1, v___x_63_);
return v___x_64_;
}
v___jp_65_:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_70_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_70_, 0, v___y_69_);
lean_ctor_set(v___x_70_, 1, v___y_66_);
v___x_71_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_71_, 0, v___y_68_);
lean_ctor_set(v___x_71_, 1, v___x_70_);
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v___y_67_);
lean_ctor_set(v___x_72_, 1, v___x_71_);
return v___x_72_;
}
v___jp_73_:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
lean_inc(v___y_74_);
v___x_76_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_76_, 0, v___y_74_);
v___x_77_ = lean_box(0);
v___x_78_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_78_, 0, v___x_76_);
lean_ctor_set(v___x_78_, 1, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___y_75_);
lean_ctor_set(v___x_79_, 1, v___x_78_);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v___y_74_);
lean_ctor_set(v___x_80_, 1, v___x_79_);
return v___x_80_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(lean_object* v_x_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___boxed(lean_object* v_x_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v_x_267_);
lean_dec_ref(v_x_267_);
return v_res_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1(lean_object* v___y_269_, lean_object* v_x_270_){
_start:
{
lean_inc(v___y_269_);
return v___y_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1___boxed(lean_object* v___y_271_, lean_object* v_x_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1(v___y_271_, v_x_272_);
lean_dec(v___y_271_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma(lean_object* v_twoH_317_, uint8_t v_debug_318_){
_start:
{
lean_object* v___y_320_; lean_object* v___y_321_; lean_object* v___y_332_; lean_object* v_snd_338_; lean_object* v_fst_339_; lean_object* v_fst_340_; lean_object* v_snd_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_585_; 
v_snd_338_ = lean_ctor_get(v_twoH_317_, 1);
lean_inc(v_snd_338_);
v_fst_339_ = lean_ctor_get(v_twoH_317_, 0);
v_fst_340_ = lean_ctor_get(v_snd_338_, 0);
v_snd_341_ = lean_ctor_get(v_snd_338_, 1);
v_isSharedCheck_585_ = !lean_is_exclusive(v_snd_338_);
if (v_isSharedCheck_585_ == 0)
{
v___x_343_ = v_snd_338_;
v_isShared_344_ = v_isSharedCheck_585_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_snd_341_);
lean_inc(v_fst_340_);
lean_dec(v_snd_338_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_585_;
goto v_resetjp_342_;
}
v___jp_319_:
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_322_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__1));
v___x_323_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_323_, 0, v___y_321_);
v___x_324_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_322_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
v___x_325_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__3));
v___x_326_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_324_);
lean_ctor_set(v___x_326_, 1, v___x_325_);
v___x_327_ = l_Std_Format_defWidth;
v___x_328_ = lean_unsigned_to_nat(0u);
v___x_329_ = l_Std_Format_pretty(v___x_326_, v___x_327_, v___x_328_, v___x_328_);
v___x_330_ = lean_dbg_trace(v___x_329_, v___y_320_);
return v___x_330_;
}
v___jp_331_:
{
if (v_debug_318_ == 0)
{
return v___y_332_;
}
else
{
lean_object* v___f_333_; lean_object* v_last_334_; lean_object* v___x_335_; uint8_t v___x_336_; 
lean_inc_n(v___y_332_, 2);
v___f_333_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1___boxed), 2, 1);
lean_closure_set(v___f_333_, 0, v___y_332_);
v_last_334_ = lp_mathlib_Lean_Name_lastComponentAsString(v___y_332_);
v___x_335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__4));
v___x_336_ = lean_string_dec_eq(v_last_334_, v___x_335_);
if (v___x_336_ == 0)
{
lean_dec(v___y_332_);
v___y_320_ = v___f_333_;
v___y_321_ = v_last_334_;
goto v___jp_319_;
}
else
{
lean_object* v___x_337_; 
lean_dec_ref(v_last_334_);
v___x_337_ = l_Lean_Name_toString(v___y_332_, v_debug_318_);
v___y_320_ = v___f_333_;
v___y_321_ = v___x_337_;
goto v___jp_319_;
}
}
}
v_resetjp_342_:
{
lean_object* v___y_346_; lean_object* v___y_349_; 
if (lean_obj_tag(v_fst_340_) == 1)
{
lean_object* v_pre_351_; 
v_pre_351_ = lean_ctor_get(v_fst_340_, 0);
lean_inc(v_pre_351_);
switch(lean_obj_tag(v_pre_351_))
{
case 1:
{
lean_object* v_pre_352_; 
v_pre_352_ = lean_ctor_get(v_pre_351_, 0);
lean_inc(v_pre_352_);
if (lean_obj_tag(v_pre_352_) == 0)
{
lean_object* v_str_353_; lean_object* v_str_354_; lean_object* v___x_355_; uint8_t v___x_356_; 
v_str_353_ = lean_ctor_get(v_fst_340_, 1);
lean_inc_ref(v_str_353_);
lean_dec_ref_known(v_fst_340_, 2);
v_str_354_ = lean_ctor_get(v_pre_351_, 1);
lean_inc_ref(v_str_354_);
lean_dec_ref_known(v_pre_351_, 2);
v___x_355_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_356_ = lean_string_dec_eq(v_str_354_, v___x_355_);
if (v___x_356_ == 0)
{
lean_object* v___x_357_; uint8_t v___x_358_; 
v___x_357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__29));
v___x_358_ = lean_string_dec_eq(v_str_354_, v___x_357_);
lean_dec_ref(v_str_354_);
if (v___x_358_ == 0)
{
lean_dec_ref(v_str_353_);
lean_del_object(v___x_343_);
lean_dec(v_snd_341_);
v___y_349_ = v_twoH_317_;
goto v___jp_348_;
}
else
{
lean_object* v___x_360_; uint8_t v_isShared_361_; uint8_t v_isSharedCheck_396_; 
lean_inc(v_fst_339_);
v_isSharedCheck_396_ = !lean_is_exclusive(v_twoH_317_);
if (v_isSharedCheck_396_ == 0)
{
lean_object* v_unused_397_; lean_object* v_unused_398_; 
v_unused_397_ = lean_ctor_get(v_twoH_317_, 1);
lean_dec(v_unused_397_);
v_unused_398_ = lean_ctor_get(v_twoH_317_, 0);
lean_dec(v_unused_398_);
v___x_360_ = v_twoH_317_;
v_isShared_361_ = v_isSharedCheck_396_;
goto v_resetjp_359_;
}
else
{
lean_dec(v_twoH_317_);
v___x_360_ = lean_box(0);
v_isShared_361_ = v_isSharedCheck_396_;
goto v_resetjp_359_;
}
v_resetjp_359_:
{
lean_object* v___x_362_; uint8_t v___x_363_; 
v___x_362_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__30));
v___x_363_ = lean_string_dec_eq(v_str_353_, v___x_362_);
if (v___x_363_ == 0)
{
lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_367_; 
v___x_364_ = l_Lean_Name_str___override(v_pre_352_, v___x_357_);
v___x_365_ = l_Lean_Name_str___override(v___x_364_, v_str_353_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_365_);
v___x_367_ = v___x_343_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_365_);
lean_ctor_set(v_reuseFailAlloc_372_, 1, v_snd_341_);
v___x_367_ = v_reuseFailAlloc_372_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
lean_object* v___x_369_; 
if (v_isShared_361_ == 0)
{
lean_ctor_set(v___x_360_, 1, v___x_367_);
v___x_369_ = v___x_360_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_371_, 1, v___x_367_);
v___x_369_ = v_reuseFailAlloc_371_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
lean_object* v___x_370_; 
v___x_370_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_369_);
lean_dec_ref(v___x_369_);
v___y_332_ = v___x_370_;
goto v___jp_331_;
}
}
}
else
{
lean_dec_ref(v_str_353_);
if (lean_obj_tag(v_snd_341_) == 1)
{
lean_object* v_tail_373_; 
v_tail_373_ = lean_ctor_get(v_snd_341_, 1);
if (lean_obj_tag(v_tail_373_) == 0)
{
lean_object* v_head_374_; uint8_t v___x_375_; 
lean_del_object(v___x_360_);
lean_del_object(v___x_343_);
lean_dec(v_fst_339_);
v_head_374_ = lean_ctor_get(v_snd_341_, 0);
lean_inc(v_head_374_);
lean_dec_ref_known(v_snd_341_, 2);
v___x_375_ = lean_unbox(v_head_374_);
lean_dec(v_head_374_);
if (v___x_375_ == 0)
{
lean_object* v___x_376_; 
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__6));
v___y_332_ = v___x_376_;
goto v___jp_331_;
}
else
{
lean_object* v___x_377_; 
v___x_377_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
v___y_332_ = v___x_377_;
goto v___jp_331_;
}
}
else
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_381_; 
v___x_378_ = l_Lean_Name_str___override(v_pre_352_, v___x_357_);
v___x_379_ = l_Lean_Name_str___override(v___x_378_, v___x_362_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_379_);
v___x_381_ = v___x_343_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v___x_379_);
lean_ctor_set(v_reuseFailAlloc_386_, 1, v_snd_341_);
v___x_381_ = v_reuseFailAlloc_386_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_383_; 
if (v_isShared_361_ == 0)
{
lean_ctor_set(v___x_360_, 1, v___x_381_);
v___x_383_ = v___x_360_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_385_, 1, v___x_381_);
v___x_383_ = v_reuseFailAlloc_385_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_383_);
lean_dec_ref(v___x_383_);
v___y_332_ = v___x_384_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_390_; 
v___x_387_ = l_Lean_Name_str___override(v_pre_352_, v___x_357_);
v___x_388_ = l_Lean_Name_str___override(v___x_387_, v___x_362_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_388_);
v___x_390_ = v___x_343_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v___x_388_);
lean_ctor_set(v_reuseFailAlloc_395_, 1, v_snd_341_);
v___x_390_ = v_reuseFailAlloc_395_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
lean_object* v___x_392_; 
if (v_isShared_361_ == 0)
{
lean_ctor_set(v___x_360_, 1, v___x_390_);
v___x_392_ = v___x_360_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_394_, 1, v___x_390_);
v___x_392_ = v_reuseFailAlloc_394_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_392_);
lean_dec_ref(v___x_392_);
v___y_332_ = v___x_393_;
goto v___jp_331_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_400_; uint8_t v_isShared_401_; uint8_t v_isSharedCheck_436_; 
lean_inc(v_fst_339_);
lean_dec_ref(v_str_354_);
v_isSharedCheck_436_ = !lean_is_exclusive(v_twoH_317_);
if (v_isSharedCheck_436_ == 0)
{
lean_object* v_unused_437_; lean_object* v_unused_438_; 
v_unused_437_ = lean_ctor_get(v_twoH_317_, 1);
lean_dec(v_unused_437_);
v_unused_438_ = lean_ctor_get(v_twoH_317_, 0);
lean_dec(v_unused_438_);
v___x_400_ = v_twoH_317_;
v_isShared_401_ = v_isSharedCheck_436_;
goto v_resetjp_399_;
}
else
{
lean_dec(v_twoH_317_);
v___x_400_ = lean_box(0);
v_isShared_401_ = v_isSharedCheck_436_;
goto v_resetjp_399_;
}
v_resetjp_399_:
{
lean_object* v___x_402_; uint8_t v___x_403_; 
v___x_402_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_403_ = lean_string_dec_eq(v_str_353_, v___x_402_);
if (v___x_403_ == 0)
{
lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_407_; 
v___x_404_ = l_Lean_Name_str___override(v_pre_352_, v___x_355_);
v___x_405_ = l_Lean_Name_str___override(v___x_404_, v_str_353_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_405_);
v___x_407_ = v___x_343_;
goto v_reusejp_406_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_405_);
lean_ctor_set(v_reuseFailAlloc_412_, 1, v_snd_341_);
v___x_407_ = v_reuseFailAlloc_412_;
goto v_reusejp_406_;
}
v_reusejp_406_:
{
lean_object* v___x_409_; 
if (v_isShared_401_ == 0)
{
lean_ctor_set(v___x_400_, 1, v___x_407_);
v___x_409_ = v___x_400_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_411_, 1, v___x_407_);
v___x_409_ = v_reuseFailAlloc_411_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
lean_object* v___x_410_; 
v___x_410_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_409_);
lean_dec_ref(v___x_409_);
v___y_332_ = v___x_410_;
goto v___jp_331_;
}
}
}
else
{
lean_dec_ref(v_str_353_);
if (lean_obj_tag(v_snd_341_) == 1)
{
lean_object* v_tail_413_; 
v_tail_413_ = lean_ctor_get(v_snd_341_, 1);
if (lean_obj_tag(v_tail_413_) == 0)
{
lean_object* v_head_414_; uint8_t v___x_415_; 
lean_del_object(v___x_400_);
lean_del_object(v___x_343_);
lean_dec(v_fst_339_);
v_head_414_ = lean_ctor_get(v_snd_341_, 0);
lean_inc(v_head_414_);
lean_dec_ref_known(v_snd_341_, 2);
v___x_415_ = lean_unbox(v_head_414_);
lean_dec(v_head_414_);
if (v___x_415_ == 0)
{
lean_object* v___x_416_; 
v___x_416_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__8));
v___y_332_ = v___x_416_;
goto v___jp_331_;
}
else
{
lean_object* v___x_417_; 
v___x_417_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
v___y_332_ = v___x_417_;
goto v___jp_331_;
}
}
else
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_421_; 
v___x_418_ = l_Lean_Name_str___override(v_pre_352_, v___x_355_);
v___x_419_ = l_Lean_Name_str___override(v___x_418_, v___x_402_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_419_);
v___x_421_ = v___x_343_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v___x_419_);
lean_ctor_set(v_reuseFailAlloc_426_, 1, v_snd_341_);
v___x_421_ = v_reuseFailAlloc_426_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
lean_object* v___x_423_; 
if (v_isShared_401_ == 0)
{
lean_ctor_set(v___x_400_, 1, v___x_421_);
v___x_423_ = v___x_400_;
goto v_reusejp_422_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_425_, 1, v___x_421_);
v___x_423_ = v_reuseFailAlloc_425_;
goto v_reusejp_422_;
}
v_reusejp_422_:
{
lean_object* v___x_424_; 
v___x_424_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_423_);
lean_dec_ref(v___x_423_);
v___y_332_ = v___x_424_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_430_; 
v___x_427_ = l_Lean_Name_str___override(v_pre_352_, v___x_355_);
v___x_428_ = l_Lean_Name_str___override(v___x_427_, v___x_402_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_428_);
v___x_430_ = v___x_343_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v___x_428_);
lean_ctor_set(v_reuseFailAlloc_435_, 1, v_snd_341_);
v___x_430_ = v_reuseFailAlloc_435_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
lean_object* v___x_432_; 
if (v_isShared_401_ == 0)
{
lean_ctor_set(v___x_400_, 1, v___x_430_);
v___x_432_ = v___x_400_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_434_, 1, v___x_430_);
v___x_432_ = v_reuseFailAlloc_434_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
lean_object* v___x_433_; 
v___x_433_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_432_);
lean_dec_ref(v___x_432_);
v___y_332_ = v___x_433_;
goto v___jp_331_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_439_; 
lean_dec_ref_known(v_pre_351_, 2);
lean_dec(v_pre_352_);
lean_dec_ref_known(v_fst_340_, 2);
lean_del_object(v___x_343_);
lean_dec(v_snd_341_);
v___x_439_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v_twoH_317_);
lean_dec_ref(v_twoH_317_);
v___y_332_ = v___x_439_;
goto v___jp_331_;
}
}
case 0:
{
lean_object* v_str_440_; lean_object* v___x_441_; uint8_t v___x_442_; 
v_str_440_ = lean_ctor_get(v_fst_340_, 1);
lean_inc_ref(v_str_440_);
lean_dec_ref_known(v_fst_340_, 2);
v___x_441_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27));
v___x_442_ = lean_string_dec_eq(v_str_440_, v___x_441_);
lean_dec_ref(v_str_440_);
if (v___x_442_ == 0)
{
lean_del_object(v___x_343_);
lean_dec(v_snd_341_);
v___y_349_ = v_twoH_317_;
goto v___jp_348_;
}
else
{
lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_580_; 
lean_inc(v_fst_339_);
v_isSharedCheck_580_ = !lean_is_exclusive(v_twoH_317_);
if (v_isSharedCheck_580_ == 0)
{
lean_object* v_unused_581_; lean_object* v_unused_582_; 
v_unused_581_ = lean_ctor_get(v_twoH_317_, 1);
lean_dec(v_unused_581_);
v_unused_582_ = lean_ctor_get(v_twoH_317_, 0);
lean_dec(v_unused_582_);
v___x_444_ = v_twoH_317_;
v_isShared_445_ = v_isSharedCheck_580_;
goto v_resetjp_443_;
}
else
{
lean_dec(v_twoH_317_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_580_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
if (lean_obj_tag(v_snd_341_) == 1)
{
lean_object* v_tail_446_; 
v_tail_446_ = lean_ctor_get(v_snd_341_, 1);
if (lean_obj_tag(v_tail_446_) == 0)
{
if (lean_obj_tag(v_fst_339_) == 1)
{
lean_object* v_pre_447_; 
v_pre_447_ = lean_ctor_get(v_fst_339_, 0);
if (lean_obj_tag(v_pre_447_) == 1)
{
lean_object* v_pre_448_; 
v_pre_448_ = lean_ctor_get(v_pre_447_, 0);
if (lean_obj_tag(v_pre_448_) == 0)
{
lean_object* v_head_449_; lean_object* v_str_450_; lean_object* v_str_451_; lean_object* v___x_452_; uint8_t v___x_453_; 
v_head_449_ = lean_ctor_get(v_snd_341_, 0);
v_str_450_ = lean_ctor_get(v_fst_339_, 1);
v_str_451_ = lean_ctor_get(v_pre_447_, 1);
v___x_452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7));
v___x_453_ = lean_string_dec_eq(v_str_451_, v___x_452_);
if (v___x_453_ == 0)
{
lean_object* v___x_454_; lean_object* v___x_456_; 
v___x_454_ = l_Lean_Name_str___override(v_pre_448_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_454_);
v___x_456_ = v___x_343_;
goto v_reusejp_455_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v___x_454_);
lean_ctor_set(v_reuseFailAlloc_460_, 1, v_snd_341_);
v___x_456_ = v_reuseFailAlloc_460_;
goto v_reusejp_455_;
}
v_reusejp_455_:
{
lean_object* v___x_458_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_456_);
v___x_458_ = v___x_444_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v___x_456_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
v___y_346_ = v___x_458_;
goto v___jp_345_;
}
}
}
else
{
lean_object* v___x_461_; uint8_t v___x_462_; 
lean_inc_ref(v_str_450_);
lean_inc(v_pre_448_);
lean_dec_ref_known(v_fst_339_, 2);
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__25));
v___x_462_ = lean_string_dec_eq(v_str_450_, v___x_461_);
if (v___x_462_ == 0)
{
lean_object* v___x_463_; uint8_t v___x_464_; 
v___x_463_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__26));
v___x_464_ = lean_string_dec_eq(v_str_450_, v___x_463_);
if (v___x_464_ == 0)
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_469_; 
v___x_465_ = l_Lean_Name_str___override(v_pre_448_, v___x_452_);
v___x_466_ = l_Lean_Name_str___override(v___x_465_, v_str_450_);
v___x_467_ = l_Lean_Name_str___override(v_pre_448_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_467_);
v___x_469_ = v___x_343_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v___x_467_);
lean_ctor_set(v_reuseFailAlloc_474_, 1, v_snd_341_);
v___x_469_ = v_reuseFailAlloc_474_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
lean_object* v___x_471_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_469_);
lean_ctor_set(v___x_444_, 0, v___x_466_);
v___x_471_ = v___x_444_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v___x_466_);
lean_ctor_set(v_reuseFailAlloc_473_, 1, v___x_469_);
v___x_471_ = v_reuseFailAlloc_473_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
lean_object* v___x_472_; 
v___x_472_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_471_);
lean_dec_ref(v___x_471_);
v___y_332_ = v___x_472_;
goto v___jp_331_;
}
}
}
else
{
uint8_t v___x_475_; 
lean_inc(v_head_449_);
lean_dec_ref(v_str_450_);
lean_dec_ref_known(v_snd_341_, 2);
lean_del_object(v___x_444_);
lean_del_object(v___x_343_);
v___x_475_ = lean_unbox(v_head_449_);
lean_dec(v_head_449_);
if (v___x_475_ == 0)
{
lean_object* v___x_476_; 
v___x_476_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__13));
v___y_332_ = v___x_476_;
goto v___jp_331_;
}
else
{
lean_object* v___x_477_; 
v___x_477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
v___y_332_ = v___x_477_;
goto v___jp_331_;
}
}
}
else
{
uint8_t v___x_478_; 
lean_inc(v_head_449_);
lean_dec_ref(v_str_450_);
lean_dec_ref_known(v_snd_341_, 2);
lean_del_object(v___x_444_);
lean_del_object(v___x_343_);
v___x_478_ = lean_unbox(v_head_449_);
lean_dec(v_head_449_);
if (v___x_478_ == 0)
{
lean_object* v___x_479_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__15));
v___y_332_ = v___x_479_;
goto v___jp_331_;
}
else
{
lean_object* v___x_480_; 
v___x_480_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
v___y_332_ = v___x_480_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_481_; lean_object* v___x_483_; 
v___x_481_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_481_);
v___x_483_ = v___x_343_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v___x_481_);
lean_ctor_set(v_reuseFailAlloc_488_, 1, v_snd_341_);
v___x_483_ = v_reuseFailAlloc_488_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_485_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_483_);
v___x_485_ = v___x_444_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_487_, 1, v___x_483_);
v___x_485_ = v_reuseFailAlloc_487_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_485_);
lean_dec_ref(v___x_485_);
v___y_332_ = v___x_486_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_489_; lean_object* v___x_491_; 
v___x_489_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_489_);
v___x_491_ = v___x_343_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v___x_489_);
lean_ctor_set(v_reuseFailAlloc_496_, 1, v_snd_341_);
v___x_491_ = v_reuseFailAlloc_496_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
lean_object* v___x_493_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_491_);
v___x_493_ = v___x_444_;
goto v_reusejp_492_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_495_, 1, v___x_491_);
v___x_493_ = v_reuseFailAlloc_495_;
goto v_reusejp_492_;
}
v_reusejp_492_:
{
lean_object* v___x_494_; 
v___x_494_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_493_);
lean_dec_ref(v___x_493_);
v___y_332_ = v___x_494_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_497_; lean_object* v___x_499_; 
v___x_497_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_497_);
v___x_499_ = v___x_343_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v___x_497_);
lean_ctor_set(v_reuseFailAlloc_504_, 1, v_snd_341_);
v___x_499_ = v_reuseFailAlloc_504_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
lean_object* v___x_501_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_499_);
v___x_501_ = v___x_444_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_503_, 1, v___x_499_);
v___x_501_ = v_reuseFailAlloc_503_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
lean_object* v___x_502_; 
v___x_502_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_501_);
lean_dec_ref(v___x_501_);
v___y_332_ = v___x_502_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v_tail_505_; 
v_tail_505_ = lean_ctor_get(v_tail_446_, 1);
if (lean_obj_tag(v_tail_505_) == 0)
{
if (lean_obj_tag(v_fst_339_) == 1)
{
lean_object* v_pre_506_; 
v_pre_506_ = lean_ctor_get(v_fst_339_, 0);
if (lean_obj_tag(v_pre_506_) == 1)
{
lean_object* v_pre_507_; 
v_pre_507_ = lean_ctor_get(v_pre_506_, 0);
if (lean_obj_tag(v_pre_507_) == 0)
{
lean_object* v_head_508_; lean_object* v_head_509_; lean_object* v_str_510_; lean_object* v_str_511_; lean_object* v___x_512_; uint8_t v___x_513_; 
v_head_508_ = lean_ctor_get(v_snd_341_, 0);
v_head_509_ = lean_ctor_get(v_tail_446_, 0);
v_str_510_ = lean_ctor_get(v_fst_339_, 1);
v_str_511_ = lean_ctor_get(v_pre_506_, 1);
v___x_512_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7));
v___x_513_ = lean_string_dec_eq(v_str_511_, v___x_512_);
if (v___x_513_ == 0)
{
lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_514_ = l_Lean_Name_str___override(v_pre_507_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_514_);
v___x_516_ = v___x_343_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v___x_514_);
lean_ctor_set(v_reuseFailAlloc_520_, 1, v_snd_341_);
v___x_516_ = v_reuseFailAlloc_520_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
lean_object* v___x_518_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_516_);
v___x_518_ = v___x_444_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v___x_516_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
v___y_346_ = v___x_518_;
goto v___jp_345_;
}
}
}
else
{
lean_object* v___x_521_; uint8_t v___x_522_; 
lean_inc_ref(v_str_510_);
lean_inc(v_pre_507_);
lean_dec_ref_known(v_fst_339_, 2);
v___x_521_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23));
v___x_522_ = lean_string_dec_eq(v_str_510_, v___x_521_);
if (v___x_522_ == 0)
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_527_; 
v___x_523_ = l_Lean_Name_str___override(v_pre_507_, v___x_512_);
v___x_524_ = l_Lean_Name_str___override(v___x_523_, v_str_510_);
v___x_525_ = l_Lean_Name_str___override(v_pre_507_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_525_);
v___x_527_ = v___x_343_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v___x_525_);
lean_ctor_set(v_reuseFailAlloc_532_, 1, v_snd_341_);
v___x_527_ = v_reuseFailAlloc_532_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
lean_object* v___x_529_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_527_);
lean_ctor_set(v___x_444_, 0, v___x_524_);
v___x_529_ = v___x_444_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_531_; 
v_reuseFailAlloc_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_531_, 0, v___x_524_);
lean_ctor_set(v_reuseFailAlloc_531_, 1, v___x_527_);
v___x_529_ = v_reuseFailAlloc_531_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
lean_object* v___x_530_; 
v___x_530_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_529_);
lean_dec_ref(v___x_529_);
v___y_332_ = v___x_530_;
goto v___jp_331_;
}
}
}
else
{
uint8_t v___x_533_; 
lean_inc(v_head_509_);
lean_inc(v_head_508_);
lean_dec_ref(v_str_510_);
lean_dec_ref_known(v_snd_341_, 2);
lean_del_object(v___x_444_);
lean_del_object(v___x_343_);
v___x_533_ = lean_unbox(v_head_508_);
lean_dec(v_head_508_);
if (v___x_533_ == 0)
{
uint8_t v___x_534_; 
v___x_534_ = lean_unbox(v_head_509_);
lean_dec(v_head_509_);
if (v___x_534_ == 0)
{
lean_object* v___x_535_; 
v___x_535_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__17));
v___y_332_ = v___x_535_;
goto v___jp_331_;
}
else
{
lean_object* v___x_536_; 
v___x_536_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__18));
v___y_332_ = v___x_536_;
goto v___jp_331_;
}
}
else
{
uint8_t v___x_537_; 
v___x_537_ = lean_unbox(v_head_509_);
lean_dec(v_head_509_);
if (v___x_537_ == 0)
{
lean_object* v___x_538_; 
v___x_538_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__20));
v___y_332_ = v___x_538_;
goto v___jp_331_;
}
else
{
lean_object* v___x_539_; 
v___x_539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
v___y_332_ = v___x_539_;
goto v___jp_331_;
}
}
}
}
}
else
{
lean_object* v___x_540_; lean_object* v___x_542_; 
v___x_540_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_540_);
v___x_542_ = v___x_343_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_540_);
lean_ctor_set(v_reuseFailAlloc_547_, 1, v_snd_341_);
v___x_542_ = v_reuseFailAlloc_547_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
lean_object* v___x_544_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_542_);
v___x_544_ = v___x_444_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_546_, 1, v___x_542_);
v___x_544_ = v_reuseFailAlloc_546_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
lean_object* v___x_545_; 
v___x_545_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_544_);
lean_dec_ref(v___x_544_);
v___y_332_ = v___x_545_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_548_; lean_object* v___x_550_; 
v___x_548_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_548_);
v___x_550_ = v___x_343_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v___x_548_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v_snd_341_);
v___x_550_ = v_reuseFailAlloc_555_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
lean_object* v___x_552_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_550_);
v___x_552_ = v___x_444_;
goto v_reusejp_551_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_554_, 1, v___x_550_);
v___x_552_ = v_reuseFailAlloc_554_;
goto v_reusejp_551_;
}
v_reusejp_551_:
{
lean_object* v___x_553_; 
v___x_553_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_552_);
lean_dec_ref(v___x_552_);
v___y_332_ = v___x_553_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_556_; lean_object* v___x_558_; 
v___x_556_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_556_);
v___x_558_ = v___x_343_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_556_);
lean_ctor_set(v_reuseFailAlloc_563_, 1, v_snd_341_);
v___x_558_ = v_reuseFailAlloc_563_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
lean_object* v___x_560_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_558_);
v___x_560_ = v___x_444_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_562_; 
v_reuseFailAlloc_562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_562_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_562_, 1, v___x_558_);
v___x_560_ = v_reuseFailAlloc_562_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
lean_object* v___x_561_; 
v___x_561_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_560_);
lean_dec_ref(v___x_560_);
v___y_332_ = v___x_561_;
goto v___jp_331_;
}
}
}
}
else
{
lean_object* v___x_564_; lean_object* v___x_566_; 
v___x_564_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_564_);
v___x_566_ = v___x_343_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v___x_564_);
lean_ctor_set(v_reuseFailAlloc_571_, 1, v_snd_341_);
v___x_566_ = v_reuseFailAlloc_571_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
lean_object* v___x_568_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_566_);
v___x_568_ = v___x_444_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_570_, 1, v___x_566_);
v___x_568_ = v_reuseFailAlloc_570_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
lean_object* v___x_569_; 
v___x_569_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_568_);
lean_dec_ref(v___x_568_);
v___y_332_ = v___x_569_;
goto v___jp_331_;
}
}
}
}
}
else
{
lean_object* v___x_572_; lean_object* v___x_574_; 
v___x_572_ = l_Lean_Name_str___override(v_pre_351_, v___x_441_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 0, v___x_572_);
v___x_574_ = v___x_343_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v___x_572_);
lean_ctor_set(v_reuseFailAlloc_579_, 1, v_snd_341_);
v___x_574_ = v_reuseFailAlloc_579_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
lean_object* v___x_576_; 
if (v_isShared_445_ == 0)
{
lean_ctor_set(v___x_444_, 1, v___x_574_);
v___x_576_ = v___x_444_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v_fst_339_);
lean_ctor_set(v_reuseFailAlloc_578_, 1, v___x_574_);
v___x_576_ = v_reuseFailAlloc_578_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
lean_object* v___x_577_; 
v___x_577_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___x_576_);
lean_dec_ref(v___x_576_);
v___y_332_ = v___x_577_;
goto v___jp_331_;
}
}
}
}
}
}
default: 
{
lean_object* v___x_583_; 
lean_dec_ref_known(v_fst_340_, 2);
lean_dec(v_pre_351_);
lean_del_object(v___x_343_);
lean_dec(v_snd_341_);
v___x_583_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v_twoH_317_);
lean_dec_ref(v_twoH_317_);
v___y_332_ = v___x_583_;
goto v___jp_331_;
}
}
}
else
{
lean_object* v___x_584_; 
lean_del_object(v___x_343_);
lean_dec(v_snd_341_);
lean_dec(v_fst_340_);
v___x_584_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v_twoH_317_);
lean_dec_ref(v_twoH_317_);
v___y_332_ = v___x_584_;
goto v___jp_331_;
}
v___jp_345_:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___y_346_);
lean_dec_ref(v___y_346_);
v___y_332_ = v___x_347_;
goto v___jp_331_;
}
v___jp_348_:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0(v___y_349_);
lean_dec_ref(v___y_349_);
v___y_332_ = v___x_350_;
goto v___jp_331_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___boxed(lean_object* v_twoH_586_, lean_object* v_debug_587_){
_start:
{
uint8_t v_debug_boxed_588_; lean_object* v_res_589_; 
v_debug_boxed_588_ = lean_unbox(v_debug_587_);
v_res_589_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma(v_twoH_586_, v_debug_boxed_588_);
return v_res_589_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__1(uint8_t v___y_590_, uint8_t v___y_591_){
_start:
{
if (v___y_590_ == 0)
{
if (v___y_591_ == 0)
{
uint8_t v___x_592_; 
v___x_592_ = 1;
return v___x_592_;
}
else
{
return v___y_590_;
}
}
else
{
return v___y_591_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__1___boxed(lean_object* v___y_593_, lean_object* v___y_594_){
_start:
{
uint8_t v___y_2293__boxed_595_; uint8_t v___y_2294__boxed_596_; uint8_t v_res_597_; lean_object* v_r_598_; 
v___y_2293__boxed_595_ = lean_unbox(v___y_593_);
v___y_2294__boxed_596_ = lean_unbox(v___y_594_);
v_res_597_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__1(v___y_2293__boxed_595_, v___y_2294__boxed_596_);
v_r_598_ = lean_box(v_res_597_);
return v_r_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(lean_object* v_x_602_, lean_object* v_x_603_){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1));
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___boxed(lean_object* v_x_605_, lean_object* v_x_606_){
_start:
{
lean_object* v_res_607_; 
v_res_607_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_x_605_, v_x_606_);
lean_dec(v_x_606_);
lean_dec(v_x_605_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(lean_object* v_x_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__1));
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___boxed(lean_object* v_x_613_){
_start:
{
lean_object* v_res_614_; 
v_res_614_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v_x_613_);
lean_dec(v_x_613_);
return v_res_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma(lean_object* v_twoH_829_, uint8_t v_debug_830_){
_start:
{
lean_object* v___y_834_; lean_object* v___y_835_; lean_object* v_fst_846_; 
v_fst_846_ = lean_ctor_get(v_twoH_829_, 0);
lean_inc(v_fst_846_);
if (lean_obj_tag(v_fst_846_) == 0)
{
lean_dec_ref(v_twoH_829_);
goto v___jp_831_;
}
else
{
lean_object* v_snd_847_; lean_object* v___x_849_; uint8_t v_isShared_850_; uint8_t v_isSharedCheck_1211_; 
v_snd_847_ = lean_ctor_get(v_twoH_829_, 1);
v_isSharedCheck_1211_ = !lean_is_exclusive(v_twoH_829_);
if (v_isSharedCheck_1211_ == 0)
{
lean_object* v_unused_1212_; 
v_unused_1212_ = lean_ctor_get(v_twoH_829_, 0);
lean_dec(v_unused_1212_);
v___x_849_ = v_twoH_829_;
v_isShared_850_ = v_isSharedCheck_1211_;
goto v_resetjp_848_;
}
else
{
lean_inc(v_snd_847_);
lean_dec(v_twoH_829_);
v___x_849_ = lean_box(0);
v_isShared_850_ = v_isSharedCheck_1211_;
goto v_resetjp_848_;
}
v_resetjp_848_:
{
lean_object* v_fst_851_; 
v_fst_851_ = lean_ctor_get(v_snd_847_, 0);
lean_inc(v_fst_851_);
if (lean_obj_tag(v_fst_851_) == 0)
{
lean_del_object(v___x_849_);
lean_dec(v_snd_847_);
lean_dec(v_fst_846_);
goto v___jp_831_;
}
else
{
lean_object* v_snd_852_; lean_object* v___x_854_; uint8_t v_isShared_855_; uint8_t v_isSharedCheck_1209_; 
v_snd_852_ = lean_ctor_get(v_snd_847_, 1);
v_isSharedCheck_1209_ = !lean_is_exclusive(v_snd_847_);
if (v_isSharedCheck_1209_ == 0)
{
lean_object* v_unused_1210_; 
v_unused_1210_ = lean_ctor_get(v_snd_847_, 0);
lean_dec(v_unused_1210_);
v___x_854_ = v_snd_847_;
v_isShared_855_ = v_isSharedCheck_1209_;
goto v_resetjp_853_;
}
else
{
lean_inc(v_snd_852_);
lean_dec(v_snd_847_);
v___x_854_ = lean_box(0);
v_isShared_855_ = v_isSharedCheck_1209_;
goto v_resetjp_853_;
}
v_resetjp_853_:
{
lean_object* v_fst_856_; lean_object* v_snd_857_; lean_object* v___x_859_; uint8_t v_isShared_860_; uint8_t v_isSharedCheck_1208_; 
v_fst_856_ = lean_ctor_get(v_snd_852_, 0);
v_snd_857_ = lean_ctor_get(v_snd_852_, 1);
v_isSharedCheck_1208_ = !lean_is_exclusive(v_snd_852_);
if (v_isSharedCheck_1208_ == 0)
{
v___x_859_ = v_snd_852_;
v_isShared_860_ = v_isSharedCheck_1208_;
goto v_resetjp_858_;
}
else
{
lean_inc(v_snd_857_);
lean_inc(v_fst_856_);
lean_dec(v_snd_852_);
v___x_859_ = lean_box(0);
v_isShared_860_ = v_isSharedCheck_1208_;
goto v_resetjp_858_;
}
v_resetjp_858_:
{
lean_object* v___y_862_; lean_object* v___y_863_; lean_object* v___y_864_; lean_object* v___y_867_; lean_object* v_natDegLE_868_; lean_object* v_degLE_869_; lean_object* v_coeff_870_; lean_object* v___y_1064_; lean_object* v___y_1069_; lean_object* v___y_1073_; lean_object* v___x_1077_; lean_object* v___y_1079_; 
v___x_1077_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__16));
if (lean_obj_tag(v_fst_856_) == 0)
{
lean_object* v_val_1192_; lean_object* v___x_1193_; uint8_t v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v_val_1192_ = lean_ctor_get(v_fst_856_, 0);
v___x_1193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__104));
v___x_1194_ = 1;
lean_inc(v_val_1192_);
v___x_1195_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1192_, v___x_1194_);
v___x_1196_ = l_addParenHeuristic(v___x_1195_);
v___x_1197_ = lean_string_append(v___x_1193_, v___x_1196_);
lean_dec_ref(v___x_1196_);
v___x_1198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105));
v___x_1199_ = lean_string_append(v___x_1197_, v___x_1198_);
v___y_1079_ = v___x_1199_;
goto v___jp_1078_;
}
else
{
lean_object* v_val_1200_; lean_object* v___x_1201_; uint8_t v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; 
v_val_1200_ = lean_ctor_get(v_fst_856_, 0);
v___x_1201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__106));
v___x_1202_ = 1;
lean_inc(v_val_1200_);
v___x_1203_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1200_, v___x_1202_);
v___x_1204_ = l_addParenHeuristic(v___x_1203_);
v___x_1205_ = lean_string_append(v___x_1201_, v___x_1204_);
lean_dec_ref(v___x_1204_);
v___x_1206_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105));
v___x_1207_ = lean_string_append(v___x_1205_, v___x_1206_);
v___y_1079_ = v___x_1207_;
goto v___jp_1078_;
}
v___jp_861_:
{
lean_object* v___x_865_; 
v___x_865_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___y_863_, v___y_864_);
lean_dec(v___y_864_);
lean_dec(v___y_863_);
v___y_834_ = v___y_862_;
v___y_835_ = v___x_865_;
goto v___jp_833_;
}
v___jp_866_:
{
if (lean_obj_tag(v_fst_846_) == 1)
{
lean_object* v_pre_871_; 
v_pre_871_ = lean_ctor_get(v_fst_846_, 0);
if (lean_obj_tag(v_pre_871_) == 1)
{
lean_object* v_pre_872_; 
v_pre_872_ = lean_ctor_get(v_pre_871_, 0);
if (lean_obj_tag(v_pre_872_) == 0)
{
lean_object* v_str_873_; lean_object* v_str_874_; lean_object* v___x_875_; uint8_t v___x_876_; 
v_str_873_ = lean_ctor_get(v_fst_846_, 1);
v_str_874_ = lean_ctor_get(v_pre_871_, 1);
v___x_875_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7));
v___x_876_ = lean_string_dec_eq(v_str_874_, v___x_875_);
if (v___x_876_ == 0)
{
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_877_; 
v_pre_877_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_877_) == 1)
{
lean_object* v_pre_878_; 
v_pre_878_ = lean_ctor_get(v_pre_877_, 0);
if (lean_obj_tag(v_pre_878_) == 0)
{
lean_object* v_str_879_; lean_object* v_str_880_; lean_object* v___x_881_; uint8_t v___x_882_; 
lean_inc_ref(v_str_874_);
lean_inc_ref(v_str_873_);
lean_dec_ref_known(v_fst_846_, 2);
v_str_879_ = lean_ctor_get(v_fst_851_, 1);
v_str_880_ = lean_ctor_get(v_pre_877_, 1);
v___x_881_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_882_ = lean_string_dec_eq(v_str_880_, v___x_881_);
if (v___x_882_ == 0)
{
lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; 
v___x_883_ = l_Lean_Name_str___override(v_pre_878_, v_str_874_);
v___x_884_ = l_Lean_Name_str___override(v___x_883_, v_str_873_);
v___x_885_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_884_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_884_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_885_;
goto v___jp_833_;
}
else
{
lean_object* v___x_886_; uint8_t v___x_887_; 
lean_inc(v_pre_878_);
lean_inc_ref(v_str_879_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_886_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_887_ = lean_string_dec_eq(v_str_879_, v___x_886_);
if (v___x_887_ == 0)
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_888_ = l_Lean_Name_str___override(v_pre_878_, v_str_874_);
v___x_889_ = l_Lean_Name_str___override(v___x_888_, v_str_873_);
v___x_890_ = l_Lean_Name_str___override(v_pre_878_, v___x_881_);
v___x_891_ = l_Lean_Name_str___override(v___x_890_, v_str_879_);
v___x_892_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_889_, v___x_891_);
lean_dec(v___x_891_);
lean_dec(v___x_889_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_892_;
goto v___jp_833_;
}
else
{
lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; 
lean_dec_ref(v_str_879_);
v___x_893_ = l_Lean_Name_str___override(v_pre_878_, v_str_874_);
v___x_894_ = l_Lean_Name_str___override(v___x_893_, v_str_873_);
v___x_895_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v___x_894_);
lean_dec(v___x_894_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_895_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_896_; 
v___x_896_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_896_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_897_; 
v___x_897_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_897_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_898_; 
v___x_898_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_898_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_899_; uint8_t v___x_900_; 
lean_inc_ref(v_str_873_);
lean_inc(v_pre_872_);
lean_dec_ref_known(v_fst_846_, 2);
v___x_899_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__25));
v___x_900_ = lean_string_dec_eq(v_str_873_, v___x_899_);
if (v___x_900_ == 0)
{
lean_object* v___x_901_; uint8_t v___x_902_; 
v___x_901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__26));
v___x_902_ = lean_string_dec_eq(v_str_873_, v___x_901_);
if (v___x_902_ == 0)
{
lean_object* v___x_903_; uint8_t v___x_904_; 
v___x_903_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__23));
v___x_904_ = lean_string_dec_eq(v_str_873_, v___x_903_);
if (v___x_904_ == 0)
{
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_905_; 
v_pre_905_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_905_) == 1)
{
lean_object* v_pre_906_; 
v_pre_906_ = lean_ctor_get(v_pre_905_, 0);
if (lean_obj_tag(v_pre_906_) == 0)
{
lean_object* v_str_907_; lean_object* v_str_908_; lean_object* v___x_909_; uint8_t v___x_910_; 
v_str_907_ = lean_ctor_get(v_fst_851_, 1);
v_str_908_ = lean_ctor_get(v_pre_905_, 1);
v___x_909_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_910_ = lean_string_dec_eq(v_str_908_, v___x_909_);
if (v___x_910_ == 0)
{
lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; 
v___x_911_ = l_Lean_Name_str___override(v_pre_906_, v___x_875_);
v___x_912_ = l_Lean_Name_str___override(v___x_911_, v_str_873_);
v___x_913_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_912_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_912_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_913_;
goto v___jp_833_;
}
else
{
lean_object* v___x_914_; uint8_t v___x_915_; 
lean_inc_ref(v_str_907_);
lean_inc(v_pre_906_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_915_ = lean_string_dec_eq(v_str_907_, v___x_914_);
if (v___x_915_ == 0)
{
lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; 
v___x_916_ = l_Lean_Name_str___override(v_pre_906_, v___x_875_);
v___x_917_ = l_Lean_Name_str___override(v___x_916_, v_str_873_);
v___x_918_ = l_Lean_Name_str___override(v_pre_906_, v___x_909_);
v___x_919_ = l_Lean_Name_str___override(v___x_918_, v_str_907_);
v___x_920_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_917_, v___x_919_);
lean_dec(v___x_919_);
lean_dec(v___x_917_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_920_;
goto v___jp_833_;
}
else
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
lean_dec_ref(v_str_907_);
v___x_921_ = l_Lean_Name_str___override(v_pre_906_, v___x_875_);
v___x_922_ = l_Lean_Name_str___override(v___x_921_, v_str_873_);
v___x_923_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v___x_922_);
lean_dec(v___x_922_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_923_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; 
v___x_924_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_925_ = l_Lean_Name_str___override(v___x_924_, v_str_873_);
v___x_926_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_925_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_925_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_926_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; 
v___x_927_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_928_ = l_Lean_Name_str___override(v___x_927_, v_str_873_);
v___x_929_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_928_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_928_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_929_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; 
v___x_930_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_931_ = l_Lean_Name_str___override(v___x_930_, v_str_873_);
v___x_932_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_931_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec(v___x_931_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_932_;
goto v___jp_833_;
}
}
else
{
lean_dec_ref(v_str_873_);
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_933_; 
v_pre_933_ = lean_ctor_get(v_fst_851_, 0);
switch(lean_obj_tag(v_pre_933_))
{
case 0:
{
lean_object* v_str_934_; lean_object* v___x_935_; uint8_t v___x_936_; 
v_str_934_ = lean_ctor_get(v_fst_851_, 1);
v___x_935_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__27));
v___x_936_ = lean_string_dec_eq(v_str_934_, v___x_935_);
if (v___x_936_ == 0)
{
lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_937_ = l_Lean_Name_str___override(v_pre_933_, v___x_875_);
v___x_938_ = l_Lean_Name_str___override(v___x_937_, v___x_903_);
v___y_862_ = v___y_867_;
v___y_863_ = v___x_938_;
v___y_864_ = v_fst_851_;
goto v___jp_861_;
}
else
{
lean_dec_ref_known(v_fst_851_, 2);
lean_inc(v_coeff_870_);
v___y_834_ = v___y_867_;
v___y_835_ = v_coeff_870_;
goto v___jp_833_;
}
}
case 1:
{
lean_object* v_pre_939_; 
v_pre_939_ = lean_ctor_get(v_pre_933_, 0);
if (lean_obj_tag(v_pre_939_) == 0)
{
lean_object* v_str_940_; lean_object* v_str_941_; lean_object* v___x_942_; uint8_t v___x_943_; 
v_str_940_ = lean_ctor_get(v_fst_851_, 1);
v_str_941_ = lean_ctor_get(v_pre_933_, 1);
v___x_942_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_943_ = lean_string_dec_eq(v_str_941_, v___x_942_);
if (v___x_943_ == 0)
{
lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_944_ = l_Lean_Name_str___override(v_pre_939_, v___x_875_);
v___x_945_ = l_Lean_Name_str___override(v___x_944_, v___x_903_);
v___y_862_ = v___y_867_;
v___y_863_ = v___x_945_;
v___y_864_ = v_fst_851_;
goto v___jp_861_;
}
else
{
lean_object* v___x_946_; uint8_t v___x_947_; 
lean_inc(v_pre_939_);
lean_inc_ref(v_str_940_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_946_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_947_ = lean_string_dec_eq(v_str_940_, v___x_946_);
if (v___x_947_ == 0)
{
lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; 
v___x_948_ = l_Lean_Name_str___override(v_pre_939_, v___x_875_);
v___x_949_ = l_Lean_Name_str___override(v___x_948_, v___x_903_);
v___x_950_ = l_Lean_Name_str___override(v_pre_939_, v___x_942_);
v___x_951_ = l_Lean_Name_str___override(v___x_950_, v_str_940_);
v___x_952_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_949_, v___x_951_);
lean_dec(v___x_951_);
lean_dec(v___x_949_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_952_;
goto v___jp_833_;
}
else
{
lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; 
lean_dec_ref(v_str_940_);
v___x_953_ = l_Lean_Name_str___override(v_pre_939_, v___x_875_);
v___x_954_ = l_Lean_Name_str___override(v___x_953_, v___x_903_);
v___x_955_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v___x_954_);
lean_dec(v___x_954_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_955_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; 
v___x_956_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_957_ = l_Lean_Name_str___override(v___x_956_, v___x_903_);
v___x_958_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_957_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_957_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_958_;
goto v___jp_833_;
}
}
default: 
{
lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; 
v___x_959_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_960_ = l_Lean_Name_str___override(v___x_959_, v___x_903_);
v___x_961_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_960_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_960_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_961_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; 
v___x_962_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_963_ = l_Lean_Name_str___override(v___x_962_, v___x_903_);
v___x_964_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_963_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec(v___x_963_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_964_;
goto v___jp_833_;
}
}
}
else
{
lean_dec_ref(v_str_873_);
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_965_; 
v_pre_965_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_965_) == 1)
{
lean_object* v_pre_966_; 
v_pre_966_ = lean_ctor_get(v_pre_965_, 0);
if (lean_obj_tag(v_pre_966_) == 0)
{
lean_object* v_str_967_; lean_object* v_str_968_; lean_object* v___x_969_; uint8_t v___x_970_; 
v_str_967_ = lean_ctor_get(v_fst_851_, 1);
v_str_968_ = lean_ctor_get(v_pre_965_, 1);
v___x_969_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_970_ = lean_string_dec_eq(v_str_968_, v___x_969_);
if (v___x_970_ == 0)
{
lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; 
v___x_971_ = l_Lean_Name_str___override(v_pre_966_, v___x_875_);
v___x_972_ = l_Lean_Name_str___override(v___x_971_, v___x_901_);
v___x_973_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_972_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_972_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_973_;
goto v___jp_833_;
}
else
{
lean_object* v___x_974_; uint8_t v___x_975_; 
lean_inc(v_pre_966_);
lean_inc_ref(v_str_967_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_974_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_975_ = lean_string_dec_eq(v_str_967_, v___x_974_);
if (v___x_975_ == 0)
{
lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_976_ = l_Lean_Name_str___override(v_pre_966_, v___x_875_);
v___x_977_ = l_Lean_Name_str___override(v___x_976_, v___x_901_);
v___x_978_ = l_Lean_Name_str___override(v_pre_966_, v___x_969_);
v___x_979_ = l_Lean_Name_str___override(v___x_978_, v_str_967_);
v___x_980_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_977_, v___x_979_);
lean_dec(v___x_979_);
lean_dec(v___x_977_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_980_;
goto v___jp_833_;
}
else
{
lean_dec_ref(v_str_967_);
lean_inc(v_degLE_869_);
v___y_834_ = v___y_867_;
v___y_835_ = v_degLE_869_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_981_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_982_ = l_Lean_Name_str___override(v___x_981_, v___x_901_);
v___x_983_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_982_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_982_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_983_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; 
v___x_984_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_985_ = l_Lean_Name_str___override(v___x_984_, v___x_901_);
v___x_986_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_985_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_985_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_986_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; 
v___x_987_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_988_ = l_Lean_Name_str___override(v___x_987_, v___x_901_);
v___x_989_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_988_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec(v___x_988_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_989_;
goto v___jp_833_;
}
}
}
else
{
lean_dec_ref(v_str_873_);
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_990_; 
v_pre_990_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_990_) == 1)
{
lean_object* v_pre_991_; 
v_pre_991_ = lean_ctor_get(v_pre_990_, 0);
if (lean_obj_tag(v_pre_991_) == 0)
{
lean_object* v_str_992_; lean_object* v_str_993_; lean_object* v___x_994_; uint8_t v___x_995_; 
v_str_992_ = lean_ctor_get(v_fst_851_, 1);
v_str_993_ = lean_ctor_get(v_pre_990_, 1);
v___x_994_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_995_ = lean_string_dec_eq(v_str_993_, v___x_994_);
if (v___x_995_ == 0)
{
lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; 
v___x_996_ = l_Lean_Name_str___override(v_pre_991_, v___x_875_);
v___x_997_ = l_Lean_Name_str___override(v___x_996_, v___x_899_);
v___x_998_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_997_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_997_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_998_;
goto v___jp_833_;
}
else
{
lean_object* v___x_999_; uint8_t v___x_1000_; 
lean_inc_ref(v_str_992_);
lean_inc(v_pre_991_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_999_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_1000_ = lean_string_dec_eq(v_str_992_, v___x_999_);
if (v___x_1000_ == 0)
{
lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; 
v___x_1001_ = l_Lean_Name_str___override(v_pre_991_, v___x_875_);
v___x_1002_ = l_Lean_Name_str___override(v___x_1001_, v___x_899_);
v___x_1003_ = l_Lean_Name_str___override(v_pre_991_, v___x_994_);
v___x_1004_ = l_Lean_Name_str___override(v___x_1003_, v_str_992_);
v___x_1005_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_1002_, v___x_1004_);
lean_dec(v___x_1004_);
lean_dec(v___x_1002_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1005_;
goto v___jp_833_;
}
else
{
lean_dec_ref(v_str_992_);
lean_inc(v_natDegLE_868_);
v___y_834_ = v___y_867_;
v___y_835_ = v_natDegLE_868_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; 
v___x_1006_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_1007_ = l_Lean_Name_str___override(v___x_1006_, v___x_899_);
v___x_1008_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_1007_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_1007_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1008_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1009_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_1010_ = l_Lean_Name_str___override(v___x_1009_, v___x_899_);
v___x_1011_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_1010_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v___x_1010_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1011_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1012_ = l_Lean_Name_str___override(v_pre_872_, v___x_875_);
v___x_1013_ = l_Lean_Name_str___override(v___x_1012_, v___x_899_);
v___x_1014_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v___x_1013_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec(v___x_1013_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1014_;
goto v___jp_833_;
}
}
}
}
else
{
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_1015_; 
v_pre_1015_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_1015_) == 1)
{
lean_object* v_pre_1016_; 
v_pre_1016_ = lean_ctor_get(v_pre_1015_, 0);
if (lean_obj_tag(v_pre_1016_) == 0)
{
lean_object* v_str_1017_; lean_object* v_str_1018_; lean_object* v___x_1019_; uint8_t v___x_1020_; 
v_str_1017_ = lean_ctor_get(v_fst_851_, 1);
v_str_1018_ = lean_ctor_get(v_pre_1015_, 1);
v___x_1019_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_1020_ = lean_string_dec_eq(v_str_1018_, v___x_1019_);
if (v___x_1020_ == 0)
{
lean_object* v___x_1021_; 
v___x_1021_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1021_;
goto v___jp_833_;
}
else
{
lean_object* v___x_1022_; uint8_t v___x_1023_; 
lean_inc_ref(v_str_1017_);
lean_inc(v_pre_1016_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_1022_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_1023_ = lean_string_dec_eq(v_str_1017_, v___x_1022_);
if (v___x_1023_ == 0)
{
lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1024_ = l_Lean_Name_str___override(v_pre_1016_, v___x_1019_);
v___x_1025_ = l_Lean_Name_str___override(v___x_1024_, v_str_1017_);
v___x_1026_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v___x_1025_);
lean_dec(v___x_1025_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1026_;
goto v___jp_833_;
}
else
{
lean_object* v___x_1027_; 
lean_dec_ref(v_str_1017_);
v___x_1027_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v_fst_846_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1027_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_1028_; 
v___x_1028_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1028_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1029_; 
v___x_1029_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1029_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1030_; 
v___x_1030_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1030_;
goto v___jp_833_;
}
}
}
else
{
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_1031_; 
v_pre_1031_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_1031_) == 1)
{
lean_object* v_pre_1032_; 
v_pre_1032_ = lean_ctor_get(v_pre_1031_, 0);
if (lean_obj_tag(v_pre_1032_) == 0)
{
lean_object* v_str_1033_; lean_object* v_str_1034_; lean_object* v___x_1035_; uint8_t v___x_1036_; 
v_str_1033_ = lean_ctor_get(v_fst_851_, 1);
v_str_1034_ = lean_ctor_get(v_pre_1031_, 1);
v___x_1035_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_1036_ = lean_string_dec_eq(v_str_1034_, v___x_1035_);
if (v___x_1036_ == 0)
{
lean_object* v___x_1037_; 
v___x_1037_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1037_;
goto v___jp_833_;
}
else
{
lean_object* v___x_1038_; uint8_t v___x_1039_; 
lean_inc(v_pre_1032_);
lean_inc_ref(v_str_1033_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_1038_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_1039_ = lean_string_dec_eq(v_str_1033_, v___x_1038_);
if (v___x_1039_ == 0)
{
lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; 
v___x_1040_ = l_Lean_Name_str___override(v_pre_1032_, v___x_1035_);
v___x_1041_ = l_Lean_Name_str___override(v___x_1040_, v_str_1033_);
v___x_1042_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v___x_1041_);
lean_dec(v___x_1041_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1042_;
goto v___jp_833_;
}
else
{
lean_object* v___x_1043_; 
lean_dec_ref(v_str_1033_);
v___x_1043_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v_fst_846_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1043_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_1044_; 
v___x_1044_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1044_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1045_; 
v___x_1045_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1045_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1046_; 
v___x_1046_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec_ref_known(v_fst_846_, 2);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1046_;
goto v___jp_833_;
}
}
}
else
{
if (lean_obj_tag(v_fst_851_) == 1)
{
lean_object* v_pre_1047_; 
v_pre_1047_ = lean_ctor_get(v_fst_851_, 0);
if (lean_obj_tag(v_pre_1047_) == 1)
{
lean_object* v_pre_1048_; 
v_pre_1048_ = lean_ctor_get(v_pre_1047_, 0);
if (lean_obj_tag(v_pre_1048_) == 0)
{
lean_object* v_str_1049_; lean_object* v_str_1050_; lean_object* v___x_1051_; uint8_t v___x_1052_; 
v_str_1049_ = lean_ctor_get(v_fst_851_, 1);
v_str_1050_ = lean_ctor_get(v_pre_1047_, 1);
v___x_1051_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__28));
v___x_1052_ = lean_string_dec_eq(v_str_1050_, v___x_1051_);
if (v___x_1052_ == 0)
{
lean_object* v___x_1053_; 
v___x_1053_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v_fst_846_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1053_;
goto v___jp_833_;
}
else
{
lean_object* v___x_1054_; uint8_t v___x_1055_; 
lean_inc_ref(v_str_1049_);
lean_inc(v_pre_1048_);
lean_dec_ref_known(v_fst_851_, 2);
v___x_1054_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__31));
v___x_1055_ = lean_string_dec_eq(v_str_1049_, v___x_1054_);
if (v___x_1055_ == 0)
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1056_ = l_Lean_Name_str___override(v_pre_1048_, v___x_1051_);
v___x_1057_ = l_Lean_Name_str___override(v___x_1056_, v_str_1049_);
v___x_1058_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v___x_1057_);
lean_dec(v___x_1057_);
lean_dec(v_fst_846_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1058_;
goto v___jp_833_;
}
else
{
lean_object* v___x_1059_; 
lean_dec_ref(v_str_1049_);
v___x_1059_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2(v_fst_846_);
lean_dec(v_fst_846_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1059_;
goto v___jp_833_;
}
}
}
else
{
lean_object* v___x_1060_; 
v___x_1060_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v_fst_846_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1060_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1061_; 
v___x_1061_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec_ref_known(v_fst_851_, 2);
lean_dec(v_fst_846_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1061_;
goto v___jp_833_;
}
}
else
{
lean_object* v___x_1062_; 
v___x_1062_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0(v_fst_846_, v_fst_851_);
lean_dec(v_fst_851_);
lean_dec(v_fst_846_);
v___y_834_ = v___y_867_;
v___y_835_ = v___x_1062_;
goto v___jp_833_;
}
}
}
v___jp_1063_:
{
lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; 
v___x_1065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__4));
v___x_1066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__6));
v___x_1067_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__8));
v___y_867_ = v___y_1064_;
v_natDegLE_868_ = v___x_1065_;
v_degLE_869_ = v___x_1066_;
v_coeff_870_ = v___x_1067_;
goto v___jp_866_;
}
v___jp_1068_:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; 
v___x_1070_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__1));
v___x_1071_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1));
v___y_867_ = v___y_1069_;
v_natDegLE_868_ = v___x_1070_;
v_degLE_869_ = v___x_1070_;
v_coeff_870_ = v___x_1071_;
goto v___jp_866_;
}
v___jp_1072_:
{
lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; 
v___x_1074_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__10));
v___x_1075_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__12));
v___x_1076_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__14));
v___y_867_ = v___y_1073_;
v_natDegLE_868_ = v___x_1074_;
v_degLE_869_ = v___x_1075_;
v_coeff_870_ = v___x_1076_;
goto v___jp_866_;
}
v___jp_1078_:
{
lean_object* v___f_1080_; uint8_t v___x_1081_; lean_object* v___x_1082_; uint8_t v___x_1083_; 
v___f_1080_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__17));
v___x_1081_ = 0;
v___x_1082_ = lean_box(v___x_1081_);
lean_inc(v_snd_857_);
v___x_1083_ = l_List_elem___redArg(v___f_1080_, v___x_1082_, v_snd_857_);
if (v___x_1083_ == 0)
{
lean_object* v___x_1084_; lean_object* v_msg_1086_; 
lean_del_object(v___x_859_);
lean_dec(v_snd_857_);
lean_del_object(v___x_854_);
v___x_1084_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1084_, 0, v___y_1079_);
if (v_isShared_850_ == 0)
{
lean_ctor_set_tag(v___x_849_, 5);
lean_ctor_set(v___x_849_, 1, v___x_1084_);
lean_ctor_set(v___x_849_, 0, v___x_1077_);
v_msg_1086_ = v___x_849_;
goto v_reusejp_1085_;
}
else
{
lean_object* v_reuseFailAlloc_1184_; 
v_reuseFailAlloc_1184_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1184_, 0, v___x_1077_);
lean_ctor_set(v_reuseFailAlloc_1184_, 1, v___x_1084_);
v_msg_1086_ = v_reuseFailAlloc_1184_;
goto v_reusejp_1085_;
}
v_reusejp_1085_:
{
if (lean_obj_tag(v_fst_856_) == 0)
{
lean_object* v_val_1087_; 
v_val_1087_ = lean_ctor_get(v_fst_856_, 0);
lean_inc(v_val_1087_);
lean_dec_ref_known(v_fst_856_, 1);
switch(lean_obj_tag(v_val_1087_))
{
case 1:
{
lean_object* v_pre_1088_; 
v_pre_1088_ = lean_ctor_get(v_val_1087_, 0);
if (lean_obj_tag(v_pre_1088_) == 0)
{
lean_object* v_str_1089_; lean_object* v___x_1090_; uint8_t v___x_1091_; 
v_str_1089_ = lean_ctor_get(v_val_1087_, 1);
lean_inc_ref(v_str_1089_);
lean_dec_ref_known(v_val_1087_, 2);
v___x_1090_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__20));
v___x_1091_ = lean_string_dec_eq(v_str_1089_, v___x_1090_);
if (v___x_1091_ == 0)
{
lean_object* v___x_1092_; uint8_t v___x_1093_; 
v___x_1092_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__17));
v___x_1093_ = lean_string_dec_eq(v_str_1089_, v___x_1092_);
if (v___x_1093_ == 0)
{
lean_object* v___x_1094_; uint8_t v___x_1095_; 
v___x_1094_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__14));
v___x_1095_ = lean_string_dec_eq(v_str_1089_, v___x_1094_);
lean_dec_ref(v_str_1089_);
if (v___x_1095_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
v___y_1073_ = v_msg_1086_;
goto v___jp_1072_;
}
}
else
{
lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; 
lean_dec_ref(v_str_1089_);
v___x_1096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__19));
v___x_1097_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__21));
v___x_1098_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__23));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1096_;
v_degLE_869_ = v___x_1097_;
v_coeff_870_ = v___x_1098_;
goto v___jp_866_;
}
}
else
{
lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; 
lean_dec_ref(v_str_1089_);
v___x_1099_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__25));
v___x_1100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__27));
v___x_1101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__29));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1099_;
v_degLE_869_ = v___x_1100_;
v_coeff_870_ = v___x_1101_;
goto v___jp_866_;
}
}
else
{
lean_dec_ref_known(v_val_1087_, 2);
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
}
case 0:
{
lean_object* v___x_1102_; lean_object* v___x_1103_; 
v___x_1102_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__2___closed__1));
v___x_1103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1102_;
v_degLE_869_ = v___x_1102_;
v_coeff_870_ = v___x_1103_;
goto v___jp_866_;
}
default: 
{
lean_dec(v_val_1087_);
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
}
}
else
{
lean_object* v_val_1104_; 
v_val_1104_ = lean_ctor_get(v_fst_856_, 0);
lean_inc(v_val_1104_);
lean_dec_ref_known(v_fst_856_, 1);
if (lean_obj_tag(v_val_1104_) == 1)
{
lean_object* v_pre_1105_; 
v_pre_1105_ = lean_ctor_get(v_val_1104_, 0);
lean_inc(v_pre_1105_);
if (lean_obj_tag(v_pre_1105_) == 1)
{
lean_object* v_pre_1106_; 
v_pre_1106_ = lean_ctor_get(v_pre_1105_, 0);
if (lean_obj_tag(v_pre_1106_) == 0)
{
lean_object* v_str_1107_; lean_object* v_str_1108_; lean_object* v___x_1109_; uint8_t v___x_1110_; 
v_str_1107_ = lean_ctor_get(v_val_1104_, 1);
lean_inc_ref(v_str_1107_);
lean_dec_ref_known(v_val_1104_, 2);
v_str_1108_ = lean_ctor_get(v_pre_1105_, 1);
lean_inc_ref(v_str_1108_);
lean_dec_ref_known(v_pre_1105_, 2);
v___x_1109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__30));
v___x_1110_ = lean_string_dec_eq(v_str_1108_, v___x_1109_);
if (v___x_1110_ == 0)
{
lean_object* v___x_1111_; uint8_t v___x_1112_; 
v___x_1111_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__31));
v___x_1112_ = lean_string_dec_eq(v_str_1108_, v___x_1111_);
if (v___x_1112_ == 0)
{
lean_object* v___x_1113_; uint8_t v___x_1114_; 
v___x_1113_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__32));
v___x_1114_ = lean_string_dec_eq(v_str_1108_, v___x_1113_);
if (v___x_1114_ == 0)
{
lean_object* v___x_1115_; uint8_t v___x_1116_; 
v___x_1115_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__33));
v___x_1116_ = lean_string_dec_eq(v_str_1108_, v___x_1115_);
if (v___x_1116_ == 0)
{
lean_object* v___x_1117_; uint8_t v___x_1118_; 
v___x_1117_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__34));
v___x_1118_ = lean_string_dec_eq(v_str_1108_, v___x_1117_);
if (v___x_1118_ == 0)
{
lean_object* v___x_1119_; uint8_t v___x_1120_; 
v___x_1119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__7));
v___x_1120_ = lean_string_dec_eq(v_str_1108_, v___x_1119_);
if (v___x_1120_ == 0)
{
lean_object* v___x_1121_; uint8_t v___x_1122_; 
v___x_1121_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__35));
v___x_1122_ = lean_string_dec_eq(v_str_1108_, v___x_1121_);
if (v___x_1122_ == 0)
{
lean_object* v___x_1123_; uint8_t v___x_1124_; 
v___x_1123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__36));
v___x_1124_ = lean_string_dec_eq(v_str_1108_, v___x_1123_);
if (v___x_1124_ == 0)
{
lean_object* v___x_1125_; uint8_t v___x_1126_; 
v___x_1125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__37));
v___x_1126_ = lean_string_dec_eq(v_str_1108_, v___x_1125_);
if (v___x_1126_ == 0)
{
lean_object* v___x_1127_; uint8_t v___x_1128_; 
v___x_1127_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__38));
v___x_1128_ = lean_string_dec_eq(v_str_1108_, v___x_1127_);
if (v___x_1128_ == 0)
{
lean_object* v___x_1129_; uint8_t v___x_1130_; 
v___x_1129_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__39));
v___x_1130_ = lean_string_dec_eq(v_str_1108_, v___x_1129_);
lean_dec_ref(v_str_1108_);
if (v___x_1130_ == 0)
{
lean_dec_ref(v_str_1107_);
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1131_; uint8_t v___x_1132_; 
v___x_1131_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__40));
v___x_1132_ = lean_string_dec_eq(v_str_1107_, v___x_1131_);
lean_dec_ref(v_str_1107_);
if (v___x_1132_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; 
v___x_1133_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__42));
v___x_1134_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__44));
v___x_1135_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__46));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1133_;
v_degLE_869_ = v___x_1134_;
v_coeff_870_ = v___x_1135_;
goto v___jp_866_;
}
}
}
else
{
lean_object* v___x_1136_; uint8_t v___x_1137_; 
lean_dec_ref(v_str_1108_);
v___x_1136_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__47));
v___x_1137_ = lean_string_dec_eq(v_str_1107_, v___x_1136_);
lean_dec_ref(v_str_1107_);
if (v___x_1137_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
v___y_1064_ = v_msg_1086_;
goto v___jp_1063_;
}
}
}
else
{
lean_object* v___x_1138_; uint8_t v___x_1139_; 
lean_dec_ref(v_str_1108_);
v___x_1138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__48));
v___x_1139_ = lean_string_dec_eq(v_str_1107_, v___x_1138_);
lean_dec_ref(v_str_1107_);
if (v___x_1139_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
v___y_1064_ = v_msg_1086_;
goto v___jp_1063_;
}
}
}
else
{
lean_object* v___x_1140_; uint8_t v___x_1141_; 
lean_dec_ref(v_str_1108_);
v___x_1140_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__49));
v___x_1141_ = lean_string_dec_eq(v_str_1107_, v___x_1140_);
lean_dec_ref(v_str_1107_);
if (v___x_1141_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
v___y_1073_ = v_msg_1086_;
goto v___jp_1072_;
}
}
}
else
{
lean_object* v___x_1142_; uint8_t v___x_1143_; 
lean_dec_ref(v_str_1108_);
v___x_1142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__48));
v___x_1143_ = lean_string_dec_eq(v_str_1107_, v___x_1142_);
lean_dec_ref(v_str_1107_);
if (v___x_1143_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
v___y_1073_ = v_msg_1086_;
goto v___jp_1072_;
}
}
}
else
{
lean_object* v___x_1144_; uint8_t v___x_1145_; 
lean_dec_ref(v_str_1108_);
v___x_1144_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__50));
v___x_1145_ = lean_string_dec_eq(v_str_1107_, v___x_1144_);
if (v___x_1145_ == 0)
{
lean_object* v___x_1146_; uint8_t v___x_1147_; 
v___x_1146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__8));
v___x_1147_ = lean_string_dec_eq(v_str_1107_, v___x_1146_);
if (v___x_1147_ == 0)
{
lean_object* v___x_1148_; uint8_t v___x_1149_; 
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs___closed__10));
v___x_1149_ = lean_string_dec_eq(v_str_1107_, v___x_1148_);
lean_dec_ref(v_str_1107_);
if (v___x_1149_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; 
v___x_1150_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__52));
v___x_1151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__54));
v___x_1152_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__56));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1150_;
v_degLE_869_ = v___x_1151_;
v_coeff_870_ = v___x_1152_;
goto v___jp_866_;
}
}
else
{
lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; 
lean_dec_ref(v_str_1107_);
v___x_1153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__58));
v___x_1154_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__60));
v___x_1155_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__62));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1153_;
v_degLE_869_ = v___x_1154_;
v_coeff_870_ = v___x_1155_;
goto v___jp_866_;
}
}
else
{
lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; 
lean_dec_ref(v_str_1107_);
v___x_1156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__64));
v___x_1157_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__66));
v___x_1158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__68));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1156_;
v_degLE_869_ = v___x_1157_;
v_coeff_870_ = v___x_1158_;
goto v___jp_866_;
}
}
}
else
{
lean_object* v___x_1159_; uint8_t v___x_1160_; 
lean_dec_ref(v_str_1108_);
v___x_1159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__69));
v___x_1160_ = lean_string_dec_eq(v_str_1107_, v___x_1159_);
lean_dec_ref(v_str_1107_);
if (v___x_1160_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1161_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__71));
v___x_1162_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__73));
v___x_1163_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__75));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1161_;
v_degLE_869_ = v___x_1162_;
v_coeff_870_ = v___x_1163_;
goto v___jp_866_;
}
}
}
else
{
lean_object* v___x_1164_; uint8_t v___x_1165_; 
lean_dec_ref(v_str_1108_);
v___x_1164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__76));
v___x_1165_ = lean_string_dec_eq(v_str_1107_, v___x_1164_);
lean_dec_ref(v_str_1107_);
if (v___x_1165_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__78));
v___x_1167_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__80));
v___x_1168_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__82));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1166_;
v_degLE_869_ = v___x_1167_;
v_coeff_870_ = v___x_1168_;
goto v___jp_866_;
}
}
}
else
{
lean_object* v___x_1169_; uint8_t v___x_1170_; 
lean_dec_ref(v_str_1108_);
v___x_1169_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__83));
v___x_1170_ = lean_string_dec_eq(v_str_1107_, v___x_1169_);
lean_dec_ref(v_str_1107_);
if (v___x_1170_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; 
v___x_1171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__85));
v___x_1172_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__87));
v___x_1173_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__89));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1171_;
v_degLE_869_ = v___x_1172_;
v_coeff_870_ = v___x_1173_;
goto v___jp_866_;
}
}
}
else
{
lean_object* v___x_1174_; uint8_t v___x_1175_; 
lean_dec_ref(v_str_1108_);
v___x_1174_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__90));
v___x_1175_ = lean_string_dec_eq(v_str_1107_, v___x_1174_);
lean_dec_ref(v_str_1107_);
if (v___x_1175_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1176_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__92));
v___x_1177_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__94));
v___x_1178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__96));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1176_;
v_degLE_869_ = v___x_1177_;
v_coeff_870_ = v___x_1178_;
goto v___jp_866_;
}
}
}
else
{
lean_object* v___x_1179_; uint8_t v___x_1180_; 
lean_dec_ref(v_str_1108_);
v___x_1179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__97));
v___x_1180_ = lean_string_dec_eq(v_str_1107_, v___x_1179_);
lean_dec_ref(v_str_1107_);
if (v___x_1180_ == 0)
{
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
else
{
lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; 
v___x_1181_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__99));
v___x_1182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__101));
v___x_1183_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__103));
v___y_867_ = v_msg_1086_;
v_natDegLE_868_ = v___x_1181_;
v_degLE_869_ = v___x_1182_;
v_coeff_870_ = v___x_1183_;
goto v___jp_866_;
}
}
}
else
{
lean_dec_ref_known(v_pre_1105_, 2);
lean_dec_ref_known(v_val_1104_, 2);
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
}
else
{
lean_dec_ref_known(v_val_1104_, 2);
lean_dec(v_pre_1105_);
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
}
else
{
lean_dec(v_val_1104_);
v___y_1069_ = v_msg_1086_;
goto v___jp_1068_;
}
}
}
}
else
{
lean_object* v___x_1186_; 
lean_dec_ref(v___y_1079_);
lean_dec(v_fst_856_);
lean_del_object(v___x_849_);
if (v_isShared_860_ == 0)
{
lean_ctor_set(v___x_859_, 0, v_fst_851_);
v___x_1186_ = v___x_859_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1191_; 
v_reuseFailAlloc_1191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1191_, 0, v_fst_851_);
lean_ctor_set(v_reuseFailAlloc_1191_, 1, v_snd_857_);
v___x_1186_ = v_reuseFailAlloc_1191_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
lean_object* v___x_1188_; 
if (v_isShared_855_ == 0)
{
lean_ctor_set(v___x_854_, 1, v___x_1186_);
lean_ctor_set(v___x_854_, 0, v_fst_846_);
v___x_1188_ = v___x_854_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1190_; 
v_reuseFailAlloc_1190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1190_, 0, v_fst_846_);
lean_ctor_set(v_reuseFailAlloc_1190_, 1, v___x_1186_);
v___x_1188_ = v_reuseFailAlloc_1190_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
lean_object* v___x_1189_; 
v___x_1189_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma(v___x_1188_, v_debug_830_);
return v___x_1189_;
}
}
}
}
}
}
}
}
}
v___jp_831_:
{
lean_object* v___x_832_; 
v___x_832_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
return v___x_832_;
}
v___jp_833_:
{
if (v_debug_830_ == 0)
{
lean_dec(v___y_834_);
return v___y_835_;
}
else
{
lean_object* v___f_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; 
lean_inc(v___y_835_);
v___f_836_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__1___boxed), 2, 1);
lean_closure_set(v___f_836_, 0, v___y_835_);
v___x_837_ = lp_mathlib_Lean_Name_lastComponentAsString(v___y_835_);
v___x_838_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_838_, 0, v___x_837_);
v___x_839_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__1));
v___x_840_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_840_, 0, v___x_838_);
lean_ctor_set(v___x_840_, 1, v___x_839_);
v___x_841_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_841_, 0, v___x_840_);
lean_ctor_set(v___x_841_, 1, v___y_834_);
v___x_842_ = l_Std_Format_defWidth;
v___x_843_ = lean_unsigned_to_nat(0u);
v___x_844_ = l_Std_Format_pretty(v___x_841_, v___x_842_, v___x_843_, v___x_843_);
v___x_845_ = lean_dbg_trace(v___x_844_, v___f_836_);
return v___x_845_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___boxed(lean_object* v_twoH_1213_, lean_object* v_debug_1214_){
_start:
{
uint8_t v_debug_boxed_1215_; lean_object* v_res_1216_; 
v_debug_boxed_1215_ = lean_unbox(v_debug_1214_);
v_res_1216_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma(v_twoH_1213_, v_debug_boxed_1215_);
return v_res_1216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg(lean_object* v_e_1217_, lean_object* v___y_1218_){
_start:
{
uint8_t v___x_1220_; 
v___x_1220_ = l_Lean_Expr_hasMVar(v_e_1217_);
if (v___x_1220_ == 0)
{
lean_object* v___x_1221_; 
v___x_1221_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1221_, 0, v_e_1217_);
return v___x_1221_;
}
else
{
lean_object* v___x_1222_; lean_object* v_mctx_1223_; lean_object* v___x_1224_; lean_object* v_fst_1225_; lean_object* v_snd_1226_; lean_object* v___x_1227_; lean_object* v_cache_1228_; lean_object* v_zetaDeltaFVarIds_1229_; lean_object* v_postponed_1230_; lean_object* v_diag_1231_; lean_object* v___x_1233_; uint8_t v_isShared_1234_; uint8_t v_isSharedCheck_1240_; 
v___x_1222_ = lean_st_ref_get(v___y_1218_);
v_mctx_1223_ = lean_ctor_get(v___x_1222_, 0);
lean_inc_ref(v_mctx_1223_);
lean_dec(v___x_1222_);
v___x_1224_ = l_Lean_instantiateMVarsCore(v_mctx_1223_, v_e_1217_);
v_fst_1225_ = lean_ctor_get(v___x_1224_, 0);
lean_inc(v_fst_1225_);
v_snd_1226_ = lean_ctor_get(v___x_1224_, 1);
lean_inc(v_snd_1226_);
lean_dec_ref(v___x_1224_);
v___x_1227_ = lean_st_ref_take(v___y_1218_);
v_cache_1228_ = lean_ctor_get(v___x_1227_, 1);
v_zetaDeltaFVarIds_1229_ = lean_ctor_get(v___x_1227_, 2);
v_postponed_1230_ = lean_ctor_get(v___x_1227_, 3);
v_diag_1231_ = lean_ctor_get(v___x_1227_, 4);
v_isSharedCheck_1240_ = !lean_is_exclusive(v___x_1227_);
if (v_isSharedCheck_1240_ == 0)
{
lean_object* v_unused_1241_; 
v_unused_1241_ = lean_ctor_get(v___x_1227_, 0);
lean_dec(v_unused_1241_);
v___x_1233_ = v___x_1227_;
v_isShared_1234_ = v_isSharedCheck_1240_;
goto v_resetjp_1232_;
}
else
{
lean_inc(v_diag_1231_);
lean_inc(v_postponed_1230_);
lean_inc(v_zetaDeltaFVarIds_1229_);
lean_inc(v_cache_1228_);
lean_dec(v___x_1227_);
v___x_1233_ = lean_box(0);
v_isShared_1234_ = v_isSharedCheck_1240_;
goto v_resetjp_1232_;
}
v_resetjp_1232_:
{
lean_object* v___x_1236_; 
if (v_isShared_1234_ == 0)
{
lean_ctor_set(v___x_1233_, 0, v_snd_1226_);
v___x_1236_ = v___x_1233_;
goto v_reusejp_1235_;
}
else
{
lean_object* v_reuseFailAlloc_1239_; 
v_reuseFailAlloc_1239_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1239_, 0, v_snd_1226_);
lean_ctor_set(v_reuseFailAlloc_1239_, 1, v_cache_1228_);
lean_ctor_set(v_reuseFailAlloc_1239_, 2, v_zetaDeltaFVarIds_1229_);
lean_ctor_set(v_reuseFailAlloc_1239_, 3, v_postponed_1230_);
lean_ctor_set(v_reuseFailAlloc_1239_, 4, v_diag_1231_);
v___x_1236_ = v_reuseFailAlloc_1239_;
goto v_reusejp_1235_;
}
v_reusejp_1235_:
{
lean_object* v___x_1237_; lean_object* v___x_1238_; 
v___x_1237_ = lean_st_ref_set(v___y_1218_, v___x_1236_);
v___x_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1238_, 0, v_fst_1225_);
return v___x_1238_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg___boxed(lean_object* v_e_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg(v_e_1242_, v___y_1243_);
lean_dec(v___y_1243_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0(lean_object* v_e_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
lean_object* v___x_1252_; 
v___x_1252_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg(v_e_1246_, v___y_1248_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___boxed(lean_object* v_e_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_){
_start:
{
lean_object* v_res_1259_; 
v_res_1259_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0(v_e_1253_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_);
lean_dec(v___y_1257_);
lean_dec_ref(v___y_1256_);
lean_dec(v___y_1255_);
lean_dec_ref(v___y_1254_);
return v_res_1259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__1(lean_object* v_a_1260_, lean_object* v_a_1261_, lean_object* v_a_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_){
_start:
{
if (lean_obj_tag(v_a_1260_) == 0)
{
lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; 
v___x_1268_ = lean_array_to_list(v_a_1261_);
v___x_1269_ = lean_array_to_list(v_a_1262_);
v___x_1270_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1268_);
lean_ctor_set(v___x_1270_, 1, v___x_1269_);
v___x_1271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1270_);
return v___x_1271_;
}
else
{
lean_object* v_head_1272_; lean_object* v_tail_1273_; lean_object* v___x_1274_; 
v_head_1272_ = lean_ctor_get(v_a_1260_, 0);
lean_inc_n(v_head_1272_, 2);
v_tail_1273_ = lean_ctor_get(v_a_1260_, 1);
lean_inc(v_tail_1273_);
lean_dec_ref_known(v_a_1260_, 2);
v___x_1274_ = l_Lean_MVarId_getDecl(v_head_1272_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_);
if (lean_obj_tag(v___x_1274_) == 0)
{
lean_object* v_a_1275_; lean_object* v_type_1276_; lean_object* v___x_1277_; lean_object* v_a_1278_; uint8_t v___x_1279_; 
v_a_1275_ = lean_ctor_get(v___x_1274_, 0);
lean_inc(v_a_1275_);
lean_dec_ref_known(v___x_1274_, 1);
v_type_1276_ = lean_ctor_get(v_a_1275_, 2);
lean_inc_ref(v_type_1276_);
lean_dec(v_a_1275_);
v___x_1277_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg(v_type_1276_, v___y_1264_);
v_a_1278_ = lean_ctor_get(v___x_1277_, 0);
lean_inc(v_a_1278_);
lean_dec_ref(v___x_1277_);
v___x_1279_ = l_Lean_Expr_hasExprMVar(v_a_1278_);
lean_dec(v_a_1278_);
if (v___x_1279_ == 0)
{
lean_object* v___x_1280_; 
v___x_1280_ = lean_array_push(v_a_1262_, v_head_1272_);
v_a_1260_ = v_tail_1273_;
v_a_1262_ = v___x_1280_;
goto _start;
}
else
{
lean_object* v___x_1282_; 
v___x_1282_ = lean_array_push(v_a_1261_, v_head_1272_);
v_a_1260_ = v_tail_1273_;
v_a_1261_ = v___x_1282_;
goto _start;
}
}
else
{
lean_object* v_a_1284_; lean_object* v___x_1286_; uint8_t v_isShared_1287_; uint8_t v_isSharedCheck_1291_; 
lean_dec(v_tail_1273_);
lean_dec(v_head_1272_);
lean_dec_ref(v_a_1262_);
lean_dec_ref(v_a_1261_);
v_a_1284_ = lean_ctor_get(v___x_1274_, 0);
v_isSharedCheck_1291_ = !lean_is_exclusive(v___x_1274_);
if (v_isSharedCheck_1291_ == 0)
{
v___x_1286_ = v___x_1274_;
v_isShared_1287_ = v_isSharedCheck_1291_;
goto v_resetjp_1285_;
}
else
{
lean_inc(v_a_1284_);
lean_dec(v___x_1274_);
v___x_1286_ = lean_box(0);
v_isShared_1287_ = v_isSharedCheck_1291_;
goto v_resetjp_1285_;
}
v_resetjp_1285_:
{
lean_object* v___x_1289_; 
if (v_isShared_1287_ == 0)
{
v___x_1289_ = v___x_1286_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v_a_1284_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__1___boxed(lean_object* v_a_1292_, lean_object* v_a_1293_, lean_object* v_a_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_){
_start:
{
lean_object* v_res_1300_; 
v_res_1300_ = lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__1(v_a_1292_, v_a_1293_, v_a_1294_, v___y_1295_, v___y_1296_, v___y_1297_, v___y_1298_);
lean_dec(v___y_1298_);
lean_dec_ref(v___y_1297_);
lean_dec(v___y_1296_);
lean_dec_ref(v___y_1295_);
return v_res_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3(lean_object* v_x_1303_, lean_object* v_x_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_){
_start:
{
if (lean_obj_tag(v_x_1303_) == 0)
{
lean_object* v___x_1310_; lean_object* v___x_1311_; 
v___x_1310_ = l_List_reverse___redArg(v_x_1304_);
v___x_1311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1311_, 0, v___x_1310_);
return v___x_1311_;
}
else
{
lean_object* v_head_1312_; lean_object* v_tail_1313_; lean_object* v___x_1315_; uint8_t v_isShared_1316_; uint8_t v_isSharedCheck_1371_; 
v_head_1312_ = lean_ctor_get(v_x_1303_, 0);
v_tail_1313_ = lean_ctor_get(v_x_1303_, 1);
v_isSharedCheck_1371_ = !lean_is_exclusive(v_x_1303_);
if (v_isSharedCheck_1371_ == 0)
{
v___x_1315_ = v_x_1303_;
v_isShared_1316_ = v_isSharedCheck_1371_;
goto v_resetjp_1314_;
}
else
{
lean_inc(v_tail_1313_);
lean_inc(v_head_1312_);
lean_dec(v_x_1303_);
v___x_1315_ = lean_box(0);
v_isShared_1316_ = v_isSharedCheck_1371_;
goto v_resetjp_1314_;
}
v_resetjp_1314_:
{
lean_object* v_a_1318_; lean_object* v___x_1323_; 
lean_inc(v_head_1312_);
v___x_1323_ = l_Lean_MVarId_getDecl(v_head_1312_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_);
if (lean_obj_tag(v___x_1323_) == 0)
{
lean_object* v_a_1324_; lean_object* v_type_1325_; lean_object* v___x_1326_; lean_object* v_a_1327_; lean_object* v___x_1331_; lean_object* v___x_1332_; uint8_t v___x_1333_; uint8_t v___y_1350_; 
v_a_1324_ = lean_ctor_get(v___x_1323_, 0);
lean_inc(v_a_1324_);
lean_dec_ref_known(v___x_1323_, 1);
v_type_1325_ = lean_ctor_get(v_a_1324_, 2);
lean_inc_ref(v_type_1325_);
lean_dec(v_a_1324_);
v___x_1326_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__0___redArg(v_type_1325_, v___y_1306_);
v_a_1327_ = lean_ctor_get(v___x_1326_, 0);
lean_inc(v_a_1327_);
lean_dec_ref(v___x_1326_);
v___x_1331_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___closed__0));
v___x_1332_ = lean_unsigned_to_nat(3u);
v___x_1333_ = l_Lean_Expr_isAppOfArity(v_a_1327_, v___x_1331_, v___x_1332_);
if (v___x_1333_ == 0)
{
lean_object* v___x_1351_; lean_object* v___x_1352_; 
lean_dec(v_a_1327_);
v___x_1351_ = lean_box(0);
v___x_1352_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1352_, 0, v_head_1312_);
lean_ctor_set(v___x_1352_, 1, v___x_1351_);
v_a_1318_ = v___x_1352_;
goto v___jp_1317_;
}
else
{
lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; uint8_t v___y_1360_; uint8_t v___x_1361_; 
v___x_1353_ = l_Lean_Expr_appFn_x21(v_a_1327_);
v___x_1354_ = l_Lean_Expr_appArg_x21(v___x_1353_);
lean_dec_ref(v___x_1353_);
v___x_1355_ = l_Lean_Expr_appArg_x21(v_a_1327_);
lean_dec(v_a_1327_);
v___x_1361_ = l_Lean_Expr_isMVar(v___x_1355_);
if (v___x_1361_ == 0)
{
v___y_1360_ = v___x_1361_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1362_; 
v___x_1362_ = l_Lean_Expr_hasExprMVar(v___x_1354_);
if (v___x_1362_ == 0)
{
v___y_1360_ = v___x_1361_;
goto v___jp_1359_;
}
else
{
goto v___jp_1356_;
}
}
v___jp_1356_:
{
uint8_t v___x_1357_; 
v___x_1357_ = l_Lean_Expr_isMVar(v___x_1354_);
lean_dec_ref(v___x_1354_);
if (v___x_1357_ == 0)
{
lean_dec_ref(v___x_1355_);
v___y_1350_ = v___x_1357_;
goto v___jp_1349_;
}
else
{
uint8_t v___x_1358_; 
v___x_1358_ = l_Lean_Expr_hasExprMVar(v___x_1355_);
lean_dec_ref(v___x_1355_);
if (v___x_1358_ == 0)
{
v___y_1350_ = v___x_1357_;
goto v___jp_1349_;
}
else
{
goto v___jp_1328_;
}
}
}
v___jp_1359_:
{
if (v___y_1360_ == 0)
{
goto v___jp_1356_;
}
else
{
lean_dec_ref(v___x_1355_);
lean_dec_ref(v___x_1354_);
goto v___jp_1334_;
}
}
}
v___jp_1328_:
{
lean_object* v___x_1329_; lean_object* v___x_1330_; 
v___x_1329_ = lean_box(0);
v___x_1330_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1330_, 0, v_head_1312_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
v_a_1318_ = v___x_1330_;
goto v___jp_1317_;
}
v___jp_1334_:
{
lean_object* v___x_1335_; uint8_t v___x_1336_; uint8_t v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; 
v___x_1335_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1));
v___x_1336_ = 0;
v___x_1337_ = 0;
v___x_1338_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_1338_, 0, v___x_1336_);
lean_ctor_set_uint8(v___x_1338_, 1, v___x_1333_);
lean_ctor_set_uint8(v___x_1338_, 2, v___x_1337_);
lean_ctor_set_uint8(v___x_1338_, 3, v___x_1333_);
v___x_1339_ = l_Lean_MVarId_applyConst(v_head_1312_, v___x_1335_, v___x_1338_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_);
if (lean_obj_tag(v___x_1339_) == 0)
{
lean_object* v_a_1340_; 
v_a_1340_ = lean_ctor_get(v___x_1339_, 0);
lean_inc(v_a_1340_);
lean_dec_ref_known(v___x_1339_, 1);
v_a_1318_ = v_a_1340_;
goto v___jp_1317_;
}
else
{
lean_object* v_a_1341_; lean_object* v___x_1343_; uint8_t v_isShared_1344_; uint8_t v_isSharedCheck_1348_; 
lean_del_object(v___x_1315_);
lean_dec(v_tail_1313_);
lean_dec(v_x_1304_);
v_a_1341_ = lean_ctor_get(v___x_1339_, 0);
v_isSharedCheck_1348_ = !lean_is_exclusive(v___x_1339_);
if (v_isSharedCheck_1348_ == 0)
{
v___x_1343_ = v___x_1339_;
v_isShared_1344_ = v_isSharedCheck_1348_;
goto v_resetjp_1342_;
}
else
{
lean_inc(v_a_1341_);
lean_dec(v___x_1339_);
v___x_1343_ = lean_box(0);
v_isShared_1344_ = v_isSharedCheck_1348_;
goto v_resetjp_1342_;
}
v_resetjp_1342_:
{
lean_object* v___x_1346_; 
if (v_isShared_1344_ == 0)
{
v___x_1346_ = v___x_1343_;
goto v_reusejp_1345_;
}
else
{
lean_object* v_reuseFailAlloc_1347_; 
v_reuseFailAlloc_1347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1347_, 0, v_a_1341_);
v___x_1346_ = v_reuseFailAlloc_1347_;
goto v_reusejp_1345_;
}
v_reusejp_1345_:
{
return v___x_1346_;
}
}
}
}
v___jp_1349_:
{
if (v___y_1350_ == 0)
{
goto v___jp_1328_;
}
else
{
goto v___jp_1334_;
}
}
}
else
{
lean_object* v_a_1363_; lean_object* v___x_1365_; uint8_t v_isShared_1366_; uint8_t v_isSharedCheck_1370_; 
lean_del_object(v___x_1315_);
lean_dec(v_tail_1313_);
lean_dec(v_head_1312_);
lean_dec(v_x_1304_);
v_a_1363_ = lean_ctor_get(v___x_1323_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1323_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1365_ = v___x_1323_;
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
else
{
lean_inc(v_a_1363_);
lean_dec(v___x_1323_);
v___x_1365_ = lean_box(0);
v_isShared_1366_ = v_isSharedCheck_1370_;
goto v_resetjp_1364_;
}
v_resetjp_1364_:
{
lean_object* v___x_1368_; 
if (v_isShared_1366_ == 0)
{
v___x_1368_ = v___x_1365_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v_a_1363_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
v___jp_1317_:
{
lean_object* v___x_1320_; 
if (v_isShared_1316_ == 0)
{
lean_ctor_set(v___x_1315_, 1, v_x_1304_);
lean_ctor_set(v___x_1315_, 0, v_a_1318_);
v___x_1320_ = v___x_1315_;
goto v_reusejp_1319_;
}
else
{
lean_object* v_reuseFailAlloc_1322_; 
v_reuseFailAlloc_1322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1322_, 0, v_a_1318_);
lean_ctor_set(v_reuseFailAlloc_1322_, 1, v_x_1304_);
v___x_1320_ = v_reuseFailAlloc_1322_;
goto v_reusejp_1319_;
}
v_reusejp_1319_:
{
v_x_1303_ = v_tail_1313_;
v_x_1304_ = v___x_1320_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___boxed(lean_object* v_x_1372_, lean_object* v_x_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_){
_start:
{
lean_object* v_res_1379_; 
v_res_1379_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3(v_x_1372_, v_x_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_);
lean_dec(v___y_1377_);
lean_dec_ref(v___y_1376_);
lean_dec(v___y_1375_);
lean_dec_ref(v___y_1374_);
return v_res_1379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2(lean_object* v_x_1384_, lean_object* v_x_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_){
_start:
{
if (lean_obj_tag(v_x_1384_) == 0)
{
lean_object* v___x_1391_; lean_object* v___x_1392_; 
v___x_1391_ = l_List_reverse___redArg(v_x_1385_);
v___x_1392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1392_, 0, v___x_1391_);
return v___x_1392_;
}
else
{
lean_object* v_head_1393_; lean_object* v_tail_1394_; lean_object* v___x_1396_; uint8_t v_isShared_1397_; uint8_t v_isSharedCheck_1444_; 
v_head_1393_ = lean_ctor_get(v_x_1384_, 0);
v_tail_1394_ = lean_ctor_get(v_x_1384_, 1);
v_isSharedCheck_1444_ = !lean_is_exclusive(v_x_1384_);
if (v_isSharedCheck_1444_ == 0)
{
v___x_1396_ = v_x_1384_;
v_isShared_1397_ = v_isSharedCheck_1444_;
goto v_resetjp_1395_;
}
else
{
lean_inc(v_tail_1394_);
lean_inc(v_head_1393_);
lean_dec(v_x_1384_);
v___x_1396_ = lean_box(0);
v_isShared_1397_ = v_isSharedCheck_1444_;
goto v_resetjp_1395_;
}
v_resetjp_1395_:
{
lean_object* v_a_1399_; lean_object* v___y_1405_; lean_object* v___x_1415_; lean_object* v___x_1416_; 
v___x_1415_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___closed__0));
v___x_1416_ = l_Lean_Meta_saveState___redArg(v___y_1387_, v___y_1389_);
if (lean_obj_tag(v___x_1416_) == 0)
{
lean_object* v_a_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; 
v_a_1417_ = lean_ctor_get(v___x_1416_, 0);
lean_inc(v_a_1417_);
lean_dec_ref_known(v___x_1416_, 1);
v___x_1418_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___lam__0___closed__1));
lean_inc(v_head_1393_);
v___x_1419_ = l_Lean_MVarId_applyConst(v_head_1393_, v___x_1418_, v___x_1415_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_);
if (lean_obj_tag(v___x_1419_) == 0)
{
lean_dec(v_a_1417_);
lean_dec(v_head_1393_);
v___y_1405_ = v___x_1419_;
goto v___jp_1404_;
}
else
{
lean_object* v_a_1420_; uint8_t v___y_1422_; uint8_t v___x_1434_; 
v_a_1420_ = lean_ctor_get(v___x_1419_, 0);
lean_inc(v_a_1420_);
v___x_1434_ = l_Lean_Exception_isInterrupt(v_a_1420_);
if (v___x_1434_ == 0)
{
uint8_t v___x_1435_; 
v___x_1435_ = l_Lean_Exception_isRuntime(v_a_1420_);
v___y_1422_ = v___x_1435_;
goto v___jp_1421_;
}
else
{
lean_dec(v_a_1420_);
v___y_1422_ = v___x_1434_;
goto v___jp_1421_;
}
v___jp_1421_:
{
if (v___y_1422_ == 0)
{
lean_object* v___x_1423_; 
lean_dec_ref_known(v___x_1419_, 1);
v___x_1423_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1417_, v___y_1387_, v___y_1389_);
lean_dec(v_a_1417_);
if (lean_obj_tag(v___x_1423_) == 0)
{
lean_object* v___x_1424_; lean_object* v___x_1425_; 
lean_dec_ref_known(v___x_1423_, 1);
v___x_1424_ = lean_box(0);
v___x_1425_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1425_, 0, v_head_1393_);
lean_ctor_set(v___x_1425_, 1, v___x_1424_);
v_a_1399_ = v___x_1425_;
goto v___jp_1398_;
}
else
{
lean_object* v_a_1426_; lean_object* v___x_1428_; uint8_t v_isShared_1429_; uint8_t v_isSharedCheck_1433_; 
lean_del_object(v___x_1396_);
lean_dec(v_tail_1394_);
lean_dec(v_head_1393_);
lean_dec(v_x_1385_);
v_a_1426_ = lean_ctor_get(v___x_1423_, 0);
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1423_);
if (v_isSharedCheck_1433_ == 0)
{
v___x_1428_ = v___x_1423_;
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_a_1426_);
lean_dec(v___x_1423_);
v___x_1428_ = lean_box(0);
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
v_resetjp_1427_:
{
lean_object* v___x_1431_; 
if (v_isShared_1429_ == 0)
{
v___x_1431_ = v___x_1428_;
goto v_reusejp_1430_;
}
else
{
lean_object* v_reuseFailAlloc_1432_; 
v_reuseFailAlloc_1432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1432_, 0, v_a_1426_);
v___x_1431_ = v_reuseFailAlloc_1432_;
goto v_reusejp_1430_;
}
v_reusejp_1430_:
{
return v___x_1431_;
}
}
}
}
else
{
lean_dec(v_a_1417_);
lean_dec(v_head_1393_);
v___y_1405_ = v___x_1419_;
goto v___jp_1404_;
}
}
}
}
else
{
lean_object* v_a_1436_; lean_object* v___x_1438_; uint8_t v_isShared_1439_; uint8_t v_isSharedCheck_1443_; 
lean_del_object(v___x_1396_);
lean_dec(v_tail_1394_);
lean_dec(v_head_1393_);
lean_dec(v_x_1385_);
v_a_1436_ = lean_ctor_get(v___x_1416_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1438_ = v___x_1416_;
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
else
{
lean_inc(v_a_1436_);
lean_dec(v___x_1416_);
v___x_1438_ = lean_box(0);
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
v_resetjp_1437_:
{
lean_object* v___x_1441_; 
if (v_isShared_1439_ == 0)
{
v___x_1441_ = v___x_1438_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1442_; 
v_reuseFailAlloc_1442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1442_, 0, v_a_1436_);
v___x_1441_ = v_reuseFailAlloc_1442_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
return v___x_1441_;
}
}
}
v___jp_1398_:
{
lean_object* v___x_1401_; 
if (v_isShared_1397_ == 0)
{
lean_ctor_set(v___x_1396_, 1, v_x_1385_);
lean_ctor_set(v___x_1396_, 0, v_a_1399_);
v___x_1401_ = v___x_1396_;
goto v_reusejp_1400_;
}
else
{
lean_object* v_reuseFailAlloc_1403_; 
v_reuseFailAlloc_1403_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1403_, 0, v_a_1399_);
lean_ctor_set(v_reuseFailAlloc_1403_, 1, v_x_1385_);
v___x_1401_ = v_reuseFailAlloc_1403_;
goto v_reusejp_1400_;
}
v_reusejp_1400_:
{
v_x_1384_ = v_tail_1394_;
v_x_1385_ = v___x_1401_;
goto _start;
}
}
v___jp_1404_:
{
if (lean_obj_tag(v___y_1405_) == 0)
{
lean_object* v_a_1406_; 
v_a_1406_ = lean_ctor_get(v___y_1405_, 0);
lean_inc(v_a_1406_);
lean_dec_ref_known(v___y_1405_, 1);
v_a_1399_ = v_a_1406_;
goto v___jp_1398_;
}
else
{
lean_object* v_a_1407_; lean_object* v___x_1409_; uint8_t v_isShared_1410_; uint8_t v_isSharedCheck_1414_; 
lean_del_object(v___x_1396_);
lean_dec(v_tail_1394_);
lean_dec(v_x_1385_);
v_a_1407_ = lean_ctor_get(v___y_1405_, 0);
v_isSharedCheck_1414_ = !lean_is_exclusive(v___y_1405_);
if (v_isSharedCheck_1414_ == 0)
{
v___x_1409_ = v___y_1405_;
v_isShared_1410_ = v_isSharedCheck_1414_;
goto v_resetjp_1408_;
}
else
{
lean_inc(v_a_1407_);
lean_dec(v___y_1405_);
v___x_1409_ = lean_box(0);
v_isShared_1410_ = v_isSharedCheck_1414_;
goto v_resetjp_1408_;
}
v_resetjp_1408_:
{
lean_object* v___x_1412_; 
if (v_isShared_1410_ == 0)
{
v___x_1412_ = v___x_1409_;
goto v_reusejp_1411_;
}
else
{
lean_object* v_reuseFailAlloc_1413_; 
v_reuseFailAlloc_1413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1413_, 0, v_a_1407_);
v___x_1412_ = v_reuseFailAlloc_1413_;
goto v_reusejp_1411_;
}
v_reusejp_1411_:
{
return v___x_1412_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___boxed(lean_object* v_x_1445_, lean_object* v_x_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_){
_start:
{
lean_object* v_res_1452_; 
v_res_1452_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2(v_x_1445_, v_x_1446_, v___y_1447_, v___y_1448_, v___y_1449_, v___y_1450_);
lean_dec(v___y_1450_);
lean_dec_ref(v___y_1449_);
lean_dec(v___y_1448_);
lean_dec_ref(v___y_1447_);
return v_res_1452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__4(lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
if (lean_obj_tag(v_a_1453_) == 0)
{
lean_object* v___x_1455_; 
v___x_1455_ = lean_array_to_list(v_a_1454_);
return v___x_1455_;
}
else
{
lean_object* v_head_1456_; lean_object* v_tail_1457_; lean_object* v___x_1458_; 
v_head_1456_ = lean_ctor_get(v_a_1453_, 0);
lean_inc(v_head_1456_);
v_tail_1457_ = lean_ctor_get(v_a_1453_, 1);
lean_inc(v_tail_1457_);
lean_dec_ref_known(v_a_1453_, 2);
v___x_1458_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_1454_, v_head_1456_);
v_a_1453_ = v_tail_1457_;
v_a_1454_ = v___x_1458_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl(lean_object* v_mvs_1462_, lean_object* v_a_1463_, lean_object* v_a_1464_, lean_object* v_a_1465_, lean_object* v_a_1466_){
_start:
{
lean_object* v___x_1468_; lean_object* v___x_1469_; 
v___x_1468_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0));
v___x_1469_ = lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__1(v_mvs_1462_, v___x_1468_, v___x_1468_, v_a_1463_, v_a_1464_, v_a_1465_, v_a_1466_);
if (lean_obj_tag(v___x_1469_) == 0)
{
lean_object* v_a_1470_; lean_object* v_fst_1471_; lean_object* v_snd_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; 
v_a_1470_ = lean_ctor_get(v___x_1469_, 0);
lean_inc(v_a_1470_);
lean_dec_ref_known(v___x_1469_, 1);
v_fst_1471_ = lean_ctor_get(v_a_1470_, 0);
lean_inc(v_fst_1471_);
v_snd_1472_ = lean_ctor_get(v_a_1470_, 1);
lean_inc(v_snd_1472_);
lean_dec(v_a_1470_);
v___x_1473_ = lean_box(0);
v___x_1474_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2(v_snd_1472_, v___x_1473_, v_a_1463_, v_a_1464_, v_a_1465_, v_a_1466_);
if (lean_obj_tag(v___x_1474_) == 0)
{
lean_object* v_a_1475_; lean_object* v___x_1476_; 
v_a_1475_ = lean_ctor_get(v___x_1474_, 0);
lean_inc(v_a_1475_);
lean_dec_ref_known(v___x_1474_, 1);
v___x_1476_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3(v_fst_1471_, v___x_1473_, v_a_1463_, v_a_1464_, v_a_1465_, v_a_1466_);
if (lean_obj_tag(v___x_1476_) == 0)
{
lean_object* v_a_1477_; lean_object* v___x_1479_; uint8_t v_isShared_1480_; uint8_t v_isSharedCheck_1487_; 
v_a_1477_ = lean_ctor_get(v___x_1476_, 0);
v_isSharedCheck_1487_ = !lean_is_exclusive(v___x_1476_);
if (v_isSharedCheck_1487_ == 0)
{
v___x_1479_ = v___x_1476_;
v_isShared_1480_ = v_isSharedCheck_1487_;
goto v_resetjp_1478_;
}
else
{
lean_inc(v_a_1477_);
lean_dec(v___x_1476_);
v___x_1479_ = lean_box(0);
v_isShared_1480_ = v_isSharedCheck_1487_;
goto v_resetjp_1478_;
}
v_resetjp_1478_:
{
lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1485_; 
v___x_1481_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__4(v_a_1477_, v___x_1468_);
v___x_1482_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__4(v_a_1475_, v___x_1468_);
v___x_1483_ = l_List_appendTR___redArg(v___x_1481_, v___x_1482_);
if (v_isShared_1480_ == 0)
{
lean_ctor_set(v___x_1479_, 0, v___x_1483_);
v___x_1485_ = v___x_1479_;
goto v_reusejp_1484_;
}
else
{
lean_object* v_reuseFailAlloc_1486_; 
v_reuseFailAlloc_1486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1486_, 0, v___x_1483_);
v___x_1485_ = v_reuseFailAlloc_1486_;
goto v_reusejp_1484_;
}
v_reusejp_1484_:
{
return v___x_1485_;
}
}
}
else
{
lean_object* v_a_1488_; lean_object* v___x_1490_; uint8_t v_isShared_1491_; uint8_t v_isSharedCheck_1495_; 
lean_dec(v_a_1475_);
v_a_1488_ = lean_ctor_get(v___x_1476_, 0);
v_isSharedCheck_1495_ = !lean_is_exclusive(v___x_1476_);
if (v_isSharedCheck_1495_ == 0)
{
v___x_1490_ = v___x_1476_;
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
else
{
lean_inc(v_a_1488_);
lean_dec(v___x_1476_);
v___x_1490_ = lean_box(0);
v_isShared_1491_ = v_isSharedCheck_1495_;
goto v_resetjp_1489_;
}
v_resetjp_1489_:
{
lean_object* v___x_1493_; 
if (v_isShared_1491_ == 0)
{
v___x_1493_ = v___x_1490_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1494_; 
v_reuseFailAlloc_1494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1494_, 0, v_a_1488_);
v___x_1493_ = v_reuseFailAlloc_1494_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
return v___x_1493_;
}
}
}
}
else
{
lean_object* v_a_1496_; lean_object* v___x_1498_; uint8_t v_isShared_1499_; uint8_t v_isSharedCheck_1503_; 
lean_dec(v_fst_1471_);
v_a_1496_ = lean_ctor_get(v___x_1474_, 0);
v_isSharedCheck_1503_ = !lean_is_exclusive(v___x_1474_);
if (v_isSharedCheck_1503_ == 0)
{
v___x_1498_ = v___x_1474_;
v_isShared_1499_ = v_isSharedCheck_1503_;
goto v_resetjp_1497_;
}
else
{
lean_inc(v_a_1496_);
lean_dec(v___x_1474_);
v___x_1498_ = lean_box(0);
v_isShared_1499_ = v_isSharedCheck_1503_;
goto v_resetjp_1497_;
}
v_resetjp_1497_:
{
lean_object* v___x_1501_; 
if (v_isShared_1499_ == 0)
{
v___x_1501_ = v___x_1498_;
goto v_reusejp_1500_;
}
else
{
lean_object* v_reuseFailAlloc_1502_; 
v_reuseFailAlloc_1502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1502_, 0, v_a_1496_);
v___x_1501_ = v_reuseFailAlloc_1502_;
goto v_reusejp_1500_;
}
v_reusejp_1500_:
{
return v___x_1501_;
}
}
}
}
else
{
lean_object* v_a_1504_; lean_object* v___x_1506_; uint8_t v_isShared_1507_; uint8_t v_isSharedCheck_1511_; 
v_a_1504_ = lean_ctor_get(v___x_1469_, 0);
v_isSharedCheck_1511_ = !lean_is_exclusive(v___x_1469_);
if (v_isSharedCheck_1511_ == 0)
{
v___x_1506_ = v___x_1469_;
v_isShared_1507_ = v_isSharedCheck_1511_;
goto v_resetjp_1505_;
}
else
{
lean_inc(v_a_1504_);
lean_dec(v___x_1469_);
v___x_1506_ = lean_box(0);
v_isShared_1507_ = v_isSharedCheck_1511_;
goto v_resetjp_1505_;
}
v_resetjp_1505_:
{
lean_object* v___x_1509_; 
if (v_isShared_1507_ == 0)
{
v___x_1509_ = v___x_1506_;
goto v_reusejp_1508_;
}
else
{
lean_object* v_reuseFailAlloc_1510_; 
v_reuseFailAlloc_1510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1510_, 0, v_a_1504_);
v___x_1509_ = v_reuseFailAlloc_1510_;
goto v_reusejp_1508_;
}
v_reusejp_1508_:
{
return v___x_1509_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___boxed(lean_object* v_mvs_1512_, lean_object* v_a_1513_, lean_object* v_a_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_){
_start:
{
lean_object* v_res_1518_; 
v_res_1518_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl(v_mvs_1512_, v_a_1513_, v_a_1514_, v_a_1515_, v_a_1516_);
lean_dec(v_a_1516_);
lean_dec_ref(v_a_1515_);
lean_dec(v_a_1514_);
lean_dec_ref(v_a_1513_);
return v_res_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_try__rfl(lean_object* v_mvs_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_){
_start:
{
lean_object* v___x_1525_; 
v___x_1525_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl(v_mvs_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_);
return v___x_1525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_try__rfl___boxed(lean_object* v_mvs_1526_, lean_object* v_a_1527_, lean_object* v_a_1528_, lean_object* v_a_1529_, lean_object* v_a_1530_, lean_object* v_a_1531_){
_start:
{
lean_object* v_res_1532_; 
v_res_1532_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_try__rfl(v_mvs_1526_, v_a_1527_, v_a_1528_, v_a_1529_, v_a_1530_);
lean_dec(v_a_1530_);
lean_dec_ref(v_a_1529_);
lean_dec(v_a_1528_);
lean_dec_ref(v_a_1527_);
return v_res_1532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__1(lean_object* v_x_1533_, lean_object* v_x_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_, lean_object* v___y_1537_, lean_object* v___y_1538_){
_start:
{
if (lean_obj_tag(v_x_1533_) == 0)
{
lean_object* v___x_1540_; lean_object* v___x_1541_; 
v___x_1540_ = l_List_reverse___redArg(v_x_1534_);
v___x_1541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1541_, 0, v___x_1540_);
return v___x_1541_;
}
else
{
lean_object* v_head_1542_; lean_object* v_tail_1543_; lean_object* v___x_1545_; uint8_t v_isShared_1546_; uint8_t v_isSharedCheck_1575_; 
v_head_1542_ = lean_ctor_get(v_x_1533_, 0);
v_tail_1543_ = lean_ctor_get(v_x_1533_, 1);
v_isSharedCheck_1575_ = !lean_is_exclusive(v_x_1533_);
if (v_isSharedCheck_1575_ == 0)
{
v___x_1545_ = v_x_1533_;
v_isShared_1546_ = v_isSharedCheck_1575_;
goto v_resetjp_1544_;
}
else
{
lean_inc(v_tail_1543_);
lean_inc(v_head_1542_);
lean_dec(v_x_1533_);
v___x_1545_ = lean_box(0);
v_isShared_1546_ = v_isSharedCheck_1575_;
goto v_resetjp_1544_;
}
v_resetjp_1544_:
{
lean_object* v___x_1547_; 
lean_inc(v_head_1542_);
v___x_1547_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_head_1542_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_);
if (lean_obj_tag(v___x_1547_) == 0)
{
lean_object* v_a_1548_; lean_object* v___x_1549_; uint8_t v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; 
v_a_1548_ = lean_ctor_get(v___x_1547_, 0);
lean_inc(v_a_1548_);
lean_dec_ref_known(v___x_1547_, 1);
v___x_1549_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs(v_a_1548_);
v___x_1550_ = 0;
v___x_1551_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma(v___x_1549_, v___x_1550_);
v___x_1552_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__2___closed__0));
v___x_1553_ = l_Lean_MVarId_applyConst(v_head_1542_, v___x_1551_, v___x_1552_, v___y_1535_, v___y_1536_, v___y_1537_, v___y_1538_);
if (lean_obj_tag(v___x_1553_) == 0)
{
lean_object* v_a_1554_; lean_object* v___x_1556_; 
v_a_1554_ = lean_ctor_get(v___x_1553_, 0);
lean_inc(v_a_1554_);
lean_dec_ref_known(v___x_1553_, 1);
if (v_isShared_1546_ == 0)
{
lean_ctor_set(v___x_1545_, 1, v_x_1534_);
lean_ctor_set(v___x_1545_, 0, v_a_1554_);
v___x_1556_ = v___x_1545_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v_a_1554_);
lean_ctor_set(v_reuseFailAlloc_1558_, 1, v_x_1534_);
v___x_1556_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
v_x_1533_ = v_tail_1543_;
v_x_1534_ = v___x_1556_;
goto _start;
}
}
else
{
lean_object* v_a_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1566_; 
lean_del_object(v___x_1545_);
lean_dec(v_tail_1543_);
lean_dec(v_x_1534_);
v_a_1559_ = lean_ctor_get(v___x_1553_, 0);
v_isSharedCheck_1566_ = !lean_is_exclusive(v___x_1553_);
if (v_isSharedCheck_1566_ == 0)
{
v___x_1561_ = v___x_1553_;
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_a_1559_);
lean_dec(v___x_1553_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
lean_object* v___x_1564_; 
if (v_isShared_1562_ == 0)
{
v___x_1564_ = v___x_1561_;
goto v_reusejp_1563_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v_a_1559_);
v___x_1564_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1563_;
}
v_reusejp_1563_:
{
return v___x_1564_;
}
}
}
}
else
{
lean_object* v_a_1567_; lean_object* v___x_1569_; uint8_t v_isShared_1570_; uint8_t v_isSharedCheck_1574_; 
lean_del_object(v___x_1545_);
lean_dec(v_tail_1543_);
lean_dec(v_head_1542_);
lean_dec(v_x_1534_);
v_a_1567_ = lean_ctor_get(v___x_1547_, 0);
v_isSharedCheck_1574_ = !lean_is_exclusive(v___x_1547_);
if (v_isSharedCheck_1574_ == 0)
{
v___x_1569_ = v___x_1547_;
v_isShared_1570_ = v_isSharedCheck_1574_;
goto v_resetjp_1568_;
}
else
{
lean_inc(v_a_1567_);
lean_dec(v___x_1547_);
v___x_1569_ = lean_box(0);
v_isShared_1570_ = v_isSharedCheck_1574_;
goto v_resetjp_1568_;
}
v_resetjp_1568_:
{
lean_object* v___x_1572_; 
if (v_isShared_1570_ == 0)
{
v___x_1572_ = v___x_1569_;
goto v_reusejp_1571_;
}
else
{
lean_object* v_reuseFailAlloc_1573_; 
v_reuseFailAlloc_1573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1573_, 0, v_a_1567_);
v___x_1572_ = v_reuseFailAlloc_1573_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
return v___x_1572_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__1___boxed(lean_object* v_x_1576_, lean_object* v_x_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_){
_start:
{
lean_object* v_res_1583_; 
v_res_1583_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__1(v_x_1576_, v_x_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_);
lean_dec(v___y_1581_);
lean_dec_ref(v___y_1580_);
lean_dec(v___y_1579_);
lean_dec_ref(v___y_1578_);
return v_res_1583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__0(lean_object* v_a_1584_, lean_object* v_a_1585_, lean_object* v_a_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_, lean_object* v___y_1590_){
_start:
{
if (lean_obj_tag(v_a_1584_) == 0)
{
lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; 
v___x_1592_ = lean_array_to_list(v_a_1585_);
v___x_1593_ = lean_array_to_list(v_a_1586_);
v___x_1594_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1594_, 0, v___x_1592_);
lean_ctor_set(v___x_1594_, 1, v___x_1593_);
v___x_1595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1595_, 0, v___x_1594_);
return v___x_1595_;
}
else
{
lean_object* v_head_1596_; lean_object* v_tail_1597_; lean_object* v___x_1598_; 
v_head_1596_ = lean_ctor_get(v_a_1584_, 0);
lean_inc_n(v_head_1596_, 2);
v_tail_1597_ = lean_ctor_get(v_a_1584_, 1);
lean_inc(v_tail_1597_);
lean_dec_ref_known(v_a_1584_, 2);
v___x_1598_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_head_1596_, v___y_1587_, v___y_1588_, v___y_1589_, v___y_1590_);
if (lean_obj_tag(v___x_1598_) == 0)
{
lean_object* v_a_1599_; lean_object* v___x_1600_; uint8_t v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; uint8_t v___x_1604_; 
v_a_1599_ = lean_ctor_get(v___x_1598_, 0);
lean_inc(v_a_1599_);
lean_dec_ref_known(v___x_1598_, 1);
v___x_1600_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs(v_a_1599_);
v___x_1601_ = 0;
v___x_1602_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma(v___x_1600_, v___x_1601_);
v___x_1603_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___lam__0___closed__1));
v___x_1604_ = lean_name_eq(v___x_1602_, v___x_1603_);
lean_dec(v___x_1602_);
if (v___x_1604_ == 0)
{
lean_object* v___x_1605_; 
v___x_1605_ = lean_array_push(v_a_1585_, v_head_1596_);
v_a_1584_ = v_tail_1597_;
v_a_1585_ = v___x_1605_;
goto _start;
}
else
{
lean_object* v___x_1607_; 
v___x_1607_ = lean_array_push(v_a_1586_, v_head_1596_);
v_a_1584_ = v_tail_1597_;
v_a_1586_ = v___x_1607_;
goto _start;
}
}
else
{
lean_object* v_a_1609_; lean_object* v___x_1611_; uint8_t v_isShared_1612_; uint8_t v_isSharedCheck_1616_; 
lean_dec(v_tail_1597_);
lean_dec(v_head_1596_);
lean_dec_ref(v_a_1586_);
lean_dec_ref(v_a_1585_);
v_a_1609_ = lean_ctor_get(v___x_1598_, 0);
v_isSharedCheck_1616_ = !lean_is_exclusive(v___x_1598_);
if (v_isSharedCheck_1616_ == 0)
{
v___x_1611_ = v___x_1598_;
v_isShared_1612_ = v_isSharedCheck_1616_;
goto v_resetjp_1610_;
}
else
{
lean_inc(v_a_1609_);
lean_dec(v___x_1598_);
v___x_1611_ = lean_box(0);
v_isShared_1612_ = v_isSharedCheck_1616_;
goto v_resetjp_1610_;
}
v_resetjp_1610_:
{
lean_object* v___x_1614_; 
if (v_isShared_1612_ == 0)
{
v___x_1614_ = v___x_1611_;
goto v_reusejp_1613_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v_a_1609_);
v___x_1614_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1613_;
}
v_reusejp_1613_:
{
return v___x_1614_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__0___boxed(lean_object* v_a_1617_, lean_object* v_a_1618_, lean_object* v_a_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_){
_start:
{
lean_object* v_res_1625_; 
v_res_1625_ = lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__0(v_a_1617_, v_a_1618_, v_a_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_);
lean_dec(v___y_1623_);
lean_dec_ref(v___y_1622_);
lean_dec(v___y_1621_);
lean_dec_ref(v___y_1620_);
return v_res_1625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_splitApply(lean_object* v_mvs_1626_, lean_object* v_static_1627_, lean_object* v_a_1628_, lean_object* v_a_1629_, lean_object* v_a_1630_, lean_object* v_a_1631_){
_start:
{
lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1633_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl___closed__0));
v___x_1634_ = lp_mathlib___private_Init_Data_List_BasicAux_0__List_partitionM_go___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__0(v_mvs_1626_, v___x_1633_, v___x_1633_, v_a_1628_, v_a_1629_, v_a_1630_, v_a_1631_);
if (lean_obj_tag(v___x_1634_) == 0)
{
lean_object* v_a_1635_; lean_object* v_fst_1636_; lean_object* v_snd_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1664_; 
v_a_1635_ = lean_ctor_get(v___x_1634_, 0);
lean_inc(v_a_1635_);
lean_dec_ref_known(v___x_1634_, 1);
v_fst_1636_ = lean_ctor_get(v_a_1635_, 0);
v_snd_1637_ = lean_ctor_get(v_a_1635_, 1);
v_isSharedCheck_1664_ = !lean_is_exclusive(v_a_1635_);
if (v_isSharedCheck_1664_ == 0)
{
v___x_1639_ = v_a_1635_;
v_isShared_1640_ = v_isSharedCheck_1664_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_snd_1637_);
lean_inc(v_fst_1636_);
lean_dec(v_a_1635_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1664_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___x_1641_; lean_object* v___x_1642_; 
v___x_1641_ = lean_box(0);
v___x_1642_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_splitApply_spec__1(v_fst_1636_, v___x_1641_, v_a_1628_, v_a_1629_, v_a_1630_, v_a_1631_);
if (lean_obj_tag(v___x_1642_) == 0)
{
lean_object* v_a_1643_; lean_object* v___x_1645_; uint8_t v_isShared_1646_; uint8_t v_isSharedCheck_1655_; 
v_a_1643_ = lean_ctor_get(v___x_1642_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1642_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1645_ = v___x_1642_;
v_isShared_1646_ = v_isSharedCheck_1655_;
goto v_resetjp_1644_;
}
else
{
lean_inc(v_a_1643_);
lean_dec(v___x_1642_);
v___x_1645_ = lean_box(0);
v_isShared_1646_ = v_isSharedCheck_1655_;
goto v_resetjp_1644_;
}
v_resetjp_1644_:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1650_; 
v___x_1647_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__4(v_a_1643_, v___x_1633_);
v___x_1648_ = l_List_appendTR___redArg(v_static_1627_, v_snd_1637_);
if (v_isShared_1640_ == 0)
{
lean_ctor_set(v___x_1639_, 1, v___x_1648_);
lean_ctor_set(v___x_1639_, 0, v___x_1647_);
v___x_1650_ = v___x_1639_;
goto v_reusejp_1649_;
}
else
{
lean_object* v_reuseFailAlloc_1654_; 
v_reuseFailAlloc_1654_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1654_, 0, v___x_1647_);
lean_ctor_set(v_reuseFailAlloc_1654_, 1, v___x_1648_);
v___x_1650_ = v_reuseFailAlloc_1654_;
goto v_reusejp_1649_;
}
v_reusejp_1649_:
{
lean_object* v___x_1652_; 
if (v_isShared_1646_ == 0)
{
lean_ctor_set(v___x_1645_, 0, v___x_1650_);
v___x_1652_ = v___x_1645_;
goto v_reusejp_1651_;
}
else
{
lean_object* v_reuseFailAlloc_1653_; 
v_reuseFailAlloc_1653_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1653_, 0, v___x_1650_);
v___x_1652_ = v_reuseFailAlloc_1653_;
goto v_reusejp_1651_;
}
v_reusejp_1651_:
{
return v___x_1652_;
}
}
}
}
else
{
lean_object* v_a_1656_; lean_object* v___x_1658_; uint8_t v_isShared_1659_; uint8_t v_isSharedCheck_1663_; 
lean_del_object(v___x_1639_);
lean_dec(v_snd_1637_);
lean_dec(v_static_1627_);
v_a_1656_ = lean_ctor_get(v___x_1642_, 0);
v_isSharedCheck_1663_ = !lean_is_exclusive(v___x_1642_);
if (v_isSharedCheck_1663_ == 0)
{
v___x_1658_ = v___x_1642_;
v_isShared_1659_ = v_isSharedCheck_1663_;
goto v_resetjp_1657_;
}
else
{
lean_inc(v_a_1656_);
lean_dec(v___x_1642_);
v___x_1658_ = lean_box(0);
v_isShared_1659_ = v_isSharedCheck_1663_;
goto v_resetjp_1657_;
}
v_resetjp_1657_:
{
lean_object* v___x_1661_; 
if (v_isShared_1659_ == 0)
{
v___x_1661_ = v___x_1658_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1662_; 
v_reuseFailAlloc_1662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1662_, 0, v_a_1656_);
v___x_1661_ = v_reuseFailAlloc_1662_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
return v___x_1661_;
}
}
}
}
}
else
{
lean_dec(v_static_1627_);
return v___x_1634_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_splitApply___boxed(lean_object* v_mvs_1665_, lean_object* v_static_1666_, lean_object* v_a_1667_, lean_object* v_a_1668_, lean_object* v_a_1669_, lean_object* v_a_1670_, lean_object* v_a_1671_){
_start:
{
lean_object* v_res_1672_; 
v_res_1672_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_splitApply(v_mvs_1665_, v_static_1666_, v_a_1667_, v_a_1668_, v_a_1669_, v_a_1670_);
lean_dec(v_a_1670_);
lean_dec_ref(v_a_1669_);
lean_dec(v_a_1668_);
lean_dec_ref(v_a_1667_);
return v_res_1672_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__1(void){
_start:
{
lean_object* v___x_1674_; lean_object* v___x_1675_; 
v___x_1674_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__0));
v___x_1675_ = l_Lean_stringToMessageData(v___x_1674_);
return v___x_1675_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__6(void){
_start:
{
lean_object* v___x_1683_; lean_object* v___x_1684_; 
v___x_1683_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__5));
v___x_1684_ = l_Lean_stringToMessageData(v___x_1683_);
return v___x_1684_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__8(void){
_start:
{
lean_object* v___x_1686_; lean_object* v___x_1687_; 
v___x_1686_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__7));
v___x_1687_ = l_Lean_stringToMessageData(v___x_1686_);
return v___x_1687_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__10(void){
_start:
{
lean_object* v___x_1689_; lean_object* v___x_1690_; 
v___x_1689_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__9));
v___x_1690_ = l_Lean_stringToMessageData(v___x_1689_);
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f(lean_object* v_deg_1691_, lean_object* v_x_1692_){
_start:
{
if (lean_obj_tag(v_x_1692_) == 0)
{
lean_object* v___x_1693_; 
lean_dec_ref(v_deg_1691_);
v___x_1693_ = lean_box(0);
return v___x_1693_;
}
else
{
lean_object* v_head_1694_; lean_object* v_tail_1695_; lean_object* v___x_1697_; uint8_t v_isShared_1698_; uint8_t v_isSharedCheck_1739_; 
v_head_1694_ = lean_ctor_get(v_x_1692_, 0);
v_tail_1695_ = lean_ctor_get(v_x_1692_, 1);
v_isSharedCheck_1739_ = !lean_is_exclusive(v_x_1692_);
if (v_isSharedCheck_1739_ == 0)
{
v___x_1697_ = v_x_1692_;
v_isShared_1698_ = v_isSharedCheck_1739_;
goto v_resetjp_1696_;
}
else
{
lean_inc(v_tail_1695_);
lean_inc(v_head_1694_);
lean_dec(v_x_1692_);
v___x_1697_ = lean_box(0);
v_isShared_1698_ = v_isSharedCheck_1739_;
goto v_resetjp_1696_;
}
v_resetjp_1696_:
{
lean_object* v_rest_1699_; lean_object* v___x_1712_; lean_object* v___x_1713_; uint8_t v___x_1714_; 
lean_inc_ref(v_deg_1691_);
v_rest_1699_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f(v_deg_1691_, v_tail_1695_);
v___x_1712_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__3));
v___x_1713_ = lean_unsigned_to_nat(3u);
v___x_1714_ = l_Lean_Expr_isAppOfArity(v_head_1694_, v___x_1712_, v___x_1713_);
if (v___x_1714_ == 0)
{
lean_object* v___x_1715_; lean_object* v___x_1716_; uint8_t v___x_1717_; 
lean_dec_ref(v_deg_1691_);
v___x_1715_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__4));
v___x_1716_ = lean_unsigned_to_nat(4u);
v___x_1717_ = l_Lean_Expr_isAppOfArity(v_head_1694_, v___x_1715_, v___x_1716_);
if (v___x_1717_ == 0)
{
goto v___jp_1700_;
}
else
{
lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; 
v___x_1718_ = l_Lean_Expr_appFn_x21(v_head_1694_);
v___x_1719_ = l_Lean_Expr_appFn_x21(v___x_1718_);
v___x_1720_ = l_Lean_Expr_appFn_x21(v___x_1719_);
lean_dec_ref(v___x_1719_);
v___x_1721_ = l_Lean_Expr_appArg_x21(v___x_1720_);
lean_dec_ref(v___x_1720_);
if (lean_obj_tag(v___x_1721_) == 4)
{
lean_object* v_declName_1722_; 
v_declName_1722_ = lean_ctor_get(v___x_1721_, 0);
lean_inc(v_declName_1722_);
if (lean_obj_tag(v_declName_1722_) == 1)
{
lean_object* v_pre_1723_; 
v_pre_1723_ = lean_ctor_get(v_declName_1722_, 0);
if (lean_obj_tag(v_pre_1723_) == 0)
{
lean_object* v_us_1724_; lean_object* v_str_1725_; lean_object* v___x_1726_; uint8_t v___x_1727_; 
v_us_1724_ = lean_ctor_get(v___x_1721_, 1);
lean_inc(v_us_1724_);
lean_dec_ref_known(v___x_1721_, 2);
v_str_1725_ = lean_ctor_get(v_declName_1722_, 1);
lean_inc_ref(v_str_1725_);
lean_dec_ref_known(v_declName_1722_, 2);
v___x_1726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__35));
v___x_1727_ = lean_string_dec_eq(v_str_1725_, v___x_1726_);
lean_dec_ref(v_str_1725_);
if (v___x_1727_ == 0)
{
lean_dec(v_us_1724_);
lean_dec_ref(v___x_1718_);
goto v___jp_1700_;
}
else
{
if (lean_obj_tag(v_us_1724_) == 0)
{
lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; 
lean_del_object(v___x_1697_);
lean_dec(v_head_1694_);
v___x_1728_ = l_Lean_Expr_appArg_x21(v___x_1718_);
lean_dec_ref(v___x_1718_);
v___x_1729_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__6, &lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__6);
v___x_1730_ = l_Lean_MessageData_ofExpr(v___x_1728_);
v___x_1731_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1731_, 0, v___x_1729_);
lean_ctor_set(v___x_1731_, 1, v___x_1730_);
v___x_1732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1732_, 0, v___x_1731_);
lean_ctor_set(v___x_1732_, 1, v_rest_1699_);
return v___x_1732_;
}
else
{
lean_dec(v_us_1724_);
lean_dec_ref(v___x_1718_);
goto v___jp_1700_;
}
}
}
else
{
lean_dec_ref_known(v_declName_1722_, 2);
lean_dec_ref_known(v___x_1721_, 2);
lean_dec_ref(v___x_1718_);
goto v___jp_1700_;
}
}
else
{
lean_dec(v_declName_1722_);
lean_dec_ref_known(v___x_1721_, 2);
lean_dec_ref(v___x_1718_);
goto v___jp_1700_;
}
}
else
{
lean_dec_ref(v___x_1721_);
lean_dec_ref(v___x_1718_);
goto v___jp_1700_;
}
}
}
else
{
lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; 
lean_del_object(v___x_1697_);
lean_dec(v_head_1694_);
v___x_1733_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__8, &lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__8);
v___x_1734_ = l_Lean_MessageData_ofExpr(v_deg_1691_);
v___x_1735_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1735_, 0, v___x_1733_);
lean_ctor_set(v___x_1735_, 1, v___x_1734_);
v___x_1736_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__10, &lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__10);
v___x_1737_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1737_, 0, v___x_1735_);
lean_ctor_set(v___x_1737_, 1, v___x_1736_);
v___x_1738_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1738_, 0, v___x_1737_);
lean_ctor_set(v___x_1738_, 1, v_rest_1699_);
return v___x_1738_;
}
v___jp_1700_:
{
lean_object* v___x_1701_; lean_object* v___x_1702_; uint8_t v___x_1703_; 
v___x_1701_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___closed__0));
v___x_1702_ = lean_unsigned_to_nat(3u);
v___x_1703_ = l_Lean_Expr_isAppOfArity(v_head_1694_, v___x_1701_, v___x_1702_);
if (v___x_1703_ == 0)
{
lean_del_object(v___x_1697_);
lean_dec(v_head_1694_);
return v_rest_1699_;
}
else
{
lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1710_; 
v___x_1704_ = l_Lean_Expr_appFn_x21(v_head_1694_);
lean_dec(v_head_1694_);
v___x_1705_ = l_Lean_Expr_appArg_x21(v___x_1704_);
lean_dec_ref(v___x_1704_);
v___x_1706_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__1, &lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f___closed__1);
v___x_1707_ = l_Lean_MessageData_ofExpr(v___x_1705_);
v___x_1708_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1708_, 0, v___x_1706_);
lean_ctor_set(v___x_1708_, 1, v___x_1707_);
if (v_isShared_1698_ == 0)
{
lean_ctor_set(v___x_1697_, 1, v_rest_1699_);
lean_ctor_set(v___x_1697_, 0, v___x_1708_);
v___x_1710_ = v___x_1697_;
goto v_reusejp_1709_;
}
else
{
lean_object* v_reuseFailAlloc_1711_; 
v_reuseFailAlloc_1711_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1711_, 0, v___x_1708_);
lean_ctor_set(v_reuseFailAlloc_1711_, 1, v_rest_1699_);
v___x_1710_ = v_reuseFailAlloc_1711_;
goto v_reusejp_1709_;
}
v_reusejp_1709_:
{
return v___x_1710_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1831_; uint8_t v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; 
v___x_1831_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__0_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_));
v___x_1832_ = 0;
v___x_1833_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn___closed__22_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_));
v___x_1834_ = l_Lean_registerTraceClass(v___x_1831_, v___x_1832_, v___x_1833_);
return v___x_1834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2____boxed(lean_object* v_a_1835_){
_start:
{
lean_object* v_res_1836_; 
v_res_1836_ = lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_();
return v_res_1836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1(lean_object* v_x_1855_, lean_object* v_a_1856_, lean_object* v_a_1857_){
_start:
{
lean_object* v___x_1858_; uint8_t v___x_1859_; 
v___x_1858_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1));
v___x_1859_ = l_Lean_Syntax_isOfKind(v_x_1855_, v___x_1858_);
if (v___x_1859_ == 0)
{
lean_object* v___x_1860_; lean_object* v___x_1861_; 
v___x_1860_ = lean_box(1);
v___x_1861_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1861_, 0, v___x_1860_);
lean_ctor_set(v___x_1861_, 1, v_a_1857_);
return v___x_1861_;
}
else
{
lean_object* v_ref_1862_; uint8_t v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; 
v_ref_1862_ = lean_ctor_get(v_a_1856_, 5);
v___x_1863_ = 0;
v___x_1864_ = l_Lean_SourceInfo_fromRef(v_ref_1862_, v___x_1863_);
v___x_1865_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1));
v___x_1866_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4));
lean_inc_n(v___x_1864_, 3);
v___x_1867_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1867_, 0, v___x_1864_);
lean_ctor_set(v___x_1867_, 1, v___x_1866_);
v___x_1868_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1));
v___x_1869_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__8));
v___x_1870_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1870_, 0, v___x_1864_);
lean_ctor_set(v___x_1870_, 1, v___x_1869_);
v___x_1871_ = l_Lean_Syntax_node1(v___x_1864_, v___x_1868_, v___x_1870_);
v___x_1872_ = l_Lean_Syntax_node2(v___x_1864_, v___x_1865_, v___x_1867_, v___x_1871_);
v___x_1873_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1873_, 0, v___x_1872_);
lean_ctor_set(v___x_1873_, 1, v_a_1857_);
return v___x_1873_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___boxed(lean_object* v_x_1874_, lean_object* v_a_1875_, lean_object* v_a_1876_){
_start:
{
lean_object* v_res_1877_; 
v_res_1877_ = lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1(v_x_1874_, v_a_1875_, v_a_1876_);
lean_dec_ref(v_a_1875_);
return v_res_1877_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; 
v___x_1878_ = lean_box(0);
v___x_1879_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1880_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1880_, 0, v___x_1879_);
lean_ctor_set(v___x_1880_, 1, v___x_1878_);
return v___x_1880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg(){
_start:
{
lean_object* v___x_1882_; lean_object* v___x_1883_; 
v___x_1882_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___closed__0);
v___x_1883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1883_, 0, v___x_1882_);
return v___x_1883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg___boxed(lean_object* v___y_1884_){
_start:
{
lean_object* v_res_1885_; 
v_res_1885_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg();
return v_res_1885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1(lean_object* v_00_u03b1_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_){
_start:
{
lean_object* v___x_1896_; 
v___x_1896_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg();
return v___x_1896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___boxed(lean_object* v_00_u03b1_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_){
_start:
{
lean_object* v_res_1907_; 
v_res_1907_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1(v_00_u03b1_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v___y_1901_);
lean_dec_ref(v___y_1900_);
lean_dec(v___y_1899_);
lean_dec_ref(v___y_1898_);
return v_res_1907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___lam__0(lean_object* v_x_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
lean_object* v___x_1918_; 
lean_inc(v___y_1912_);
lean_inc_ref(v___y_1911_);
lean_inc(v___y_1910_);
lean_inc_ref(v___y_1909_);
v___x_1918_ = lean_apply_9(v_x_1908_, v___y_1909_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_, v___y_1916_, lean_box(0));
return v___x_1918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___lam__0___boxed(lean_object* v_x_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_){
_start:
{
lean_object* v_res_1929_; 
v_res_1929_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___lam__0(v_x_1919_, v___y_1920_, v___y_1921_, v___y_1922_, v___y_1923_, v___y_1924_, v___y_1925_, v___y_1926_, v___y_1927_);
lean_dec(v___y_1923_);
lean_dec_ref(v___y_1922_);
lean_dec(v___y_1921_);
lean_dec_ref(v___y_1920_);
return v_res_1929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg(lean_object* v_mvarId_1930_, lean_object* v_x_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_){
_start:
{
lean_object* v___f_1941_; lean_object* v___x_1942_; 
lean_inc(v___y_1935_);
lean_inc_ref(v___y_1934_);
lean_inc(v___y_1933_);
lean_inc_ref(v___y_1932_);
v___f_1941_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1941_, 0, v_x_1931_);
lean_closure_set(v___f_1941_, 1, v___y_1932_);
lean_closure_set(v___f_1941_, 2, v___y_1933_);
lean_closure_set(v___f_1941_, 3, v___y_1934_);
lean_closure_set(v___f_1941_, 4, v___y_1935_);
v___x_1942_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1930_, v___f_1941_, v___y_1936_, v___y_1937_, v___y_1938_, v___y_1939_);
if (lean_obj_tag(v___x_1942_) == 0)
{
return v___x_1942_;
}
else
{
lean_object* v_a_1943_; lean_object* v___x_1945_; uint8_t v_isShared_1946_; uint8_t v_isSharedCheck_1950_; 
v_a_1943_ = lean_ctor_get(v___x_1942_, 0);
v_isSharedCheck_1950_ = !lean_is_exclusive(v___x_1942_);
if (v_isSharedCheck_1950_ == 0)
{
v___x_1945_ = v___x_1942_;
v_isShared_1946_ = v_isSharedCheck_1950_;
goto v_resetjp_1944_;
}
else
{
lean_inc(v_a_1943_);
lean_dec(v___x_1942_);
v___x_1945_ = lean_box(0);
v_isShared_1946_ = v_isSharedCheck_1950_;
goto v_resetjp_1944_;
}
v_resetjp_1944_:
{
lean_object* v___x_1948_; 
if (v_isShared_1946_ == 0)
{
v___x_1948_ = v___x_1945_;
goto v_reusejp_1947_;
}
else
{
lean_object* v_reuseFailAlloc_1949_; 
v_reuseFailAlloc_1949_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1949_, 0, v_a_1943_);
v___x_1948_ = v_reuseFailAlloc_1949_;
goto v_reusejp_1947_;
}
v_reusejp_1947_:
{
return v___x_1948_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg___boxed(lean_object* v_mvarId_1951_, lean_object* v_x_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_){
_start:
{
lean_object* v_res_1962_; 
v_res_1962_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg(v_mvarId_1951_, v_x_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
lean_dec(v___y_1956_);
lean_dec_ref(v___y_1955_);
lean_dec(v___y_1954_);
lean_dec_ref(v___y_1953_);
return v_res_1962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2(lean_object* v_00_u03b1_1963_, lean_object* v_mvarId_1964_, lean_object* v_x_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_){
_start:
{
lean_object* v___x_1975_; 
v___x_1975_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg(v_mvarId_1964_, v_x_1965_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_, v___y_1971_, v___y_1972_, v___y_1973_);
return v___x_1975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___boxed(lean_object* v_00_u03b1_1976_, lean_object* v_mvarId_1977_, lean_object* v_x_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_){
_start:
{
lean_object* v_res_1988_; 
v_res_1988_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2(v_00_u03b1_1976_, v_mvarId_1977_, v_x_1978_, v___y_1979_, v___y_1980_, v___y_1981_, v___y_1982_, v___y_1983_, v___y_1984_, v___y_1985_, v___y_1986_);
lean_dec(v___y_1986_);
lean_dec_ref(v___y_1985_);
lean_dec(v___y_1984_);
lean_dec_ref(v___y_1983_);
lean_dec(v___y_1982_);
lean_dec_ref(v___y_1981_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
return v_res_1988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4(lean_object* v_msgData_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_){
_start:
{
lean_object* v___x_1995_; lean_object* v_env_1996_; lean_object* v___x_1997_; lean_object* v_mctx_1998_; lean_object* v_lctx_1999_; lean_object* v_options_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; 
v___x_1995_ = lean_st_ref_get(v___y_1993_);
v_env_1996_ = lean_ctor_get(v___x_1995_, 0);
lean_inc_ref(v_env_1996_);
lean_dec(v___x_1995_);
v___x_1997_ = lean_st_ref_get(v___y_1991_);
v_mctx_1998_ = lean_ctor_get(v___x_1997_, 0);
lean_inc_ref(v_mctx_1998_);
lean_dec(v___x_1997_);
v_lctx_1999_ = lean_ctor_get(v___y_1990_, 2);
v_options_2000_ = lean_ctor_get(v___y_1992_, 2);
lean_inc_ref(v_options_2000_);
lean_inc_ref(v_lctx_1999_);
v___x_2001_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2001_, 0, v_env_1996_);
lean_ctor_set(v___x_2001_, 1, v_mctx_1998_);
lean_ctor_set(v___x_2001_, 2, v_lctx_1999_);
lean_ctor_set(v___x_2001_, 3, v_options_2000_);
v___x_2002_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2002_, 0, v___x_2001_);
lean_ctor_set(v___x_2002_, 1, v_msgData_1989_);
v___x_2003_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2003_, 0, v___x_2002_);
return v___x_2003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4___boxed(lean_object* v_msgData_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_){
_start:
{
lean_object* v_res_2010_; 
v_res_2010_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4(v_msgData_2004_, v___y_2005_, v___y_2006_, v___y_2007_, v___y_2008_);
lean_dec(v___y_2008_);
lean_dec_ref(v___y_2007_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
return v_res_2010_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__0(void){
_start:
{
lean_object* v___x_2011_; double v___x_2012_; 
v___x_2011_ = lean_unsigned_to_nat(0u);
v___x_2012_ = lean_float_of_nat(v___x_2011_);
return v___x_2012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg(lean_object* v_cls_2016_, lean_object* v_msg_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_){
_start:
{
lean_object* v_ref_2023_; lean_object* v___x_2024_; lean_object* v_a_2025_; lean_object* v___x_2027_; uint8_t v_isShared_2028_; uint8_t v_isSharedCheck_2069_; 
v_ref_2023_ = lean_ctor_get(v___y_2020_, 5);
v___x_2024_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4(v_msg_2017_, v___y_2018_, v___y_2019_, v___y_2020_, v___y_2021_);
v_a_2025_ = lean_ctor_get(v___x_2024_, 0);
v_isSharedCheck_2069_ = !lean_is_exclusive(v___x_2024_);
if (v_isSharedCheck_2069_ == 0)
{
v___x_2027_ = v___x_2024_;
v_isShared_2028_ = v_isSharedCheck_2069_;
goto v_resetjp_2026_;
}
else
{
lean_inc(v_a_2025_);
lean_dec(v___x_2024_);
v___x_2027_ = lean_box(0);
v_isShared_2028_ = v_isSharedCheck_2069_;
goto v_resetjp_2026_;
}
v_resetjp_2026_:
{
lean_object* v___x_2029_; lean_object* v_traceState_2030_; lean_object* v_env_2031_; lean_object* v_nextMacroScope_2032_; lean_object* v_ngen_2033_; lean_object* v_auxDeclNGen_2034_; lean_object* v_cache_2035_; lean_object* v_messages_2036_; lean_object* v_infoState_2037_; lean_object* v_snapshotTasks_2038_; lean_object* v___x_2040_; uint8_t v_isShared_2041_; uint8_t v_isSharedCheck_2068_; 
v___x_2029_ = lean_st_ref_take(v___y_2021_);
v_traceState_2030_ = lean_ctor_get(v___x_2029_, 4);
v_env_2031_ = lean_ctor_get(v___x_2029_, 0);
v_nextMacroScope_2032_ = lean_ctor_get(v___x_2029_, 1);
v_ngen_2033_ = lean_ctor_get(v___x_2029_, 2);
v_auxDeclNGen_2034_ = lean_ctor_get(v___x_2029_, 3);
v_cache_2035_ = lean_ctor_get(v___x_2029_, 5);
v_messages_2036_ = lean_ctor_get(v___x_2029_, 6);
v_infoState_2037_ = lean_ctor_get(v___x_2029_, 7);
v_snapshotTasks_2038_ = lean_ctor_get(v___x_2029_, 8);
v_isSharedCheck_2068_ = !lean_is_exclusive(v___x_2029_);
if (v_isSharedCheck_2068_ == 0)
{
v___x_2040_ = v___x_2029_;
v_isShared_2041_ = v_isSharedCheck_2068_;
goto v_resetjp_2039_;
}
else
{
lean_inc(v_snapshotTasks_2038_);
lean_inc(v_infoState_2037_);
lean_inc(v_messages_2036_);
lean_inc(v_cache_2035_);
lean_inc(v_traceState_2030_);
lean_inc(v_auxDeclNGen_2034_);
lean_inc(v_ngen_2033_);
lean_inc(v_nextMacroScope_2032_);
lean_inc(v_env_2031_);
lean_dec(v___x_2029_);
v___x_2040_ = lean_box(0);
v_isShared_2041_ = v_isSharedCheck_2068_;
goto v_resetjp_2039_;
}
v_resetjp_2039_:
{
uint64_t v_tid_2042_; lean_object* v_traces_2043_; lean_object* v___x_2045_; uint8_t v_isShared_2046_; uint8_t v_isSharedCheck_2067_; 
v_tid_2042_ = lean_ctor_get_uint64(v_traceState_2030_, sizeof(void*)*1);
v_traces_2043_ = lean_ctor_get(v_traceState_2030_, 0);
v_isSharedCheck_2067_ = !lean_is_exclusive(v_traceState_2030_);
if (v_isSharedCheck_2067_ == 0)
{
v___x_2045_ = v_traceState_2030_;
v_isShared_2046_ = v_isSharedCheck_2067_;
goto v_resetjp_2044_;
}
else
{
lean_inc(v_traces_2043_);
lean_dec(v_traceState_2030_);
v___x_2045_ = lean_box(0);
v_isShared_2046_ = v_isSharedCheck_2067_;
goto v_resetjp_2044_;
}
v_resetjp_2044_:
{
lean_object* v___x_2047_; double v___x_2048_; uint8_t v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2057_; 
v___x_2047_ = lean_box(0);
v___x_2048_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__0);
v___x_2049_ = 0;
v___x_2050_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__1));
v___x_2051_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2051_, 0, v_cls_2016_);
lean_ctor_set(v___x_2051_, 1, v___x_2047_);
lean_ctor_set(v___x_2051_, 2, v___x_2050_);
lean_ctor_set_float(v___x_2051_, sizeof(void*)*3, v___x_2048_);
lean_ctor_set_float(v___x_2051_, sizeof(void*)*3 + 8, v___x_2048_);
lean_ctor_set_uint8(v___x_2051_, sizeof(void*)*3 + 16, v___x_2049_);
v___x_2052_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___closed__2));
v___x_2053_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2053_, 0, v___x_2051_);
lean_ctor_set(v___x_2053_, 1, v_a_2025_);
lean_ctor_set(v___x_2053_, 2, v___x_2052_);
lean_inc(v_ref_2023_);
v___x_2054_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2054_, 0, v_ref_2023_);
lean_ctor_set(v___x_2054_, 1, v___x_2053_);
v___x_2055_ = l_Lean_PersistentArray_push___redArg(v_traces_2043_, v___x_2054_);
if (v_isShared_2046_ == 0)
{
lean_ctor_set(v___x_2045_, 0, v___x_2055_);
v___x_2057_ = v___x_2045_;
goto v_reusejp_2056_;
}
else
{
lean_object* v_reuseFailAlloc_2066_; 
v_reuseFailAlloc_2066_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2066_, 0, v___x_2055_);
lean_ctor_set_uint64(v_reuseFailAlloc_2066_, sizeof(void*)*1, v_tid_2042_);
v___x_2057_ = v_reuseFailAlloc_2066_;
goto v_reusejp_2056_;
}
v_reusejp_2056_:
{
lean_object* v___x_2059_; 
if (v_isShared_2041_ == 0)
{
lean_ctor_set(v___x_2040_, 4, v___x_2057_);
v___x_2059_ = v___x_2040_;
goto v_reusejp_2058_;
}
else
{
lean_object* v_reuseFailAlloc_2065_; 
v_reuseFailAlloc_2065_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2065_, 0, v_env_2031_);
lean_ctor_set(v_reuseFailAlloc_2065_, 1, v_nextMacroScope_2032_);
lean_ctor_set(v_reuseFailAlloc_2065_, 2, v_ngen_2033_);
lean_ctor_set(v_reuseFailAlloc_2065_, 3, v_auxDeclNGen_2034_);
lean_ctor_set(v_reuseFailAlloc_2065_, 4, v___x_2057_);
lean_ctor_set(v_reuseFailAlloc_2065_, 5, v_cache_2035_);
lean_ctor_set(v_reuseFailAlloc_2065_, 6, v_messages_2036_);
lean_ctor_set(v_reuseFailAlloc_2065_, 7, v_infoState_2037_);
lean_ctor_set(v_reuseFailAlloc_2065_, 8, v_snapshotTasks_2038_);
v___x_2059_ = v_reuseFailAlloc_2065_;
goto v_reusejp_2058_;
}
v_reusejp_2058_:
{
lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2063_; 
v___x_2060_ = lean_st_ref_set(v___y_2021_, v___x_2059_);
v___x_2061_ = lean_box(0);
if (v_isShared_2028_ == 0)
{
lean_ctor_set(v___x_2027_, 0, v___x_2061_);
v___x_2063_ = v___x_2027_;
goto v_reusejp_2062_;
}
else
{
lean_object* v_reuseFailAlloc_2064_; 
v_reuseFailAlloc_2064_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2064_, 0, v___x_2061_);
v___x_2063_ = v_reuseFailAlloc_2064_;
goto v_reusejp_2062_;
}
v_reusejp_2062_:
{
return v___x_2063_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg___boxed(lean_object* v_cls_2070_, lean_object* v_msg_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_){
_start:
{
lean_object* v_res_2077_; 
v_res_2077_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg(v_cls_2070_, v_msg_2071_, v___y_2072_, v___y_2073_, v___y_2074_, v___y_2075_);
lean_dec(v___y_2075_);
lean_dec_ref(v___y_2074_);
lean_dec(v___y_2073_);
lean_dec_ref(v___y_2072_);
return v_res_2077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg(size_t v_sz_2078_, size_t v_i_2079_, lean_object* v_bs_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_){
_start:
{
uint8_t v___x_2086_; 
v___x_2086_ = lean_usize_dec_lt(v_i_2079_, v_sz_2078_);
if (v___x_2086_ == 0)
{
lean_object* v___x_2087_; 
v___x_2087_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2087_, 0, v_bs_2080_);
return v___x_2087_;
}
else
{
lean_object* v_v_2088_; lean_object* v___x_2089_; 
v_v_2088_ = lean_array_uget_borrowed(v_bs_2080_, v_i_2079_);
lean_inc(v_v_2088_);
v___x_2089_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_v_2088_, v___y_2081_, v___y_2082_, v___y_2083_, v___y_2084_);
if (lean_obj_tag(v___x_2089_) == 0)
{
lean_object* v_a_2090_; lean_object* v___x_2091_; lean_object* v_bs_x27_2092_; size_t v___x_2093_; size_t v___x_2094_; lean_object* v___x_2095_; 
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
lean_inc(v_a_2090_);
lean_dec_ref_known(v___x_2089_, 1);
v___x_2091_ = lean_unsigned_to_nat(0u);
v_bs_x27_2092_ = lean_array_uset(v_bs_2080_, v_i_2079_, v___x_2091_);
v___x_2093_ = ((size_t)1ULL);
v___x_2094_ = lean_usize_add(v_i_2079_, v___x_2093_);
v___x_2095_ = lean_array_uset(v_bs_x27_2092_, v_i_2079_, v_a_2090_);
v_i_2079_ = v___x_2094_;
v_bs_2080_ = v___x_2095_;
goto _start;
}
else
{
lean_object* v_a_2097_; lean_object* v___x_2099_; uint8_t v_isShared_2100_; uint8_t v_isSharedCheck_2104_; 
lean_dec_ref(v_bs_2080_);
v_a_2097_ = lean_ctor_get(v___x_2089_, 0);
v_isSharedCheck_2104_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2104_ == 0)
{
v___x_2099_ = v___x_2089_;
v_isShared_2100_ = v_isSharedCheck_2104_;
goto v_resetjp_2098_;
}
else
{
lean_inc(v_a_2097_);
lean_dec(v___x_2089_);
v___x_2099_ = lean_box(0);
v_isShared_2100_ = v_isSharedCheck_2104_;
goto v_resetjp_2098_;
}
v_resetjp_2098_:
{
lean_object* v___x_2102_; 
if (v_isShared_2100_ == 0)
{
v___x_2102_ = v___x_2099_;
goto v_reusejp_2101_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v_a_2097_);
v___x_2102_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2101_;
}
v_reusejp_2101_:
{
return v___x_2102_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg___boxed(lean_object* v_sz_2105_, lean_object* v_i_2106_, lean_object* v_bs_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_){
_start:
{
size_t v_sz_boxed_2113_; size_t v_i_boxed_2114_; lean_object* v_res_2115_; 
v_sz_boxed_2113_ = lean_unbox_usize(v_sz_2105_);
lean_dec(v_sz_2105_);
v_i_boxed_2114_ = lean_unbox_usize(v_i_2106_);
lean_dec(v_i_2106_);
v_res_2115_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg(v_sz_boxed_2113_, v_i_boxed_2114_, v_bs_2107_, v___y_2108_, v___y_2109_, v___y_2110_, v___y_2111_);
lean_dec(v___y_2111_);
lean_dec_ref(v___y_2110_);
lean_dec(v___y_2109_);
lean_dec_ref(v___y_2108_);
return v_res_2115_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__0(lean_object* v_x_2116_, lean_object* v_x_2117_){
_start:
{
if (lean_obj_tag(v_x_2116_) == 0)
{
if (lean_obj_tag(v_x_2117_) == 0)
{
uint8_t v___x_2118_; 
v___x_2118_ = 1;
return v___x_2118_;
}
else
{
uint8_t v___x_2119_; 
v___x_2119_ = 0;
return v___x_2119_;
}
}
else
{
if (lean_obj_tag(v_x_2117_) == 0)
{
uint8_t v___x_2120_; 
v___x_2120_ = 0;
return v___x_2120_;
}
else
{
lean_object* v_head_2121_; lean_object* v_tail_2122_; lean_object* v_head_2123_; lean_object* v_tail_2124_; uint8_t v___x_2125_; 
v_head_2121_ = lean_ctor_get(v_x_2116_, 0);
v_tail_2122_ = lean_ctor_get(v_x_2116_, 1);
v_head_2123_ = lean_ctor_get(v_x_2117_, 0);
v_tail_2124_ = lean_ctor_get(v_x_2117_, 1);
v___x_2125_ = l_Lean_instBEqMVarId_beq(v_head_2121_, v_head_2123_);
if (v___x_2125_ == 0)
{
return v___x_2125_;
}
else
{
v_x_2116_ = v_tail_2122_;
v_x_2117_ = v_tail_2124_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__0___boxed(lean_object* v_x_2127_, lean_object* v_x_2128_){
_start:
{
uint8_t v_res_2129_; lean_object* v_r_2130_; 
v_res_2129_ = lp_mathlib_List_beq___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__0(v_x_2127_, v_x_2128_);
lean_dec(v_x_2128_);
lean_dec(v_x_2127_);
v_r_2130_ = lean_box(v_res_2129_);
return v_r_2130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg(uint8_t v___x_2131_, lean_object* v_a_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_){
_start:
{
lean_object* v_fst_2138_; lean_object* v_snd_2139_; lean_object* v___x_2141_; uint8_t v_isShared_2142_; uint8_t v_isSharedCheck_2153_; 
v_fst_2138_ = lean_ctor_get(v_a_2132_, 0);
v_snd_2139_ = lean_ctor_get(v_a_2132_, 1);
v_isSharedCheck_2153_ = !lean_is_exclusive(v_a_2132_);
if (v_isSharedCheck_2153_ == 0)
{
v___x_2141_ = v_a_2132_;
v_isShared_2142_ = v_isSharedCheck_2153_;
goto v_resetjp_2140_;
}
else
{
lean_inc(v_snd_2139_);
lean_inc(v_fst_2138_);
lean_dec(v_a_2132_);
v___x_2141_ = lean_box(0);
v_isShared_2142_ = v_isSharedCheck_2153_;
goto v_resetjp_2140_;
}
v_resetjp_2140_:
{
lean_object* v___x_2148_; uint8_t v___x_2149_; 
v___x_2148_ = lean_box(0);
v___x_2149_ = lp_mathlib_List_beq___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__0(v_fst_2138_, v___x_2148_);
if (v___x_2149_ == 0)
{
if (v___x_2131_ == 0)
{
goto v___jp_2143_;
}
else
{
lean_object* v___x_2150_; 
lean_del_object(v___x_2141_);
v___x_2150_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_splitApply(v_fst_2138_, v_snd_2139_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_);
if (lean_obj_tag(v___x_2150_) == 0)
{
lean_object* v_a_2151_; 
v_a_2151_ = lean_ctor_get(v___x_2150_, 0);
lean_inc(v_a_2151_);
lean_dec_ref_known(v___x_2150_, 1);
v_a_2132_ = v_a_2151_;
goto _start;
}
else
{
return v___x_2150_;
}
}
}
else
{
goto v___jp_2143_;
}
v___jp_2143_:
{
lean_object* v___x_2145_; 
if (v_isShared_2142_ == 0)
{
v___x_2145_ = v___x_2141_;
goto v_reusejp_2144_;
}
else
{
lean_object* v_reuseFailAlloc_2147_; 
v_reuseFailAlloc_2147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2147_, 0, v_fst_2138_);
lean_ctor_set(v_reuseFailAlloc_2147_, 1, v_snd_2139_);
v___x_2145_ = v_reuseFailAlloc_2147_;
goto v_reusejp_2144_;
}
v_reusejp_2144_:
{
lean_object* v___x_2146_; 
v___x_2146_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2146_, 0, v___x_2145_);
return v___x_2146_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg___boxed(lean_object* v___x_2154_, lean_object* v_a_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_, lean_object* v___y_2160_){
_start:
{
uint8_t v___x_30812__boxed_2161_; lean_object* v_res_2162_; 
v___x_30812__boxed_2161_ = lean_unbox(v___x_2154_);
v_res_2162_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg(v___x_30812__boxed_2161_, v_a_2155_, v___y_2156_, v___y_2157_, v___y_2158_, v___y_2159_);
lean_dec(v___y_2159_);
lean_dec_ref(v___y_2158_);
lean_dec(v___y_2157_);
lean_dec_ref(v___y_2156_);
return v_res_2162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(lean_object* v_msg_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_){
_start:
{
lean_object* v_ref_2169_; lean_object* v___x_2170_; lean_object* v_a_2171_; lean_object* v___x_2173_; uint8_t v_isShared_2174_; uint8_t v_isSharedCheck_2179_; 
v_ref_2169_ = lean_ctor_get(v___y_2166_, 5);
v___x_2170_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4_spec__4(v_msg_2163_, v___y_2164_, v___y_2165_, v___y_2166_, v___y_2167_);
v_a_2171_ = lean_ctor_get(v___x_2170_, 0);
v_isSharedCheck_2179_ = !lean_is_exclusive(v___x_2170_);
if (v_isSharedCheck_2179_ == 0)
{
v___x_2173_ = v___x_2170_;
v_isShared_2174_ = v_isSharedCheck_2179_;
goto v_resetjp_2172_;
}
else
{
lean_inc(v_a_2171_);
lean_dec(v___x_2170_);
v___x_2173_ = lean_box(0);
v_isShared_2174_ = v_isSharedCheck_2179_;
goto v_resetjp_2172_;
}
v_resetjp_2172_:
{
lean_object* v___x_2175_; lean_object* v___x_2177_; 
lean_inc(v_ref_2169_);
v___x_2175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2175_, 0, v_ref_2169_);
lean_ctor_set(v___x_2175_, 1, v_a_2171_);
if (v_isShared_2174_ == 0)
{
lean_ctor_set_tag(v___x_2173_, 1);
lean_ctor_set(v___x_2173_, 0, v___x_2175_);
v___x_2177_ = v___x_2173_;
goto v_reusejp_2176_;
}
else
{
lean_object* v_reuseFailAlloc_2178_; 
v_reuseFailAlloc_2178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2178_, 0, v___x_2175_);
v___x_2177_ = v_reuseFailAlloc_2178_;
goto v_reusejp_2176_;
}
v_reusejp_2176_:
{
return v___x_2177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg___boxed(lean_object* v_msg_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_){
_start:
{
lean_object* v_res_2186_; 
v_res_2186_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(v_msg_2180_, v___y_2181_, v___y_2182_, v___y_2183_, v___y_2184_);
lean_dec(v___y_2184_);
lean_dec_ref(v___y_2183_);
lean_dec(v___y_2182_);
lean_dec_ref(v___y_2181_);
return v_res_2186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0(lean_object* v_head_2190_, lean_object* v___y_2191_, lean_object* v___y_2192_, lean_object* v___y_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_){
_start:
{
lean_object* v___x_2200_; 
v___x_2200_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_head_2190_, v___y_2195_, v___y_2196_, v___y_2197_, v___y_2198_);
if (lean_obj_tag(v___x_2200_) == 0)
{
lean_object* v_a_2201_; lean_object* v___x_2203_; uint8_t v_isShared_2204_; uint8_t v_isSharedCheck_2211_; 
v_a_2201_ = lean_ctor_get(v___x_2200_, 0);
v_isSharedCheck_2211_ = !lean_is_exclusive(v___x_2200_);
if (v_isSharedCheck_2211_ == 0)
{
v___x_2203_ = v___x_2200_;
v_isShared_2204_ = v_isSharedCheck_2211_;
goto v_resetjp_2202_;
}
else
{
lean_inc(v_a_2201_);
lean_dec(v___x_2200_);
v___x_2203_ = lean_box(0);
v_isShared_2204_ = v_isSharedCheck_2211_;
goto v_resetjp_2202_;
}
v_resetjp_2202_:
{
lean_object* v___x_2205_; uint8_t v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2209_; 
v___x_2205_ = ((lean_object*)(lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___closed__1));
v___x_2206_ = l_Lean_Expr_isConstOf(v_a_2201_, v___x_2205_);
lean_dec(v_a_2201_);
v___x_2207_ = lean_box(v___x_2206_);
if (v_isShared_2204_ == 0)
{
lean_ctor_set(v___x_2203_, 0, v___x_2207_);
v___x_2209_ = v___x_2203_;
goto v_reusejp_2208_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v___x_2207_);
v___x_2209_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2208_;
}
v_reusejp_2208_:
{
return v___x_2209_;
}
}
}
else
{
lean_object* v_a_2212_; lean_object* v___x_2214_; uint8_t v_isShared_2215_; uint8_t v_isSharedCheck_2219_; 
v_a_2212_ = lean_ctor_get(v___x_2200_, 0);
v_isSharedCheck_2219_ = !lean_is_exclusive(v___x_2200_);
if (v_isSharedCheck_2219_ == 0)
{
v___x_2214_ = v___x_2200_;
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
else
{
lean_inc(v_a_2212_);
lean_dec(v___x_2200_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___boxed(lean_object* v_head_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_){
_start:
{
lean_object* v_res_2230_; 
v_res_2230_ = lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0(v_head_2220_, v___y_2221_, v___y_2222_, v___y_2223_, v___y_2224_, v___y_2225_, v___y_2226_, v___y_2227_, v___y_2228_);
lean_dec(v___y_2228_);
lean_dec_ref(v___y_2227_);
lean_dec(v___y_2226_);
lean_dec_ref(v___y_2225_);
lean_dec(v___y_2224_);
lean_dec_ref(v___y_2223_);
lean_dec(v___y_2222_);
lean_dec_ref(v___y_2221_);
return v_res_2230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3(lean_object* v_x_2231_, lean_object* v___y_2232_, lean_object* v___y_2233_, lean_object* v___y_2234_, lean_object* v___y_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_){
_start:
{
if (lean_obj_tag(v_x_2231_) == 0)
{
uint8_t v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; 
v___x_2241_ = 0;
v___x_2242_ = lean_box(v___x_2241_);
v___x_2243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2242_);
return v___x_2243_;
}
else
{
lean_object* v_head_2244_; lean_object* v_tail_2245_; lean_object* v___f_2246_; lean_object* v___x_2247_; 
v_head_2244_ = lean_ctor_get(v_x_2231_, 0);
lean_inc_n(v_head_2244_, 2);
v_tail_2245_ = lean_ctor_get(v_x_2231_, 1);
lean_inc(v_tail_2245_);
lean_dec_ref_known(v_x_2231_, 2);
v___f_2246_ = lean_alloc_closure((void*)(lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___lam__0___boxed), 10, 1);
lean_closure_set(v___f_2246_, 0, v_head_2244_);
v___x_2247_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__2___redArg(v_head_2244_, v___f_2246_, v___y_2232_, v___y_2233_, v___y_2234_, v___y_2235_, v___y_2236_, v___y_2237_, v___y_2238_, v___y_2239_);
if (lean_obj_tag(v___x_2247_) == 0)
{
lean_object* v_a_2248_; uint8_t v___x_2249_; 
v_a_2248_ = lean_ctor_get(v___x_2247_, 0);
lean_inc(v_a_2248_);
v___x_2249_ = lean_unbox(v_a_2248_);
lean_dec(v_a_2248_);
if (v___x_2249_ == 0)
{
lean_dec_ref_known(v___x_2247_, 1);
v_x_2231_ = v_tail_2245_;
goto _start;
}
else
{
lean_dec(v_tail_2245_);
return v___x_2247_;
}
}
else
{
lean_dec(v_tail_2245_);
return v___x_2247_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3___boxed(lean_object* v_x_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_){
_start:
{
lean_object* v_res_2261_; 
v_res_2261_ = lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3(v_x_2251_, v___y_2252_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_, v___y_2257_, v___y_2258_, v___y_2259_);
lean_dec(v___y_2259_);
lean_dec_ref(v___y_2258_);
lean_dec(v___y_2257_);
lean_dec_ref(v___y_2256_);
lean_dec(v___y_2255_);
lean_dec_ref(v___y_2254_);
lean_dec(v___y_2253_);
lean_dec_ref(v___y_2252_);
return v_res_2261_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14(void){
_start:
{
lean_object* v___x_2276_; 
v___x_2276_ = l_Array_mkArray0(lean_box(0));
return v___x_2276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0(lean_object* v___x_2281_, lean_object* v___x_2282_, lean_object* v___y_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_){
_start:
{
lean_object* v_ref_2292_; uint8_t v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; lean_object* v___x_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; 
v_ref_2292_ = lean_ctor_get(v___y_2289_, 5);
v___x_2293_ = 0;
v___x_2294_ = l_Lean_SourceInfo_fromRef(v_ref_2292_, v___x_2293_);
v___x_2295_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0));
v___x_2296_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1));
v___x_2297_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__2));
lean_inc_ref_n(v___x_2281_, 9);
v___x_2298_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2297_);
v___x_2299_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__3));
lean_inc_n(v___x_2294_, 30);
v___x_2300_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2300_, 0, v___x_2294_);
lean_ctor_set(v___x_2300_, 1, v___x_2299_);
v___x_2301_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__4));
v___x_2302_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2301_);
v___x_2303_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__5));
v___x_2304_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2303_);
v___x_2305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1));
v___x_2306_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__6));
v___x_2307_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2306_);
v___x_2308_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7));
v___x_2309_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2309_, 0, v___x_2294_);
lean_ctor_set(v___x_2309_, 1, v___x_2308_);
v___x_2310_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__8));
v___x_2311_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2310_);
v___x_2312_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__9));
v___x_2313_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2313_, 0, v___x_2294_);
lean_ctor_set(v___x_2313_, 1, v___x_2312_);
v___x_2314_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__10));
v___x_2315_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2314_);
v___x_2316_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__11));
v___x_2317_ = l_Lean_Name_mkStr3(v___x_2282_, v___x_2281_, v___x_2316_);
v___x_2318_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__12));
v___x_2319_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2319_, 0, v___x_2294_);
lean_ctor_set(v___x_2319_, 1, v___x_2318_);
v___x_2320_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__13));
v___x_2321_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2320_);
v___x_2322_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14);
v___x_2323_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2323_, 0, v___x_2294_);
lean_ctor_set(v___x_2323_, 1, v___x_2305_);
lean_ctor_set(v___x_2323_, 2, v___x_2322_);
lean_inc_ref_n(v___x_2323_, 4);
v___x_2324_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2321_, v___x_2323_);
lean_inc(v___x_2324_);
v___x_2325_ = l_Lean_Syntax_node5(v___x_2294_, v___x_2317_, v___x_2319_, v___x_2324_, v___x_2323_, v___x_2323_, v___x_2323_);
v___x_2326_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__15));
v___x_2327_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2327_, 0, v___x_2294_);
lean_ctor_set(v___x_2327_, 1, v___x_2326_);
v___x_2328_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__16));
v___x_2329_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2328_);
v___x_2330_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__17));
v___x_2331_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2331_, 0, v___x_2294_);
lean_ctor_set(v___x_2331_, 1, v___x_2330_);
v___x_2332_ = l_Lean_Syntax_node3(v___x_2294_, v___x_2329_, v___x_2331_, v___x_2324_, v___x_2323_);
lean_inc_ref(v___x_2327_);
lean_inc(v___x_2315_);
v___x_2333_ = l_Lean_Syntax_node3(v___x_2294_, v___x_2315_, v___x_2325_, v___x_2327_, v___x_2332_);
v___x_2334_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__18));
v___x_2335_ = l_Lean_Name_mkStr4(v___x_2295_, v___x_2296_, v___x_2281_, v___x_2334_);
v___x_2336_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2336_, 0, v___x_2294_);
lean_ctor_set(v___x_2336_, 1, v___x_2334_);
v___x_2337_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2335_, v___x_2336_);
v___x_2338_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2305_, v___x_2337_);
lean_inc_n(v___x_2304_, 3);
v___x_2339_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2304_, v___x_2338_);
lean_inc_n(v___x_2302_, 3);
v___x_2340_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2302_, v___x_2339_);
lean_inc_ref(v___x_2300_);
lean_inc(v___x_2298_);
v___x_2341_ = l_Lean_Syntax_node2(v___x_2294_, v___x_2298_, v___x_2300_, v___x_2340_);
v___x_2342_ = l_Lean_Syntax_node3(v___x_2294_, v___x_2315_, v___x_2333_, v___x_2327_, v___x_2341_);
v___x_2343_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2305_, v___x_2342_);
v___x_2344_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2304_, v___x_2343_);
v___x_2345_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2302_, v___x_2344_);
v___x_2346_ = l_Lean_Syntax_node2(v___x_2294_, v___x_2311_, v___x_2313_, v___x_2345_);
v___x_2347_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2305_, v___x_2346_);
v___x_2348_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2304_, v___x_2347_);
v___x_2349_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2302_, v___x_2348_);
v___x_2350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105));
v___x_2351_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2351_, 0, v___x_2294_);
lean_ctor_set(v___x_2351_, 1, v___x_2350_);
v___x_2352_ = l_Lean_Syntax_node3(v___x_2294_, v___x_2307_, v___x_2309_, v___x_2349_, v___x_2351_);
v___x_2353_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2305_, v___x_2352_);
v___x_2354_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2304_, v___x_2353_);
v___x_2355_ = l_Lean_Syntax_node1(v___x_2294_, v___x_2302_, v___x_2354_);
v___x_2356_ = l_Lean_Syntax_node2(v___x_2294_, v___x_2298_, v___x_2300_, v___x_2355_);
v___x_2357_ = l_Lean_Elab_Tactic_evalTactic(v___x_2356_, v___y_2283_, v___y_2284_, v___y_2285_, v___y_2286_, v___y_2287_, v___y_2288_, v___y_2289_, v___y_2290_);
return v___x_2357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___boxed(lean_object* v___x_2358_, lean_object* v___x_2359_, lean_object* v___y_2360_, lean_object* v___y_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_){
_start:
{
lean_object* v_res_2369_; 
v_res_2369_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0(v___x_2358_, v___x_2359_, v___y_2360_, v___y_2361_, v___y_2362_, v___y_2363_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_);
lean_dec(v___y_2367_);
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2362_);
lean_dec(v___y_2361_);
lean_dec_ref(v___y_2360_);
return v_res_2369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg(lean_object* v_as_x27_2373_, lean_object* v_b_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_, lean_object* v___y_2382_){
_start:
{
if (lean_obj_tag(v_as_x27_2373_) == 0)
{
lean_object* v___x_2384_; 
v___x_2384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2384_, 0, v_b_2374_);
return v___x_2384_;
}
else
{
lean_object* v_head_2385_; lean_object* v_tail_2386_; lean_object* v___f_2387_; lean_object* v___x_2388_; 
v_head_2385_ = lean_ctor_get(v_as_x27_2373_, 0);
v_tail_2386_ = lean_ctor_get(v_as_x27_2373_, 1);
v___f_2387_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___closed__0));
lean_inc(v_head_2385_);
v___x_2388_ = l_Lean_Elab_Tactic_run(v_head_2385_, v___f_2387_, v___y_2377_, v___y_2378_, v___y_2379_, v___y_2380_, v___y_2381_, v___y_2382_);
if (lean_obj_tag(v___x_2388_) == 0)
{
lean_object* v_a_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; 
v_a_2389_ = lean_ctor_get(v___x_2388_, 0);
lean_inc_n(v_a_2389_, 2);
lean_dec_ref_known(v___x_2388_, 1);
v___x_2390_ = lean_array_mk(v_a_2389_);
v___x_2391_ = lp_mathlib_List_anyM___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__3(v_a_2389_, v___y_2375_, v___y_2376_, v___y_2377_, v___y_2378_, v___y_2379_, v___y_2380_, v___y_2381_, v___y_2382_);
if (lean_obj_tag(v___x_2391_) == 0)
{
lean_object* v_a_2392_; lean_object* v_fst_2393_; lean_object* v_snd_2394_; lean_object* v___x_2396_; uint8_t v_isShared_2397_; uint8_t v_isSharedCheck_2409_; 
v_a_2392_ = lean_ctor_get(v___x_2391_, 0);
lean_inc(v_a_2392_);
lean_dec_ref_known(v___x_2391_, 1);
v_fst_2393_ = lean_ctor_get(v_b_2374_, 0);
v_snd_2394_ = lean_ctor_get(v_b_2374_, 1);
v_isSharedCheck_2409_ = !lean_is_exclusive(v_b_2374_);
if (v_isSharedCheck_2409_ == 0)
{
v___x_2396_ = v_b_2374_;
v_isShared_2397_ = v_isSharedCheck_2409_;
goto v_resetjp_2395_;
}
else
{
lean_inc(v_snd_2394_);
lean_inc(v_fst_2393_);
lean_dec(v_b_2374_);
v___x_2396_ = lean_box(0);
v_isShared_2397_ = v_isSharedCheck_2409_;
goto v_resetjp_2395_;
}
v_resetjp_2395_:
{
lean_object* v___x_2398_; uint8_t v___x_2399_; 
v___x_2398_ = l_Array_append___redArg(v_snd_2394_, v___x_2390_);
lean_dec_ref(v___x_2390_);
v___x_2399_ = lean_unbox(v_a_2392_);
lean_dec(v_a_2392_);
if (v___x_2399_ == 0)
{
lean_object* v___x_2401_; 
if (v_isShared_2397_ == 0)
{
lean_ctor_set(v___x_2396_, 1, v___x_2398_);
v___x_2401_ = v___x_2396_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2403_; 
v_reuseFailAlloc_2403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2403_, 0, v_fst_2393_);
lean_ctor_set(v_reuseFailAlloc_2403_, 1, v___x_2398_);
v___x_2401_ = v_reuseFailAlloc_2403_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
v_as_x27_2373_ = v_tail_2386_;
v_b_2374_ = v___x_2401_;
goto _start;
}
}
else
{
lean_object* v___x_2404_; lean_object* v___x_2406_; 
lean_inc(v_head_2385_);
v___x_2404_ = lean_array_push(v_fst_2393_, v_head_2385_);
if (v_isShared_2397_ == 0)
{
lean_ctor_set(v___x_2396_, 1, v___x_2398_);
lean_ctor_set(v___x_2396_, 0, v___x_2404_);
v___x_2406_ = v___x_2396_;
goto v_reusejp_2405_;
}
else
{
lean_object* v_reuseFailAlloc_2408_; 
v_reuseFailAlloc_2408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2408_, 0, v___x_2404_);
lean_ctor_set(v_reuseFailAlloc_2408_, 1, v___x_2398_);
v___x_2406_ = v_reuseFailAlloc_2408_;
goto v_reusejp_2405_;
}
v_reusejp_2405_:
{
v_as_x27_2373_ = v_tail_2386_;
v_b_2374_ = v___x_2406_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_2410_; lean_object* v___x_2412_; uint8_t v_isShared_2413_; uint8_t v_isSharedCheck_2417_; 
lean_dec_ref(v___x_2390_);
lean_dec_ref(v_b_2374_);
v_a_2410_ = lean_ctor_get(v___x_2391_, 0);
v_isSharedCheck_2417_ = !lean_is_exclusive(v___x_2391_);
if (v_isSharedCheck_2417_ == 0)
{
v___x_2412_ = v___x_2391_;
v_isShared_2413_ = v_isSharedCheck_2417_;
goto v_resetjp_2411_;
}
else
{
lean_inc(v_a_2410_);
lean_dec(v___x_2391_);
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
lean_object* v_a_2418_; lean_object* v___x_2420_; uint8_t v_isShared_2421_; uint8_t v_isSharedCheck_2425_; 
lean_dec_ref(v_b_2374_);
v_a_2418_ = lean_ctor_get(v___x_2388_, 0);
v_isSharedCheck_2425_ = !lean_is_exclusive(v___x_2388_);
if (v_isSharedCheck_2425_ == 0)
{
v___x_2420_ = v___x_2388_;
v_isShared_2421_ = v_isSharedCheck_2425_;
goto v_resetjp_2419_;
}
else
{
lean_inc(v_a_2418_);
lean_dec(v___x_2388_);
v___x_2420_ = lean_box(0);
v_isShared_2421_ = v_isSharedCheck_2425_;
goto v_resetjp_2419_;
}
v_resetjp_2419_:
{
lean_object* v___x_2423_; 
if (v_isShared_2421_ == 0)
{
v___x_2423_ = v___x_2420_;
goto v_reusejp_2422_;
}
else
{
lean_object* v_reuseFailAlloc_2424_; 
v_reuseFailAlloc_2424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2424_, 0, v_a_2418_);
v___x_2423_ = v_reuseFailAlloc_2424_;
goto v_reusejp_2422_;
}
v_reusejp_2422_:
{
return v___x_2423_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___boxed(lean_object* v_as_x27_2426_, lean_object* v_b_2427_, lean_object* v___y_2428_, lean_object* v___y_2429_, lean_object* v___y_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_, lean_object* v___y_2435_, lean_object* v___y_2436_){
_start:
{
lean_object* v_res_2437_; 
v_res_2437_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg(v_as_x27_2426_, v_b_2427_, v___y_2428_, v___y_2429_, v___y_2430_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_, v___y_2435_);
lean_dec(v___y_2435_);
lean_dec_ref(v___y_2434_);
lean_dec(v___y_2433_);
lean_dec_ref(v___y_2432_);
lean_dec(v___y_2431_);
lean_dec_ref(v___y_2430_);
lean_dec(v___y_2429_);
lean_dec_ref(v___y_2428_);
lean_dec(v_as_x27_2426_);
return v_res_2437_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__11(void){
_start:
{
lean_object* v___x_2449_; lean_object* v___x_2450_; 
v___x_2449_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__10));
v___x_2450_ = l_String_toRawSubstring_x27(v___x_2449_);
return v___x_2450_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__21(void){
_start:
{
lean_object* v___x_2467_; lean_object* v___x_2468_; 
v___x_2467_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__20));
v___x_2468_ = l_String_toRawSubstring_x27(v___x_2467_);
return v___x_2468_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__31(void){
_start:
{
lean_object* v___x_2485_; lean_object* v___x_2486_; 
v___x_2485_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__30));
v___x_2486_ = l_Lean_stringToMessageData(v___x_2485_);
return v___x_2486_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__33(void){
_start:
{
lean_object* v___x_2488_; lean_object* v___x_2489_; 
v___x_2488_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__32));
v___x_2489_ = l_Lean_stringToMessageData(v___x_2488_);
return v___x_2489_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__34(void){
_start:
{
lean_object* v___x_2490_; lean_object* v___x_2491_; 
v___x_2490_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__1));
v___x_2491_ = l_Lean_MessageData_ofFormat(v___x_2490_);
return v___x_2491_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__36(void){
_start:
{
lean_object* v___x_2493_; lean_object* v___x_2494_; 
v___x_2493_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__35));
v___x_2494_ = l_Lean_stringToMessageData(v___x_2493_);
return v___x_2494_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__38(void){
_start:
{
lean_object* v___x_2496_; lean_object* v___x_2497_; 
v___x_2496_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__37));
v___x_2497_ = l_Lean_stringToMessageData(v___x_2496_);
return v___x_2497_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__40(void){
_start:
{
lean_object* v___x_2499_; lean_object* v___x_2500_; 
v___x_2499_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__39));
v___x_2500_ = l_Lean_stringToMessageData(v___x_2499_);
return v___x_2500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0(uint8_t v___x_2507_, lean_object* v___x_2508_, lean_object* v___x_2509_, lean_object* v_bang_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_){
_start:
{
lean_object* v___x_2523_; 
v___x_2523_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2512_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
if (lean_obj_tag(v___x_2523_) == 0)
{
lean_object* v_a_2524_; lean_object* v___y_2526_; uint8_t v___y_2527_; lean_object* v___y_2528_; lean_object* v___y_2529_; lean_object* v___y_2530_; lean_object* v___y_2531_; lean_object* v___y_2532_; lean_object* v___y_2533_; lean_object* v___y_2534_; lean_object* v___y_2535_; lean_object* v___y_2536_; lean_object* v___x_2754_; 
v_a_2524_ = lean_ctor_get(v___x_2523_, 0);
lean_inc_n(v_a_2524_, 2);
lean_dec_ref_known(v___x_2523_, 1);
v___x_2754_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_a_2524_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
if (lean_obj_tag(v___x_2754_) == 0)
{
lean_object* v_a_2755_; lean_object* v___y_2757_; lean_object* v___x_2806_; lean_object* v___x_2807_; uint8_t v___x_2808_; 
v_a_2755_ = lean_ctor_get(v___x_2754_, 0);
lean_inc(v_a_2755_);
lean_dec_ref_known(v___x_2754_, 1);
v___x_2806_ = ((lean_object*)(lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ComputeDegree_tryRfl_spec__3___closed__0));
v___x_2807_ = lean_unsigned_to_nat(3u);
v___x_2808_ = l_Lean_Expr_isAppOfArity(v_a_2755_, v___x_2806_, v___x_2807_);
if (v___x_2808_ == 0)
{
lean_object* v___x_2809_; 
v___x_2809_ = lean_box(0);
v___y_2757_ = v___x_2809_;
goto v___jp_2756_;
}
else
{
lean_object* v___x_2810_; lean_object* v___x_2811_; 
v___x_2810_ = l_Lean_Expr_appArg_x21(v_a_2755_);
v___x_2811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2811_, 0, v___x_2810_);
v___y_2757_ = v___x_2811_;
goto v___jp_2756_;
}
v___jp_2756_:
{
lean_object* v___x_2758_; lean_object* v_snd_2759_; lean_object* v_fst_2760_; lean_object* v___x_2762_; uint8_t v_isShared_2763_; uint8_t v_isSharedCheck_2804_; 
lean_inc(v_a_2755_);
v___x_2758_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_twoHeadsArgs(v_a_2755_);
v_snd_2759_ = lean_ctor_get(v___x_2758_, 1);
lean_inc(v_snd_2759_);
v_fst_2760_ = lean_ctor_get(v_snd_2759_, 0);
v_isSharedCheck_2804_ = !lean_is_exclusive(v_snd_2759_);
if (v_isSharedCheck_2804_ == 0)
{
lean_object* v_unused_2805_; 
v_unused_2805_ = lean_ctor_get(v_snd_2759_, 1);
lean_dec(v_unused_2805_);
v___x_2762_ = v_snd_2759_;
v_isShared_2763_ = v_isSharedCheck_2804_;
goto v_resetjp_2761_;
}
else
{
lean_inc(v_fst_2760_);
lean_dec(v_snd_2759_);
v___x_2762_ = lean_box(0);
v_isShared_2763_ = v_isSharedCheck_2804_;
goto v_resetjp_2761_;
}
v_resetjp_2761_:
{
if (lean_obj_tag(v_fst_2760_) == 0)
{
lean_object* v___x_2765_; uint8_t v_isShared_2766_; uint8_t v_isSharedCheck_2778_; 
lean_dec(v___y_2757_);
lean_dec(v_a_2524_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v_isSharedCheck_2778_ = !lean_is_exclusive(v___x_2758_);
if (v_isSharedCheck_2778_ == 0)
{
lean_object* v_unused_2779_; lean_object* v_unused_2780_; 
v_unused_2779_ = lean_ctor_get(v___x_2758_, 1);
lean_dec(v_unused_2779_);
v_unused_2780_ = lean_ctor_get(v___x_2758_, 0);
lean_dec(v_unused_2780_);
v___x_2765_ = v___x_2758_;
v_isShared_2766_ = v_isSharedCheck_2778_;
goto v_resetjp_2764_;
}
else
{
lean_dec(v___x_2758_);
v___x_2765_ = lean_box(0);
v_isShared_2766_ = v_isSharedCheck_2778_;
goto v_resetjp_2764_;
}
v_resetjp_2764_:
{
lean_object* v___x_2767_; lean_object* v___x_2768_; lean_object* v___x_2769_; lean_object* v___x_2771_; 
v___x_2767_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__36, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__36_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__36);
v___x_2768_ = l_Lean_MessageData_ofExpr(v_a_2755_);
v___x_2769_ = l_Lean_indentD(v___x_2768_);
if (v_isShared_2763_ == 0)
{
lean_ctor_set_tag(v___x_2762_, 7);
lean_ctor_set(v___x_2762_, 1, v___x_2769_);
lean_ctor_set(v___x_2762_, 0, v___x_2767_);
v___x_2771_ = v___x_2762_;
goto v_reusejp_2770_;
}
else
{
lean_object* v_reuseFailAlloc_2777_; 
v_reuseFailAlloc_2777_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2777_, 0, v___x_2767_);
lean_ctor_set(v_reuseFailAlloc_2777_, 1, v___x_2769_);
v___x_2771_ = v_reuseFailAlloc_2777_;
goto v_reusejp_2770_;
}
v_reusejp_2770_:
{
lean_object* v___x_2772_; lean_object* v___x_2774_; 
v___x_2772_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__38, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__38_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__38);
if (v_isShared_2766_ == 0)
{
lean_ctor_set_tag(v___x_2765_, 7);
lean_ctor_set(v___x_2765_, 1, v___x_2772_);
lean_ctor_set(v___x_2765_, 0, v___x_2771_);
v___x_2774_ = v___x_2765_;
goto v_reusejp_2773_;
}
else
{
lean_object* v_reuseFailAlloc_2776_; 
v_reuseFailAlloc_2776_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2776_, 0, v___x_2771_);
lean_ctor_set(v_reuseFailAlloc_2776_, 1, v___x_2772_);
v___x_2774_ = v_reuseFailAlloc_2776_;
goto v_reusejp_2773_;
}
v_reusejp_2773_:
{
lean_object* v___x_2775_; 
v___x_2775_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(v___x_2774_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
lean_dec_ref(v___y_2517_);
return v___x_2775_;
}
}
}
}
else
{
lean_object* v_fst_2781_; 
lean_dec(v_fst_2760_);
lean_dec(v_a_2755_);
v_fst_2781_ = lean_ctor_get(v___x_2758_, 0);
lean_inc(v_fst_2781_);
if (lean_obj_tag(v_fst_2781_) == 0)
{
lean_object* v___x_2782_; lean_object* v___x_2783_; 
lean_del_object(v___x_2762_);
lean_dec_ref(v___x_2758_);
lean_dec(v___y_2757_);
lean_dec(v_a_2524_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v___x_2782_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__40, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__40_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__40);
v___x_2783_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(v___x_2782_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
lean_dec_ref(v___y_2517_);
return v___x_2783_;
}
else
{
lean_object* v_options_2784_; lean_object* v_inheritedTraceOptions_2785_; uint8_t v_hasTrace_2786_; uint8_t v___x_2787_; lean_object* v___x_2788_; 
lean_dec(v_fst_2781_);
v_options_2784_ = lean_ctor_get(v___y_2517_, 2);
v_inheritedTraceOptions_2785_ = lean_ctor_get(v___y_2517_, 13);
v_hasTrace_2786_ = lean_ctor_get_uint8(v_options_2784_, sizeof(void*)*1);
v___x_2787_ = 0;
v___x_2788_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma(v___x_2758_, v___x_2787_);
if (v_hasTrace_2786_ == 0)
{
lean_del_object(v___x_2762_);
v___y_2526_ = v___x_2788_;
v___y_2527_ = v___x_2787_;
v___y_2528_ = v___y_2757_;
v___y_2529_ = v___y_2511_;
v___y_2530_ = v___y_2512_;
v___y_2531_ = v___y_2513_;
v___y_2532_ = v___y_2514_;
v___y_2533_ = v___y_2515_;
v___y_2534_ = v___y_2516_;
v___y_2535_ = v___y_2517_;
v___y_2536_ = v___y_2518_;
goto v___jp_2525_;
}
else
{
lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; lean_object* v___x_2792_; uint8_t v___x_2793_; 
v___x_2789_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4));
lean_inc_ref(v___x_2508_);
v___x_2790_ = l_Lean_Name_mkStr2(v___x_2508_, v___x_2789_);
v___x_2791_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__42));
lean_inc(v___x_2790_);
v___x_2792_ = l_Lean_Name_append(v___x_2791_, v___x_2790_);
v___x_2793_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2785_, v_options_2784_, v___x_2792_);
lean_dec(v___x_2792_);
if (v___x_2793_ == 0)
{
lean_dec(v___x_2790_);
lean_del_object(v___x_2762_);
v___y_2526_ = v___x_2788_;
v___y_2527_ = v___x_2787_;
v___y_2528_ = v___y_2757_;
v___y_2529_ = v___y_2511_;
v___y_2530_ = v___y_2512_;
v___y_2531_ = v___y_2513_;
v___y_2532_ = v___y_2514_;
v___y_2533_ = v___y_2515_;
v___y_2534_ = v___y_2516_;
v___y_2535_ = v___y_2517_;
v___y_2536_ = v___y_2518_;
goto v___jp_2525_;
}
else
{
lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___x_2798_; 
v___x_2794_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__44));
lean_inc(v___x_2788_);
v___x_2795_ = lp_mathlib_Lean_Name_lastComponentAsString(v___x_2788_);
v___x_2796_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2796_, 0, v___x_2795_);
if (v_isShared_2763_ == 0)
{
lean_ctor_set_tag(v___x_2762_, 5);
lean_ctor_set(v___x_2762_, 1, v___x_2796_);
lean_ctor_set(v___x_2762_, 0, v___x_2794_);
v___x_2798_ = v___x_2762_;
goto v_reusejp_2797_;
}
else
{
lean_object* v_reuseFailAlloc_2803_; 
v_reuseFailAlloc_2803_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2803_, 0, v___x_2794_);
lean_ctor_set(v_reuseFailAlloc_2803_, 1, v___x_2796_);
v___x_2798_ = v_reuseFailAlloc_2803_;
goto v_reusejp_2797_;
}
v_reusejp_2797_:
{
lean_object* v___x_2799_; lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; 
v___x_2799_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__3));
v___x_2800_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2800_, 0, v___x_2798_);
lean_ctor_set(v___x_2800_, 1, v___x_2799_);
v___x_2801_ = l_Lean_MessageData_ofFormat(v___x_2800_);
v___x_2802_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg(v___x_2790_, v___x_2801_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_);
if (lean_obj_tag(v___x_2802_) == 0)
{
lean_dec_ref_known(v___x_2802_, 1);
v___y_2526_ = v___x_2788_;
v___y_2527_ = v___x_2787_;
v___y_2528_ = v___y_2757_;
v___y_2529_ = v___y_2511_;
v___y_2530_ = v___y_2512_;
v___y_2531_ = v___y_2513_;
v___y_2532_ = v___y_2514_;
v___y_2533_ = v___y_2515_;
v___y_2534_ = v___y_2516_;
v___y_2535_ = v___y_2517_;
v___y_2536_ = v___y_2518_;
goto v___jp_2525_;
}
else
{
lean_dec(v___x_2788_);
lean_dec(v___y_2757_);
lean_dec(v_a_2524_);
lean_dec_ref(v___y_2517_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
return v___x_2802_;
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
lean_object* v_a_2812_; lean_object* v___x_2814_; uint8_t v_isShared_2815_; uint8_t v_isSharedCheck_2819_; 
lean_dec(v_a_2524_);
lean_dec_ref(v___y_2517_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v_a_2812_ = lean_ctor_get(v___x_2754_, 0);
v_isSharedCheck_2819_ = !lean_is_exclusive(v___x_2754_);
if (v_isSharedCheck_2819_ == 0)
{
v___x_2814_ = v___x_2754_;
v_isShared_2815_ = v_isSharedCheck_2819_;
goto v_resetjp_2813_;
}
else
{
lean_inc(v_a_2812_);
lean_dec(v___x_2754_);
v___x_2814_ = lean_box(0);
v_isShared_2815_ = v_isSharedCheck_2819_;
goto v_resetjp_2813_;
}
v_resetjp_2813_:
{
lean_object* v___x_2817_; 
if (v_isShared_2815_ == 0)
{
v___x_2817_ = v___x_2814_;
goto v_reusejp_2816_;
}
else
{
lean_object* v_reuseFailAlloc_2818_; 
v_reuseFailAlloc_2818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2818_, 0, v_a_2812_);
v___x_2817_ = v_reuseFailAlloc_2818_;
goto v_reusejp_2816_;
}
v_reusejp_2816_:
{
return v___x_2817_;
}
}
}
v___jp_2525_:
{
uint8_t v___x_2537_; lean_object* v___x_2538_; lean_object* v___x_2539_; 
v___x_2537_ = 0;
v___x_2538_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_2538_, 0, v___x_2537_);
lean_ctor_set_uint8(v___x_2538_, 1, v___x_2507_);
lean_ctor_set_uint8(v___x_2538_, 2, v___y_2527_);
lean_ctor_set_uint8(v___x_2538_, 3, v___x_2507_);
v___x_2539_ = l_Lean_MVarId_applyConst(v_a_2524_, v___y_2526_, v___x_2538_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
if (lean_obj_tag(v___x_2539_) == 0)
{
lean_object* v_a_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; lean_object* v___x_2543_; 
v_a_2540_ = lean_ctor_get(v___x_2539_, 0);
lean_inc(v_a_2540_);
lean_dec_ref_known(v___x_2539_, 1);
v___x_2541_ = lean_box(0);
v___x_2542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2542_, 0, v_a_2540_);
lean_ctor_set(v___x_2542_, 1, v___x_2541_);
v___x_2543_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg(v___x_2507_, v___x_2542_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
if (lean_obj_tag(v___x_2543_) == 0)
{
lean_object* v_a_2544_; lean_object* v_snd_2545_; lean_object* v___x_2547_; uint8_t v_isShared_2548_; uint8_t v_isSharedCheck_2736_; 
v_a_2544_ = lean_ctor_get(v___x_2543_, 0);
lean_inc(v_a_2544_);
lean_dec_ref_known(v___x_2543_, 1);
v_snd_2545_ = lean_ctor_get(v_a_2544_, 1);
v_isSharedCheck_2736_ = !lean_is_exclusive(v_a_2544_);
if (v_isSharedCheck_2736_ == 0)
{
lean_object* v_unused_2737_; 
v_unused_2737_ = lean_ctor_get(v_a_2544_, 0);
lean_dec(v_unused_2737_);
v___x_2547_ = v_a_2544_;
v_isShared_2548_ = v_isSharedCheck_2736_;
goto v_resetjp_2546_;
}
else
{
lean_inc(v_snd_2545_);
lean_dec(v_a_2544_);
v___x_2547_ = lean_box(0);
v_isShared_2548_ = v_isSharedCheck_2736_;
goto v_resetjp_2546_;
}
v_resetjp_2546_:
{
lean_object* v___x_2549_; 
v___x_2549_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_tryRfl(v_snd_2545_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
if (lean_obj_tag(v___x_2549_) == 0)
{
lean_object* v_a_2550_; lean_object* v___x_2551_; 
v_a_2550_ = lean_ctor_get(v___x_2549_, 0);
lean_inc(v_a_2550_);
lean_dec_ref_known(v___x_2549_, 1);
v___x_2551_ = l_Lean_Elab_Tactic_setGoals___redArg(v_a_2550_, v___y_2530_);
if (lean_obj_tag(v___x_2551_) == 0)
{
lean_object* v_ref_2552_; lean_object* v_quotContext_2553_; lean_object* v_currMacroScope_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2562_; 
lean_dec_ref_known(v___x_2551_, 1);
v_ref_2552_ = lean_ctor_get(v___y_2535_, 5);
v_quotContext_2553_ = lean_ctor_get(v___y_2535_, 10);
v_currMacroScope_2554_ = lean_ctor_get(v___y_2535_, 11);
v___x_2555_ = l_Lean_SourceInfo_fromRef(v_ref_2552_, v___y_2527_);
v___x_2556_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__0));
v___x_2557_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__1));
v___x_2558_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__2));
lean_inc_ref(v___x_2508_);
v___x_2559_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2558_);
v___x_2560_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__3));
lean_inc(v___x_2555_);
if (v_isShared_2548_ == 0)
{
lean_ctor_set_tag(v___x_2547_, 2);
lean_ctor_set(v___x_2547_, 1, v___x_2560_);
lean_ctor_set(v___x_2547_, 0, v___x_2555_);
v___x_2562_ = v___x_2547_;
goto v_reusejp_2561_;
}
else
{
lean_object* v_reuseFailAlloc_2727_; 
v_reuseFailAlloc_2727_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2727_, 0, v___x_2555_);
lean_ctor_set(v_reuseFailAlloc_2727_, 1, v___x_2560_);
v___x_2562_ = v_reuseFailAlloc_2727_;
goto v_reusejp_2561_;
}
v_reusejp_2561_:
{
lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2582_; lean_object* v___x_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; lean_object* v___x_2587_; lean_object* v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; lean_object* v___x_2602_; lean_object* v___x_2603_; lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; lean_object* v___x_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2653_; 
v___x_2563_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__4));
lean_inc_ref_n(v___x_2508_, 12);
v___x_2564_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2563_);
v___x_2565_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__5));
v___x_2566_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2565_);
v___x_2567_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1));
v___x_2568_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__8));
v___x_2569_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2568_);
v___x_2570_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__9));
lean_inc_n(v___x_2555_, 41);
v___x_2571_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2571_, 0, v___x_2555_);
lean_ctor_set(v___x_2571_, 1, v___x_2570_);
v___x_2572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__0));
v___x_2573_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__1));
lean_inc_ref(v___x_2509_);
v___x_2574_ = l_Lean_Name_mkStr4(v___x_2509_, v___x_2508_, v___x_2572_, v___x_2573_);
v___x_2575_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__2));
v___x_2576_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2576_, 0, v___x_2555_);
lean_ctor_set(v___x_2576_, 1, v___x_2575_);
v___x_2577_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14);
v___x_2578_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2578_, 0, v___x_2555_);
lean_ctor_set(v___x_2578_, 1, v___x_2567_);
lean_ctor_set(v___x_2578_, 2, v___x_2577_);
v___x_2579_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__3));
v___x_2580_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2580_, 0, v___x_2555_);
lean_ctor_set(v___x_2580_, 1, v___x_2579_);
v___x_2581_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__4));
v___x_2582_ = l_Lean_Name_mkStr5(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2572_, v___x_2581_);
v___x_2583_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__5));
v___x_2584_ = l_Lean_Name_mkStr5(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2572_, v___x_2583_);
v___x_2585_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__6));
v___x_2586_ = l_Lean_Name_mkStr5(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2572_, v___x_2585_);
v___x_2587_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7));
v___x_2588_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2588_, 0, v___x_2555_);
lean_ctor_set(v___x_2588_, 1, v___x_2587_);
v___x_2589_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__6));
v___x_2590_ = l_Lean_Name_mkStr5(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2572_, v___x_2589_);
v___x_2591_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2591_, 0, v___x_2555_);
lean_ctor_set(v___x_2591_, 1, v___x_2589_);
v___x_2592_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__13));
v___x_2593_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2592_);
v___x_2594_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__7));
v___x_2595_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2594_);
v___x_2596_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__8));
v___x_2597_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2596_);
v___x_2598_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__9));
v___x_2599_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2599_, 0, v___x_2555_);
lean_ctor_set(v___x_2599_, 1, v___x_2598_);
v___x_2600_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__11, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__11);
v___x_2601_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__12));
lean_inc_n(v_currMacroScope_2554_, 2);
lean_inc_n(v_quotContext_2553_, 2);
v___x_2602_ = l_Lean_addMacroScope(v_quotContext_2553_, v___x_2601_, v_currMacroScope_2554_);
v___x_2603_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__16));
v___x_2604_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2604_, 0, v___x_2555_);
lean_ctor_set(v___x_2604_, 1, v___x_2600_);
lean_ctor_set(v___x_2604_, 2, v___x_2602_);
lean_ctor_set(v___x_2604_, 3, v___x_2603_);
v___x_2605_ = l_Lean_Syntax_node2(v___x_2555_, v___x_2597_, v___x_2599_, v___x_2604_);
v___x_2606_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2595_, v___x_2605_);
v___x_2607_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2567_, v___x_2606_);
lean_inc(v___x_2593_);
v___x_2608_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2593_, v___x_2607_);
v___x_2609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__17));
v___x_2610_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2610_, 0, v___x_2555_);
lean_ctor_set(v___x_2610_, 1, v___x_2609_);
v___x_2611_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2567_, v___x_2610_);
v___x_2612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__18));
v___x_2613_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2613_, 0, v___x_2555_);
lean_ctor_set(v___x_2613_, 1, v___x_2612_);
v___x_2614_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__19));
v___x_2615_ = l_Lean_Name_mkStr4(v___x_2556_, v___x_2557_, v___x_2508_, v___x_2614_);
v___x_2616_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__21, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__21);
v___x_2617_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__23));
v___x_2618_ = l_Lean_addMacroScope(v_quotContext_2553_, v___x_2617_, v_currMacroScope_2554_);
v___x_2619_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__25));
v___x_2620_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2620_, 0, v___x_2555_);
lean_ctor_set(v___x_2620_, 1, v___x_2616_);
lean_ctor_set(v___x_2620_, 2, v___x_2618_);
lean_ctor_set(v___x_2620_, 3, v___x_2619_);
lean_inc_ref_n(v___x_2578_, 7);
v___x_2621_ = l_Lean_Syntax_node3(v___x_2555_, v___x_2615_, v___x_2578_, v___x_2578_, v___x_2620_);
v___x_2622_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2567_, v___x_2621_);
v___x_2623_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__26));
v___x_2624_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2624_, 0, v___x_2555_);
lean_ctor_set(v___x_2624_, 1, v___x_2623_);
v___x_2625_ = l_Lean_Syntax_node3(v___x_2555_, v___x_2567_, v___x_2613_, v___x_2622_, v___x_2624_);
v___x_2626_ = l_Lean_Syntax_node5(v___x_2555_, v___x_2590_, v___x_2591_, v___x_2608_, v___x_2578_, v___x_2611_, v___x_2625_);
v___x_2627_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__27));
v___x_2628_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2628_, 0, v___x_2555_);
lean_ctor_set(v___x_2628_, 1, v___x_2627_);
v___x_2629_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__28));
v___x_2630_ = l_Lean_Name_mkStr3(v___x_2509_, v___x_2508_, v___x_2629_);
v___x_2631_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__12));
v___x_2632_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2632_, 0, v___x_2555_);
lean_ctor_set(v___x_2632_, 1, v___x_2631_);
v___x_2633_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2593_, v___x_2578_);
v___x_2634_ = l_Lean_Syntax_node4(v___x_2555_, v___x_2630_, v___x_2632_, v___x_2633_, v___x_2578_, v___x_2578_);
v___x_2635_ = l_Lean_Syntax_node3(v___x_2555_, v___x_2567_, v___x_2626_, v___x_2628_, v___x_2634_);
lean_inc(v___x_2584_);
v___x_2636_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2584_, v___x_2635_);
lean_inc(v___x_2582_);
v___x_2637_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2582_, v___x_2636_);
v___x_2638_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105));
v___x_2639_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2639_, 0, v___x_2555_);
lean_ctor_set(v___x_2639_, 1, v___x_2638_);
v___x_2640_ = l_Lean_Syntax_node3(v___x_2555_, v___x_2586_, v___x_2588_, v___x_2637_, v___x_2639_);
v___x_2641_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2567_, v___x_2640_);
v___x_2642_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2584_, v___x_2641_);
v___x_2643_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2582_, v___x_2642_);
v___x_2644_ = l_Lean_Syntax_node5(v___x_2555_, v___x_2574_, v___x_2576_, v___x_2578_, v___x_2578_, v___x_2580_, v___x_2643_);
v___x_2645_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2567_, v___x_2644_);
lean_inc(v___x_2566_);
v___x_2646_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2566_, v___x_2645_);
lean_inc(v___x_2564_);
v___x_2647_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2564_, v___x_2646_);
v___x_2648_ = l_Lean_Syntax_node2(v___x_2555_, v___x_2569_, v___x_2571_, v___x_2647_);
v___x_2649_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2567_, v___x_2648_);
v___x_2650_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2566_, v___x_2649_);
v___x_2651_ = l_Lean_Syntax_node1(v___x_2555_, v___x_2564_, v___x_2650_);
v___x_2652_ = l_Lean_Syntax_node2(v___x_2555_, v___x_2559_, v___x_2562_, v___x_2651_);
v___x_2653_ = l_Lean_Elab_Tactic_evalTactic(v___x_2652_, v___y_2529_, v___y_2530_, v___y_2531_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
if (lean_obj_tag(v___x_2653_) == 0)
{
lean_dec_ref_known(v___x_2653_, 1);
if (lean_obj_tag(v_bang_2510_) == 0)
{
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
goto v___jp_2520_;
}
else
{
if (v___x_2507_ == 0)
{
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
goto v___jp_2520_;
}
else
{
lean_object* v___x_2654_; 
v___x_2654_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_2530_);
if (lean_obj_tag(v___x_2654_) == 0)
{
lean_object* v_a_2655_; lean_object* v___x_2656_; lean_object* v___x_2657_; 
v_a_2655_ = lean_ctor_get(v___x_2654_, 0);
lean_inc(v_a_2655_);
lean_dec_ref_known(v___x_2654_, 1);
v___x_2656_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__29));
v___x_2657_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg(v_a_2655_, v___x_2656_, v___y_2529_, v___y_2530_, v___y_2531_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
lean_dec(v_a_2655_);
if (lean_obj_tag(v___x_2657_) == 0)
{
lean_object* v_a_2658_; lean_object* v_fst_2659_; lean_object* v_snd_2660_; lean_object* v___x_2662_; uint8_t v_isShared_2663_; uint8_t v_isSharedCheck_2710_; 
v_a_2658_ = lean_ctor_get(v___x_2657_, 0);
lean_inc(v_a_2658_);
lean_dec_ref_known(v___x_2657_, 1);
v_fst_2659_ = lean_ctor_get(v_a_2658_, 0);
v_snd_2660_ = lean_ctor_get(v_a_2658_, 1);
v_isSharedCheck_2710_ = !lean_is_exclusive(v_a_2658_);
if (v_isSharedCheck_2710_ == 0)
{
v___x_2662_ = v_a_2658_;
v_isShared_2663_ = v_isSharedCheck_2710_;
goto v_resetjp_2661_;
}
else
{
lean_inc(v_snd_2660_);
lean_inc(v_fst_2659_);
lean_dec(v_a_2658_);
v___x_2662_ = lean_box(0);
v_isShared_2663_ = v_isSharedCheck_2710_;
goto v_resetjp_2661_;
}
v_resetjp_2661_:
{
lean_object* v___x_2664_; lean_object* v___x_2665_; 
v___x_2664_ = lean_array_to_list(v_snd_2660_);
v___x_2665_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_2664_, v___y_2530_);
if (lean_obj_tag(v___x_2665_) == 0)
{
lean_object* v___x_2667_; uint8_t v_isShared_2668_; uint8_t v_isSharedCheck_2708_; 
v_isSharedCheck_2708_ = !lean_is_exclusive(v___x_2665_);
if (v_isSharedCheck_2708_ == 0)
{
lean_object* v_unused_2709_; 
v_unused_2709_ = lean_ctor_get(v___x_2665_, 0);
lean_dec(v_unused_2709_);
v___x_2667_ = v___x_2665_;
v_isShared_2668_ = v_isSharedCheck_2708_;
goto v_resetjp_2666_;
}
else
{
lean_dec(v___x_2665_);
v___x_2667_ = lean_box(0);
v_isShared_2668_ = v_isSharedCheck_2708_;
goto v_resetjp_2666_;
}
v_resetjp_2666_:
{
if (lean_obj_tag(v___y_2528_) == 1)
{
lean_object* v_val_2669_; size_t v_sz_2670_; size_t v___x_2671_; lean_object* v___x_2672_; 
lean_del_object(v___x_2667_);
v_val_2669_ = lean_ctor_get(v___y_2528_, 0);
lean_inc(v_val_2669_);
lean_dec_ref_known(v___y_2528_, 1);
v_sz_2670_ = lean_array_size(v_fst_2659_);
v___x_2671_ = ((size_t)0ULL);
v___x_2672_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg(v_sz_2670_, v___x_2671_, v_fst_2659_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
if (lean_obj_tag(v___x_2672_) == 0)
{
lean_object* v_a_2673_; lean_object* v___x_2675_; uint8_t v_isShared_2676_; uint8_t v_isSharedCheck_2695_; 
v_a_2673_ = lean_ctor_get(v___x_2672_, 0);
v_isSharedCheck_2695_ = !lean_is_exclusive(v___x_2672_);
if (v_isSharedCheck_2695_ == 0)
{
v___x_2675_ = v___x_2672_;
v_isShared_2676_ = v_isSharedCheck_2695_;
goto v_resetjp_2674_;
}
else
{
lean_inc(v_a_2673_);
lean_dec(v___x_2672_);
v___x_2675_ = lean_box(0);
v_isShared_2676_ = v_isSharedCheck_2695_;
goto v_resetjp_2674_;
}
v_resetjp_2674_:
{
lean_object* v___x_2677_; lean_object* v___x_2678_; uint8_t v___x_2679_; 
v___x_2677_ = lean_array_to_list(v_a_2673_);
lean_inc(v_val_2669_);
v___x_2678_ = lp_mathlib_Mathlib_Tactic_ComputeDegree_miscomputedDegree_x3f(v_val_2669_, v___x_2677_);
v___x_2679_ = l_List_isEmpty___redArg(v___x_2678_);
if (v___x_2679_ == 0)
{
lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2683_; 
lean_del_object(v___x_2675_);
v___x_2680_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__31, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__31);
v___x_2681_ = l_Lean_MessageData_ofExpr(v_val_2669_);
if (v_isShared_2663_ == 0)
{
lean_ctor_set_tag(v___x_2662_, 7);
lean_ctor_set(v___x_2662_, 1, v___x_2681_);
lean_ctor_set(v___x_2662_, 0, v___x_2680_);
v___x_2683_ = v___x_2662_;
goto v_reusejp_2682_;
}
else
{
lean_object* v_reuseFailAlloc_2690_; 
v_reuseFailAlloc_2690_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2690_, 0, v___x_2680_);
lean_ctor_set(v_reuseFailAlloc_2690_, 1, v___x_2681_);
v___x_2683_ = v_reuseFailAlloc_2690_;
goto v_reusejp_2682_;
}
v_reusejp_2682_:
{
lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; 
v___x_2684_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__33, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__33);
v___x_2685_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2685_, 0, v___x_2683_);
lean_ctor_set(v___x_2685_, 1, v___x_2684_);
v___x_2686_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2686_, 0, v___x_2685_);
lean_ctor_set(v___x_2686_, 1, v___x_2678_);
v___x_2687_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__34, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__34_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___closed__34);
v___x_2688_ = l_Lean_MessageData_joinSep(v___x_2686_, v___x_2687_);
v___x_2689_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(v___x_2688_, v___y_2533_, v___y_2534_, v___y_2535_, v___y_2536_);
lean_dec_ref(v___y_2535_);
return v___x_2689_;
}
}
else
{
lean_object* v___x_2691_; lean_object* v___x_2693_; 
lean_dec(v___x_2678_);
lean_dec(v_val_2669_);
lean_del_object(v___x_2662_);
lean_dec_ref(v___y_2535_);
v___x_2691_ = lean_box(0);
if (v_isShared_2676_ == 0)
{
lean_ctor_set(v___x_2675_, 0, v___x_2691_);
v___x_2693_ = v___x_2675_;
goto v_reusejp_2692_;
}
else
{
lean_object* v_reuseFailAlloc_2694_; 
v_reuseFailAlloc_2694_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2694_, 0, v___x_2691_);
v___x_2693_ = v_reuseFailAlloc_2694_;
goto v_reusejp_2692_;
}
v_reusejp_2692_:
{
return v___x_2693_;
}
}
}
}
else
{
lean_object* v_a_2696_; lean_object* v___x_2698_; uint8_t v_isShared_2699_; uint8_t v_isSharedCheck_2703_; 
lean_dec(v_val_2669_);
lean_del_object(v___x_2662_);
lean_dec_ref(v___y_2535_);
v_a_2696_ = lean_ctor_get(v___x_2672_, 0);
v_isSharedCheck_2703_ = !lean_is_exclusive(v___x_2672_);
if (v_isSharedCheck_2703_ == 0)
{
v___x_2698_ = v___x_2672_;
v_isShared_2699_ = v_isSharedCheck_2703_;
goto v_resetjp_2697_;
}
else
{
lean_inc(v_a_2696_);
lean_dec(v___x_2672_);
v___x_2698_ = lean_box(0);
v_isShared_2699_ = v_isSharedCheck_2703_;
goto v_resetjp_2697_;
}
v_resetjp_2697_:
{
lean_object* v___x_2701_; 
if (v_isShared_2699_ == 0)
{
v___x_2701_ = v___x_2698_;
goto v_reusejp_2700_;
}
else
{
lean_object* v_reuseFailAlloc_2702_; 
v_reuseFailAlloc_2702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2702_, 0, v_a_2696_);
v___x_2701_ = v_reuseFailAlloc_2702_;
goto v_reusejp_2700_;
}
v_reusejp_2700_:
{
return v___x_2701_;
}
}
}
}
else
{
lean_object* v___x_2704_; lean_object* v___x_2706_; 
lean_del_object(v___x_2662_);
lean_dec(v_fst_2659_);
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
v___x_2704_ = lean_box(0);
if (v_isShared_2668_ == 0)
{
lean_ctor_set(v___x_2667_, 0, v___x_2704_);
v___x_2706_ = v___x_2667_;
goto v_reusejp_2705_;
}
else
{
lean_object* v_reuseFailAlloc_2707_; 
v_reuseFailAlloc_2707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2707_, 0, v___x_2704_);
v___x_2706_ = v_reuseFailAlloc_2707_;
goto v_reusejp_2705_;
}
v_reusejp_2705_:
{
return v___x_2706_;
}
}
}
}
else
{
lean_del_object(v___x_2662_);
lean_dec(v_fst_2659_);
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
return v___x_2665_;
}
}
}
else
{
lean_object* v_a_2711_; lean_object* v___x_2713_; uint8_t v_isShared_2714_; uint8_t v_isSharedCheck_2718_; 
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
v_a_2711_ = lean_ctor_get(v___x_2657_, 0);
v_isSharedCheck_2718_ = !lean_is_exclusive(v___x_2657_);
if (v_isSharedCheck_2718_ == 0)
{
v___x_2713_ = v___x_2657_;
v_isShared_2714_ = v_isSharedCheck_2718_;
goto v_resetjp_2712_;
}
else
{
lean_inc(v_a_2711_);
lean_dec(v___x_2657_);
v___x_2713_ = lean_box(0);
v_isShared_2714_ = v_isSharedCheck_2718_;
goto v_resetjp_2712_;
}
v_resetjp_2712_:
{
lean_object* v___x_2716_; 
if (v_isShared_2714_ == 0)
{
v___x_2716_ = v___x_2713_;
goto v_reusejp_2715_;
}
else
{
lean_object* v_reuseFailAlloc_2717_; 
v_reuseFailAlloc_2717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2717_, 0, v_a_2711_);
v___x_2716_ = v_reuseFailAlloc_2717_;
goto v_reusejp_2715_;
}
v_reusejp_2715_:
{
return v___x_2716_;
}
}
}
}
else
{
lean_object* v_a_2719_; lean_object* v___x_2721_; uint8_t v_isShared_2722_; uint8_t v_isSharedCheck_2726_; 
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
v_a_2719_ = lean_ctor_get(v___x_2654_, 0);
v_isSharedCheck_2726_ = !lean_is_exclusive(v___x_2654_);
if (v_isSharedCheck_2726_ == 0)
{
v___x_2721_ = v___x_2654_;
v_isShared_2722_ = v_isSharedCheck_2726_;
goto v_resetjp_2720_;
}
else
{
lean_inc(v_a_2719_);
lean_dec(v___x_2654_);
v___x_2721_ = lean_box(0);
v_isShared_2722_ = v_isSharedCheck_2726_;
goto v_resetjp_2720_;
}
v_resetjp_2720_:
{
lean_object* v___x_2724_; 
if (v_isShared_2722_ == 0)
{
v___x_2724_ = v___x_2721_;
goto v_reusejp_2723_;
}
else
{
lean_object* v_reuseFailAlloc_2725_; 
v_reuseFailAlloc_2725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2725_, 0, v_a_2719_);
v___x_2724_ = v_reuseFailAlloc_2725_;
goto v_reusejp_2723_;
}
v_reusejp_2723_:
{
return v___x_2724_;
}
}
}
}
}
}
else
{
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
return v___x_2653_;
}
}
}
else
{
lean_del_object(v___x_2547_);
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
return v___x_2551_;
}
}
else
{
lean_object* v_a_2728_; lean_object* v___x_2730_; uint8_t v_isShared_2731_; uint8_t v_isSharedCheck_2735_; 
lean_del_object(v___x_2547_);
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v_a_2728_ = lean_ctor_get(v___x_2549_, 0);
v_isSharedCheck_2735_ = !lean_is_exclusive(v___x_2549_);
if (v_isSharedCheck_2735_ == 0)
{
v___x_2730_ = v___x_2549_;
v_isShared_2731_ = v_isSharedCheck_2735_;
goto v_resetjp_2729_;
}
else
{
lean_inc(v_a_2728_);
lean_dec(v___x_2549_);
v___x_2730_ = lean_box(0);
v_isShared_2731_ = v_isSharedCheck_2735_;
goto v_resetjp_2729_;
}
v_resetjp_2729_:
{
lean_object* v___x_2733_; 
if (v_isShared_2731_ == 0)
{
v___x_2733_ = v___x_2730_;
goto v_reusejp_2732_;
}
else
{
lean_object* v_reuseFailAlloc_2734_; 
v_reuseFailAlloc_2734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2734_, 0, v_a_2728_);
v___x_2733_ = v_reuseFailAlloc_2734_;
goto v_reusejp_2732_;
}
v_reusejp_2732_:
{
return v___x_2733_;
}
}
}
}
}
else
{
lean_object* v_a_2738_; lean_object* v___x_2740_; uint8_t v_isShared_2741_; uint8_t v_isSharedCheck_2745_; 
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v_a_2738_ = lean_ctor_get(v___x_2543_, 0);
v_isSharedCheck_2745_ = !lean_is_exclusive(v___x_2543_);
if (v_isSharedCheck_2745_ == 0)
{
v___x_2740_ = v___x_2543_;
v_isShared_2741_ = v_isSharedCheck_2745_;
goto v_resetjp_2739_;
}
else
{
lean_inc(v_a_2738_);
lean_dec(v___x_2543_);
v___x_2740_ = lean_box(0);
v_isShared_2741_ = v_isSharedCheck_2745_;
goto v_resetjp_2739_;
}
v_resetjp_2739_:
{
lean_object* v___x_2743_; 
if (v_isShared_2741_ == 0)
{
v___x_2743_ = v___x_2740_;
goto v_reusejp_2742_;
}
else
{
lean_object* v_reuseFailAlloc_2744_; 
v_reuseFailAlloc_2744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2744_, 0, v_a_2738_);
v___x_2743_ = v_reuseFailAlloc_2744_;
goto v_reusejp_2742_;
}
v_reusejp_2742_:
{
return v___x_2743_;
}
}
}
}
else
{
lean_object* v_a_2746_; lean_object* v___x_2748_; uint8_t v_isShared_2749_; uint8_t v_isSharedCheck_2753_; 
lean_dec_ref(v___y_2535_);
lean_dec(v___y_2528_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v_a_2746_ = lean_ctor_get(v___x_2539_, 0);
v_isSharedCheck_2753_ = !lean_is_exclusive(v___x_2539_);
if (v_isSharedCheck_2753_ == 0)
{
v___x_2748_ = v___x_2539_;
v_isShared_2749_ = v_isSharedCheck_2753_;
goto v_resetjp_2747_;
}
else
{
lean_inc(v_a_2746_);
lean_dec(v___x_2539_);
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
else
{
lean_object* v_a_2820_; lean_object* v___x_2822_; uint8_t v_isShared_2823_; uint8_t v_isSharedCheck_2827_; 
lean_dec_ref(v___y_2517_);
lean_dec_ref(v___x_2509_);
lean_dec_ref(v___x_2508_);
v_a_2820_ = lean_ctor_get(v___x_2523_, 0);
v_isSharedCheck_2827_ = !lean_is_exclusive(v___x_2523_);
if (v_isSharedCheck_2827_ == 0)
{
v___x_2822_ = v___x_2523_;
v_isShared_2823_ = v_isSharedCheck_2827_;
goto v_resetjp_2821_;
}
else
{
lean_inc(v_a_2820_);
lean_dec(v___x_2523_);
v___x_2822_ = lean_box(0);
v_isShared_2823_ = v_isSharedCheck_2827_;
goto v_resetjp_2821_;
}
v_resetjp_2821_:
{
lean_object* v___x_2825_; 
if (v_isShared_2823_ == 0)
{
v___x_2825_ = v___x_2822_;
goto v_reusejp_2824_;
}
else
{
lean_object* v_reuseFailAlloc_2826_; 
v_reuseFailAlloc_2826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2826_, 0, v_a_2820_);
v___x_2825_ = v_reuseFailAlloc_2826_;
goto v_reusejp_2824_;
}
v_reusejp_2824_:
{
return v___x_2825_;
}
}
}
v___jp_2520_:
{
lean_object* v___x_2521_; lean_object* v___x_2522_; 
v___x_2521_ = lean_box(0);
v___x_2522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2522_, 0, v___x_2521_);
return v___x_2522_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___boxed(lean_object* v___x_2828_, lean_object* v___x_2829_, lean_object* v___x_2830_, lean_object* v_bang_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_, lean_object* v___y_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_){
_start:
{
uint8_t v___x_31537__boxed_2841_; lean_object* v_res_2842_; 
v___x_31537__boxed_2841_ = lean_unbox(v___x_2828_);
v_res_2842_ = lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0(v___x_31537__boxed_2841_, v___x_2829_, v___x_2830_, v_bang_2831_, v___y_2832_, v___y_2833_, v___y_2834_, v___y_2835_, v___y_2836_, v___y_2837_, v___y_2838_, v___y_2839_);
lean_dec(v___y_2839_);
lean_dec(v___y_2837_);
lean_dec_ref(v___y_2836_);
lean_dec(v___y_2835_);
lean_dec_ref(v___y_2834_);
lean_dec(v___y_2833_);
lean_dec_ref(v___y_2832_);
lean_dec(v_bang_2831_);
return v_res_2842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1(lean_object* v_x_2843_, lean_object* v_a_2844_, lean_object* v_a_2845_, lean_object* v_a_2846_, lean_object* v_a_2847_, lean_object* v_a_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_){
_start:
{
lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; uint8_t v___x_2856_; lean_object* v_bang_2858_; lean_object* v___y_2859_; lean_object* v___y_2860_; lean_object* v___y_2861_; lean_object* v___y_2862_; lean_object* v___y_2863_; lean_object* v___y_2864_; lean_object* v___y_2865_; lean_object* v___y_2866_; 
v___x_2853_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__9));
v___x_2854_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_getCongrLemma___closed__10));
v___x_2855_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1));
lean_inc(v_x_2843_);
v___x_2856_ = l_Lean_Syntax_isOfKind(v_x_2843_, v___x_2855_);
if (v___x_2856_ == 0)
{
lean_object* v___x_2871_; 
lean_dec(v_x_2843_);
v___x_2871_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg();
return v___x_2871_;
}
else
{
lean_object* v___x_2872_; lean_object* v___x_2873_; uint8_t v___x_2874_; 
v___x_2872_ = lean_unsigned_to_nat(1u);
v___x_2873_ = l_Lean_Syntax_getArg(v_x_2843_, v___x_2872_);
lean_dec(v_x_2843_);
v___x_2874_ = l_Lean_Syntax_isNone(v___x_2873_);
if (v___x_2874_ == 0)
{
uint8_t v___x_2875_; 
lean_inc(v___x_2873_);
v___x_2875_ = l_Lean_Syntax_matchesNull(v___x_2873_, v___x_2872_);
if (v___x_2875_ == 0)
{
lean_object* v___x_2876_; 
lean_dec(v___x_2873_);
v___x_2876_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__1___redArg();
return v___x_2876_;
}
else
{
lean_object* v___x_2877_; lean_object* v_bang_2878_; lean_object* v___x_2879_; 
v___x_2877_ = lean_unsigned_to_nat(0u);
v_bang_2878_ = l_Lean_Syntax_getArg(v___x_2873_, v___x_2877_);
lean_dec(v___x_2873_);
v___x_2879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2879_, 0, v_bang_2878_);
v_bang_2858_ = v___x_2879_;
v___y_2859_ = v_a_2844_;
v___y_2860_ = v_a_2845_;
v___y_2861_ = v_a_2846_;
v___y_2862_ = v_a_2847_;
v___y_2863_ = v_a_2848_;
v___y_2864_ = v_a_2849_;
v___y_2865_ = v_a_2850_;
v___y_2866_ = v_a_2851_;
goto v___jp_2857_;
}
}
else
{
lean_object* v___x_2880_; 
lean_dec(v___x_2873_);
v___x_2880_ = lean_box(0);
v_bang_2858_ = v___x_2880_;
v___y_2859_ = v_a_2844_;
v___y_2860_ = v_a_2845_;
v___y_2861_ = v_a_2846_;
v___y_2862_ = v_a_2847_;
v___y_2863_ = v_a_2848_;
v___y_2864_ = v_a_2849_;
v___y_2865_ = v_a_2850_;
v___y_2866_ = v_a_2851_;
goto v___jp_2857_;
}
}
v___jp_2857_:
{
lean_object* v___x_2867_; lean_object* v___f_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; 
v___x_2867_ = lean_box(v___x_2856_);
v___f_2868_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_2868_, 0, v___x_2867_);
lean_closure_set(v___f_2868_, 1, v___x_2854_);
lean_closure_set(v___f_2868_, 2, v___x_2853_);
lean_closure_set(v___f_2868_, 3, v_bang_2858_);
v___x_2869_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withMainContext___boxed), 11, 2);
lean_closure_set(v___x_2869_, 0, lean_box(0));
lean_closure_set(v___x_2869_, 1, v___f_2868_);
v___x_2870_ = l_Lean_Elab_Tactic_focus___redArg(v___x_2869_, v___y_2859_, v___y_2860_, v___y_2861_, v___y_2862_, v___y_2863_, v___y_2864_, v___y_2865_, v___y_2866_);
return v___x_2870_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1___boxed(lean_object* v_x_2881_, lean_object* v_a_2882_, lean_object* v_a_2883_, lean_object* v_a_2884_, lean_object* v_a_2885_, lean_object* v_a_2886_, lean_object* v_a_2887_, lean_object* v_a_2888_, lean_object* v_a_2889_, lean_object* v_a_2890_){
_start:
{
lean_object* v_res_2891_; 
v_res_2891_ = lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1(v_x_2881_, v_a_2882_, v_a_2883_, v_a_2884_, v_a_2885_, v_a_2886_, v_a_2887_, v_a_2888_, v_a_2889_);
lean_dec(v_a_2889_);
lean_dec_ref(v_a_2888_);
lean_dec(v_a_2887_);
lean_dec_ref(v_a_2886_);
lean_dec(v_a_2885_);
lean_dec_ref(v_a_2884_);
lean_dec(v_a_2883_);
lean_dec_ref(v_a_2882_);
return v_res_2891_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4(lean_object* v_00_u03b1_2892_, lean_object* v_msg_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_, lean_object* v___y_2897_, lean_object* v___y_2898_, lean_object* v___y_2899_, lean_object* v___y_2900_, lean_object* v___y_2901_){
_start:
{
lean_object* v___x_2903_; 
v___x_2903_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___redArg(v_msg_2893_, v___y_2898_, v___y_2899_, v___y_2900_, v___y_2901_);
return v___x_2903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4___boxed(lean_object* v_00_u03b1_2904_, lean_object* v_msg_2905_, lean_object* v___y_2906_, lean_object* v___y_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_, lean_object* v___y_2914_){
_start:
{
lean_object* v_res_2915_; 
v_res_2915_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__4(v_00_u03b1_2904_, v_msg_2905_, v___y_2906_, v___y_2907_, v___y_2908_, v___y_2909_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_);
lean_dec(v___y_2913_);
lean_dec_ref(v___y_2912_);
lean_dec(v___y_2911_);
lean_dec_ref(v___y_2910_);
lean_dec(v___y_2909_);
lean_dec_ref(v___y_2908_);
lean_dec(v___y_2907_);
lean_dec_ref(v___y_2906_);
return v_res_2915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5(uint8_t v___x_2916_, lean_object* v_inst_2917_, lean_object* v_a_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_, lean_object* v___y_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_){
_start:
{
lean_object* v___x_2928_; 
v___x_2928_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___redArg(v___x_2916_, v_a_2918_, v___y_2923_, v___y_2924_, v___y_2925_, v___y_2926_);
return v___x_2928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5___boxed(lean_object* v___x_2929_, lean_object* v_inst_2930_, lean_object* v_a_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_, lean_object* v___y_2934_, lean_object* v___y_2935_, lean_object* v___y_2936_, lean_object* v___y_2937_, lean_object* v___y_2938_, lean_object* v___y_2939_, lean_object* v___y_2940_){
_start:
{
uint8_t v___x_32341__boxed_2941_; lean_object* v_res_2942_; 
v___x_32341__boxed_2941_ = lean_unbox(v___x_2929_);
v_res_2942_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__5(v___x_32341__boxed_2941_, v_inst_2930_, v_a_2931_, v___y_2932_, v___y_2933_, v___y_2934_, v___y_2935_, v___y_2936_, v___y_2937_, v___y_2938_, v___y_2939_);
lean_dec(v___y_2939_);
lean_dec_ref(v___y_2938_);
lean_dec(v___y_2937_);
lean_dec_ref(v___y_2936_);
lean_dec(v___y_2935_);
lean_dec_ref(v___y_2934_);
lean_dec(v___y_2933_);
lean_dec_ref(v___y_2932_);
return v_res_2942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6(lean_object* v_as_2943_, lean_object* v_as_x27_2944_, lean_object* v_b_2945_, lean_object* v_a_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_, lean_object* v___y_2953_, lean_object* v___y_2954_){
_start:
{
lean_object* v___x_2956_; 
v___x_2956_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg(v_as_x27_2944_, v_b_2945_, v___y_2947_, v___y_2948_, v___y_2949_, v___y_2950_, v___y_2951_, v___y_2952_, v___y_2953_, v___y_2954_);
return v___x_2956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___boxed(lean_object* v_as_2957_, lean_object* v_as_x27_2958_, lean_object* v_b_2959_, lean_object* v_a_2960_, lean_object* v___y_2961_, lean_object* v___y_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_, lean_object* v___y_2966_, lean_object* v___y_2967_, lean_object* v___y_2968_, lean_object* v___y_2969_){
_start:
{
lean_object* v_res_2970_; 
v_res_2970_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6(v_as_2957_, v_as_x27_2958_, v_b_2959_, v_a_2960_, v___y_2961_, v___y_2962_, v___y_2963_, v___y_2964_, v___y_2965_, v___y_2966_, v___y_2967_, v___y_2968_);
lean_dec(v___y_2968_);
lean_dec_ref(v___y_2967_);
lean_dec(v___y_2966_);
lean_dec_ref(v___y_2965_);
lean_dec(v___y_2964_);
lean_dec_ref(v___y_2963_);
lean_dec(v___y_2962_);
lean_dec_ref(v___y_2961_);
lean_dec(v_as_x27_2958_);
lean_dec(v_as_2957_);
return v_res_2970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7(size_t v_sz_2971_, size_t v_i_2972_, lean_object* v_bs_2973_, lean_object* v___y_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_){
_start:
{
lean_object* v___x_2983_; 
v___x_2983_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___redArg(v_sz_2971_, v_i_2972_, v_bs_2973_, v___y_2978_, v___y_2979_, v___y_2980_, v___y_2981_);
return v___x_2983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7___boxed(lean_object* v_sz_2984_, lean_object* v_i_2985_, lean_object* v_bs_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_){
_start:
{
size_t v_sz_boxed_2996_; size_t v_i_boxed_2997_; lean_object* v_res_2998_; 
v_sz_boxed_2996_ = lean_unbox_usize(v_sz_2984_);
lean_dec(v_sz_2984_);
v_i_boxed_2997_ = lean_unbox_usize(v_i_2985_);
lean_dec(v_i_2985_);
v_res_2998_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__7(v_sz_boxed_2996_, v_i_boxed_2997_, v_bs_2986_, v___y_2987_, v___y_2988_, v___y_2989_, v___y_2990_, v___y_2991_, v___y_2992_, v___y_2993_, v___y_2994_);
lean_dec(v___y_2994_);
lean_dec_ref(v___y_2993_);
lean_dec(v___y_2992_);
lean_dec_ref(v___y_2991_);
lean_dec(v___y_2990_);
lean_dec_ref(v___y_2989_);
lean_dec(v___y_2988_);
lean_dec_ref(v___y_2987_);
return v_res_2998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8(lean_object* v_cls_2999_, lean_object* v_msg_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_){
_start:
{
lean_object* v___x_3010_; 
v___x_3010_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___redArg(v_cls_2999_, v_msg_3000_, v___y_3005_, v___y_3006_, v___y_3007_, v___y_3008_);
return v___x_3010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8___boxed(lean_object* v_cls_3011_, lean_object* v_msg_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_, lean_object* v___y_3019_, lean_object* v___y_3020_, lean_object* v___y_3021_){
_start:
{
lean_object* v_res_3022_; 
v_res_3022_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__8(v_cls_3011_, v_msg_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_, v___y_3017_, v___y_3018_, v___y_3019_, v___y_3020_);
lean_dec(v___y_3020_);
lean_dec_ref(v___y_3019_);
lean_dec(v___y_3018_);
lean_dec_ref(v___y_3017_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
lean_dec(v___y_3014_);
lean_dec_ref(v___y_3013_);
return v_res_3022_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7(void){
_start:
{
lean_object* v___x_3065_; lean_object* v___x_3066_; 
v___x_3065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__6));
v___x_3066_ = l_String_toRawSubstring_x27(v___x_3065_);
return v___x_3066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1(lean_object* v_x_3078_, lean_object* v_a_3079_, lean_object* v_a_3080_){
_start:
{
lean_object* v___x_3081_; uint8_t v___x_3082_; 
v___x_3081_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_monicityMacro___closed__1));
v___x_3082_ = l_Lean_Syntax_isOfKind(v_x_3078_, v___x_3081_);
if (v___x_3082_ == 0)
{
lean_object* v___x_3083_; lean_object* v___x_3084_; 
v___x_3083_ = lean_box(1);
v___x_3084_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3084_, 0, v___x_3083_);
lean_ctor_set(v___x_3084_, 1, v_a_3080_);
return v___x_3084_;
}
else
{
lean_object* v_quotContext_3085_; lean_object* v_currMacroScope_3086_; lean_object* v_ref_3087_; uint8_t v___x_3088_; lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; lean_object* v___x_3092_; lean_object* v___x_3093_; lean_object* v___x_3094_; lean_object* v___x_3095_; lean_object* v___x_3096_; lean_object* v___x_3097_; lean_object* v___x_3098_; lean_object* v___x_3099_; lean_object* v___x_3100_; lean_object* v___x_3101_; lean_object* v___x_3102_; lean_object* v___x_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; lean_object* v___x_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; lean_object* v___x_3109_; lean_object* v___x_3110_; lean_object* v___x_3111_; lean_object* v___x_3112_; lean_object* v___x_3113_; lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v___x_3116_; lean_object* v___x_3117_; lean_object* v___x_3118_; lean_object* v___x_3119_; lean_object* v___x_3120_; lean_object* v___x_3121_; 
v_quotContext_3085_ = lean_ctor_get(v_a_3079_, 1);
v_currMacroScope_3086_ = lean_ctor_get(v_a_3079_, 2);
v_ref_3087_ = lean_ctor_get(v_a_3079_, 5);
v___x_3088_ = 0;
v___x_3089_ = l_Lean_SourceInfo_fromRef(v_ref_3087_, v___x_3088_);
v___x_3090_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0));
v___x_3091_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7));
lean_inc_n(v___x_3089_, 13);
v___x_3092_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3092_, 0, v___x_3089_);
lean_ctor_set(v___x_3092_, 1, v___x_3091_);
v___x_3093_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1));
v___x_3094_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2));
v___x_3095_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1));
v___x_3096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3));
v___x_3097_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__4));
v___x_3098_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5));
v___x_3099_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3099_, 0, v___x_3089_);
lean_ctor_set(v___x_3099_, 1, v___x_3097_);
v___x_3100_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7);
v___x_3101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__8));
lean_inc(v_currMacroScope_3086_);
lean_inc(v_quotContext_3085_);
v___x_3102_ = l_Lean_addMacroScope(v_quotContext_3085_, v___x_3101_, v_currMacroScope_3086_);
v___x_3103_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__11));
v___x_3104_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3104_, 0, v___x_3089_);
lean_ctor_set(v___x_3104_, 1, v___x_3100_);
lean_ctor_set(v___x_3104_, 2, v___x_3102_);
lean_ctor_set(v___x_3104_, 3, v___x_3103_);
v___x_3105_ = l_Lean_Syntax_node2(v___x_3089_, v___x_3098_, v___x_3099_, v___x_3104_);
v___x_3106_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__15));
v___x_3107_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3107_, 0, v___x_3089_);
lean_ctor_set(v___x_3107_, 1, v___x_3106_);
v___x_3108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__1));
v___x_3109_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4));
v___x_3110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3110_, 0, v___x_3089_);
lean_ctor_set(v___x_3110_, 1, v___x_3109_);
v___x_3111_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14, &lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14_once, _init_lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__14);
v___x_3112_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3112_, 0, v___x_3089_);
lean_ctor_set(v___x_3112_, 1, v___x_3095_);
lean_ctor_set(v___x_3112_, 2, v___x_3111_);
v___x_3113_ = l_Lean_Syntax_node2(v___x_3089_, v___x_3108_, v___x_3110_, v___x_3112_);
v___x_3114_ = l_Lean_Syntax_node3(v___x_3089_, v___x_3096_, v___x_3105_, v___x_3107_, v___x_3113_);
v___x_3115_ = l_Lean_Syntax_node1(v___x_3089_, v___x_3095_, v___x_3114_);
v___x_3116_ = l_Lean_Syntax_node1(v___x_3089_, v___x_3094_, v___x_3115_);
v___x_3117_ = l_Lean_Syntax_node1(v___x_3089_, v___x_3093_, v___x_3116_);
v___x_3118_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105));
v___x_3119_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3119_, 0, v___x_3089_);
lean_ctor_set(v___x_3119_, 1, v___x_3118_);
v___x_3120_ = l_Lean_Syntax_node3(v___x_3089_, v___x_3090_, v___x_3092_, v___x_3117_, v___x_3119_);
v___x_3121_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3121_, 0, v___x_3120_);
lean_ctor_set(v___x_3121_, 1, v_a_3080_);
return v___x_3121_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___boxed(lean_object* v_x_3122_, lean_object* v_a_3123_, lean_object* v_a_3124_){
_start:
{
lean_object* v_res_3125_; 
v_res_3125_ = lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1(v_x_3122_, v_a_3123_, v_a_3124_);
lean_dec_ref(v_a_3123_);
return v_res_3125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticMonicity_x21__1(lean_object* v_x_3141_, lean_object* v_a_3142_, lean_object* v_a_3143_){
_start:
{
lean_object* v___x_3144_; uint8_t v___x_3145_; 
v___x_3144_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticMonicity_x21___closed__1));
v___x_3145_ = l_Lean_Syntax_isOfKind(v_x_3141_, v___x_3144_);
if (v___x_3145_ == 0)
{
lean_object* v___x_3146_; lean_object* v___x_3147_; 
v___x_3146_ = lean_box(1);
v___x_3147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3147_, 0, v___x_3146_);
lean_ctor_set(v___x_3147_, 1, v_a_3143_);
return v___x_3147_;
}
else
{
lean_object* v_quotContext_3148_; lean_object* v_currMacroScope_3149_; lean_object* v_ref_3150_; uint8_t v___x_3151_; lean_object* v___x_3152_; lean_object* v___x_3153_; lean_object* v___x_3154_; lean_object* v___x_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; lean_object* v___x_3158_; lean_object* v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; lean_object* v___x_3162_; lean_object* v___x_3163_; lean_object* v___x_3164_; lean_object* v___x_3165_; lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; lean_object* v___x_3174_; lean_object* v___x_3175_; lean_object* v___x_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; lean_object* v___x_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; 
v_quotContext_3148_ = lean_ctor_get(v_a_3142_, 1);
v_currMacroScope_3149_ = lean_ctor_get(v_a_3142_, 2);
v_ref_3150_ = lean_ctor_get(v_a_3142_, 5);
v___x_3151_ = 0;
v___x_3152_ = l_Lean_SourceInfo_fromRef(v_ref_3150_, v___x_3151_);
v___x_3153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__0));
v___x_3154_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__7));
lean_inc_n(v___x_3152_, 12);
v___x_3155_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3155_, 0, v___x_3152_);
lean_ctor_set(v___x_3155_, 1, v___x_3154_);
v___x_3156_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__1));
v___x_3157_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__2));
v___x_3158_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticCompute__degree_x21__1___closed__1));
v___x_3159_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__3));
v___x_3160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__4));
v___x_3161_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__5));
v___x_3162_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3162_, 0, v___x_3152_);
lean_ctor_set(v___x_3162_, 1, v___x_3160_);
v___x_3163_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7, &lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__7);
v___x_3164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__8));
lean_inc(v_currMacroScope_3149_);
lean_inc(v_quotContext_3148_);
v___x_3165_ = l_Lean_addMacroScope(v_quotContext_3148_, v___x_3164_, v_currMacroScope_3149_);
v___x_3166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__monicityMacro__1___closed__11));
v___x_3167_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3167_, 0, v___x_3152_);
lean_ctor_set(v___x_3167_, 1, v___x_3163_);
lean_ctor_set(v___x_3167_, 2, v___x_3165_);
lean_ctor_set(v___x_3167_, 3, v___x_3166_);
v___x_3168_ = l_Lean_Syntax_node2(v___x_3152_, v___x_3161_, v___x_3162_, v___x_3167_);
v___x_3169_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______elabRules__Mathlib__Tactic__ComputeDegree__computeDegree__1_spec__6___redArg___lam__0___closed__15));
v___x_3170_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3170_, 0, v___x_3152_);
lean_ctor_set(v___x_3170_, 1, v___x_3169_);
v___x_3171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__1));
v___x_3172_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_tacticCompute__degree_x21___closed__2));
v___x_3173_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3173_, 0, v___x_3152_);
lean_ctor_set(v___x_3173_, 1, v___x_3172_);
v___x_3174_ = l_Lean_Syntax_node1(v___x_3152_, v___x_3171_, v___x_3173_);
v___x_3175_ = l_Lean_Syntax_node3(v___x_3152_, v___x_3159_, v___x_3168_, v___x_3170_, v___x_3174_);
v___x_3176_ = l_Lean_Syntax_node1(v___x_3152_, v___x_3158_, v___x_3175_);
v___x_3177_ = l_Lean_Syntax_node1(v___x_3152_, v___x_3157_, v___x_3176_);
v___x_3178_ = l_Lean_Syntax_node1(v___x_3152_, v___x_3156_, v___x_3177_);
v___x_3179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_dispatchLemma___closed__105));
v___x_3180_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3180_, 0, v___x_3152_);
lean_ctor_set(v___x_3180_, 1, v___x_3179_);
v___x_3181_ = l_Lean_Syntax_node3(v___x_3152_, v___x_3153_, v___x_3155_, v___x_3178_, v___x_3180_);
v___x_3182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3182_, 0, v___x_3181_);
lean_ctor_set(v___x_3182_, 1, v_a_3143_);
return v___x_3182_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticMonicity_x21__1___boxed(lean_object* v_x_3183_, lean_object* v_a_3184_, lean_object* v_a_3185_){
_start:
{
lean_object* v_res_3186_; 
v_res_3186_ = lp_mathlib_Mathlib_Tactic_ComputeDegree___aux__Mathlib__Tactic__ComputeDegree______macroRules__Mathlib__Tactic__ComputeDegree__tacticMonicity_x21__1(v_x_3183_, v_a_3184_, v_a_3185_);
lean_dec_ref(v_a_3184_);
return v_res_3186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg(lean_object* v_a_3193_){
_start:
{
lean_object* v___x_3195_; lean_object* v_env_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; 
v___x_3195_ = lean_st_ref_get(v_a_3193_);
v_env_3196_ = lean_ctor_get(v___x_3195_, 0);
lean_inc_ref(v_env_3196_);
lean_dec(v___x_3195_);
v___x_3197_ = ((lean_object*)(lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__1));
v___x_3198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ComputeDegree_computeDegree___closed__4));
v___x_3199_ = ((lean_object*)(lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__2));
v___x_3200_ = l_Lean_Parser_runParserCategory(v_env_3196_, v___x_3197_, v___x_3198_, v___x_3199_);
if (lean_obj_tag(v___x_3200_) == 0)
{
lean_object* v___x_3202_; uint8_t v_isShared_3203_; uint8_t v_isSharedCheck_3208_; 
v_isSharedCheck_3208_ = !lean_is_exclusive(v___x_3200_);
if (v_isSharedCheck_3208_ == 0)
{
lean_object* v_unused_3209_; 
v_unused_3209_ = lean_ctor_get(v___x_3200_, 0);
lean_dec(v_unused_3209_);
v___x_3202_ = v___x_3200_;
v_isShared_3203_ = v_isSharedCheck_3208_;
goto v_resetjp_3201_;
}
else
{
lean_dec(v___x_3200_);
v___x_3202_ = lean_box(0);
v_isShared_3203_ = v_isSharedCheck_3208_;
goto v_resetjp_3201_;
}
v_resetjp_3201_:
{
lean_object* v___x_3204_; lean_object* v___x_3206_; 
v___x_3204_ = ((lean_object*)(lp_mathlib___auxTryTactic12152987033550515202___redArg___closed__3));
if (v_isShared_3203_ == 0)
{
lean_ctor_set(v___x_3202_, 0, v___x_3204_);
v___x_3206_ = v___x_3202_;
goto v_reusejp_3205_;
}
else
{
lean_object* v_reuseFailAlloc_3207_; 
v_reuseFailAlloc_3207_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3207_, 0, v___x_3204_);
v___x_3206_ = v_reuseFailAlloc_3207_;
goto v_reusejp_3205_;
}
v_reusejp_3205_:
{
return v___x_3206_;
}
}
}
else
{
lean_object* v_a_3210_; lean_object* v___x_3212_; uint8_t v_isShared_3213_; uint8_t v_isSharedCheck_3220_; 
v_a_3210_ = lean_ctor_get(v___x_3200_, 0);
v_isSharedCheck_3220_ = !lean_is_exclusive(v___x_3200_);
if (v_isSharedCheck_3220_ == 0)
{
v___x_3212_ = v___x_3200_;
v_isShared_3213_ = v_isSharedCheck_3220_;
goto v_resetjp_3211_;
}
else
{
lean_inc(v_a_3210_);
lean_dec(v___x_3200_);
v___x_3212_ = lean_box(0);
v_isShared_3213_ = v_isSharedCheck_3220_;
goto v_resetjp_3211_;
}
v_resetjp_3211_:
{
lean_object* v___x_3214_; lean_object* v___x_3215_; lean_object* v___x_3216_; lean_object* v___x_3218_; 
v___x_3214_ = lean_unsigned_to_nat(1u);
v___x_3215_ = lean_mk_empty_array_with_capacity(v___x_3214_);
v___x_3216_ = lean_array_push(v___x_3215_, v_a_3210_);
if (v_isShared_3213_ == 0)
{
lean_ctor_set_tag(v___x_3212_, 0);
lean_ctor_set(v___x_3212_, 0, v___x_3216_);
v___x_3218_ = v___x_3212_;
goto v_reusejp_3217_;
}
else
{
lean_object* v_reuseFailAlloc_3219_; 
v_reuseFailAlloc_3219_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3219_, 0, v___x_3216_);
v___x_3218_ = v_reuseFailAlloc_3219_;
goto v_reusejp_3217_;
}
v_reusejp_3217_:
{
return v___x_3218_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202___redArg___boxed(lean_object* v_a_3221_, lean_object* v_a_3222_){
_start:
{
lean_object* v_res_3223_; 
v_res_3223_ = lp_mathlib___auxTryTactic12152987033550515202___redArg(v_a_3221_);
lean_dec(v_a_3221_);
return v_res_3223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202(lean_object* v___goal_3224_, lean_object* v___info_3225_, lean_object* v_a_3226_, lean_object* v_a_3227_, lean_object* v_a_3228_, lean_object* v_a_3229_){
_start:
{
lean_object* v___x_3231_; 
v___x_3231_ = lp_mathlib___auxTryTactic12152987033550515202___redArg(v_a_3229_);
return v___x_3231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___auxTryTactic12152987033550515202___boxed(lean_object* v___goal_3232_, lean_object* v___info_3233_, lean_object* v_a_3234_, lean_object* v_a_3235_, lean_object* v_a_3236_, lean_object* v_a_3237_, lean_object* v_a_3238_){
_start:
{
lean_object* v_res_3239_; 
v_res_3239_ = lp_mathlib___auxTryTactic12152987033550515202(v___goal_3232_, v___info_3233_, v_a_3234_, v_a_3235_, v_a_3236_, v_a_3237_);
lean_dec(v_a_3237_);
lean_dec_ref(v_a_3236_);
lean_dec(v_a_3235_);
lean_dec_ref(v_a_3234_);
lean_dec_ref(v___info_3233_);
lean_dec(v___goal_3232_);
return v_res_3239_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ComputeDegree(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ComputeDegree(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_ComputeDegree_0__Mathlib_Tactic_ComputeDegree_initFn_00___x40_Mathlib_Tactic_ComputeDegree_1830119225____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ComputeDegree(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ComputeDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ComputeDegree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ComputeDegree(builtin);
}
#ifdef __cplusplus
}
#endif
