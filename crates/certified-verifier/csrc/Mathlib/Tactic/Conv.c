// Lean compiler output
// Module: Mathlib.Tactic.Conv
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Conv.Basic public meta import Lean.Elab.Command
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
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getGoals___redArg(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_refl(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_inferInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Elab_Tactic_pruneSolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray3___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
extern lean_object* l_Lean_Parser_Tactic_Conv_convSeq;
extern lean_object* l_Lean_Parser_Tactic_Conv_occs;
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
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_simpArgs;
lean_object* l_Lean_Elab_Term_elabTermAndSynthesize(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_mkConvGoalFor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_runTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Conv_getLhsRhsCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqTrue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_done(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convLHS"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 199, 252, 210, 27, 127, 215, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "conv_lhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " at "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__22;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__23_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__28;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__31;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__33;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_convLHS;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "convSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "in"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(202, 81, 30, 13, 252, 23, 29, 64)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(192, 39, 103, 162, 58, 5, 181, 114)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "convRHS"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 141, 229, 94, 0, 2, 204, 200)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "conv_rhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRHS;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rhs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "convRun_conv_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(1, 67, 38, 23, 102, 217, 2, 116)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "run_conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "doSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 108, 208, 147, 238, 58, 86, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRun__conv__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "nestedTacticCore"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(169, 23, 62, 30, 134, 160, 158, 203)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "tactic'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "runTac"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 93, 177, 70, 42, 135, 100, 249)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "run_tac"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "convConvIn__=>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(162, 192, 123, 110, 106, 248, 24, 242)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e__;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(4, 172, 52, 155, 88, 222, 189, 57)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "convConvSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(136, 85, 97, 250, 95, 212, 248, 253)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(249, 35, 202, 76, 198, 168, 114, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "pattern"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(59, 139, 144, 223, 221, 17, 152, 53)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "dischargeConv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 125, 66, 83, 245, 64, 99, 127)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "discharge"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_dischargeConv = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "target is not a proposition"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "convRefine_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(127, 59, 91, 31, 116, 134, 176, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "refine "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_convRefine__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "nestedTactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 28, 213, 2, 207, 8, 223, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "command#conv_=>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(254, 184, 104, 200, 83, 192, 77, 132)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "#conv "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(232, 67, 39, 189, 45, 247, 54, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 188, 245, 35, 76, 176, 121, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "with_reducible "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_withReducible;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 44, 223, 192, 8, 197, 146, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "convTactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(174, 239, 251, 250, 179, 30, 250, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "conv'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "command#whnf_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 172, 20, 244, 172, 130, 42, 69)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "#whnf "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#conv"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "whnf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 111, 86, 148, 119, 255, 116, 73)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "command#whnfR_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(141, 87, 144, 153, 243, 198, 0, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#whnfR "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnfR____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnfR____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "command#simpOnly_=>__"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(82, 42, 44, 190, 203, 225, 51, 226)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(198, 48, 238, 216, 47, 218, 159, 19)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " =>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__13;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e____;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(235, 165, 42, 136, 187, 206, 234, 202)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "simpArgs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(158, 198, 190, 154, 66, 126, 242, 208)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = l_Lean_Parser_Tactic_Conv_occs;
v___x_43_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10));
v___x_44_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___x_42_);
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__22(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_45_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21);
v___x_46_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__20));
v___x_47_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_48_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v___x_46_);
lean_ctor_set(v___x_48_, 2, v___x_45_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__26(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25));
v___x_56_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__22, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__22_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__22);
v___x_57_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_58_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v___x_56_);
lean_ctor_set(v___x_58_, 2, v___x_55_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_59_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__26, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__26);
v___x_60_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10));
v___x_61_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v___x_59_);
return v___x_61_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__28(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_62_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27);
v___x_63_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__18));
v___x_64_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_65_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_65_, 0, v___x_64_);
lean_ctor_set(v___x_65_, 1, v___x_63_);
lean_ctor_set(v___x_65_, 2, v___x_62_);
return v___x_65_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__31(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_69_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30));
v___x_70_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__28, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__28);
v___x_71_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_72_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v___x_70_);
lean_ctor_set(v___x_72_, 2, v___x_69_);
return v___x_72_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__32(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_73_ = l_Lean_Parser_Tactic_Conv_convSeq;
v___x_74_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__31, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__31_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__31);
v___x_75_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_76_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_74_);
lean_ctor_set(v___x_76_, 2, v___x_73_);
return v___x_76_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__33(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_77_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__32, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__32_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__32);
v___x_78_ = lean_unsigned_to_nat(1022u);
v___x_79_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4));
v___x_80_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v___x_78_);
lean_ctor_set(v___x_80_, 2, v___x_77_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS(void){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__33, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__33);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(lean_object* v_x_84_, lean_object* v_x_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___boxed(lean_object* v_x_87_, lean_object* v_x_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(v_x_87_, v_x_88_);
lean_dec(v_x_88_);
lean_dec(v_x_87_);
return v_res_89_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16(void){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = l_Array_mkArray0(lean_box(0));
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1(lean_object* v_x_119_, lean_object* v_a_120_, lean_object* v_a_121_){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___y_125_; lean_object* v___y_126_; lean_object* v___y_127_; lean_object* v___y_128_; lean_object* v___y_129_; lean_object* v___y_130_; lean_object* v___y_131_; lean_object* v___y_132_; lean_object* v___y_133_; lean_object* v___y_134_; lean_object* v___y_135_; lean_object* v___y_136_; lean_object* v___y_162_; lean_object* v___y_163_; lean_object* v___y_164_; lean_object* v___y_165_; lean_object* v___y_166_; lean_object* v___y_167_; lean_object* v___y_168_; lean_object* v___y_169_; lean_object* v___y_170_; lean_object* v___y_171_; lean_object* v___y_172_; lean_object* v___y_173_; lean_object* v___y_174_; lean_object* v___y_175_; lean_object* v___x_179_; uint8_t v___x_180_; 
v___x_122_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1));
v___x_123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2));
v___x_179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__4));
lean_inc(v_x_119_);
v___x_180_ = l_Lean_Syntax_isOfKind(v_x_119_, v___x_179_);
if (v___x_180_ == 0)
{
lean_object* v___x_181_; lean_object* v___x_182_; 
lean_dec(v_x_119_);
v___x_181_ = lean_box(1);
v___x_182_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v_a_121_);
return v___x_182_;
}
else
{
lean_object* v___x_183_; lean_object* v___y_185_; lean_object* v___y_186_; lean_object* v___y_187_; lean_object* v___y_188_; lean_object* v___y_189_; lean_object* v___y_190_; lean_object* v___y_191_; lean_object* v___y_192_; lean_object* v___y_193_; lean_object* v___y_194_; lean_object* v___y_195_; lean_object* v___y_196_; lean_object* v___y_197_; lean_object* v___y_210_; lean_object* v_occs_211_; lean_object* v_pat_212_; lean_object* v___y_213_; lean_object* v___y_214_; lean_object* v___y_234_; lean_object* v___y_235_; lean_object* v___y_236_; lean_object* v_occs_237_; lean_object* v___y_238_; lean_object* v___y_239_; lean_object* v___x_243_; lean_object* v_id_245_; lean_object* v___y_246_; lean_object* v___y_247_; lean_object* v___x_264_; uint8_t v___x_265_; 
v___x_183_ = lean_unsigned_to_nat(0u);
v___x_243_ = lean_unsigned_to_nat(1u);
v___x_264_ = l_Lean_Syntax_getArg(v_x_119_, v___x_243_);
v___x_265_ = l_Lean_Syntax_isNone(v___x_264_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; uint8_t v___x_267_; 
v___x_266_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_264_);
v___x_267_ = l_Lean_Syntax_matchesNull(v___x_264_, v___x_266_);
if (v___x_267_ == 0)
{
lean_object* v___x_268_; lean_object* v___x_269_; 
lean_dec(v___x_264_);
lean_dec(v_x_119_);
v___x_268_ = lean_box(1);
v___x_269_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_269_, 0, v___x_268_);
lean_ctor_set(v___x_269_, 1, v_a_121_);
return v___x_269_;
}
else
{
lean_object* v_id_270_; lean_object* v___x_271_; 
v_id_270_ = l_Lean_Syntax_getArg(v___x_264_, v___x_243_);
lean_dec(v___x_264_);
v___x_271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_271_, 0, v_id_270_);
v_id_245_ = v___x_271_;
v___y_246_ = v_a_120_;
v___y_247_ = v_a_121_;
goto v___jp_244_;
}
}
else
{
lean_object* v___x_272_; 
lean_dec(v___x_264_);
v___x_272_ = lean_box(0);
v_id_245_ = v___x_272_;
v___y_246_ = v_a_120_;
v___y_247_ = v_a_121_;
goto v___jp_244_;
}
v___jp_184_:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
lean_inc_ref(v___y_187_);
v___x_198_ = l_Array_append___redArg(v___y_187_, v___y_197_);
lean_dec_ref(v___y_197_);
lean_inc(v___y_192_);
lean_inc(v___y_188_);
v___x_199_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_199_, 0, v___y_188_);
lean_ctor_set(v___x_199_, 1, v___y_192_);
lean_ctor_set(v___x_199_, 2, v___x_198_);
if (lean_obj_tag(v___y_189_) == 1)
{
if (lean_obj_tag(v___y_185_) == 1)
{
lean_object* v_val_200_; lean_object* v_val_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v_val_200_ = lean_ctor_get(v___y_189_, 0);
lean_inc(v_val_200_);
lean_dec_ref_known(v___y_189_, 1);
v_val_201_ = lean_ctor_get(v___y_185_, 0);
lean_inc(v_val_201_);
lean_dec_ref_known(v___y_185_, 1);
v___x_202_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__7));
lean_inc(v___y_188_);
v___x_203_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_203_, 0, v___y_188_);
lean_ctor_set(v___x_203_, 1, v___x_202_);
if (lean_obj_tag(v_val_200_) == 1)
{
lean_object* v_val_204_; lean_object* v___x_205_; 
v_val_204_ = lean_ctor_get(v_val_200_, 0);
lean_inc(v_val_204_);
lean_dec_ref_known(v_val_200_, 1);
v___x_205_ = l_Array_mkArray1___redArg(v_val_204_);
v___y_162_ = v_val_201_;
v___y_163_ = v___y_188_;
v___y_164_ = v___y_191_;
v___y_165_ = v___y_190_;
v___y_166_ = v___y_193_;
v___y_167_ = v___y_194_;
v___y_168_ = v___x_199_;
v___y_169_ = v___x_203_;
v___y_170_ = v___y_196_;
v___y_171_ = v___y_186_;
v___y_172_ = v___y_187_;
v___y_173_ = v___y_192_;
v___y_174_ = v___y_195_;
v___y_175_ = v___x_205_;
goto v___jp_161_;
}
else
{
lean_object* v___x_206_; 
lean_dec(v_val_200_);
v___x_206_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_162_ = v_val_201_;
v___y_163_ = v___y_188_;
v___y_164_ = v___y_191_;
v___y_165_ = v___y_190_;
v___y_166_ = v___y_193_;
v___y_167_ = v___y_194_;
v___y_168_ = v___x_199_;
v___y_169_ = v___x_203_;
v___y_170_ = v___y_196_;
v___y_171_ = v___y_186_;
v___y_172_ = v___y_187_;
v___y_173_ = v___y_192_;
v___y_174_ = v___y_195_;
v___y_175_ = v___x_206_;
goto v___jp_161_;
}
}
else
{
lean_object* v___x_207_; 
v___x_207_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(v___y_189_, v___y_185_);
lean_dec(v___y_185_);
lean_dec_ref_known(v___y_189_, 1);
v___y_125_ = v___y_186_;
v___y_126_ = v___y_187_;
v___y_127_ = v___y_188_;
v___y_128_ = v___y_191_;
v___y_129_ = v___y_190_;
v___y_130_ = v___y_192_;
v___y_131_ = v___y_193_;
v___y_132_ = v___y_194_;
v___y_133_ = v___x_199_;
v___y_134_ = v___y_195_;
v___y_135_ = v___y_196_;
v___y_136_ = v___x_207_;
goto v___jp_124_;
}
}
else
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(v___y_189_, v___y_185_);
lean_dec(v___y_185_);
lean_dec(v___y_189_);
v___y_125_ = v___y_186_;
v___y_126_ = v___y_187_;
v___y_127_ = v___y_188_;
v___y_128_ = v___y_191_;
v___y_129_ = v___y_190_;
v___y_130_ = v___y_192_;
v___y_131_ = v___y_193_;
v___y_132_ = v___y_194_;
v___y_133_ = v___x_199_;
v___y_134_ = v___y_195_;
v___y_135_ = v___y_196_;
v___y_136_ = v___x_208_;
goto v___jp_124_;
}
}
v___jp_209_:
{
lean_object* v_ref_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; uint8_t v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v_ref_215_ = lean_ctor_get(v___y_213_, 5);
v___x_216_ = lean_unsigned_to_nat(4u);
v___x_217_ = l_Lean_Syntax_getArg(v_x_119_, v___x_216_);
lean_dec(v_x_119_);
v___x_218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8));
v___x_219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9));
v___x_220_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11));
v___x_221_ = 0;
v___x_222_ = l_Lean_SourceInfo_fromRef(v_ref_215_, v___x_221_);
v___x_223_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12));
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13));
lean_inc(v___x_222_);
v___x_225_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_222_);
lean_ctor_set(v___x_225_, 1, v___x_223_);
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_227_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16, &lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16);
if (lean_obj_tag(v___y_210_) == 1)
{
lean_object* v_val_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; 
v_val_228_ = lean_ctor_get(v___y_210_, 0);
lean_inc(v_val_228_);
lean_dec_ref_known(v___y_210_, 1);
v___x_229_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__17));
lean_inc(v___x_222_);
v___x_230_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_230_, 0, v___x_222_);
lean_ctor_set(v___x_230_, 1, v___x_229_);
v___x_231_ = l_Array_mkArray2___redArg(v___x_230_, v_val_228_);
v___y_185_ = v_pat_212_;
v___y_186_ = v___y_214_;
v___y_187_ = v___x_227_;
v___y_188_ = v___x_222_;
v___y_189_ = v_occs_211_;
v___y_190_ = v___x_220_;
v___y_191_ = v___x_225_;
v___y_192_ = v___x_226_;
v___y_193_ = v___x_218_;
v___y_194_ = v___x_219_;
v___y_195_ = v___x_217_;
v___y_196_ = v___x_224_;
v___y_197_ = v___x_231_;
goto v___jp_184_;
}
else
{
lean_object* v___x_232_; 
lean_dec(v___y_210_);
v___x_232_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_185_ = v_pat_212_;
v___y_186_ = v___y_214_;
v___y_187_ = v___x_227_;
v___y_188_ = v___x_222_;
v___y_189_ = v_occs_211_;
v___y_190_ = v___x_220_;
v___y_191_ = v___x_225_;
v___y_192_ = v___x_226_;
v___y_193_ = v___x_218_;
v___y_194_ = v___x_219_;
v___y_195_ = v___x_217_;
v___y_196_ = v___x_224_;
v___y_197_ = v___x_232_;
goto v___jp_184_;
}
}
v___jp_233_:
{
lean_object* v_pat_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v_pat_240_ = l_Lean_Syntax_getArg(v___y_234_, v___y_236_);
lean_dec(v___y_234_);
v___x_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_241_, 0, v_occs_237_);
v___x_242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_242_, 0, v_pat_240_);
v___y_210_ = v___y_235_;
v_occs_211_ = v___x_241_;
v_pat_212_ = v___x_242_;
v___y_213_ = v___y_238_;
v___y_214_ = v___y_239_;
goto v___jp_209_;
}
v___jp_244_:
{
lean_object* v___x_248_; lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_248_ = lean_unsigned_to_nat(2u);
v___x_249_ = l_Lean_Syntax_getArg(v_x_119_, v___x_248_);
v___x_250_ = l_Lean_Syntax_isNone(v___x_249_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_251_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_249_);
v___x_252_ = l_Lean_Syntax_matchesNull(v___x_249_, v___x_251_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; lean_object* v___x_254_; 
lean_dec(v___x_249_);
lean_dec(v_id_245_);
lean_dec(v_x_119_);
v___x_253_ = lean_box(1);
v___x_254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v___y_247_);
return v___x_254_;
}
else
{
lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_255_ = l_Lean_Syntax_getArg(v___x_249_, v___x_243_);
v___x_256_ = l_Lean_Syntax_isNone(v___x_255_);
if (v___x_256_ == 0)
{
uint8_t v___x_257_; 
lean_inc(v___x_255_);
v___x_257_ = l_Lean_Syntax_matchesNull(v___x_255_, v___x_243_);
if (v___x_257_ == 0)
{
lean_object* v___x_258_; lean_object* v___x_259_; 
lean_dec(v___x_255_);
lean_dec(v___x_249_);
lean_dec(v_id_245_);
lean_dec(v_x_119_);
v___x_258_ = lean_box(1);
v___x_259_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_258_);
lean_ctor_set(v___x_259_, 1, v___y_247_);
return v___x_259_;
}
else
{
lean_object* v_occs_260_; lean_object* v___x_261_; 
v_occs_260_ = l_Lean_Syntax_getArg(v___x_255_, v___x_183_);
lean_dec(v___x_255_);
v___x_261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_261_, 0, v_occs_260_);
v___y_234_ = v___x_249_;
v___y_235_ = v_id_245_;
v___y_236_ = v___x_248_;
v_occs_237_ = v___x_261_;
v___y_238_ = v___y_246_;
v___y_239_ = v___y_247_;
goto v___jp_233_;
}
}
else
{
lean_object* v___x_262_; 
lean_dec(v___x_255_);
v___x_262_ = lean_box(0);
v___y_234_ = v___x_249_;
v___y_235_ = v_id_245_;
v___y_236_ = v___x_248_;
v_occs_237_ = v___x_262_;
v___y_238_ = v___y_246_;
v___y_239_ = v___y_247_;
goto v___jp_233_;
}
}
}
else
{
lean_object* v___x_263_; 
lean_dec(v___x_249_);
v___x_263_ = lean_box(0);
v___y_210_ = v_id_245_;
v_occs_211_ = v___x_263_;
v_pat_212_ = v___x_263_;
v___y_213_ = v___y_246_;
v___y_214_ = v___y_247_;
goto v___jp_209_;
}
}
}
v___jp_124_:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
lean_inc_ref(v___y_126_);
v___x_137_ = l_Array_append___redArg(v___y_126_, v___y_136_);
lean_dec_ref(v___y_136_);
lean_inc_n(v___y_130_, 2);
lean_inc_n(v___y_127_, 11);
v___x_138_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_138_, 0, v___y_127_);
lean_ctor_set(v___x_138_, 1, v___y_130_);
lean_ctor_set(v___x_138_, 2, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_140_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_140_, 0, v___y_127_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v___x_141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__1));
lean_inc_ref_n(v___y_132_, 3);
lean_inc_ref_n(v___y_131_, 3);
v___x_142_ = l_Lean_Name_mkStr5(v___y_131_, v___y_132_, v___x_122_, v___x_123_, v___x_141_);
v___x_143_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__2));
v___x_144_ = l_Lean_Name_mkStr5(v___y_131_, v___y_132_, v___x_122_, v___x_123_, v___x_143_);
v___x_145_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_145_, 0, v___y_127_);
lean_ctor_set(v___x_145_, 1, v___x_143_);
v___x_146_ = l_Lean_Syntax_node1(v___y_127_, v___x_144_, v___x_145_);
v___x_147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__3));
v___x_148_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_148_, 0, v___y_127_);
lean_ctor_set(v___x_148_, 1, v___x_147_);
v___x_149_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__4));
v___x_150_ = l_Lean_Name_mkStr5(v___y_131_, v___y_132_, v___x_122_, v___x_123_, v___x_149_);
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__5));
v___x_152_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_152_, 0, v___y_127_);
lean_ctor_set(v___x_152_, 1, v___x_151_);
v___x_153_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__6));
v___x_154_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_154_, 0, v___y_127_);
lean_ctor_set(v___x_154_, 1, v___x_153_);
v___x_155_ = l_Lean_Syntax_node3(v___y_127_, v___x_150_, v___x_152_, v___y_134_, v___x_154_);
v___x_156_ = l_Lean_Syntax_node3(v___y_127_, v___y_130_, v___x_146_, v___x_148_, v___x_155_);
v___x_157_ = l_Lean_Syntax_node1(v___y_127_, v___x_142_, v___x_156_);
lean_inc(v___y_129_);
v___x_158_ = l_Lean_Syntax_node1(v___y_127_, v___y_129_, v___x_157_);
lean_inc(v___y_135_);
v___x_159_ = l_Lean_Syntax_node5(v___y_127_, v___y_135_, v___y_128_, v___y_133_, v___x_138_, v___x_140_, v___x_158_);
v___x_160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_159_);
lean_ctor_set(v___x_160_, 1, v___y_125_);
return v___x_160_;
}
v___jp_161_:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
lean_inc_ref(v___y_172_);
v___x_176_ = l_Array_append___redArg(v___y_172_, v___y_175_);
lean_dec_ref(v___y_175_);
lean_inc(v___y_173_);
lean_inc(v___y_163_);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v___y_163_);
lean_ctor_set(v___x_177_, 1, v___y_173_);
lean_ctor_set(v___x_177_, 2, v___x_176_);
v___x_178_ = l_Array_mkArray3___redArg(v___y_169_, v___x_177_, v___y_162_);
v___y_125_ = v___y_171_;
v___y_126_ = v___y_172_;
v___y_127_ = v___y_163_;
v___y_128_ = v___y_164_;
v___y_129_ = v___y_165_;
v___y_130_ = v___y_173_;
v___y_131_ = v___y_166_;
v___y_132_ = v___y_167_;
v___y_133_ = v___y_168_;
v___y_134_ = v___y_174_;
v___y_135_ = v___y_170_;
v___y_136_ = v___x_178_;
goto v___jp_124_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___boxed(lean_object* v_x_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1(v_x_273_, v_a_274_, v_a_275_);
lean_dec_ref(v_a_274_);
return v_res_276_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__5(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_291_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__27);
v___x_292_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__4));
v___x_293_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_294_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v___x_292_);
lean_ctor_set(v___x_294_, 2, v___x_291_);
return v___x_294_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__6(void){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_295_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30));
v___x_296_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__5, &lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__5);
v___x_297_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_298_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_298_, 0, v___x_297_);
lean_ctor_set(v___x_298_, 1, v___x_296_);
lean_ctor_set(v___x_298_, 2, v___x_295_);
return v___x_298_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__7(void){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_299_ = l_Lean_Parser_Tactic_Conv_convSeq;
v___x_300_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__6, &lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__6);
v___x_301_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_302_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
lean_ctor_set(v___x_302_, 1, v___x_300_);
lean_ctor_set(v___x_302_, 2, v___x_299_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__8(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; 
v___x_303_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__7, &lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__7);
v___x_304_ = lean_unsigned_to_nat(1022u);
v___x_305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1));
v___x_306_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_306_, 0, v___x_305_);
lean_ctor_set(v___x_306_, 1, v___x_304_);
lean_ctor_set(v___x_306_, 2, v___x_303_);
return v___x_306_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS(void){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__8, &lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__8);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1(lean_object* v_x_309_, lean_object* v_a_310_, lean_object* v_a_311_){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___y_315_; lean_object* v___y_316_; lean_object* v___y_317_; lean_object* v___y_318_; lean_object* v___y_319_; lean_object* v___y_320_; lean_object* v___y_321_; lean_object* v___y_322_; lean_object* v___y_323_; lean_object* v___y_324_; lean_object* v___y_325_; lean_object* v___y_326_; lean_object* v___y_352_; lean_object* v___y_353_; lean_object* v___y_354_; lean_object* v___y_355_; lean_object* v___y_356_; lean_object* v___y_357_; lean_object* v___y_358_; lean_object* v___y_359_; lean_object* v___y_360_; lean_object* v___y_361_; lean_object* v___y_362_; lean_object* v___y_363_; lean_object* v___y_364_; lean_object* v___y_365_; lean_object* v___x_369_; uint8_t v___x_370_; 
v___x_312_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1));
v___x_313_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__2));
v___x_369_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convRHS___closed__1));
lean_inc(v_x_309_);
v___x_370_ = l_Lean_Syntax_isOfKind(v_x_309_, v___x_369_);
if (v___x_370_ == 0)
{
lean_object* v___x_371_; lean_object* v___x_372_; 
lean_dec(v_x_309_);
v___x_371_ = lean_box(1);
v___x_372_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v_a_311_);
return v___x_372_;
}
else
{
lean_object* v___x_373_; lean_object* v___y_375_; lean_object* v___y_376_; lean_object* v___y_377_; lean_object* v___y_378_; lean_object* v___y_379_; lean_object* v___y_380_; lean_object* v___y_381_; lean_object* v___y_382_; lean_object* v___y_383_; lean_object* v___y_384_; lean_object* v___y_385_; lean_object* v___y_386_; lean_object* v___y_387_; lean_object* v___y_400_; lean_object* v_occs_401_; lean_object* v_pat_402_; lean_object* v___y_403_; lean_object* v___y_404_; lean_object* v___y_424_; lean_object* v___y_425_; lean_object* v___y_426_; lean_object* v_occs_427_; lean_object* v___y_428_; lean_object* v___y_429_; lean_object* v___x_433_; lean_object* v_id_435_; lean_object* v___y_436_; lean_object* v___y_437_; lean_object* v___x_454_; uint8_t v___x_455_; 
v___x_373_ = lean_unsigned_to_nat(0u);
v___x_433_ = lean_unsigned_to_nat(1u);
v___x_454_ = l_Lean_Syntax_getArg(v_x_309_, v___x_433_);
v___x_455_ = l_Lean_Syntax_isNone(v___x_454_);
if (v___x_455_ == 0)
{
lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_456_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_454_);
v___x_457_ = l_Lean_Syntax_matchesNull(v___x_454_, v___x_456_);
if (v___x_457_ == 0)
{
lean_object* v___x_458_; lean_object* v___x_459_; 
lean_dec(v___x_454_);
lean_dec(v_x_309_);
v___x_458_ = lean_box(1);
v___x_459_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_459_, 0, v___x_458_);
lean_ctor_set(v___x_459_, 1, v_a_311_);
return v___x_459_;
}
else
{
lean_object* v_id_460_; lean_object* v___x_461_; 
v_id_460_ = l_Lean_Syntax_getArg(v___x_454_, v___x_433_);
lean_dec(v___x_454_);
v___x_461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_461_, 0, v_id_460_);
v_id_435_ = v___x_461_;
v___y_436_ = v_a_310_;
v___y_437_ = v_a_311_;
goto v___jp_434_;
}
}
else
{
lean_object* v___x_462_; 
lean_dec(v___x_454_);
v___x_462_ = lean_box(0);
v_id_435_ = v___x_462_;
v___y_436_ = v_a_310_;
v___y_437_ = v_a_311_;
goto v___jp_434_;
}
v___jp_374_:
{
lean_object* v___x_388_; lean_object* v___x_389_; 
lean_inc_ref(v___y_382_);
v___x_388_ = l_Array_append___redArg(v___y_382_, v___y_387_);
lean_dec_ref(v___y_387_);
lean_inc(v___y_386_);
lean_inc(v___y_376_);
v___x_389_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_389_, 0, v___y_376_);
lean_ctor_set(v___x_389_, 1, v___y_386_);
lean_ctor_set(v___x_389_, 2, v___x_388_);
if (lean_obj_tag(v___y_378_) == 1)
{
if (lean_obj_tag(v___y_375_) == 1)
{
lean_object* v_val_390_; lean_object* v_val_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v_val_390_ = lean_ctor_get(v___y_378_, 0);
lean_inc(v_val_390_);
lean_dec_ref_known(v___y_378_, 1);
v_val_391_ = lean_ctor_get(v___y_375_, 0);
lean_inc(v_val_391_);
lean_dec_ref_known(v___y_375_, 1);
v___x_392_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__7));
lean_inc(v___y_376_);
v___x_393_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_393_, 0, v___y_376_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
if (lean_obj_tag(v_val_390_) == 1)
{
lean_object* v_val_394_; lean_object* v___x_395_; 
v_val_394_ = lean_ctor_get(v_val_390_, 0);
lean_inc(v_val_394_);
lean_dec_ref_known(v_val_390_, 1);
v___x_395_ = l_Array_mkArray1___redArg(v_val_394_);
v___y_352_ = v___y_379_;
v___y_353_ = v___y_380_;
v___y_354_ = v___y_381_;
v___y_355_ = v___y_383_;
v___y_356_ = v_val_391_;
v___y_357_ = v___y_384_;
v___y_358_ = v___y_385_;
v___y_359_ = v___y_386_;
v___y_360_ = v___y_376_;
v___y_361_ = v___y_377_;
v___y_362_ = v___y_382_;
v___y_363_ = v___x_389_;
v___y_364_ = v___x_393_;
v___y_365_ = v___x_395_;
goto v___jp_351_;
}
else
{
lean_object* v___x_396_; 
lean_dec(v_val_390_);
v___x_396_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_352_ = v___y_379_;
v___y_353_ = v___y_380_;
v___y_354_ = v___y_381_;
v___y_355_ = v___y_383_;
v___y_356_ = v_val_391_;
v___y_357_ = v___y_384_;
v___y_358_ = v___y_385_;
v___y_359_ = v___y_386_;
v___y_360_ = v___y_376_;
v___y_361_ = v___y_377_;
v___y_362_ = v___y_382_;
v___y_363_ = v___x_389_;
v___y_364_ = v___x_393_;
v___y_365_ = v___x_396_;
goto v___jp_351_;
}
}
else
{
lean_object* v___x_397_; 
v___x_397_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(v___y_378_, v___y_375_);
lean_dec(v___y_375_);
lean_dec_ref_known(v___y_378_, 1);
v___y_315_ = v___y_376_;
v___y_316_ = v___y_377_;
v___y_317_ = v___y_379_;
v___y_318_ = v___y_380_;
v___y_319_ = v___y_381_;
v___y_320_ = v___y_382_;
v___y_321_ = v___y_383_;
v___y_322_ = v___x_389_;
v___y_323_ = v___y_384_;
v___y_324_ = v___y_385_;
v___y_325_ = v___y_386_;
v___y_326_ = v___x_397_;
goto v___jp_314_;
}
}
else
{
lean_object* v___x_398_; 
v___x_398_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0(v___y_378_, v___y_375_);
lean_dec(v___y_375_);
lean_dec(v___y_378_);
v___y_315_ = v___y_376_;
v___y_316_ = v___y_377_;
v___y_317_ = v___y_379_;
v___y_318_ = v___y_380_;
v___y_319_ = v___y_381_;
v___y_320_ = v___y_382_;
v___y_321_ = v___y_383_;
v___y_322_ = v___x_389_;
v___y_323_ = v___y_384_;
v___y_324_ = v___y_385_;
v___y_325_ = v___y_386_;
v___y_326_ = v___x_398_;
goto v___jp_314_;
}
}
v___jp_399_:
{
lean_object* v_ref_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; uint8_t v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v_ref_405_ = lean_ctor_get(v___y_403_, 5);
v___x_406_ = lean_unsigned_to_nat(4u);
v___x_407_ = l_Lean_Syntax_getArg(v_x_309_, v___x_406_);
lean_dec(v_x_309_);
v___x_408_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__8));
v___x_409_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__9));
v___x_410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11));
v___x_411_ = 0;
v___x_412_ = l_Lean_SourceInfo_fromRef(v_ref_405_, v___x_411_);
v___x_413_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12));
v___x_414_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__13));
lean_inc(v___x_412_);
v___x_415_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_412_);
lean_ctor_set(v___x_415_, 1, v___x_413_);
v___x_416_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_417_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16, &lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16);
if (lean_obj_tag(v___y_400_) == 1)
{
lean_object* v_val_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; 
v_val_418_ = lean_ctor_get(v___y_400_, 0);
lean_inc(v_val_418_);
lean_dec_ref_known(v___y_400_, 1);
v___x_419_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__17));
lean_inc(v___x_412_);
v___x_420_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_420_, 0, v___x_412_);
lean_ctor_set(v___x_420_, 1, v___x_419_);
v___x_421_ = l_Array_mkArray2___redArg(v___x_420_, v_val_418_);
v___y_375_ = v_pat_402_;
v___y_376_ = v___x_412_;
v___y_377_ = v___y_404_;
v___y_378_ = v_occs_401_;
v___y_379_ = v___x_407_;
v___y_380_ = v___x_415_;
v___y_381_ = v___x_410_;
v___y_382_ = v___x_417_;
v___y_383_ = v___x_409_;
v___y_384_ = v___x_408_;
v___y_385_ = v___x_414_;
v___y_386_ = v___x_416_;
v___y_387_ = v___x_421_;
goto v___jp_374_;
}
else
{
lean_object* v___x_422_; 
lean_dec(v___y_400_);
v___x_422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_375_ = v_pat_402_;
v___y_376_ = v___x_412_;
v___y_377_ = v___y_404_;
v___y_378_ = v_occs_401_;
v___y_379_ = v___x_407_;
v___y_380_ = v___x_415_;
v___y_381_ = v___x_410_;
v___y_382_ = v___x_417_;
v___y_383_ = v___x_409_;
v___y_384_ = v___x_408_;
v___y_385_ = v___x_414_;
v___y_386_ = v___x_416_;
v___y_387_ = v___x_422_;
goto v___jp_374_;
}
}
v___jp_423_:
{
lean_object* v_pat_430_; lean_object* v___x_431_; lean_object* v___x_432_; 
v_pat_430_ = l_Lean_Syntax_getArg(v___y_425_, v___y_426_);
lean_dec(v___y_425_);
v___x_431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_431_, 0, v_occs_427_);
v___x_432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_432_, 0, v_pat_430_);
v___y_400_ = v___y_424_;
v_occs_401_ = v___x_431_;
v_pat_402_ = v___x_432_;
v___y_403_ = v___y_428_;
v___y_404_ = v___y_429_;
goto v___jp_399_;
}
v___jp_434_:
{
lean_object* v___x_438_; lean_object* v___x_439_; uint8_t v___x_440_; 
v___x_438_ = lean_unsigned_to_nat(2u);
v___x_439_ = l_Lean_Syntax_getArg(v_x_309_, v___x_438_);
v___x_440_ = l_Lean_Syntax_isNone(v___x_439_);
if (v___x_440_ == 0)
{
lean_object* v___x_441_; uint8_t v___x_442_; 
v___x_441_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_439_);
v___x_442_ = l_Lean_Syntax_matchesNull(v___x_439_, v___x_441_);
if (v___x_442_ == 0)
{
lean_object* v___x_443_; lean_object* v___x_444_; 
lean_dec(v___x_439_);
lean_dec(v_id_435_);
lean_dec(v_x_309_);
v___x_443_ = lean_box(1);
v___x_444_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_443_);
lean_ctor_set(v___x_444_, 1, v___y_437_);
return v___x_444_;
}
else
{
lean_object* v___x_445_; uint8_t v___x_446_; 
v___x_445_ = l_Lean_Syntax_getArg(v___x_439_, v___x_433_);
v___x_446_ = l_Lean_Syntax_isNone(v___x_445_);
if (v___x_446_ == 0)
{
uint8_t v___x_447_; 
lean_inc(v___x_445_);
v___x_447_ = l_Lean_Syntax_matchesNull(v___x_445_, v___x_433_);
if (v___x_447_ == 0)
{
lean_object* v___x_448_; lean_object* v___x_449_; 
lean_dec(v___x_445_);
lean_dec(v___x_439_);
lean_dec(v_id_435_);
lean_dec(v_x_309_);
v___x_448_ = lean_box(1);
v___x_449_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_449_, 0, v___x_448_);
lean_ctor_set(v___x_449_, 1, v___y_437_);
return v___x_449_;
}
else
{
lean_object* v_occs_450_; lean_object* v___x_451_; 
v_occs_450_ = l_Lean_Syntax_getArg(v___x_445_, v___x_373_);
lean_dec(v___x_445_);
v___x_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_451_, 0, v_occs_450_);
v___y_424_ = v_id_435_;
v___y_425_ = v___x_439_;
v___y_426_ = v___x_438_;
v_occs_427_ = v___x_451_;
v___y_428_ = v___y_436_;
v___y_429_ = v___y_437_;
goto v___jp_423_;
}
}
else
{
lean_object* v___x_452_; 
lean_dec(v___x_445_);
v___x_452_ = lean_box(0);
v___y_424_ = v_id_435_;
v___y_425_ = v___x_439_;
v___y_426_ = v___x_438_;
v_occs_427_ = v___x_452_;
v___y_428_ = v___y_436_;
v___y_429_ = v___y_437_;
goto v___jp_423_;
}
}
}
else
{
lean_object* v___x_453_; 
lean_dec(v___x_439_);
v___x_453_ = lean_box(0);
v___y_400_ = v_id_435_;
v_occs_401_ = v___x_453_;
v_pat_402_ = v___x_453_;
v___y_403_ = v___y_436_;
v___y_404_ = v___y_437_;
goto v___jp_399_;
}
}
}
v___jp_314_:
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
lean_inc_ref(v___y_320_);
v___x_327_ = l_Array_append___redArg(v___y_320_, v___y_326_);
lean_dec_ref(v___y_326_);
lean_inc_n(v___y_325_, 2);
lean_inc_n(v___y_315_, 11);
v___x_328_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_328_, 0, v___y_315_);
lean_ctor_set(v___x_328_, 1, v___y_325_);
lean_ctor_set(v___x_328_, 2, v___x_327_);
v___x_329_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_330_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_330_, 0, v___y_315_);
lean_ctor_set(v___x_330_, 1, v___x_329_);
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__1));
lean_inc_ref_n(v___y_321_, 3);
lean_inc_ref_n(v___y_323_, 3);
v___x_332_ = l_Lean_Name_mkStr5(v___y_323_, v___y_321_, v___x_312_, v___x_313_, v___x_331_);
v___x_333_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1___closed__0));
v___x_334_ = l_Lean_Name_mkStr5(v___y_323_, v___y_321_, v___x_312_, v___x_313_, v___x_333_);
v___x_335_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_335_, 0, v___y_315_);
lean_ctor_set(v___x_335_, 1, v___x_333_);
v___x_336_ = l_Lean_Syntax_node1(v___y_315_, v___x_334_, v___x_335_);
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__3));
v___x_338_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_338_, 0, v___y_315_);
lean_ctor_set(v___x_338_, 1, v___x_337_);
v___x_339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__4));
v___x_340_ = l_Lean_Name_mkStr5(v___y_323_, v___y_321_, v___x_312_, v___x_313_, v___x_339_);
v___x_341_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__5));
v___x_342_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_342_, 0, v___y_315_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__6));
v___x_344_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_344_, 0, v___y_315_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = l_Lean_Syntax_node3(v___y_315_, v___x_340_, v___x_342_, v___y_317_, v___x_344_);
v___x_346_ = l_Lean_Syntax_node3(v___y_315_, v___y_325_, v___x_336_, v___x_338_, v___x_345_);
v___x_347_ = l_Lean_Syntax_node1(v___y_315_, v___x_332_, v___x_346_);
lean_inc(v___y_319_);
v___x_348_ = l_Lean_Syntax_node1(v___y_315_, v___y_319_, v___x_347_);
lean_inc(v___y_324_);
v___x_349_ = l_Lean_Syntax_node5(v___y_315_, v___y_324_, v___y_318_, v___y_322_, v___x_328_, v___x_330_, v___x_348_);
v___x_350_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
lean_ctor_set(v___x_350_, 1, v___y_316_);
return v___x_350_;
}
v___jp_351_:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
lean_inc_ref(v___y_362_);
v___x_366_ = l_Array_append___redArg(v___y_362_, v___y_365_);
lean_dec_ref(v___y_365_);
lean_inc(v___y_359_);
lean_inc(v___y_360_);
v___x_367_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_367_, 0, v___y_360_);
lean_ctor_set(v___x_367_, 1, v___y_359_);
lean_ctor_set(v___x_367_, 2, v___x_366_);
v___x_368_ = l_Array_mkArray3___redArg(v___y_364_, v___x_367_, v___y_356_);
v___y_315_ = v___y_360_;
v___y_316_ = v___y_361_;
v___y_317_ = v___y_352_;
v___y_318_ = v___y_353_;
v___y_319_ = v___y_354_;
v___y_320_ = v___y_362_;
v___y_321_ = v___y_355_;
v___y_322_ = v___y_363_;
v___y_323_ = v___y_357_;
v___y_324_ = v___y_358_;
v___y_325_ = v___y_359_;
v___y_326_ = v___x_368_;
goto v___jp_314_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1___boxed(lean_object* v_x_463_, lean_object* v_a_464_, lean_object* v_a_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRHS__1(v_x_463_, v_a_464_, v_a_465_);
lean_dec_ref(v_a_464_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1(lean_object* v_x_518_, lean_object* v_a_519_, lean_object* v_a_520_){
_start:
{
lean_object* v___x_521_; uint8_t v___x_522_; 
v___x_521_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convRun__conv___00__closed__1));
lean_inc(v_x_518_);
v___x_522_ = l_Lean_Syntax_isOfKind(v_x_518_, v___x_521_);
if (v___x_522_ == 0)
{
lean_object* v___x_523_; lean_object* v___x_524_; 
lean_dec(v_x_518_);
v___x_523_ = lean_box(1);
v___x_524_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_524_, 0, v___x_523_);
lean_ctor_set(v___x_524_, 1, v_a_520_);
return v___x_524_;
}
else
{
lean_object* v_ref_525_; lean_object* v___x_526_; lean_object* v___x_527_; uint8_t v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; 
v_ref_525_ = lean_ctor_get(v_a_519_, 5);
v___x_526_ = lean_unsigned_to_nat(1u);
v___x_527_ = l_Lean_Syntax_getArg(v_x_518_, v___x_526_);
lean_dec(v_x_518_);
v___x_528_ = 0;
v___x_529_ = l_Lean_SourceInfo_fromRef(v_ref_525_, v___x_528_);
v___x_530_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1));
v___x_531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__2));
lean_inc_n(v___x_529_, 7);
v___x_532_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_532_, 0, v___x_529_);
lean_ctor_set(v___x_532_, 1, v___x_531_);
v___x_533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_534_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_529_);
lean_ctor_set(v___x_534_, 1, v___x_533_);
v___x_535_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4));
v___x_536_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6));
v___x_537_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_538_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__8));
v___x_539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__9));
v___x_540_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_540_, 0, v___x_529_);
lean_ctor_set(v___x_540_, 1, v___x_539_);
v___x_541_ = l_Lean_Syntax_node2(v___x_529_, v___x_538_, v___x_540_, v___x_527_);
v___x_542_ = l_Lean_Syntax_node1(v___x_529_, v___x_537_, v___x_541_);
v___x_543_ = l_Lean_Syntax_node1(v___x_529_, v___x_536_, v___x_542_);
v___x_544_ = l_Lean_Syntax_node1(v___x_529_, v___x_535_, v___x_543_);
v___x_545_ = l_Lean_Syntax_node3(v___x_529_, v___x_530_, v___x_532_, v___x_534_, v___x_544_);
v___x_546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_546_, 0, v___x_545_);
lean_ctor_set(v___x_546_, 1, v_a_520_);
return v___x_546_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___boxed(lean_object* v_x_547_, lean_object* v_a_548_, lean_object* v_a_549_){
_start:
{
lean_object* v_res_550_; 
v_res_550_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1(v_x_547_, v_a_548_, v_a_549_);
lean_dec_ref(v_a_548_);
return v_res_550_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__4(void){
_start:
{
lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
v___x_564_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21, &lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__21);
v___x_565_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__3));
v___x_566_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_567_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_567_, 0, v___x_566_);
lean_ctor_set(v___x_567_, 1, v___x_565_);
lean_ctor_set(v___x_567_, 2, v___x_564_);
return v___x_567_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__5(void){
_start:
{
lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; 
v___x_568_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25));
v___x_569_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__4, &lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__4);
v___x_570_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_571_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_571_, 0, v___x_570_);
lean_ctor_set(v___x_571_, 1, v___x_569_);
lean_ctor_set(v___x_571_, 2, v___x_568_);
return v___x_571_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__6(void){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; 
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__30));
v___x_573_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__5, &lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__5);
v___x_574_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_575_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_575_, 0, v___x_574_);
lean_ctor_set(v___x_575_, 1, v___x_573_);
lean_ctor_set(v___x_575_, 2, v___x_572_);
return v___x_575_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__7(void){
_start:
{
lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_576_ = l_Lean_Parser_Tactic_Conv_convSeq;
v___x_577_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__6, &lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__6);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_579_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_579_, 0, v___x_578_);
lean_ctor_set(v___x_579_, 1, v___x_577_);
lean_ctor_set(v___x_579_, 2, v___x_576_);
return v___x_579_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__8(void){
_start:
{
lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_580_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__7, &lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__7);
v___x_581_ = lean_unsigned_to_nat(1022u);
v___x_582_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1));
v___x_583_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_583_, 0, v___x_582_);
lean_ctor_set(v___x_583_, 1, v___x_581_);
lean_ctor_set(v___x_583_, 2, v___x_580_);
return v___x_583_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e__(void){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__8, &lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__8);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1(lean_object* v_x_611_, lean_object* v_a_612_, lean_object* v_a_613_){
_start:
{
lean_object* v___x_614_; uint8_t v___x_615_; 
v___x_614_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e___00__closed__1));
lean_inc(v_x_611_);
v___x_615_ = l_Lean_Syntax_isOfKind(v_x_611_, v___x_614_);
if (v___x_615_ == 0)
{
lean_object* v___x_616_; lean_object* v___x_617_; 
lean_dec(v_x_611_);
v___x_616_ = lean_box(1);
v___x_617_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_617_, 0, v___x_616_);
lean_ctor_set(v___x_617_, 1, v_a_613_);
return v___x_617_;
}
else
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___y_626_; lean_object* v___y_627_; lean_object* v___y_628_; lean_object* v___y_629_; lean_object* v___y_630_; lean_object* v___y_631_; lean_object* v___y_632_; lean_object* v___y_633_; lean_object* v___y_634_; lean_object* v___y_635_; lean_object* v___y_653_; lean_object* v___x_671_; 
v___x_618_ = lean_unsigned_to_nat(2u);
v___x_619_ = l_Lean_Syntax_getArg(v_x_611_, v___x_618_);
v___x_620_ = lean_unsigned_to_nat(3u);
v___x_621_ = l_Lean_Syntax_getArg(v_x_611_, v___x_620_);
v___x_622_ = lean_unsigned_to_nat(5u);
v___x_623_ = l_Lean_Syntax_getArg(v_x_611_, v___x_622_);
lean_dec(v_x_611_);
v___x_624_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11));
v___x_671_ = l_Lean_Syntax_getOptional_x3f(v___x_619_);
lean_dec(v___x_619_);
if (lean_obj_tag(v___x_671_) == 0)
{
lean_object* v___x_672_; 
v___x_672_ = lean_box(0);
v___y_653_ = v___x_672_;
goto v___jp_652_;
}
else
{
lean_object* v_val_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_680_; 
v_val_673_ = lean_ctor_get(v___x_671_, 0);
v_isSharedCheck_680_ = !lean_is_exclusive(v___x_671_);
if (v_isSharedCheck_680_ == 0)
{
v___x_675_ = v___x_671_;
v_isShared_676_ = v_isSharedCheck_680_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_val_673_);
lean_dec(v___x_671_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_680_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_678_; 
if (v_isShared_676_ == 0)
{
v___x_678_ = v___x_675_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_679_; 
v_reuseFailAlloc_679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_679_, 0, v_val_673_);
v___x_678_ = v_reuseFailAlloc_679_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
v___y_653_ = v___x_678_;
goto v___jp_652_;
}
}
}
v___jp_625_:
{
lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
lean_inc_ref(v___y_626_);
v___x_636_ = l_Array_append___redArg(v___y_626_, v___y_635_);
lean_dec_ref(v___y_635_);
lean_inc_n(v___y_633_, 2);
lean_inc_n(v___y_634_, 9);
v___x_637_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_637_, 0, v___y_634_);
lean_ctor_set(v___x_637_, 1, v___y_633_);
lean_ctor_set(v___x_637_, 2, v___x_636_);
lean_inc(v___y_629_);
v___x_638_ = l_Lean_Syntax_node3(v___y_634_, v___y_629_, v___y_632_, v___x_637_, v___x_621_);
v___x_639_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__3));
v___x_640_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_640_, 0, v___y_634_);
lean_ctor_set(v___x_640_, 1, v___x_639_);
v___x_641_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__0));
v___x_642_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__5));
v___x_643_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_643_, 0, v___y_634_);
lean_ctor_set(v___x_643_, 1, v___x_642_);
v___x_644_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__6));
v___x_645_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_645_, 0, v___y_634_);
lean_ctor_set(v___x_645_, 1, v___x_644_);
v___x_646_ = l_Lean_Syntax_node3(v___y_634_, v___x_641_, v___x_643_, v___x_623_, v___x_645_);
v___x_647_ = l_Lean_Syntax_node3(v___y_634_, v___y_633_, v___x_638_, v___x_640_, v___x_646_);
lean_inc(v___y_628_);
v___x_648_ = l_Lean_Syntax_node1(v___y_634_, v___y_628_, v___x_647_);
v___x_649_ = l_Lean_Syntax_node1(v___y_634_, v___x_624_, v___x_648_);
lean_inc(v___y_630_);
v___x_650_ = l_Lean_Syntax_node3(v___y_634_, v___y_630_, v___y_631_, v___y_627_, v___x_649_);
v___x_651_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_651_, 0, v___x_650_);
lean_ctor_set(v___x_651_, 1, v_a_613_);
return v___x_651_;
}
v___jp_652_:
{
lean_object* v_ref_654_; uint8_t v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
v_ref_654_ = lean_ctor_get(v_a_612_, 5);
v___x_655_ = 0;
v___x_656_ = l_Lean_SourceInfo_fromRef(v_ref_654_, v___x_655_);
v___x_657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__12));
v___x_658_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__2));
lean_inc_n(v___x_656_, 3);
v___x_659_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_659_, 0, v___x_656_);
lean_ctor_set(v___x_659_, 1, v___x_657_);
v___x_660_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_661_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_661_, 0, v___x_656_);
lean_ctor_set(v___x_661_, 1, v___x_660_);
v___x_662_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3));
v___x_663_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_664_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__4));
v___x_665_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__5));
v___x_666_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_666_, 0, v___x_656_);
lean_ctor_set(v___x_666_, 1, v___x_664_);
v___x_667_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16, &lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16);
if (lean_obj_tag(v___y_653_) == 1)
{
lean_object* v_val_668_; lean_object* v___x_669_; 
v_val_668_ = lean_ctor_get(v___y_653_, 0);
lean_inc(v_val_668_);
lean_dec_ref_known(v___y_653_, 1);
v___x_669_ = l_Array_mkArray1___redArg(v_val_668_);
v___y_626_ = v___x_667_;
v___y_627_ = v___x_661_;
v___y_628_ = v___x_662_;
v___y_629_ = v___x_665_;
v___y_630_ = v___x_658_;
v___y_631_ = v___x_659_;
v___y_632_ = v___x_666_;
v___y_633_ = v___x_663_;
v___y_634_ = v___x_656_;
v___y_635_ = v___x_669_;
goto v___jp_625_;
}
else
{
lean_object* v___x_670_; 
lean_dec(v___y_653_);
v___x_670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_626_ = v___x_667_;
v___y_627_ = v___x_661_;
v___y_628_ = v___x_662_;
v___y_629_ = v___x_665_;
v___y_630_ = v___x_658_;
v___y_631_ = v___x_659_;
v___y_632_ = v___x_666_;
v___y_633_ = v___x_663_;
v___y_634_ = v___x_656_;
v___y_635_ = v___x_670_;
goto v___jp_625_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___boxed(lean_object* v_x_681_, lean_object* v_a_682_, lean_object* v_a_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1(v_x_681_, v_a_682_, v_a_683_);
lean_dec_ref(v_a_682_);
return v_res_684_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_715_ = lean_box(0);
v___x_716_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_717_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_717_, 0, v___x_716_);
lean_ctor_set(v___x_717_, 1, v___x_715_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg(){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_719_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0);
v___x_720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_720_, 0, v___x_719_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___boxed(lean_object* v___y_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg();
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2(lean_object* v_00_u03b1_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg();
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___boxed(lean_object* v_00_u03b1_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v_res_744_; 
v_res_744_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2(v_00_u03b1_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_, v___y_742_);
lean_dec(v___y_742_);
lean_dec_ref(v___y_741_);
lean_dec(v___y_740_);
lean_dec_ref(v___y_739_);
lean_dec(v___y_738_);
lean_dec_ref(v___y_737_);
lean_dec(v___y_736_);
lean_dec_ref(v___y_735_);
return v_res_744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5_spec__6___redArg(lean_object* v_x_745_, lean_object* v_x_746_, lean_object* v_x_747_, lean_object* v_x_748_){
_start:
{
lean_object* v_ks_749_; lean_object* v_vs_750_; lean_object* v___x_752_; uint8_t v_isShared_753_; uint8_t v_isSharedCheck_774_; 
v_ks_749_ = lean_ctor_get(v_x_745_, 0);
v_vs_750_ = lean_ctor_get(v_x_745_, 1);
v_isSharedCheck_774_ = !lean_is_exclusive(v_x_745_);
if (v_isSharedCheck_774_ == 0)
{
v___x_752_ = v_x_745_;
v_isShared_753_ = v_isSharedCheck_774_;
goto v_resetjp_751_;
}
else
{
lean_inc(v_vs_750_);
lean_inc(v_ks_749_);
lean_dec(v_x_745_);
v___x_752_ = lean_box(0);
v_isShared_753_ = v_isSharedCheck_774_;
goto v_resetjp_751_;
}
v_resetjp_751_:
{
lean_object* v___x_754_; uint8_t v___x_755_; 
v___x_754_ = lean_array_get_size(v_ks_749_);
v___x_755_ = lean_nat_dec_lt(v_x_746_, v___x_754_);
if (v___x_755_ == 0)
{
lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_759_; 
lean_dec(v_x_746_);
v___x_756_ = lean_array_push(v_ks_749_, v_x_747_);
v___x_757_ = lean_array_push(v_vs_750_, v_x_748_);
if (v_isShared_753_ == 0)
{
lean_ctor_set(v___x_752_, 1, v___x_757_);
lean_ctor_set(v___x_752_, 0, v___x_756_);
v___x_759_ = v___x_752_;
goto v_reusejp_758_;
}
else
{
lean_object* v_reuseFailAlloc_760_; 
v_reuseFailAlloc_760_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_760_, 0, v___x_756_);
lean_ctor_set(v_reuseFailAlloc_760_, 1, v___x_757_);
v___x_759_ = v_reuseFailAlloc_760_;
goto v_reusejp_758_;
}
v_reusejp_758_:
{
return v___x_759_;
}
}
else
{
lean_object* v_k_x27_761_; uint8_t v___x_762_; 
v_k_x27_761_ = lean_array_fget_borrowed(v_ks_749_, v_x_746_);
v___x_762_ = l_Lean_instBEqMVarId_beq(v_x_747_, v_k_x27_761_);
if (v___x_762_ == 0)
{
lean_object* v___x_764_; 
if (v_isShared_753_ == 0)
{
v___x_764_ = v___x_752_;
goto v_reusejp_763_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v_ks_749_);
lean_ctor_set(v_reuseFailAlloc_768_, 1, v_vs_750_);
v___x_764_ = v_reuseFailAlloc_768_;
goto v_reusejp_763_;
}
v_reusejp_763_:
{
lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_765_ = lean_unsigned_to_nat(1u);
v___x_766_ = lean_nat_add(v_x_746_, v___x_765_);
lean_dec(v_x_746_);
v_x_745_ = v___x_764_;
v_x_746_ = v___x_766_;
goto _start;
}
}
else
{
lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_772_; 
v___x_769_ = lean_array_fset(v_ks_749_, v_x_746_, v_x_747_);
v___x_770_ = lean_array_fset(v_vs_750_, v_x_746_, v_x_748_);
lean_dec(v_x_746_);
if (v_isShared_753_ == 0)
{
lean_ctor_set(v___x_752_, 1, v___x_770_);
lean_ctor_set(v___x_752_, 0, v___x_769_);
v___x_772_ = v___x_752_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v___x_769_);
lean_ctor_set(v_reuseFailAlloc_773_, 1, v___x_770_);
v___x_772_ = v_reuseFailAlloc_773_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
return v___x_772_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5___redArg(lean_object* v_n_775_, lean_object* v_k_776_, lean_object* v_v_777_){
_start:
{
lean_object* v___x_778_; lean_object* v___x_779_; 
v___x_778_ = lean_unsigned_to_nat(0u);
v___x_779_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5_spec__6___redArg(v_n_775_, v___x_778_, v_k_776_, v_v_777_);
return v___x_779_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_780_; 
v___x_780_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(lean_object* v_x_781_, size_t v_x_782_, size_t v_x_783_, lean_object* v_x_784_, lean_object* v_x_785_){
_start:
{
if (lean_obj_tag(v_x_781_) == 0)
{
lean_object* v_es_786_; size_t v___x_787_; size_t v___x_788_; lean_object* v_j_789_; lean_object* v___x_790_; uint8_t v___x_791_; 
v_es_786_ = lean_ctor_get(v_x_781_, 0);
v___x_787_ = ((size_t)31ULL);
v___x_788_ = lean_usize_land(v_x_782_, v___x_787_);
v_j_789_ = lean_usize_to_nat(v___x_788_);
v___x_790_ = lean_array_get_size(v_es_786_);
v___x_791_ = lean_nat_dec_lt(v_j_789_, v___x_790_);
if (v___x_791_ == 0)
{
lean_dec(v_j_789_);
lean_dec(v_x_785_);
lean_dec(v_x_784_);
return v_x_781_;
}
else
{
lean_object* v___x_793_; uint8_t v_isShared_794_; uint8_t v_isSharedCheck_830_; 
lean_inc_ref(v_es_786_);
v_isSharedCheck_830_ = !lean_is_exclusive(v_x_781_);
if (v_isSharedCheck_830_ == 0)
{
lean_object* v_unused_831_; 
v_unused_831_ = lean_ctor_get(v_x_781_, 0);
lean_dec(v_unused_831_);
v___x_793_ = v_x_781_;
v_isShared_794_ = v_isSharedCheck_830_;
goto v_resetjp_792_;
}
else
{
lean_dec(v_x_781_);
v___x_793_ = lean_box(0);
v_isShared_794_ = v_isSharedCheck_830_;
goto v_resetjp_792_;
}
v_resetjp_792_:
{
lean_object* v_v_795_; lean_object* v___x_796_; lean_object* v_xs_x27_797_; lean_object* v___y_799_; 
v_v_795_ = lean_array_fget(v_es_786_, v_j_789_);
v___x_796_ = lean_box(0);
v_xs_x27_797_ = lean_array_fset(v_es_786_, v_j_789_, v___x_796_);
switch(lean_obj_tag(v_v_795_))
{
case 0:
{
lean_object* v_key_804_; lean_object* v_val_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_815_; 
v_key_804_ = lean_ctor_get(v_v_795_, 0);
v_val_805_ = lean_ctor_get(v_v_795_, 1);
v_isSharedCheck_815_ = !lean_is_exclusive(v_v_795_);
if (v_isSharedCheck_815_ == 0)
{
v___x_807_ = v_v_795_;
v_isShared_808_ = v_isSharedCheck_815_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_val_805_);
lean_inc(v_key_804_);
lean_dec(v_v_795_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_815_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
uint8_t v___x_809_; 
v___x_809_ = l_Lean_instBEqMVarId_beq(v_x_784_, v_key_804_);
if (v___x_809_ == 0)
{
lean_object* v___x_810_; lean_object* v___x_811_; 
lean_del_object(v___x_807_);
v___x_810_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_804_, v_val_805_, v_x_784_, v_x_785_);
v___x_811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_811_, 0, v___x_810_);
v___y_799_ = v___x_811_;
goto v___jp_798_;
}
else
{
lean_object* v___x_813_; 
lean_dec(v_val_805_);
lean_dec(v_key_804_);
if (v_isShared_808_ == 0)
{
lean_ctor_set(v___x_807_, 1, v_x_785_);
lean_ctor_set(v___x_807_, 0, v_x_784_);
v___x_813_ = v___x_807_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v_x_784_);
lean_ctor_set(v_reuseFailAlloc_814_, 1, v_x_785_);
v___x_813_ = v_reuseFailAlloc_814_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
v___y_799_ = v___x_813_;
goto v___jp_798_;
}
}
}
}
case 1:
{
lean_object* v_node_816_; lean_object* v___x_818_; uint8_t v_isShared_819_; uint8_t v_isSharedCheck_828_; 
v_node_816_ = lean_ctor_get(v_v_795_, 0);
v_isSharedCheck_828_ = !lean_is_exclusive(v_v_795_);
if (v_isSharedCheck_828_ == 0)
{
v___x_818_ = v_v_795_;
v_isShared_819_ = v_isSharedCheck_828_;
goto v_resetjp_817_;
}
else
{
lean_inc(v_node_816_);
lean_dec(v_v_795_);
v___x_818_ = lean_box(0);
v_isShared_819_ = v_isSharedCheck_828_;
goto v_resetjp_817_;
}
v_resetjp_817_:
{
size_t v___x_820_; size_t v___x_821_; size_t v___x_822_; size_t v___x_823_; lean_object* v___x_824_; lean_object* v___x_826_; 
v___x_820_ = ((size_t)5ULL);
v___x_821_ = lean_usize_shift_right(v_x_782_, v___x_820_);
v___x_822_ = ((size_t)1ULL);
v___x_823_ = lean_usize_add(v_x_783_, v___x_822_);
v___x_824_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(v_node_816_, v___x_821_, v___x_823_, v_x_784_, v_x_785_);
if (v_isShared_819_ == 0)
{
lean_ctor_set(v___x_818_, 0, v___x_824_);
v___x_826_ = v___x_818_;
goto v_reusejp_825_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v___x_824_);
v___x_826_ = v_reuseFailAlloc_827_;
goto v_reusejp_825_;
}
v_reusejp_825_:
{
v___y_799_ = v___x_826_;
goto v___jp_798_;
}
}
}
default: 
{
lean_object* v___x_829_; 
v___x_829_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_829_, 0, v_x_784_);
lean_ctor_set(v___x_829_, 1, v_x_785_);
v___y_799_ = v___x_829_;
goto v___jp_798_;
}
}
v___jp_798_:
{
lean_object* v___x_800_; lean_object* v___x_802_; 
v___x_800_ = lean_array_fset(v_xs_x27_797_, v_j_789_, v___y_799_);
lean_dec(v_j_789_);
if (v_isShared_794_ == 0)
{
lean_ctor_set(v___x_793_, 0, v___x_800_);
v___x_802_ = v___x_793_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_803_; 
v_reuseFailAlloc_803_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_803_, 0, v___x_800_);
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
}
else
{
lean_object* v_ks_832_; lean_object* v_vs_833_; lean_object* v___x_835_; uint8_t v_isShared_836_; uint8_t v_isSharedCheck_853_; 
v_ks_832_ = lean_ctor_get(v_x_781_, 0);
v_vs_833_ = lean_ctor_get(v_x_781_, 1);
v_isSharedCheck_853_ = !lean_is_exclusive(v_x_781_);
if (v_isSharedCheck_853_ == 0)
{
v___x_835_ = v_x_781_;
v_isShared_836_ = v_isSharedCheck_853_;
goto v_resetjp_834_;
}
else
{
lean_inc(v_vs_833_);
lean_inc(v_ks_832_);
lean_dec(v_x_781_);
v___x_835_ = lean_box(0);
v_isShared_836_ = v_isSharedCheck_853_;
goto v_resetjp_834_;
}
v_resetjp_834_:
{
lean_object* v___x_838_; 
if (v_isShared_836_ == 0)
{
v___x_838_ = v___x_835_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v_ks_832_);
lean_ctor_set(v_reuseFailAlloc_852_, 1, v_vs_833_);
v___x_838_ = v_reuseFailAlloc_852_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
lean_object* v_newNode_839_; uint8_t v___y_841_; size_t v___x_847_; uint8_t v___x_848_; 
v_newNode_839_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5___redArg(v___x_838_, v_x_784_, v_x_785_);
v___x_847_ = ((size_t)7ULL);
v___x_848_ = lean_usize_dec_le(v___x_847_, v_x_783_);
if (v___x_848_ == 0)
{
lean_object* v___x_849_; lean_object* v___x_850_; uint8_t v___x_851_; 
v___x_849_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_839_);
v___x_850_ = lean_unsigned_to_nat(4u);
v___x_851_ = lean_nat_dec_lt(v___x_849_, v___x_850_);
lean_dec(v___x_849_);
v___y_841_ = v___x_851_;
goto v___jp_840_;
}
else
{
v___y_841_ = v___x_848_;
goto v___jp_840_;
}
v___jp_840_:
{
if (v___y_841_ == 0)
{
lean_object* v_ks_842_; lean_object* v_vs_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; 
v_ks_842_ = lean_ctor_get(v_newNode_839_, 0);
lean_inc_ref(v_ks_842_);
v_vs_843_ = lean_ctor_get(v_newNode_839_, 1);
lean_inc_ref(v_vs_843_);
lean_dec_ref(v_newNode_839_);
v___x_844_ = lean_unsigned_to_nat(0u);
v___x_845_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___closed__0);
v___x_846_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg(v_x_783_, v_ks_842_, v_vs_843_, v___x_844_, v___x_845_);
lean_dec_ref(v_vs_843_);
lean_dec_ref(v_ks_842_);
return v___x_846_;
}
else
{
return v_newNode_839_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg(size_t v_depth_854_, lean_object* v_keys_855_, lean_object* v_vals_856_, lean_object* v_i_857_, lean_object* v_entries_858_){
_start:
{
lean_object* v___x_859_; uint8_t v___x_860_; 
v___x_859_ = lean_array_get_size(v_keys_855_);
v___x_860_ = lean_nat_dec_lt(v_i_857_, v___x_859_);
if (v___x_860_ == 0)
{
lean_dec(v_i_857_);
return v_entries_858_;
}
else
{
lean_object* v_k_861_; lean_object* v_v_862_; uint64_t v___x_863_; size_t v_h_864_; size_t v___x_865_; lean_object* v___x_866_; size_t v___x_867_; size_t v___x_868_; size_t v___x_869_; size_t v_h_870_; lean_object* v___x_871_; lean_object* v___x_872_; 
v_k_861_ = lean_array_fget_borrowed(v_keys_855_, v_i_857_);
v_v_862_ = lean_array_fget_borrowed(v_vals_856_, v_i_857_);
v___x_863_ = l_Lean_instHashableMVarId_hash(v_k_861_);
v_h_864_ = lean_uint64_to_usize(v___x_863_);
v___x_865_ = ((size_t)5ULL);
v___x_866_ = lean_unsigned_to_nat(1u);
v___x_867_ = ((size_t)1ULL);
v___x_868_ = lean_usize_sub(v_depth_854_, v___x_867_);
v___x_869_ = lean_usize_mul(v___x_865_, v___x_868_);
v_h_870_ = lean_usize_shift_right(v_h_864_, v___x_869_);
v___x_871_ = lean_nat_add(v_i_857_, v___x_866_);
lean_dec(v_i_857_);
lean_inc(v_v_862_);
lean_inc(v_k_861_);
v___x_872_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(v_entries_858_, v_h_870_, v_depth_854_, v_k_861_, v_v_862_);
v_i_857_ = v___x_871_;
v_entries_858_ = v___x_872_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg___boxed(lean_object* v_depth_874_, lean_object* v_keys_875_, lean_object* v_vals_876_, lean_object* v_i_877_, lean_object* v_entries_878_){
_start:
{
size_t v_depth_boxed_879_; lean_object* v_res_880_; 
v_depth_boxed_879_ = lean_unbox_usize(v_depth_874_);
lean_dec(v_depth_874_);
v_res_880_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg(v_depth_boxed_879_, v_keys_875_, v_vals_876_, v_i_877_, v_entries_878_);
lean_dec_ref(v_vals_876_);
lean_dec_ref(v_keys_875_);
return v_res_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_x_881_, lean_object* v_x_882_, lean_object* v_x_883_, lean_object* v_x_884_, lean_object* v_x_885_){
_start:
{
size_t v_x_5748__boxed_886_; size_t v_x_5749__boxed_887_; lean_object* v_res_888_; 
v_x_5748__boxed_886_ = lean_unbox_usize(v_x_882_);
lean_dec(v_x_882_);
v_x_5749__boxed_887_ = lean_unbox_usize(v_x_883_);
lean_dec(v_x_883_);
v_res_888_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(v_x_881_, v_x_5748__boxed_886_, v_x_5749__boxed_887_, v_x_884_, v_x_885_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0___redArg(lean_object* v_x_889_, lean_object* v_x_890_, lean_object* v_x_891_){
_start:
{
uint64_t v___x_892_; size_t v___x_893_; size_t v___x_894_; lean_object* v___x_895_; 
v___x_892_ = l_Lean_instHashableMVarId_hash(v_x_890_);
v___x_893_ = lean_uint64_to_usize(v___x_892_);
v___x_894_ = ((size_t)1ULL);
v___x_895_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(v_x_889_, v___x_893_, v___x_894_, v_x_890_, v_x_891_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg(lean_object* v_mvarId_896_, lean_object* v_val_897_, lean_object* v___y_898_){
_start:
{
lean_object* v___x_900_; lean_object* v_mctx_901_; lean_object* v_cache_902_; lean_object* v_zetaDeltaFVarIds_903_; lean_object* v_postponed_904_; lean_object* v_diag_905_; lean_object* v___x_907_; uint8_t v_isShared_908_; uint8_t v_isSharedCheck_933_; 
v___x_900_ = lean_st_ref_take(v___y_898_);
v_mctx_901_ = lean_ctor_get(v___x_900_, 0);
v_cache_902_ = lean_ctor_get(v___x_900_, 1);
v_zetaDeltaFVarIds_903_ = lean_ctor_get(v___x_900_, 2);
v_postponed_904_ = lean_ctor_get(v___x_900_, 3);
v_diag_905_ = lean_ctor_get(v___x_900_, 4);
v_isSharedCheck_933_ = !lean_is_exclusive(v___x_900_);
if (v_isSharedCheck_933_ == 0)
{
v___x_907_ = v___x_900_;
v_isShared_908_ = v_isSharedCheck_933_;
goto v_resetjp_906_;
}
else
{
lean_inc(v_diag_905_);
lean_inc(v_postponed_904_);
lean_inc(v_zetaDeltaFVarIds_903_);
lean_inc(v_cache_902_);
lean_inc(v_mctx_901_);
lean_dec(v___x_900_);
v___x_907_ = lean_box(0);
v_isShared_908_ = v_isSharedCheck_933_;
goto v_resetjp_906_;
}
v_resetjp_906_:
{
lean_object* v_depth_909_; lean_object* v_levelAssignDepth_910_; lean_object* v_lmvarCounter_911_; lean_object* v_mvarCounter_912_; lean_object* v_lDecls_913_; lean_object* v_decls_914_; lean_object* v_userNames_915_; lean_object* v_lAssignment_916_; lean_object* v_eAssignment_917_; lean_object* v_dAssignment_918_; lean_object* v___x_920_; uint8_t v_isShared_921_; uint8_t v_isSharedCheck_932_; 
v_depth_909_ = lean_ctor_get(v_mctx_901_, 0);
v_levelAssignDepth_910_ = lean_ctor_get(v_mctx_901_, 1);
v_lmvarCounter_911_ = lean_ctor_get(v_mctx_901_, 2);
v_mvarCounter_912_ = lean_ctor_get(v_mctx_901_, 3);
v_lDecls_913_ = lean_ctor_get(v_mctx_901_, 4);
v_decls_914_ = lean_ctor_get(v_mctx_901_, 5);
v_userNames_915_ = lean_ctor_get(v_mctx_901_, 6);
v_lAssignment_916_ = lean_ctor_get(v_mctx_901_, 7);
v_eAssignment_917_ = lean_ctor_get(v_mctx_901_, 8);
v_dAssignment_918_ = lean_ctor_get(v_mctx_901_, 9);
v_isSharedCheck_932_ = !lean_is_exclusive(v_mctx_901_);
if (v_isSharedCheck_932_ == 0)
{
v___x_920_ = v_mctx_901_;
v_isShared_921_ = v_isSharedCheck_932_;
goto v_resetjp_919_;
}
else
{
lean_inc(v_dAssignment_918_);
lean_inc(v_eAssignment_917_);
lean_inc(v_lAssignment_916_);
lean_inc(v_userNames_915_);
lean_inc(v_decls_914_);
lean_inc(v_lDecls_913_);
lean_inc(v_mvarCounter_912_);
lean_inc(v_lmvarCounter_911_);
lean_inc(v_levelAssignDepth_910_);
lean_inc(v_depth_909_);
lean_dec(v_mctx_901_);
v___x_920_ = lean_box(0);
v_isShared_921_ = v_isSharedCheck_932_;
goto v_resetjp_919_;
}
v_resetjp_919_:
{
lean_object* v___x_922_; lean_object* v___x_924_; 
v___x_922_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0___redArg(v_eAssignment_917_, v_mvarId_896_, v_val_897_);
if (v_isShared_921_ == 0)
{
lean_ctor_set(v___x_920_, 8, v___x_922_);
v___x_924_ = v___x_920_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_931_; 
v_reuseFailAlloc_931_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_931_, 0, v_depth_909_);
lean_ctor_set(v_reuseFailAlloc_931_, 1, v_levelAssignDepth_910_);
lean_ctor_set(v_reuseFailAlloc_931_, 2, v_lmvarCounter_911_);
lean_ctor_set(v_reuseFailAlloc_931_, 3, v_mvarCounter_912_);
lean_ctor_set(v_reuseFailAlloc_931_, 4, v_lDecls_913_);
lean_ctor_set(v_reuseFailAlloc_931_, 5, v_decls_914_);
lean_ctor_set(v_reuseFailAlloc_931_, 6, v_userNames_915_);
lean_ctor_set(v_reuseFailAlloc_931_, 7, v_lAssignment_916_);
lean_ctor_set(v_reuseFailAlloc_931_, 8, v___x_922_);
lean_ctor_set(v_reuseFailAlloc_931_, 9, v_dAssignment_918_);
v___x_924_ = v_reuseFailAlloc_931_;
goto v_reusejp_923_;
}
v_reusejp_923_:
{
lean_object* v___x_926_; 
if (v_isShared_908_ == 0)
{
lean_ctor_set(v___x_907_, 0, v___x_924_);
v___x_926_ = v___x_907_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v___x_924_);
lean_ctor_set(v_reuseFailAlloc_930_, 1, v_cache_902_);
lean_ctor_set(v_reuseFailAlloc_930_, 2, v_zetaDeltaFVarIds_903_);
lean_ctor_set(v_reuseFailAlloc_930_, 3, v_postponed_904_);
lean_ctor_set(v_reuseFailAlloc_930_, 4, v_diag_905_);
v___x_926_ = v_reuseFailAlloc_930_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; 
v___x_927_ = lean_st_ref_set(v___y_898_, v___x_926_);
v___x_928_ = lean_box(0);
v___x_929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_929_, 0, v___x_928_);
return v___x_929_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg___boxed(lean_object* v_mvarId_934_, lean_object* v_val_935_, lean_object* v___y_936_, lean_object* v___y_937_){
_start:
{
lean_object* v_res_938_; 
v_res_938_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg(v_mvarId_934_, v_val_935_, v___y_936_);
lean_dec(v___y_936_);
return v_res_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2(lean_object* v_msgData_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_){
_start:
{
lean_object* v___x_945_; lean_object* v_env_946_; lean_object* v___x_947_; lean_object* v_mctx_948_; lean_object* v_lctx_949_; lean_object* v_options_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_945_ = lean_st_ref_get(v___y_943_);
v_env_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc_ref(v_env_946_);
lean_dec(v___x_945_);
v___x_947_ = lean_st_ref_get(v___y_941_);
v_mctx_948_ = lean_ctor_get(v___x_947_, 0);
lean_inc_ref(v_mctx_948_);
lean_dec(v___x_947_);
v_lctx_949_ = lean_ctor_get(v___y_940_, 2);
v_options_950_ = lean_ctor_get(v___y_942_, 2);
lean_inc_ref(v_options_950_);
lean_inc_ref(v_lctx_949_);
v___x_951_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_951_, 0, v_env_946_);
lean_ctor_set(v___x_951_, 1, v_mctx_948_);
lean_ctor_set(v___x_951_, 2, v_lctx_949_);
lean_ctor_set(v___x_951_, 3, v_options_950_);
v___x_952_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_952_, 0, v___x_951_);
lean_ctor_set(v___x_952_, 1, v_msgData_939_);
v___x_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_953_, 0, v___x_952_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2___boxed(lean_object* v_msgData_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2(v_msgData_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg(lean_object* v_msg_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_){
_start:
{
lean_object* v_ref_967_; lean_object* v___x_968_; lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_977_; 
v_ref_967_ = lean_ctor_get(v___y_964_, 5);
v___x_968_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2(v_msg_961_, v___y_962_, v___y_963_, v___y_964_, v___y_965_);
v_a_969_ = lean_ctor_get(v___x_968_, 0);
v_isSharedCheck_977_ = !lean_is_exclusive(v___x_968_);
if (v_isSharedCheck_977_ == 0)
{
v___x_971_ = v___x_968_;
v_isShared_972_ = v_isSharedCheck_977_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_968_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_977_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_973_; lean_object* v___x_975_; 
lean_inc(v_ref_967_);
v___x_973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_973_, 0, v_ref_967_);
lean_ctor_set(v___x_973_, 1, v_a_969_);
if (v_isShared_972_ == 0)
{
lean_ctor_set_tag(v___x_971_, 1);
lean_ctor_set(v___x_971_, 0, v___x_973_);
v___x_975_ = v___x_971_;
goto v_reusejp_974_;
}
else
{
lean_object* v_reuseFailAlloc_976_; 
v_reuseFailAlloc_976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_976_, 0, v___x_973_);
v___x_975_ = v_reuseFailAlloc_976_;
goto v_reusejp_974_;
}
v_reusejp_974_:
{
return v___x_975_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg___boxed(lean_object* v_msg_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_){
_start:
{
lean_object* v_res_984_; 
v_res_984_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg(v_msg_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
return v_res_984_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__2(void){
_start:
{
lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_988_ = lean_box(0);
v___x_989_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__1));
v___x_990_ = l_Lean_mkConst(v___x_989_, v___x_988_);
return v___x_990_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__4(void){
_start:
{
lean_object* v___x_992_; lean_object* v___x_993_; 
v___x_992_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__3));
v___x_993_ = l_Lean_stringToMessageData(v___x_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv(lean_object* v_x_994_, lean_object* v_a_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_, lean_object* v_a_999_, lean_object* v_a_1000_, lean_object* v_a_1001_, lean_object* v_a_1002_){
_start:
{
lean_object* v_tac_1005_; lean_object* v___y_1006_; lean_object* v___y_1007_; lean_object* v___y_1008_; lean_object* v___y_1009_; lean_object* v___y_1010_; lean_object* v___y_1011_; lean_object* v___y_1012_; lean_object* v___y_1013_; lean_object* v___x_1105_; uint8_t v___x_1106_; 
v___x_1105_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_dischargeConv___closed__1));
lean_inc(v_x_994_);
v___x_1106_ = l_Lean_Syntax_isOfKind(v_x_994_, v___x_1105_);
if (v___x_1106_ == 0)
{
lean_object* v___x_1107_; 
lean_dec(v_x_994_);
v___x_1107_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg();
return v___x_1107_;
}
else
{
lean_object* v___x_1108_; lean_object* v___x_1109_; uint8_t v___x_1110_; 
v___x_1108_ = lean_unsigned_to_nat(1u);
v___x_1109_ = l_Lean_Syntax_getArg(v_x_994_, v___x_1108_);
lean_dec(v_x_994_);
v___x_1110_ = l_Lean_Syntax_isNone(v___x_1109_);
if (v___x_1110_ == 0)
{
lean_object* v___x_1111_; uint8_t v___x_1112_; 
v___x_1111_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_1109_);
v___x_1112_ = l_Lean_Syntax_matchesNull(v___x_1109_, v___x_1111_);
if (v___x_1112_ == 0)
{
lean_object* v___x_1113_; 
lean_dec(v___x_1109_);
v___x_1113_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg();
return v___x_1113_;
}
else
{
lean_object* v_tac_1114_; lean_object* v___x_1115_; 
v_tac_1114_ = l_Lean_Syntax_getArg(v___x_1109_, v___x_1108_);
lean_dec(v___x_1109_);
v___x_1115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1115_, 0, v_tac_1114_);
v_tac_1005_ = v___x_1115_;
v___y_1006_ = v_a_995_;
v___y_1007_ = v_a_996_;
v___y_1008_ = v_a_997_;
v___y_1009_ = v_a_998_;
v___y_1010_ = v_a_999_;
v___y_1011_ = v_a_1000_;
v___y_1012_ = v_a_1001_;
v___y_1013_ = v_a_1002_;
goto v___jp_1004_;
}
}
else
{
lean_object* v___x_1116_; 
lean_dec(v___x_1109_);
v___x_1116_ = lean_box(0);
v_tac_1005_ = v___x_1116_;
v___y_1006_ = v_a_995_;
v___y_1007_ = v_a_996_;
v___y_1008_ = v_a_997_;
v___y_1009_ = v_a_998_;
v___y_1010_ = v_a_999_;
v___y_1011_ = v_a_1000_;
v___y_1012_ = v_a_1001_;
v___y_1013_ = v_a_1002_;
goto v___jp_1004_;
}
}
v___jp_1004_:
{
lean_object* v___x_1014_; 
v___x_1014_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_1007_);
if (lean_obj_tag(v___x_1014_) == 0)
{
lean_object* v_a_1015_; 
v_a_1015_ = lean_ctor_get(v___x_1014_, 0);
lean_inc(v_a_1015_);
lean_dec_ref_known(v___x_1014_, 1);
if (lean_obj_tag(v_a_1015_) == 1)
{
lean_object* v_head_1016_; lean_object* v_tail_1017_; lean_object* v___x_1019_; uint8_t v_isShared_1020_; uint8_t v_isSharedCheck_1095_; 
v_head_1016_ = lean_ctor_get(v_a_1015_, 0);
v_tail_1017_ = lean_ctor_get(v_a_1015_, 1);
v_isSharedCheck_1095_ = !lean_is_exclusive(v_a_1015_);
if (v_isSharedCheck_1095_ == 0)
{
v___x_1019_ = v_a_1015_;
v_isShared_1020_ = v_isSharedCheck_1095_;
goto v_resetjp_1018_;
}
else
{
lean_inc(v_tail_1017_);
lean_inc(v_head_1016_);
lean_dec(v_a_1015_);
v___x_1019_ = lean_box(0);
v_isShared_1020_ = v_isSharedCheck_1095_;
goto v_resetjp_1018_;
}
v_resetjp_1018_:
{
lean_object* v___x_1021_; 
lean_inc(v_head_1016_);
v___x_1021_ = l_Lean_Elab_Tactic_Conv_getLhsRhsCore(v_head_1016_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1021_) == 0)
{
lean_object* v_a_1022_; lean_object* v_fst_1023_; lean_object* v_snd_1024_; lean_object* v___x_1025_; 
v_a_1022_ = lean_ctor_get(v___x_1021_, 0);
lean_inc(v_a_1022_);
lean_dec_ref_known(v___x_1021_, 1);
v_fst_1023_ = lean_ctor_get(v_a_1022_, 0);
lean_inc_n(v_fst_1023_, 2);
v_snd_1024_ = lean_ctor_get(v_a_1022_, 1);
lean_inc(v_snd_1024_);
lean_dec(v_a_1022_);
v___x_1025_ = l_Lean_Meta_isProp(v_fst_1023_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1025_) == 0)
{
lean_object* v_a_1026_; uint8_t v___x_1027_; 
v_a_1026_ = lean_ctor_get(v___x_1025_, 0);
lean_inc(v_a_1026_);
lean_dec_ref_known(v___x_1025_, 1);
v___x_1027_ = lean_unbox(v_a_1026_);
lean_dec(v_a_1026_);
if (v___x_1027_ == 1)
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1033_; uint8_t v_isShared_1034_; uint8_t v_isSharedCheck_1075_; 
v___x_1028_ = l_Lean_Expr_mvarId_x21(v_snd_1024_);
lean_dec(v_snd_1024_);
v___x_1029_ = lean_box(0);
v___x_1030_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__2, &lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__2);
v___x_1031_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg(v___x_1028_, v___x_1030_, v___y_1011_);
v_isSharedCheck_1075_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1075_ == 0)
{
lean_object* v_unused_1076_; 
v_unused_1076_ = lean_ctor_get(v___x_1031_, 0);
lean_dec(v_unused_1076_);
v___x_1033_ = v___x_1031_;
v_isShared_1034_ = v_isSharedCheck_1075_;
goto v_resetjp_1032_;
}
else
{
lean_dec(v___x_1031_);
v___x_1033_ = lean_box(0);
v_isShared_1034_ = v_isSharedCheck_1075_;
goto v_resetjp_1032_;
}
v_resetjp_1032_:
{
lean_object* v___x_1036_; 
if (v_isShared_1034_ == 0)
{
lean_ctor_set_tag(v___x_1033_, 1);
lean_ctor_set(v___x_1033_, 0, v_fst_1023_);
v___x_1036_ = v___x_1033_;
goto v_reusejp_1035_;
}
else
{
lean_object* v_reuseFailAlloc_1074_; 
v_reuseFailAlloc_1074_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1074_, 0, v_fst_1023_);
v___x_1036_ = v_reuseFailAlloc_1074_;
goto v_reusejp_1035_;
}
v_reusejp_1035_:
{
uint8_t v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; 
v___x_1037_ = 0;
v___x_1038_ = lean_box(0);
v___x_1039_ = l_Lean_Meta_mkFreshExprMVar(v___x_1036_, v___x_1037_, v___x_1038_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1039_) == 0)
{
lean_object* v_a_1040_; lean_object* v___x_1041_; 
v_a_1040_ = lean_ctor_get(v___x_1039_, 0);
lean_inc_n(v_a_1040_, 2);
lean_dec_ref_known(v___x_1039_, 1);
v___x_1041_ = l_Lean_Meta_mkEqTrue(v_a_1040_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1041_) == 0)
{
lean_object* v_a_1042_; lean_object* v___x_1043_; 
v_a_1042_ = lean_ctor_get(v___x_1041_, 0);
lean_inc(v_a_1042_);
lean_dec_ref_known(v___x_1041_, 1);
v___x_1043_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg(v_head_1016_, v_a_1042_, v___y_1011_);
lean_dec_ref(v___x_1043_);
if (lean_obj_tag(v_tac_1005_) == 1)
{
lean_object* v_val_1044_; lean_object* v___x_1045_; lean_object* v___x_1047_; 
v_val_1044_ = lean_ctor_get(v_tac_1005_, 0);
lean_inc(v_val_1044_);
lean_dec_ref_known(v_tac_1005_, 1);
v___x_1045_ = l_Lean_Expr_mvarId_x21(v_a_1040_);
lean_dec(v_a_1040_);
if (v_isShared_1020_ == 0)
{
lean_ctor_set(v___x_1019_, 1, v___x_1029_);
lean_ctor_set(v___x_1019_, 0, v___x_1045_);
v___x_1047_ = v___x_1019_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v___x_1045_);
lean_ctor_set(v_reuseFailAlloc_1052_, 1, v___x_1029_);
v___x_1047_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
lean_object* v___x_1048_; 
v___x_1048_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_1047_, v___y_1007_);
if (lean_obj_tag(v___x_1048_) == 0)
{
lean_object* v___x_1049_; 
lean_dec_ref_known(v___x_1048_, 1);
v___x_1049_ = l_Lean_Elab_Tactic_evalTactic(v_val_1044_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1049_) == 0)
{
lean_object* v___x_1050_; 
lean_dec_ref_known(v___x_1049_, 1);
v___x_1050_ = l_Lean_Elab_Tactic_done(v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
if (lean_obj_tag(v___x_1050_) == 0)
{
lean_object* v___x_1051_; 
lean_dec_ref_known(v___x_1050_, 1);
v___x_1051_ = l_Lean_Elab_Tactic_setGoals___redArg(v_tail_1017_, v___y_1007_);
return v___x_1051_;
}
else
{
lean_dec(v_tail_1017_);
return v___x_1050_;
}
}
else
{
lean_dec(v_tail_1017_);
return v___x_1049_;
}
}
else
{
lean_dec(v_val_1044_);
lean_dec(v_tail_1017_);
return v___x_1048_;
}
}
}
else
{
lean_object* v___x_1053_; lean_object* v___x_1055_; 
lean_dec(v_tac_1005_);
v___x_1053_ = l_Lean_Expr_mvarId_x21(v_a_1040_);
lean_dec(v_a_1040_);
if (v_isShared_1020_ == 0)
{
lean_ctor_set(v___x_1019_, 0, v___x_1053_);
v___x_1055_ = v___x_1019_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v___x_1053_);
lean_ctor_set(v_reuseFailAlloc_1057_, 1, v_tail_1017_);
v___x_1055_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
lean_object* v___x_1056_; 
v___x_1056_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_1055_, v___y_1007_);
return v___x_1056_;
}
}
}
else
{
lean_object* v_a_1058_; lean_object* v___x_1060_; uint8_t v_isShared_1061_; uint8_t v_isSharedCheck_1065_; 
lean_dec(v_a_1040_);
lean_del_object(v___x_1019_);
lean_dec(v_tail_1017_);
lean_dec(v_head_1016_);
lean_dec(v_tac_1005_);
v_a_1058_ = lean_ctor_get(v___x_1041_, 0);
v_isSharedCheck_1065_ = !lean_is_exclusive(v___x_1041_);
if (v_isSharedCheck_1065_ == 0)
{
v___x_1060_ = v___x_1041_;
v_isShared_1061_ = v_isSharedCheck_1065_;
goto v_resetjp_1059_;
}
else
{
lean_inc(v_a_1058_);
lean_dec(v___x_1041_);
v___x_1060_ = lean_box(0);
v_isShared_1061_ = v_isSharedCheck_1065_;
goto v_resetjp_1059_;
}
v_resetjp_1059_:
{
lean_object* v___x_1063_; 
if (v_isShared_1061_ == 0)
{
v___x_1063_ = v___x_1060_;
goto v_reusejp_1062_;
}
else
{
lean_object* v_reuseFailAlloc_1064_; 
v_reuseFailAlloc_1064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1064_, 0, v_a_1058_);
v___x_1063_ = v_reuseFailAlloc_1064_;
goto v_reusejp_1062_;
}
v_reusejp_1062_:
{
return v___x_1063_;
}
}
}
}
else
{
lean_object* v_a_1066_; lean_object* v___x_1068_; uint8_t v_isShared_1069_; uint8_t v_isSharedCheck_1073_; 
lean_del_object(v___x_1019_);
lean_dec(v_tail_1017_);
lean_dec(v_head_1016_);
lean_dec(v_tac_1005_);
v_a_1066_ = lean_ctor_get(v___x_1039_, 0);
v_isSharedCheck_1073_ = !lean_is_exclusive(v___x_1039_);
if (v_isSharedCheck_1073_ == 0)
{
v___x_1068_ = v___x_1039_;
v_isShared_1069_ = v_isSharedCheck_1073_;
goto v_resetjp_1067_;
}
else
{
lean_inc(v_a_1066_);
lean_dec(v___x_1039_);
v___x_1068_ = lean_box(0);
v_isShared_1069_ = v_isSharedCheck_1073_;
goto v_resetjp_1067_;
}
v_resetjp_1067_:
{
lean_object* v___x_1071_; 
if (v_isShared_1069_ == 0)
{
v___x_1071_ = v___x_1068_;
goto v_reusejp_1070_;
}
else
{
lean_object* v_reuseFailAlloc_1072_; 
v_reuseFailAlloc_1072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1072_, 0, v_a_1066_);
v___x_1071_ = v_reuseFailAlloc_1072_;
goto v_reusejp_1070_;
}
v_reusejp_1070_:
{
return v___x_1071_;
}
}
}
}
}
}
else
{
lean_object* v___x_1077_; lean_object* v___x_1078_; 
lean_dec(v_snd_1024_);
lean_dec(v_fst_1023_);
lean_del_object(v___x_1019_);
lean_dec(v_tail_1017_);
lean_dec(v_head_1016_);
lean_dec(v_tac_1005_);
v___x_1077_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__4, &lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___closed__4);
v___x_1078_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg(v___x_1077_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
return v___x_1078_;
}
}
else
{
lean_object* v_a_1079_; lean_object* v___x_1081_; uint8_t v_isShared_1082_; uint8_t v_isSharedCheck_1086_; 
lean_dec(v_snd_1024_);
lean_dec(v_fst_1023_);
lean_del_object(v___x_1019_);
lean_dec(v_tail_1017_);
lean_dec(v_head_1016_);
lean_dec(v_tac_1005_);
v_a_1079_ = lean_ctor_get(v___x_1025_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1025_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1081_ = v___x_1025_;
v_isShared_1082_ = v_isSharedCheck_1086_;
goto v_resetjp_1080_;
}
else
{
lean_inc(v_a_1079_);
lean_dec(v___x_1025_);
v___x_1081_ = lean_box(0);
v_isShared_1082_ = v_isSharedCheck_1086_;
goto v_resetjp_1080_;
}
v_resetjp_1080_:
{
lean_object* v___x_1084_; 
if (v_isShared_1082_ == 0)
{
v___x_1084_ = v___x_1081_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v_a_1079_);
v___x_1084_ = v_reuseFailAlloc_1085_;
goto v_reusejp_1083_;
}
v_reusejp_1083_:
{
return v___x_1084_;
}
}
}
}
else
{
lean_object* v_a_1087_; lean_object* v___x_1089_; uint8_t v_isShared_1090_; uint8_t v_isSharedCheck_1094_; 
lean_del_object(v___x_1019_);
lean_dec(v_tail_1017_);
lean_dec(v_head_1016_);
lean_dec(v_tac_1005_);
v_a_1087_ = lean_ctor_get(v___x_1021_, 0);
v_isSharedCheck_1094_ = !lean_is_exclusive(v___x_1021_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1089_ = v___x_1021_;
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
else
{
lean_inc(v_a_1087_);
lean_dec(v___x_1021_);
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
else
{
lean_object* v___x_1096_; 
lean_dec(v_a_1015_);
lean_dec(v_tac_1005_);
v___x_1096_ = l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
return v___x_1096_;
}
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1104_; 
lean_dec(v_tac_1005_);
v_a_1097_ = lean_ctor_get(v___x_1014_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1014_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1099_ = v___x_1014_;
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1014_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv___boxed(lean_object* v_x_1117_, lean_object* v_a_1118_, lean_object* v_a_1119_, lean_object* v_a_1120_, lean_object* v_a_1121_, lean_object* v_a_1122_, lean_object* v_a_1123_, lean_object* v_a_1124_, lean_object* v_a_1125_, lean_object* v_a_1126_){
_start:
{
lean_object* v_res_1127_; 
v_res_1127_ = lp_mathlib_Mathlib_Tactic_Conv_elabDischargeConv(v_x_1117_, v_a_1118_, v_a_1119_, v_a_1120_, v_a_1121_, v_a_1122_, v_a_1123_, v_a_1124_, v_a_1125_);
lean_dec(v_a_1125_);
lean_dec_ref(v_a_1124_);
lean_dec(v_a_1123_);
lean_dec_ref(v_a_1122_);
lean_dec(v_a_1121_);
lean_dec_ref(v_a_1120_);
lean_dec(v_a_1119_);
lean_dec_ref(v_a_1118_);
return v_res_1127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0(lean_object* v_mvarId_1128_, lean_object* v_val_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_){
_start:
{
lean_object* v___x_1139_; 
v___x_1139_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___redArg(v_mvarId_1128_, v_val_1129_, v___y_1135_);
return v___x_1139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0___boxed(lean_object* v_mvarId_1140_, lean_object* v_val_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_){
_start:
{
lean_object* v_res_1151_; 
v_res_1151_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0(v_mvarId_1140_, v_val_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_, v___y_1148_, v___y_1149_);
lean_dec(v___y_1149_);
lean_dec_ref(v___y_1148_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
return v_res_1151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1(lean_object* v_00_u03b1_1152_, lean_object* v_msg_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_){
_start:
{
lean_object* v___x_1163_; 
v___x_1163_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___redArg(v_msg_1153_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_);
return v___x_1163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1___boxed(lean_object* v_00_u03b1_1164_, lean_object* v_msg_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_){
_start:
{
lean_object* v_res_1175_; 
v_res_1175_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1(v_00_u03b1_1164_, v_msg_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_, v___y_1171_, v___y_1172_, v___y_1173_);
lean_dec(v___y_1173_);
lean_dec_ref(v___y_1172_);
lean_dec(v___y_1171_);
lean_dec_ref(v___y_1170_);
lean_dec(v___y_1169_);
lean_dec_ref(v___y_1168_);
lean_dec(v___y_1167_);
lean_dec_ref(v___y_1166_);
return v_res_1175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0(lean_object* v_00_u03b2_1176_, lean_object* v_x_1177_, lean_object* v_x_1178_, lean_object* v_x_1179_){
_start:
{
lean_object* v___x_1180_; 
v___x_1180_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0___redArg(v_x_1177_, v_x_1178_, v_x_1179_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_1181_, lean_object* v_x_1182_, size_t v_x_1183_, size_t v_x_1184_, lean_object* v_x_1185_, lean_object* v_x_1186_){
_start:
{
lean_object* v___x_1187_; 
v___x_1187_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___redArg(v_x_1182_, v_x_1183_, v_x_1184_, v_x_1185_, v_x_1186_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_1188_, lean_object* v_x_1189_, lean_object* v_x_1190_, lean_object* v_x_1191_, lean_object* v_x_1192_, lean_object* v_x_1193_){
_start:
{
size_t v_x_6362__boxed_1194_; size_t v_x_6363__boxed_1195_; lean_object* v_res_1196_; 
v_x_6362__boxed_1194_ = lean_unbox_usize(v_x_1190_);
lean_dec(v_x_1190_);
v_x_6363__boxed_1195_ = lean_unbox_usize(v_x_1191_);
lean_dec(v_x_1191_);
v_res_1196_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2(v_00_u03b2_1188_, v_x_1189_, v_x_6362__boxed_1194_, v_x_6363__boxed_1195_, v_x_1192_, v_x_1193_);
return v_res_1196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5(lean_object* v_00_u03b2_1197_, lean_object* v_n_1198_, lean_object* v_k_1199_, lean_object* v_v_1200_){
_start:
{
lean_object* v___x_1201_; 
v___x_1201_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5___redArg(v_n_1198_, v_k_1199_, v_v_1200_);
return v___x_1201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6(lean_object* v_00_u03b2_1202_, size_t v_depth_1203_, lean_object* v_keys_1204_, lean_object* v_vals_1205_, lean_object* v_heq_1206_, lean_object* v_i_1207_, lean_object* v_entries_1208_){
_start:
{
lean_object* v___x_1209_; 
v___x_1209_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___redArg(v_depth_1203_, v_keys_1204_, v_vals_1205_, v_i_1207_, v_entries_1208_);
return v___x_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6___boxed(lean_object* v_00_u03b2_1210_, lean_object* v_depth_1211_, lean_object* v_keys_1212_, lean_object* v_vals_1213_, lean_object* v_heq_1214_, lean_object* v_i_1215_, lean_object* v_entries_1216_){
_start:
{
size_t v_depth_boxed_1217_; lean_object* v_res_1218_; 
v_depth_boxed_1217_ = lean_unbox_usize(v_depth_1211_);
lean_dec(v_depth_1211_);
v_res_1218_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__6(v_00_u03b2_1210_, v_depth_boxed_1217_, v_keys_1212_, v_vals_1213_, v_heq_1214_, v_i_1215_, v_entries_1216_);
lean_dec_ref(v_vals_1213_);
lean_dec_ref(v_keys_1212_);
return v_res_1218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5_spec__6(lean_object* v_00_u03b2_1219_, lean_object* v_x_1220_, lean_object* v_x_1221_, lean_object* v_x_1222_, lean_object* v_x_1223_){
_start:
{
lean_object* v___x_1224_; 
v___x_1224_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__0_spec__0_spec__2_spec__5_spec__6___redArg(v_x_1220_, v_x_1221_, v_x_1222_, v_x_1223_);
return v___x_1224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1(lean_object* v_x_1258_, lean_object* v_a_1259_, lean_object* v_a_1260_){
_start:
{
lean_object* v___x_1261_; uint8_t v___x_1262_; 
v___x_1261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convRefine___00__closed__1));
lean_inc(v_x_1258_);
v___x_1262_ = l_Lean_Syntax_isOfKind(v_x_1258_, v___x_1261_);
if (v___x_1262_ == 0)
{
lean_object* v___x_1263_; lean_object* v___x_1264_; 
lean_dec(v_x_1258_);
v___x_1263_ = lean_box(1);
v___x_1264_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1263_);
lean_ctor_set(v___x_1264_, 1, v_a_1260_);
return v___x_1264_;
}
else
{
lean_object* v_ref_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; uint8_t v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; 
v_ref_1265_ = lean_ctor_get(v_a_1259_, 5);
v___x_1266_ = lean_unsigned_to_nat(1u);
v___x_1267_ = l_Lean_Syntax_getArg(v_x_1258_, v___x_1266_);
lean_dec(v_x_1258_);
v___x_1268_ = 0;
v___x_1269_ = l_Lean_SourceInfo_fromRef(v_ref_1265_, v___x_1268_);
v___x_1270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__1));
v___x_1271_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__2));
lean_inc_n(v___x_1269_, 7);
v___x_1272_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1269_);
lean_ctor_set(v___x_1272_, 1, v___x_1271_);
v___x_1273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_1274_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1269_);
lean_ctor_set(v___x_1274_, 1, v___x_1273_);
v___x_1275_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4));
v___x_1276_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6));
v___x_1277_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_1278_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__3));
v___x_1279_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___closed__4));
v___x_1280_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1280_, 0, v___x_1269_);
lean_ctor_set(v___x_1280_, 1, v___x_1278_);
v___x_1281_ = l_Lean_Syntax_node2(v___x_1269_, v___x_1279_, v___x_1280_, v___x_1267_);
v___x_1282_ = l_Lean_Syntax_node1(v___x_1269_, v___x_1277_, v___x_1281_);
v___x_1283_ = l_Lean_Syntax_node1(v___x_1269_, v___x_1276_, v___x_1282_);
v___x_1284_ = l_Lean_Syntax_node1(v___x_1269_, v___x_1275_, v___x_1283_);
v___x_1285_ = l_Lean_Syntax_node3(v___x_1269_, v___x_1270_, v___x_1272_, v___x_1274_, v___x_1284_);
v___x_1286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1286_, 0, v___x_1285_);
lean_ctor_set(v___x_1286_, 1, v_a_1260_);
return v___x_1286_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1___boxed(lean_object* v_x_1287_, lean_object* v_a_1288_, lean_object* v_a_1289_){
_start:
{
lean_object* v_res_1290_; 
v_res_1290_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRefine____1(v_x_1287_, v_a_1288_, v_a_1289_);
lean_dec_ref(v_a_1288_);
return v_res_1290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg(){
_start:
{
lean_object* v___x_1323_; lean_object* v___x_1324_; 
v___x_1323_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__2___redArg___closed__0);
v___x_1324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1323_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg___boxed(lean_object* v___y_1325_){
_start:
{
lean_object* v_res_1326_; 
v_res_1326_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg();
return v_res_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0(lean_object* v_00_u03b1_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_){
_start:
{
lean_object* v___x_1331_; 
v___x_1331_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg();
return v___x_1331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___boxed(lean_object* v_00_u03b1_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_){
_start:
{
lean_object* v_res_1336_; 
v_res_1336_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0(v_00_u03b1_1332_, v___y_1333_, v___y_1334_);
lean_dec(v___y_1334_);
lean_dec_ref(v___y_1333_);
return v_res_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg(lean_object* v_e_1337_, lean_object* v___y_1338_){
_start:
{
uint8_t v___x_1340_; 
v___x_1340_ = l_Lean_Expr_hasMVar(v_e_1337_);
if (v___x_1340_ == 0)
{
lean_object* v___x_1341_; 
v___x_1341_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1341_, 0, v_e_1337_);
return v___x_1341_;
}
else
{
lean_object* v___x_1342_; lean_object* v_mctx_1343_; lean_object* v___x_1344_; lean_object* v_fst_1345_; lean_object* v_snd_1346_; lean_object* v___x_1347_; lean_object* v_cache_1348_; lean_object* v_zetaDeltaFVarIds_1349_; lean_object* v_postponed_1350_; lean_object* v_diag_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1360_; 
v___x_1342_ = lean_st_ref_get(v___y_1338_);
v_mctx_1343_ = lean_ctor_get(v___x_1342_, 0);
lean_inc_ref(v_mctx_1343_);
lean_dec(v___x_1342_);
v___x_1344_ = l_Lean_instantiateMVarsCore(v_mctx_1343_, v_e_1337_);
v_fst_1345_ = lean_ctor_get(v___x_1344_, 0);
lean_inc(v_fst_1345_);
v_snd_1346_ = lean_ctor_get(v___x_1344_, 1);
lean_inc(v_snd_1346_);
lean_dec_ref(v___x_1344_);
v___x_1347_ = lean_st_ref_take(v___y_1338_);
v_cache_1348_ = lean_ctor_get(v___x_1347_, 1);
v_zetaDeltaFVarIds_1349_ = lean_ctor_get(v___x_1347_, 2);
v_postponed_1350_ = lean_ctor_get(v___x_1347_, 3);
v_diag_1351_ = lean_ctor_get(v___x_1347_, 4);
v_isSharedCheck_1360_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1360_ == 0)
{
lean_object* v_unused_1361_; 
v_unused_1361_ = lean_ctor_get(v___x_1347_, 0);
lean_dec(v_unused_1361_);
v___x_1353_ = v___x_1347_;
v_isShared_1354_ = v_isSharedCheck_1360_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_diag_1351_);
lean_inc(v_postponed_1350_);
lean_inc(v_zetaDeltaFVarIds_1349_);
lean_inc(v_cache_1348_);
lean_dec(v___x_1347_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1360_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___x_1356_; 
if (v_isShared_1354_ == 0)
{
lean_ctor_set(v___x_1353_, 0, v_snd_1346_);
v___x_1356_ = v___x_1353_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1359_; 
v_reuseFailAlloc_1359_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1359_, 0, v_snd_1346_);
lean_ctor_set(v_reuseFailAlloc_1359_, 1, v_cache_1348_);
lean_ctor_set(v_reuseFailAlloc_1359_, 2, v_zetaDeltaFVarIds_1349_);
lean_ctor_set(v_reuseFailAlloc_1359_, 3, v_postponed_1350_);
lean_ctor_set(v_reuseFailAlloc_1359_, 4, v_diag_1351_);
v___x_1356_ = v_reuseFailAlloc_1359_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
lean_object* v___x_1357_; lean_object* v___x_1358_; 
v___x_1357_ = lean_st_ref_set(v___y_1338_, v___x_1356_);
v___x_1358_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1358_, 0, v_fst_1345_);
return v___x_1358_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg___boxed(lean_object* v_e_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_){
_start:
{
lean_object* v_res_1365_; 
v_res_1365_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg(v_e_1362_, v___y_1363_);
lean_dec(v___y_1363_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2(lean_object* v_e_1366_, lean_object* v___y_1367_, lean_object* v___y_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_){
_start:
{
lean_object* v___x_1376_; 
v___x_1376_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg(v_e_1366_, v___y_1372_);
return v___x_1376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___boxed(lean_object* v_e_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_, lean_object* v___y_1380_, lean_object* v___y_1381_, lean_object* v___y_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_){
_start:
{
lean_object* v_res_1387_; 
v_res_1387_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2(v_e_1377_, v___y_1378_, v___y_1379_, v___y_1380_, v___y_1381_, v___y_1382_, v___y_1383_, v___y_1384_, v___y_1385_);
lean_dec(v___y_1385_);
lean_dec_ref(v___y_1384_);
lean_dec(v___y_1383_);
lean_dec_ref(v___y_1382_);
lean_dec(v___y_1381_);
lean_dec_ref(v___y_1380_);
lean_dec(v___y_1379_);
lean_dec_ref(v___y_1378_);
return v_res_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg(uint8_t v___x_1388_, lean_object* v_as_x27_1389_, lean_object* v_b_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_){
_start:
{
if (lean_obj_tag(v_as_x27_1389_) == 0)
{
lean_object* v___x_1396_; 
v___x_1396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1396_, 0, v_b_1390_);
return v___x_1396_;
}
else
{
lean_object* v_head_1397_; lean_object* v_tail_1398_; lean_object* v___x_1399_; 
v_head_1397_ = lean_ctor_get(v_as_x27_1389_, 0);
v_tail_1398_ = lean_ctor_get(v_as_x27_1389_, 1);
v___x_1399_ = l_Lean_Meta_saveState___redArg(v___y_1392_, v___y_1394_);
if (lean_obj_tag(v___x_1399_) == 0)
{
lean_object* v_a_1400_; lean_object* v___x_1401_; lean_object* v___y_1403_; lean_object* v___y_1406_; lean_object* v___y_1407_; uint8_t v___y_1408_; lean_object* v___x_1411_; 
v_a_1400_ = lean_ctor_get(v___x_1399_, 0);
lean_inc(v_a_1400_);
lean_dec_ref_known(v___x_1399_, 1);
v___x_1401_ = lean_box(0);
lean_inc(v_head_1397_);
v___x_1411_ = l_Lean_MVarId_refl(v_head_1397_, v___x_1388_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1411_) == 0)
{
lean_dec(v_a_1400_);
v___y_1403_ = v___x_1411_;
goto v___jp_1402_;
}
else
{
lean_object* v_a_1412_; uint8_t v___y_1414_; uint8_t v___x_1430_; 
v_a_1412_ = lean_ctor_get(v___x_1411_, 0);
lean_inc(v_a_1412_);
v___x_1430_ = l_Lean_Exception_isInterrupt(v_a_1412_);
if (v___x_1430_ == 0)
{
uint8_t v___x_1431_; 
v___x_1431_ = l_Lean_Exception_isRuntime(v_a_1412_);
v___y_1414_ = v___x_1431_;
goto v___jp_1413_;
}
else
{
lean_dec(v_a_1412_);
v___y_1414_ = v___x_1430_;
goto v___jp_1413_;
}
v___jp_1413_:
{
if (v___y_1414_ == 0)
{
lean_object* v___x_1415_; 
lean_dec_ref_known(v___x_1411_, 1);
v___x_1415_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1400_, v___y_1392_, v___y_1394_);
lean_dec(v_a_1400_);
if (lean_obj_tag(v___x_1415_) == 0)
{
lean_object* v___x_1416_; 
lean_dec_ref_known(v___x_1415_, 1);
v___x_1416_ = l_Lean_Meta_saveState___redArg(v___y_1392_, v___y_1394_);
if (lean_obj_tag(v___x_1416_) == 0)
{
lean_object* v_a_1417_; lean_object* v___x_1418_; 
v_a_1417_ = lean_ctor_get(v___x_1416_, 0);
lean_inc(v_a_1417_);
lean_dec_ref_known(v___x_1416_, 1);
lean_inc(v_head_1397_);
v___x_1418_ = l_Lean_MVarId_inferInstance(v_head_1397_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1418_) == 0)
{
lean_dec(v_a_1417_);
v___y_1403_ = v___x_1418_;
goto v___jp_1402_;
}
else
{
lean_object* v_a_1419_; uint8_t v___x_1420_; 
v_a_1419_ = lean_ctor_get(v___x_1418_, 0);
lean_inc(v_a_1419_);
v___x_1420_ = l_Lean_Exception_isInterrupt(v_a_1419_);
if (v___x_1420_ == 0)
{
uint8_t v___x_1421_; 
v___x_1421_ = l_Lean_Exception_isRuntime(v_a_1419_);
v___y_1406_ = v_a_1417_;
v___y_1407_ = v___x_1418_;
v___y_1408_ = v___x_1421_;
goto v___jp_1405_;
}
else
{
lean_dec(v_a_1419_);
v___y_1406_ = v_a_1417_;
v___y_1407_ = v___x_1418_;
v___y_1408_ = v___x_1420_;
goto v___jp_1405_;
}
}
}
else
{
lean_object* v_a_1422_; lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1429_; 
v_a_1422_ = lean_ctor_get(v___x_1416_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___x_1416_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___x_1416_);
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
else
{
v___y_1403_ = v___x_1415_;
goto v___jp_1402_;
}
}
else
{
lean_dec(v_a_1400_);
v___y_1403_ = v___x_1411_;
goto v___jp_1402_;
}
}
}
v___jp_1402_:
{
if (lean_obj_tag(v___y_1403_) == 0)
{
lean_dec_ref_known(v___y_1403_, 1);
v_as_x27_1389_ = v_tail_1398_;
v_b_1390_ = v___x_1401_;
goto _start;
}
else
{
return v___y_1403_;
}
}
v___jp_1405_:
{
if (v___y_1408_ == 0)
{
lean_object* v___x_1409_; 
lean_dec_ref(v___y_1407_);
v___x_1409_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1406_, v___y_1392_, v___y_1394_);
lean_dec_ref(v___y_1406_);
if (lean_obj_tag(v___x_1409_) == 0)
{
lean_dec_ref_known(v___x_1409_, 1);
v_as_x27_1389_ = v_tail_1398_;
v_b_1390_ = v___x_1401_;
goto _start;
}
else
{
v___y_1403_ = v___x_1409_;
goto v___jp_1402_;
}
}
else
{
lean_dec_ref(v___y_1406_);
v___y_1403_ = v___y_1407_;
goto v___jp_1402_;
}
}
}
else
{
lean_object* v_a_1432_; lean_object* v___x_1434_; uint8_t v_isShared_1435_; uint8_t v_isSharedCheck_1439_; 
v_a_1432_ = lean_ctor_get(v___x_1399_, 0);
v_isSharedCheck_1439_ = !lean_is_exclusive(v___x_1399_);
if (v_isSharedCheck_1439_ == 0)
{
v___x_1434_ = v___x_1399_;
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
else
{
lean_inc(v_a_1432_);
lean_dec(v___x_1399_);
v___x_1434_ = lean_box(0);
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
v_resetjp_1433_:
{
lean_object* v___x_1437_; 
if (v_isShared_1435_ == 0)
{
v___x_1437_ = v___x_1434_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v_a_1432_);
v___x_1437_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
return v___x_1437_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg___boxed(lean_object* v___x_1440_, lean_object* v_as_x27_1441_, lean_object* v_b_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_){
_start:
{
uint8_t v___x_8231__boxed_1448_; lean_object* v_res_1449_; 
v___x_8231__boxed_1448_ = lean_unbox(v___x_1440_);
v_res_1449_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg(v___x_8231__boxed_1448_, v_as_x27_1441_, v_b_1442_, v___y_1443_, v___y_1444_, v___y_1445_, v___y_1446_);
lean_dec(v___y_1446_);
lean_dec_ref(v___y_1445_);
lean_dec(v___y_1444_);
lean_dec_ref(v___y_1443_);
lean_dec(v_as_x27_1441_);
return v_res_1449_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0(uint8_t v___y_1457_, uint8_t v_suppressElabErrors_1458_, lean_object* v_x_1459_){
_start:
{
if (lean_obj_tag(v_x_1459_) == 1)
{
lean_object* v_pre_1460_; 
v_pre_1460_ = lean_ctor_get(v_x_1459_, 0);
switch(lean_obj_tag(v_pre_1460_))
{
case 1:
{
lean_object* v_pre_1461_; 
v_pre_1461_ = lean_ctor_get(v_pre_1460_, 0);
switch(lean_obj_tag(v_pre_1461_))
{
case 0:
{
lean_object* v_str_1462_; lean_object* v_str_1463_; lean_object* v___x_1464_; uint8_t v___x_1465_; 
v_str_1462_ = lean_ctor_get(v_x_1459_, 1);
v_str_1463_ = lean_ctor_get(v_pre_1460_, 1);
v___x_1464_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__0));
v___x_1465_ = lean_string_dec_eq(v_str_1463_, v___x_1464_);
if (v___x_1465_ == 0)
{
lean_object* v___x_1466_; uint8_t v___x_1467_; 
v___x_1466_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__1));
v___x_1467_ = lean_string_dec_eq(v_str_1463_, v___x_1466_);
if (v___x_1467_ == 0)
{
return v___y_1457_;
}
else
{
lean_object* v___x_1468_; uint8_t v___x_1469_; 
v___x_1468_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__1));
v___x_1469_ = lean_string_dec_eq(v_str_1462_, v___x_1468_);
if (v___x_1469_ == 0)
{
return v___y_1457_;
}
else
{
return v_suppressElabErrors_1458_;
}
}
}
else
{
lean_object* v___x_1470_; uint8_t v___x_1471_; 
v___x_1470_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__2));
v___x_1471_ = lean_string_dec_eq(v_str_1462_, v___x_1470_);
if (v___x_1471_ == 0)
{
return v___y_1457_;
}
else
{
return v_suppressElabErrors_1458_;
}
}
}
case 1:
{
lean_object* v_pre_1472_; 
v_pre_1472_ = lean_ctor_get(v_pre_1461_, 0);
if (lean_obj_tag(v_pre_1472_) == 0)
{
lean_object* v_str_1473_; lean_object* v_str_1474_; lean_object* v_str_1475_; lean_object* v___x_1476_; uint8_t v___x_1477_; 
v_str_1473_ = lean_ctor_get(v_x_1459_, 1);
v_str_1474_ = lean_ctor_get(v_pre_1460_, 1);
v_str_1475_ = lean_ctor_get(v_pre_1461_, 1);
v___x_1476_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__3));
v___x_1477_ = lean_string_dec_eq(v_str_1475_, v___x_1476_);
if (v___x_1477_ == 0)
{
return v___y_1457_;
}
else
{
lean_object* v___x_1478_; uint8_t v___x_1479_; 
v___x_1478_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__4));
v___x_1479_ = lean_string_dec_eq(v_str_1474_, v___x_1478_);
if (v___x_1479_ == 0)
{
return v___y_1457_;
}
else
{
lean_object* v___x_1480_; uint8_t v___x_1481_; 
v___x_1480_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__5));
v___x_1481_ = lean_string_dec_eq(v_str_1473_, v___x_1480_);
if (v___x_1481_ == 0)
{
return v___y_1457_;
}
else
{
return v_suppressElabErrors_1458_;
}
}
}
}
else
{
return v___y_1457_;
}
}
default: 
{
return v___y_1457_;
}
}
}
case 0:
{
lean_object* v_str_1482_; lean_object* v___x_1483_; uint8_t v___x_1484_; 
v_str_1482_ = lean_ctor_get(v_x_1459_, 1);
v___x_1483_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___closed__6));
v___x_1484_ = lean_string_dec_eq(v_str_1482_, v___x_1483_);
if (v___x_1484_ == 0)
{
return v___y_1457_;
}
else
{
return v_suppressElabErrors_1458_;
}
}
default: 
{
return v___y_1457_;
}
}
}
else
{
return v___y_1457_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___boxed(lean_object* v___y_1485_, lean_object* v_suppressElabErrors_1486_, lean_object* v_x_1487_){
_start:
{
uint8_t v___y_8360__boxed_1488_; uint8_t v_suppressElabErrors_boxed_1489_; uint8_t v_res_1490_; lean_object* v_r_1491_; 
v___y_8360__boxed_1488_ = lean_unbox(v___y_1485_);
v_suppressElabErrors_boxed_1489_ = lean_unbox(v_suppressElabErrors_1486_);
v_res_1490_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0(v___y_8360__boxed_1488_, v_suppressElabErrors_boxed_1489_, v_x_1487_);
lean_dec(v_x_1487_);
v_r_1491_ = lean_box(v_res_1490_);
return v_r_1491_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3_spec__4(lean_object* v_opts_1492_, lean_object* v_opt_1493_){
_start:
{
lean_object* v_name_1494_; lean_object* v_defValue_1495_; lean_object* v_map_1496_; lean_object* v___x_1497_; 
v_name_1494_ = lean_ctor_get(v_opt_1493_, 0);
v_defValue_1495_ = lean_ctor_get(v_opt_1493_, 1);
v_map_1496_ = lean_ctor_get(v_opts_1492_, 0);
v___x_1497_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1496_, v_name_1494_);
if (lean_obj_tag(v___x_1497_) == 0)
{
uint8_t v___x_1498_; 
v___x_1498_ = lean_unbox(v_defValue_1495_);
return v___x_1498_;
}
else
{
lean_object* v_val_1499_; 
v_val_1499_ = lean_ctor_get(v___x_1497_, 0);
lean_inc(v_val_1499_);
lean_dec_ref_known(v___x_1497_, 1);
if (lean_obj_tag(v_val_1499_) == 1)
{
uint8_t v_v_1500_; 
v_v_1500_ = lean_ctor_get_uint8(v_val_1499_, 0);
lean_dec_ref_known(v_val_1499_, 0);
return v_v_1500_;
}
else
{
uint8_t v___x_1501_; 
lean_dec(v_val_1499_);
v___x_1501_ = lean_unbox(v_defValue_1495_);
return v___x_1501_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3_spec__4___boxed(lean_object* v_opts_1502_, lean_object* v_opt_1503_){
_start:
{
uint8_t v_res_1504_; lean_object* v_r_1505_; 
v_res_1504_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3_spec__4(v_opts_1502_, v_opt_1503_);
lean_dec_ref(v_opt_1503_);
lean_dec_ref(v_opts_1502_);
v_r_1505_ = lean_box(v_res_1504_);
return v_r_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg(lean_object* v_ref_1507_, lean_object* v_msgData_1508_, uint8_t v_severity_1509_, uint8_t v_isSilent_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_, lean_object* v___y_1514_){
_start:
{
uint8_t v___y_1517_; lean_object* v___y_1518_; uint8_t v___y_1519_; lean_object* v___y_1520_; lean_object* v___y_1521_; lean_object* v___y_1522_; lean_object* v___y_1523_; lean_object* v___y_1524_; lean_object* v___y_1525_; lean_object* v___y_1553_; lean_object* v___y_1554_; uint8_t v___y_1555_; uint8_t v___y_1556_; uint8_t v___y_1557_; lean_object* v___y_1558_; lean_object* v___y_1559_; lean_object* v___y_1560_; lean_object* v___y_1578_; lean_object* v___y_1579_; lean_object* v___y_1580_; uint8_t v___y_1581_; uint8_t v___y_1582_; uint8_t v___y_1583_; lean_object* v___y_1584_; lean_object* v___y_1585_; lean_object* v___y_1589_; lean_object* v___y_1590_; uint8_t v___y_1591_; uint8_t v___y_1592_; lean_object* v___y_1593_; lean_object* v___y_1594_; uint8_t v___y_1595_; uint8_t v___x_1600_; lean_object* v___y_1602_; uint8_t v___y_1603_; lean_object* v___y_1604_; lean_object* v___y_1605_; lean_object* v___y_1606_; uint8_t v___y_1607_; uint8_t v___y_1608_; uint8_t v___y_1610_; uint8_t v___x_1625_; 
v___x_1600_ = 2;
v___x_1625_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1509_, v___x_1600_);
if (v___x_1625_ == 0)
{
v___y_1610_ = v___x_1625_;
goto v___jp_1609_;
}
else
{
uint8_t v___x_1626_; 
lean_inc_ref(v_msgData_1508_);
v___x_1626_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1508_);
v___y_1610_ = v___x_1626_;
goto v___jp_1609_;
}
v___jp_1516_:
{
lean_object* v___x_1526_; lean_object* v_currNamespace_1527_; lean_object* v_openDecls_1528_; lean_object* v_env_1529_; lean_object* v_nextMacroScope_1530_; lean_object* v_ngen_1531_; lean_object* v_auxDeclNGen_1532_; lean_object* v_traceState_1533_; lean_object* v_cache_1534_; lean_object* v_messages_1535_; lean_object* v_infoState_1536_; lean_object* v_snapshotTasks_1537_; lean_object* v___x_1539_; uint8_t v_isShared_1540_; uint8_t v_isSharedCheck_1551_; 
v___x_1526_ = lean_st_ref_take(v___y_1525_);
v_currNamespace_1527_ = lean_ctor_get(v___y_1524_, 6);
v_openDecls_1528_ = lean_ctor_get(v___y_1524_, 7);
v_env_1529_ = lean_ctor_get(v___x_1526_, 0);
v_nextMacroScope_1530_ = lean_ctor_get(v___x_1526_, 1);
v_ngen_1531_ = lean_ctor_get(v___x_1526_, 2);
v_auxDeclNGen_1532_ = lean_ctor_get(v___x_1526_, 3);
v_traceState_1533_ = lean_ctor_get(v___x_1526_, 4);
v_cache_1534_ = lean_ctor_get(v___x_1526_, 5);
v_messages_1535_ = lean_ctor_get(v___x_1526_, 6);
v_infoState_1536_ = lean_ctor_get(v___x_1526_, 7);
v_snapshotTasks_1537_ = lean_ctor_get(v___x_1526_, 8);
v_isSharedCheck_1551_ = !lean_is_exclusive(v___x_1526_);
if (v_isSharedCheck_1551_ == 0)
{
v___x_1539_ = v___x_1526_;
v_isShared_1540_ = v_isSharedCheck_1551_;
goto v_resetjp_1538_;
}
else
{
lean_inc(v_snapshotTasks_1537_);
lean_inc(v_infoState_1536_);
lean_inc(v_messages_1535_);
lean_inc(v_cache_1534_);
lean_inc(v_traceState_1533_);
lean_inc(v_auxDeclNGen_1532_);
lean_inc(v_ngen_1531_);
lean_inc(v_nextMacroScope_1530_);
lean_inc(v_env_1529_);
lean_dec(v___x_1526_);
v___x_1539_ = lean_box(0);
v_isShared_1540_ = v_isSharedCheck_1551_;
goto v_resetjp_1538_;
}
v_resetjp_1538_:
{
lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1546_; 
lean_inc(v_openDecls_1528_);
lean_inc(v_currNamespace_1527_);
v___x_1541_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1541_, 0, v_currNamespace_1527_);
lean_ctor_set(v___x_1541_, 1, v_openDecls_1528_);
v___x_1542_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1542_, 0, v___x_1541_);
lean_ctor_set(v___x_1542_, 1, v___y_1522_);
lean_inc_ref(v___y_1518_);
lean_inc_ref(v___y_1523_);
v___x_1543_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1543_, 0, v___y_1523_);
lean_ctor_set(v___x_1543_, 1, v___y_1521_);
lean_ctor_set(v___x_1543_, 2, v___y_1520_);
lean_ctor_set(v___x_1543_, 3, v___y_1518_);
lean_ctor_set(v___x_1543_, 4, v___x_1542_);
lean_ctor_set_uint8(v___x_1543_, sizeof(void*)*5, v___y_1517_);
lean_ctor_set_uint8(v___x_1543_, sizeof(void*)*5 + 1, v___y_1519_);
lean_ctor_set_uint8(v___x_1543_, sizeof(void*)*5 + 2, v_isSilent_1510_);
v___x_1544_ = l_Lean_MessageLog_add(v___x_1543_, v_messages_1535_);
if (v_isShared_1540_ == 0)
{
lean_ctor_set(v___x_1539_, 6, v___x_1544_);
v___x_1546_ = v___x_1539_;
goto v_reusejp_1545_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v_env_1529_);
lean_ctor_set(v_reuseFailAlloc_1550_, 1, v_nextMacroScope_1530_);
lean_ctor_set(v_reuseFailAlloc_1550_, 2, v_ngen_1531_);
lean_ctor_set(v_reuseFailAlloc_1550_, 3, v_auxDeclNGen_1532_);
lean_ctor_set(v_reuseFailAlloc_1550_, 4, v_traceState_1533_);
lean_ctor_set(v_reuseFailAlloc_1550_, 5, v_cache_1534_);
lean_ctor_set(v_reuseFailAlloc_1550_, 6, v___x_1544_);
lean_ctor_set(v_reuseFailAlloc_1550_, 7, v_infoState_1536_);
lean_ctor_set(v_reuseFailAlloc_1550_, 8, v_snapshotTasks_1537_);
v___x_1546_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1545_;
}
v_reusejp_1545_:
{
lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; 
v___x_1547_ = lean_st_ref_set(v___y_1525_, v___x_1546_);
v___x_1548_ = lean_box(0);
v___x_1549_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1549_, 0, v___x_1548_);
return v___x_1549_;
}
}
}
v___jp_1552_:
{
lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v_a_1563_; lean_object* v___x_1565_; uint8_t v_isShared_1566_; uint8_t v_isSharedCheck_1576_; 
v___x_1561_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1508_);
v___x_1562_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Conv_elabDischargeConv_spec__1_spec__2(v___x_1561_, v___y_1511_, v___y_1512_, v___y_1513_, v___y_1514_);
v_a_1563_ = lean_ctor_get(v___x_1562_, 0);
v_isSharedCheck_1576_ = !lean_is_exclusive(v___x_1562_);
if (v_isSharedCheck_1576_ == 0)
{
v___x_1565_ = v___x_1562_;
v_isShared_1566_ = v_isSharedCheck_1576_;
goto v_resetjp_1564_;
}
else
{
lean_inc(v_a_1563_);
lean_dec(v___x_1562_);
v___x_1565_ = lean_box(0);
v_isShared_1566_ = v_isSharedCheck_1576_;
goto v_resetjp_1564_;
}
v_resetjp_1564_:
{
lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; 
lean_inc_ref_n(v___y_1554_, 2);
v___x_1567_ = l_Lean_FileMap_toPosition(v___y_1554_, v___y_1558_);
lean_dec(v___y_1558_);
v___x_1568_ = l_Lean_FileMap_toPosition(v___y_1554_, v___y_1560_);
lean_dec(v___y_1560_);
v___x_1569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1569_, 0, v___x_1568_);
v___x_1570_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___closed__0));
if (v___y_1556_ == 0)
{
lean_del_object(v___x_1565_);
lean_dec_ref(v___y_1553_);
v___y_1517_ = v___y_1555_;
v___y_1518_ = v___x_1570_;
v___y_1519_ = v___y_1557_;
v___y_1520_ = v___x_1569_;
v___y_1521_ = v___x_1567_;
v___y_1522_ = v_a_1563_;
v___y_1523_ = v___y_1559_;
v___y_1524_ = v___y_1513_;
v___y_1525_ = v___y_1514_;
goto v___jp_1516_;
}
else
{
uint8_t v___x_1571_; 
lean_inc(v_a_1563_);
v___x_1571_ = l_Lean_MessageData_hasTag(v___y_1553_, v_a_1563_);
if (v___x_1571_ == 0)
{
lean_object* v___x_1572_; lean_object* v___x_1574_; 
lean_dec_ref_known(v___x_1569_, 1);
lean_dec_ref(v___x_1567_);
lean_dec(v_a_1563_);
v___x_1572_ = lean_box(0);
if (v_isShared_1566_ == 0)
{
lean_ctor_set(v___x_1565_, 0, v___x_1572_);
v___x_1574_ = v___x_1565_;
goto v_reusejp_1573_;
}
else
{
lean_object* v_reuseFailAlloc_1575_; 
v_reuseFailAlloc_1575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1575_, 0, v___x_1572_);
v___x_1574_ = v_reuseFailAlloc_1575_;
goto v_reusejp_1573_;
}
v_reusejp_1573_:
{
return v___x_1574_;
}
}
else
{
lean_del_object(v___x_1565_);
v___y_1517_ = v___y_1555_;
v___y_1518_ = v___x_1570_;
v___y_1519_ = v___y_1557_;
v___y_1520_ = v___x_1569_;
v___y_1521_ = v___x_1567_;
v___y_1522_ = v_a_1563_;
v___y_1523_ = v___y_1559_;
v___y_1524_ = v___y_1513_;
v___y_1525_ = v___y_1514_;
goto v___jp_1516_;
}
}
}
}
v___jp_1577_:
{
lean_object* v___x_1586_; 
v___x_1586_ = l_Lean_Syntax_getTailPos_x3f(v___y_1579_, v___y_1581_);
lean_dec(v___y_1579_);
if (lean_obj_tag(v___x_1586_) == 0)
{
lean_inc(v___y_1585_);
v___y_1553_ = v___y_1578_;
v___y_1554_ = v___y_1580_;
v___y_1555_ = v___y_1581_;
v___y_1556_ = v___y_1582_;
v___y_1557_ = v___y_1583_;
v___y_1558_ = v___y_1585_;
v___y_1559_ = v___y_1584_;
v___y_1560_ = v___y_1585_;
goto v___jp_1552_;
}
else
{
lean_object* v_val_1587_; 
v_val_1587_ = lean_ctor_get(v___x_1586_, 0);
lean_inc(v_val_1587_);
lean_dec_ref_known(v___x_1586_, 1);
v___y_1553_ = v___y_1578_;
v___y_1554_ = v___y_1580_;
v___y_1555_ = v___y_1581_;
v___y_1556_ = v___y_1582_;
v___y_1557_ = v___y_1583_;
v___y_1558_ = v___y_1585_;
v___y_1559_ = v___y_1584_;
v___y_1560_ = v_val_1587_;
goto v___jp_1552_;
}
}
v___jp_1588_:
{
lean_object* v_ref_1596_; lean_object* v___x_1597_; 
v_ref_1596_ = l_Lean_replaceRef(v_ref_1507_, v___y_1593_);
v___x_1597_ = l_Lean_Syntax_getPos_x3f(v_ref_1596_, v___y_1591_);
if (lean_obj_tag(v___x_1597_) == 0)
{
lean_object* v___x_1598_; 
v___x_1598_ = lean_unsigned_to_nat(0u);
v___y_1578_ = v___y_1589_;
v___y_1579_ = v_ref_1596_;
v___y_1580_ = v___y_1590_;
v___y_1581_ = v___y_1591_;
v___y_1582_ = v___y_1592_;
v___y_1583_ = v___y_1595_;
v___y_1584_ = v___y_1594_;
v___y_1585_ = v___x_1598_;
goto v___jp_1577_;
}
else
{
lean_object* v_val_1599_; 
v_val_1599_ = lean_ctor_get(v___x_1597_, 0);
lean_inc(v_val_1599_);
lean_dec_ref_known(v___x_1597_, 1);
v___y_1578_ = v___y_1589_;
v___y_1579_ = v_ref_1596_;
v___y_1580_ = v___y_1590_;
v___y_1581_ = v___y_1591_;
v___y_1582_ = v___y_1592_;
v___y_1583_ = v___y_1595_;
v___y_1584_ = v___y_1594_;
v___y_1585_ = v_val_1599_;
goto v___jp_1577_;
}
}
v___jp_1601_:
{
if (v___y_1608_ == 0)
{
v___y_1589_ = v___y_1605_;
v___y_1590_ = v___y_1602_;
v___y_1591_ = v___y_1607_;
v___y_1592_ = v___y_1603_;
v___y_1593_ = v___y_1604_;
v___y_1594_ = v___y_1606_;
v___y_1595_ = v_severity_1509_;
goto v___jp_1588_;
}
else
{
v___y_1589_ = v___y_1605_;
v___y_1590_ = v___y_1602_;
v___y_1591_ = v___y_1607_;
v___y_1592_ = v___y_1603_;
v___y_1593_ = v___y_1604_;
v___y_1594_ = v___y_1606_;
v___y_1595_ = v___x_1600_;
goto v___jp_1588_;
}
}
v___jp_1609_:
{
if (v___y_1610_ == 0)
{
lean_object* v_fileName_1611_; lean_object* v_fileMap_1612_; lean_object* v_options_1613_; lean_object* v_ref_1614_; uint8_t v_suppressElabErrors_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___f_1618_; uint8_t v___x_1619_; uint8_t v___x_1620_; 
v_fileName_1611_ = lean_ctor_get(v___y_1513_, 0);
v_fileMap_1612_ = lean_ctor_get(v___y_1513_, 1);
v_options_1613_ = lean_ctor_get(v___y_1513_, 2);
v_ref_1614_ = lean_ctor_get(v___y_1513_, 5);
v_suppressElabErrors_1615_ = lean_ctor_get_uint8(v___y_1513_, sizeof(void*)*14 + 1);
v___x_1616_ = lean_box(v___y_1610_);
v___x_1617_ = lean_box(v_suppressElabErrors_1615_);
v___f_1618_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1618_, 0, v___x_1616_);
lean_closure_set(v___f_1618_, 1, v___x_1617_);
v___x_1619_ = 1;
v___x_1620_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1509_, v___x_1619_);
if (v___x_1620_ == 0)
{
v___y_1602_ = v_fileMap_1612_;
v___y_1603_ = v_suppressElabErrors_1615_;
v___y_1604_ = v_ref_1614_;
v___y_1605_ = v___f_1618_;
v___y_1606_ = v_fileName_1611_;
v___y_1607_ = v___y_1610_;
v___y_1608_ = v___x_1620_;
goto v___jp_1601_;
}
else
{
lean_object* v___x_1621_; uint8_t v___x_1622_; 
v___x_1621_ = l_Lean_warningAsError;
v___x_1622_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3_spec__4(v_options_1613_, v___x_1621_);
v___y_1602_ = v_fileMap_1612_;
v___y_1603_ = v_suppressElabErrors_1615_;
v___y_1604_ = v_ref_1614_;
v___y_1605_ = v___f_1618_;
v___y_1606_ = v_fileName_1611_;
v___y_1607_ = v___y_1610_;
v___y_1608_ = v___x_1622_;
goto v___jp_1601_;
}
}
else
{
lean_object* v___x_1623_; lean_object* v___x_1624_; 
lean_dec_ref(v_msgData_1508_);
v___x_1623_ = lean_box(0);
v___x_1624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1624_, 0, v___x_1623_);
return v___x_1624_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg___boxed(lean_object* v_ref_1627_, lean_object* v_msgData_1628_, lean_object* v_severity_1629_, lean_object* v_isSilent_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_){
_start:
{
uint8_t v_severity_boxed_1636_; uint8_t v_isSilent_boxed_1637_; lean_object* v_res_1638_; 
v_severity_boxed_1636_ = lean_unbox(v_severity_1629_);
v_isSilent_boxed_1637_ = lean_unbox(v_isSilent_1630_);
v_res_1638_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg(v_ref_1627_, v_msgData_1628_, v_severity_boxed_1636_, v_isSilent_boxed_1637_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_);
lean_dec(v___y_1634_);
lean_dec_ref(v___y_1633_);
lean_dec(v___y_1632_);
lean_dec_ref(v___y_1631_);
lean_dec(v_ref_1627_);
return v_res_1638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3(lean_object* v_ref_1639_, lean_object* v_msgData_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_, lean_object* v___y_1648_){
_start:
{
uint8_t v___x_1650_; uint8_t v___x_1651_; lean_object* v___x_1652_; 
v___x_1650_ = 0;
v___x_1651_ = 0;
v___x_1652_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg(v_ref_1639_, v_msgData_1640_, v___x_1650_, v___x_1651_, v___y_1645_, v___y_1646_, v___y_1647_, v___y_1648_);
return v___x_1652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3___boxed(lean_object* v_ref_1653_, lean_object* v_msgData_1654_, lean_object* v___y_1655_, lean_object* v___y_1656_, lean_object* v___y_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_){
_start:
{
lean_object* v_res_1664_; 
v_res_1664_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3(v_ref_1653_, v_msgData_1654_, v___y_1655_, v___y_1656_, v___y_1657_, v___y_1658_, v___y_1659_, v___y_1660_, v___y_1661_, v___y_1662_);
lean_dec(v___y_1662_);
lean_dec_ref(v___y_1661_);
lean_dec(v___y_1660_);
lean_dec_ref(v___y_1659_);
lean_dec(v___y_1658_);
lean_dec_ref(v___y_1657_);
lean_dec(v___y_1656_);
lean_dec_ref(v___y_1655_);
lean_dec(v_ref_1653_);
return v_res_1664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__0(lean_object* v___x_1665_, uint8_t v___x_1666_, lean_object* v_fst_1667_, lean_object* v_tk_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_, lean_object* v___y_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_){
_start:
{
lean_object* v___x_1678_; 
v___x_1678_ = l_Lean_Elab_Tactic_evalTactic(v___x_1665_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_, v___y_1674_, v___y_1675_, v___y_1676_);
if (lean_obj_tag(v___x_1678_) == 0)
{
lean_object* v___x_1679_; 
lean_dec_ref_known(v___x_1678_, 1);
v___x_1679_ = l_Lean_Elab_Tactic_getGoals___redArg(v___y_1670_);
if (lean_obj_tag(v___x_1679_) == 0)
{
lean_object* v_a_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; 
v_a_1680_ = lean_ctor_get(v___x_1679_, 0);
lean_inc(v_a_1680_);
lean_dec_ref_known(v___x_1679_, 1);
v___x_1681_ = lean_box(0);
v___x_1682_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg(v___x_1666_, v_a_1680_, v___x_1681_, v___y_1673_, v___y_1674_, v___y_1675_, v___y_1676_);
lean_dec(v_a_1680_);
if (lean_obj_tag(v___x_1682_) == 0)
{
lean_object* v___x_1683_; 
lean_dec_ref_known(v___x_1682_, 1);
v___x_1683_ = l_Lean_Elab_Tactic_pruneSolvedGoals(v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_, v___y_1674_, v___y_1675_, v___y_1676_);
if (lean_obj_tag(v___x_1683_) == 0)
{
lean_object* v___x_1684_; lean_object* v_a_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; 
lean_dec_ref_known(v___x_1683_, 1);
v___x_1684_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__2___redArg(v_fst_1667_, v___y_1674_);
v_a_1685_ = lean_ctor_get(v___x_1684_, 0);
lean_inc(v_a_1685_);
lean_dec_ref(v___x_1684_);
v___x_1686_ = l_Lean_MessageData_ofExpr(v_a_1685_);
v___x_1687_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3(v_tk_1668_, v___x_1686_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_, v___y_1674_, v___y_1675_, v___y_1676_);
return v___x_1687_;
}
else
{
lean_dec_ref(v_fst_1667_);
return v___x_1683_;
}
}
else
{
lean_dec_ref(v_fst_1667_);
return v___x_1682_;
}
}
else
{
lean_object* v_a_1688_; lean_object* v___x_1690_; uint8_t v_isShared_1691_; uint8_t v_isSharedCheck_1695_; 
lean_dec_ref(v_fst_1667_);
v_a_1688_ = lean_ctor_get(v___x_1679_, 0);
v_isSharedCheck_1695_ = !lean_is_exclusive(v___x_1679_);
if (v_isSharedCheck_1695_ == 0)
{
v___x_1690_ = v___x_1679_;
v_isShared_1691_ = v_isSharedCheck_1695_;
goto v_resetjp_1689_;
}
else
{
lean_inc(v_a_1688_);
lean_dec(v___x_1679_);
v___x_1690_ = lean_box(0);
v_isShared_1691_ = v_isSharedCheck_1695_;
goto v_resetjp_1689_;
}
v_resetjp_1689_:
{
lean_object* v___x_1693_; 
if (v_isShared_1691_ == 0)
{
v___x_1693_ = v___x_1690_;
goto v_reusejp_1692_;
}
else
{
lean_object* v_reuseFailAlloc_1694_; 
v_reuseFailAlloc_1694_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1694_, 0, v_a_1688_);
v___x_1693_ = v_reuseFailAlloc_1694_;
goto v_reusejp_1692_;
}
v_reusejp_1692_:
{
return v___x_1693_;
}
}
}
}
else
{
lean_dec_ref(v_fst_1667_);
return v___x_1678_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__0___boxed(lean_object* v___x_1696_, lean_object* v___x_1697_, lean_object* v_fst_1698_, lean_object* v_tk_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_){
_start:
{
uint8_t v___x_8682__boxed_1709_; lean_object* v_res_1710_; 
v___x_8682__boxed_1709_ = lean_unbox(v___x_1697_);
v_res_1710_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__0(v___x_1696_, v___x_8682__boxed_1709_, v_fst_1698_, v_tk_1699_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_);
lean_dec(v___y_1707_);
lean_dec_ref(v___y_1706_);
lean_dec(v___y_1705_);
lean_dec_ref(v___y_1704_);
lean_dec(v___y_1703_);
lean_dec_ref(v___y_1702_);
lean_dec(v___y_1701_);
lean_dec_ref(v___y_1700_);
lean_dec(v_tk_1699_);
return v_res_1710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__1(lean_object* v___x_1711_, lean_object* v___x_1712_, uint8_t v___x_1713_, lean_object* v_tk_1714_, lean_object* v_x_1715_, lean_object* v___y_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_, lean_object* v___y_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_){
_start:
{
lean_object* v___x_1723_; lean_object* v___x_1724_; 
v___x_1723_ = lean_box(0);
v___x_1724_ = l_Lean_Elab_Term_elabTermAndSynthesize(v___x_1711_, v___x_1723_, v___y_1716_, v___y_1717_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_);
if (lean_obj_tag(v___x_1724_) == 0)
{
lean_object* v_a_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; 
v_a_1725_ = lean_ctor_get(v___x_1724_, 0);
lean_inc(v_a_1725_);
lean_dec_ref_known(v___x_1724_, 1);
v___x_1726_ = lean_box(0);
v___x_1727_ = l_Lean_Elab_Tactic_Conv_mkConvGoalFor(v_a_1725_, v___x_1726_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_);
if (lean_obj_tag(v___x_1727_) == 0)
{
lean_object* v_a_1728_; lean_object* v_fst_1729_; lean_object* v_snd_1730_; lean_object* v___x_1731_; lean_object* v___f_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; 
v_a_1728_ = lean_ctor_get(v___x_1727_, 0);
lean_inc(v_a_1728_);
lean_dec_ref_known(v___x_1727_, 1);
v_fst_1729_ = lean_ctor_get(v_a_1728_, 0);
lean_inc(v_fst_1729_);
v_snd_1730_ = lean_ctor_get(v_a_1728_, 1);
lean_inc(v_snd_1730_);
lean_dec(v_a_1728_);
v___x_1731_ = lean_box(v___x_1713_);
v___f_1732_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_1732_, 0, v___x_1712_);
lean_closure_set(v___f_1732_, 1, v___x_1731_);
lean_closure_set(v___f_1732_, 2, v_fst_1729_);
lean_closure_set(v___f_1732_, 3, v_tk_1714_);
v___x_1733_ = l_Lean_Expr_mvarId_x21(v_snd_1730_);
lean_dec(v_snd_1730_);
v___x_1734_ = l_Lean_Elab_Tactic_run(v___x_1733_, v___f_1732_, v___y_1716_, v___y_1717_, v___y_1718_, v___y_1719_, v___y_1720_, v___y_1721_);
if (lean_obj_tag(v___x_1734_) == 0)
{
lean_object* v___x_1736_; uint8_t v_isShared_1737_; uint8_t v_isSharedCheck_1742_; 
v_isSharedCheck_1742_ = !lean_is_exclusive(v___x_1734_);
if (v_isSharedCheck_1742_ == 0)
{
lean_object* v_unused_1743_; 
v_unused_1743_ = lean_ctor_get(v___x_1734_, 0);
lean_dec(v_unused_1743_);
v___x_1736_ = v___x_1734_;
v_isShared_1737_ = v_isSharedCheck_1742_;
goto v_resetjp_1735_;
}
else
{
lean_dec(v___x_1734_);
v___x_1736_ = lean_box(0);
v_isShared_1737_ = v_isSharedCheck_1742_;
goto v_resetjp_1735_;
}
v_resetjp_1735_:
{
lean_object* v___x_1738_; lean_object* v___x_1740_; 
v___x_1738_ = lean_box(0);
if (v_isShared_1737_ == 0)
{
lean_ctor_set(v___x_1736_, 0, v___x_1738_);
v___x_1740_ = v___x_1736_;
goto v_reusejp_1739_;
}
else
{
lean_object* v_reuseFailAlloc_1741_; 
v_reuseFailAlloc_1741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1741_, 0, v___x_1738_);
v___x_1740_ = v_reuseFailAlloc_1741_;
goto v_reusejp_1739_;
}
v_reusejp_1739_:
{
return v___x_1740_;
}
}
}
else
{
lean_object* v_a_1744_; lean_object* v___x_1746_; uint8_t v_isShared_1747_; uint8_t v_isSharedCheck_1751_; 
v_a_1744_ = lean_ctor_get(v___x_1734_, 0);
v_isSharedCheck_1751_ = !lean_is_exclusive(v___x_1734_);
if (v_isSharedCheck_1751_ == 0)
{
v___x_1746_ = v___x_1734_;
v_isShared_1747_ = v_isSharedCheck_1751_;
goto v_resetjp_1745_;
}
else
{
lean_inc(v_a_1744_);
lean_dec(v___x_1734_);
v___x_1746_ = lean_box(0);
v_isShared_1747_ = v_isSharedCheck_1751_;
goto v_resetjp_1745_;
}
v_resetjp_1745_:
{
lean_object* v___x_1749_; 
if (v_isShared_1747_ == 0)
{
v___x_1749_ = v___x_1746_;
goto v_reusejp_1748_;
}
else
{
lean_object* v_reuseFailAlloc_1750_; 
v_reuseFailAlloc_1750_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1750_, 0, v_a_1744_);
v___x_1749_ = v_reuseFailAlloc_1750_;
goto v_reusejp_1748_;
}
v_reusejp_1748_:
{
return v___x_1749_;
}
}
}
}
else
{
lean_object* v_a_1752_; lean_object* v___x_1754_; uint8_t v_isShared_1755_; uint8_t v_isSharedCheck_1759_; 
lean_dec(v_tk_1714_);
lean_dec(v___x_1712_);
v_a_1752_ = lean_ctor_get(v___x_1727_, 0);
v_isSharedCheck_1759_ = !lean_is_exclusive(v___x_1727_);
if (v_isSharedCheck_1759_ == 0)
{
v___x_1754_ = v___x_1727_;
v_isShared_1755_ = v_isSharedCheck_1759_;
goto v_resetjp_1753_;
}
else
{
lean_inc(v_a_1752_);
lean_dec(v___x_1727_);
v___x_1754_ = lean_box(0);
v_isShared_1755_ = v_isSharedCheck_1759_;
goto v_resetjp_1753_;
}
v_resetjp_1753_:
{
lean_object* v___x_1757_; 
if (v_isShared_1755_ == 0)
{
v___x_1757_ = v___x_1754_;
goto v_reusejp_1756_;
}
else
{
lean_object* v_reuseFailAlloc_1758_; 
v_reuseFailAlloc_1758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1758_, 0, v_a_1752_);
v___x_1757_ = v_reuseFailAlloc_1758_;
goto v_reusejp_1756_;
}
v_reusejp_1756_:
{
return v___x_1757_;
}
}
}
}
else
{
lean_object* v_a_1760_; lean_object* v___x_1762_; uint8_t v_isShared_1763_; uint8_t v_isSharedCheck_1767_; 
lean_dec(v_tk_1714_);
lean_dec(v___x_1712_);
v_a_1760_ = lean_ctor_get(v___x_1724_, 0);
v_isSharedCheck_1767_ = !lean_is_exclusive(v___x_1724_);
if (v_isSharedCheck_1767_ == 0)
{
v___x_1762_ = v___x_1724_;
v_isShared_1763_ = v_isSharedCheck_1767_;
goto v_resetjp_1761_;
}
else
{
lean_inc(v_a_1760_);
lean_dec(v___x_1724_);
v___x_1762_ = lean_box(0);
v_isShared_1763_ = v_isSharedCheck_1767_;
goto v_resetjp_1761_;
}
v_resetjp_1761_:
{
lean_object* v___x_1765_; 
if (v_isShared_1763_ == 0)
{
v___x_1765_ = v___x_1762_;
goto v_reusejp_1764_;
}
else
{
lean_object* v_reuseFailAlloc_1766_; 
v_reuseFailAlloc_1766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1766_, 0, v_a_1760_);
v___x_1765_ = v_reuseFailAlloc_1766_;
goto v_reusejp_1764_;
}
v_reusejp_1764_:
{
return v___x_1765_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__1___boxed(lean_object* v___x_1768_, lean_object* v___x_1769_, lean_object* v___x_1770_, lean_object* v_tk_1771_, lean_object* v_x_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_){
_start:
{
uint8_t v___x_8762__boxed_1780_; lean_object* v_res_1781_; 
v___x_8762__boxed_1780_ = lean_unbox(v___x_1770_);
v_res_1781_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__1(v___x_1768_, v___x_1769_, v___x_8762__boxed_1780_, v_tk_1771_, v_x_1772_, v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_, v___y_1777_, v___y_1778_);
lean_dec(v___y_1778_);
lean_dec_ref(v___y_1777_);
lean_dec(v___y_1776_);
lean_dec_ref(v___y_1775_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
lean_dec_ref(v_x_1772_);
return v_res_1781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1(lean_object* v_x_1782_, lean_object* v_a_1783_, lean_object* v_a_1784_){
_start:
{
lean_object* v___x_1786_; uint8_t v___x_1787_; 
v___x_1786_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1));
lean_inc(v_x_1782_);
v___x_1787_ = l_Lean_Syntax_isOfKind(v_x_1782_, v___x_1786_);
if (v___x_1787_ == 0)
{
lean_object* v___x_1788_; 
lean_dec(v_x_1782_);
v___x_1788_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__0___redArg();
return v___x_1788_;
}
else
{
lean_object* v___x_1789_; lean_object* v_tk_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___f_1796_; lean_object* v___x_1797_; 
v___x_1789_ = lean_unsigned_to_nat(0u);
v_tk_1790_ = l_Lean_Syntax_getArg(v_x_1782_, v___x_1789_);
v___x_1791_ = lean_unsigned_to_nat(1u);
v___x_1792_ = l_Lean_Syntax_getArg(v_x_1782_, v___x_1791_);
v___x_1793_ = lean_unsigned_to_nat(3u);
v___x_1794_ = l_Lean_Syntax_getArg(v_x_1782_, v___x_1793_);
lean_dec(v_x_1782_);
v___x_1795_ = lean_box(v___x_1787_);
v___f_1796_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___lam__1___boxed), 12, 4);
lean_closure_set(v___f_1796_, 0, v___x_1794_);
lean_closure_set(v___f_1796_, 1, v___x_1792_);
lean_closure_set(v___f_1796_, 2, v___x_1795_);
lean_closure_set(v___f_1796_, 3, v_tk_1790_);
v___x_1797_ = l_Lean_Elab_Command_runTermElabM___redArg(v___f_1796_, v_a_1783_, v_a_1784_);
return v___x_1797_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1___boxed(lean_object* v_x_1798_, lean_object* v_a_1799_, lean_object* v_a_1800_, lean_object* v_a_1801_){
_start:
{
lean_object* v_res_1802_; 
v_res_1802_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1(v_x_1798_, v_a_1799_, v_a_1800_);
lean_dec(v_a_1800_);
lean_dec_ref(v_a_1799_);
return v_res_1802_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1(uint8_t v___x_1803_, lean_object* v_as_1804_, lean_object* v_as_x27_1805_, lean_object* v_b_1806_, lean_object* v_a_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_){
_start:
{
lean_object* v___x_1817_; 
v___x_1817_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___redArg(v___x_1803_, v_as_x27_1805_, v_b_1806_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
return v___x_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1___boxed(lean_object* v___x_1818_, lean_object* v_as_1819_, lean_object* v_as_x27_1820_, lean_object* v_b_1821_, lean_object* v_a_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_){
_start:
{
uint8_t v___x_8925__boxed_1832_; lean_object* v_res_1833_; 
v___x_8925__boxed_1832_ = lean_unbox(v___x_1818_);
v_res_1833_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__1(v___x_8925__boxed_1832_, v_as_1819_, v_as_x27_1820_, v_b_1821_, v_a_1822_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec(v_as_x27_1820_);
lean_dec(v_as_1819_);
return v_res_1833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3(lean_object* v_ref_1834_, lean_object* v_msgData_1835_, uint8_t v_severity_1836_, uint8_t v_isSilent_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_){
_start:
{
lean_object* v___x_1847_; 
v___x_1847_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___redArg(v_ref_1834_, v_msgData_1835_, v_severity_1836_, v_isSilent_1837_, v___y_1842_, v___y_1843_, v___y_1844_, v___y_1845_);
return v___x_1847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3___boxed(lean_object* v_ref_1848_, lean_object* v_msgData_1849_, lean_object* v_severity_1850_, lean_object* v_isSilent_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_){
_start:
{
uint8_t v_severity_boxed_1861_; uint8_t v_isSilent_boxed_1862_; lean_object* v_res_1863_; 
v_severity_boxed_1861_ = lean_unbox(v_severity_1850_);
v_isSilent_boxed_1862_ = lean_unbox(v_isSilent_1851_);
v_res_1863_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______elabRules__Mathlib__Tactic__Conv__command_x23conv___x3d_x3e____1_spec__3_spec__3(v_ref_1848_, v_msgData_1849_, v_severity_boxed_1861_, v_isSilent_boxed_1862_, v___y_1852_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_, v___y_1857_, v___y_1858_, v___y_1859_);
lean_dec(v___y_1859_);
lean_dec_ref(v___y_1858_);
lean_dec(v___y_1857_);
lean_dec_ref(v___y_1856_);
lean_dec(v___y_1855_);
lean_dec_ref(v___y_1854_);
lean_dec(v___y_1853_);
lean_dec_ref(v___y_1852_);
lean_dec(v_ref_1848_);
return v_res_1863_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__4(void){
_start:
{
lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; 
v___x_1874_ = l_Lean_Parser_Tactic_Conv_convSeq;
v___x_1875_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__3));
v___x_1876_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_1877_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1877_, 0, v___x_1876_);
lean_ctor_set(v___x_1877_, 1, v___x_1875_);
lean_ctor_set(v___x_1877_, 2, v___x_1874_);
return v___x_1877_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__5(void){
_start:
{
lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v___x_1878_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__4, &lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__4);
v___x_1879_ = lean_unsigned_to_nat(1022u);
v___x_1880_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1));
v___x_1881_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1881_, 0, v___x_1880_);
lean_ctor_set(v___x_1881_, 1, v___x_1879_);
lean_ctor_set(v___x_1881_, 2, v___x_1878_);
return v___x_1881_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_withReducible(void){
_start:
{
lean_object* v___x_1882_; 
v___x_1882_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__5, &lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__5);
return v___x_1882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1(lean_object* v_x_1897_, lean_object* v_a_1898_, lean_object* v_a_1899_){
_start:
{
lean_object* v___x_1900_; uint8_t v___x_1901_; 
v___x_1900_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1));
lean_inc(v_x_1897_);
v___x_1901_ = l_Lean_Syntax_isOfKind(v_x_1897_, v___x_1900_);
if (v___x_1901_ == 0)
{
lean_object* v___x_1902_; lean_object* v___x_1903_; 
lean_dec(v_x_1897_);
v___x_1902_ = lean_box(1);
v___x_1903_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1903_, 0, v___x_1902_);
lean_ctor_set(v___x_1903_, 1, v_a_1899_);
return v___x_1903_;
}
else
{
lean_object* v_ref_1904_; lean_object* v___x_1905_; lean_object* v_tk_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; uint8_t v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; 
v_ref_1904_ = lean_ctor_get(v_a_1898_, 5);
v___x_1905_ = lean_unsigned_to_nat(0u);
v_tk_1906_ = l_Lean_Syntax_getArg(v_x_1897_, v___x_1905_);
v___x_1907_ = lean_unsigned_to_nat(1u);
v___x_1908_ = l_Lean_Syntax_getArg(v_x_1897_, v___x_1907_);
lean_dec(v_x_1897_);
v___x_1909_ = 0;
v___x_1910_ = l_Lean_SourceInfo_fromRef(v_ref_1904_, v___x_1909_);
v___x_1911_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__1));
v___x_1912_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__2));
lean_inc_n(v___x_1910_, 11);
v___x_1913_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1913_, 0, v___x_1910_);
lean_ctor_set(v___x_1913_, 1, v___x_1912_);
v___x_1914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_1915_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1915_, 0, v___x_1910_);
lean_ctor_set(v___x_1915_, 1, v___x_1914_);
v___x_1916_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__4));
v___x_1917_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convRun__conv____1___closed__6));
v___x_1918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_1919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__0));
v___x_1920_ = l_Lean_SourceInfo_fromRef(v_tk_1906_, v___x_1901_);
lean_dec(v_tk_1906_);
v___x_1921_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__1));
v___x_1922_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1922_, 0, v___x_1920_);
lean_ctor_set(v___x_1922_, 1, v___x_1921_);
v___x_1923_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__3));
v___x_1924_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__4));
v___x_1925_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1925_, 0, v___x_1910_);
lean_ctor_set(v___x_1925_, 1, v___x_1924_);
lean_inc_ref(v___x_1915_);
v___x_1926_ = l_Lean_Syntax_node3(v___x_1910_, v___x_1923_, v___x_1925_, v___x_1915_, v___x_1908_);
v___x_1927_ = l_Lean_Syntax_node1(v___x_1910_, v___x_1918_, v___x_1926_);
v___x_1928_ = l_Lean_Syntax_node1(v___x_1910_, v___x_1917_, v___x_1927_);
v___x_1929_ = l_Lean_Syntax_node1(v___x_1910_, v___x_1916_, v___x_1928_);
v___x_1930_ = l_Lean_Syntax_node2(v___x_1910_, v___x_1919_, v___x_1922_, v___x_1929_);
v___x_1931_ = l_Lean_Syntax_node1(v___x_1910_, v___x_1918_, v___x_1930_);
v___x_1932_ = l_Lean_Syntax_node1(v___x_1910_, v___x_1917_, v___x_1931_);
v___x_1933_ = l_Lean_Syntax_node1(v___x_1910_, v___x_1916_, v___x_1932_);
v___x_1934_ = l_Lean_Syntax_node3(v___x_1910_, v___x_1911_, v___x_1913_, v___x_1915_, v___x_1933_);
v___x_1935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1935_, 0, v___x_1934_);
lean_ctor_set(v___x_1935_, 1, v_a_1899_);
return v___x_1935_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___boxed(lean_object* v_x_1936_, lean_object* v_a_1937_, lean_object* v_a_1938_){
_start:
{
lean_object* v_res_1939_; 
v_res_1939_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1(v_x_1936_, v_a_1937_, v_a_1938_);
lean_dec_ref(v_a_1937_);
return v_res_1939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1(lean_object* v_x_1966_, lean_object* v_a_1967_, lean_object* v_a_1968_){
_start:
{
lean_object* v___x_1969_; uint8_t v___x_1970_; 
v___x_1969_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23whnf___00__closed__1));
lean_inc(v_x_1966_);
v___x_1970_ = l_Lean_Syntax_isOfKind(v_x_1966_, v___x_1969_);
if (v___x_1970_ == 0)
{
lean_object* v___x_1971_; lean_object* v___x_1972_; 
lean_dec(v_x_1966_);
v___x_1971_ = lean_box(1);
v___x_1972_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1972_, 0, v___x_1971_);
lean_ctor_set(v___x_1972_, 1, v_a_1968_);
return v___x_1972_;
}
else
{
lean_object* v_ref_1973_; lean_object* v___x_1974_; lean_object* v_tk_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; uint8_t v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; 
v_ref_1973_ = lean_ctor_get(v_a_1967_, 5);
v___x_1974_ = lean_unsigned_to_nat(0u);
v_tk_1975_ = l_Lean_Syntax_getArg(v_x_1966_, v___x_1974_);
v___x_1976_ = lean_unsigned_to_nat(1u);
v___x_1977_ = l_Lean_Syntax_getArg(v_x_1966_, v___x_1976_);
lean_dec(v_x_1966_);
v___x_1978_ = 0;
v___x_1979_ = l_Lean_SourceInfo_fromRef(v_ref_1973_, v___x_1978_);
v___x_1980_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1));
v___x_1981_ = l_Lean_SourceInfo_fromRef(v_tk_1975_, v___x_1970_);
lean_dec(v_tk_1975_);
v___x_1982_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__0));
v___x_1983_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1983_, 0, v___x_1981_);
lean_ctor_set(v___x_1983_, 1, v___x_1982_);
v___x_1984_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__1));
v___x_1985_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2));
lean_inc_n(v___x_1979_, 3);
v___x_1986_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1986_, 0, v___x_1979_);
lean_ctor_set(v___x_1986_, 1, v___x_1984_);
v___x_1987_ = l_Lean_Syntax_node1(v___x_1979_, v___x_1985_, v___x_1986_);
v___x_1988_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_1989_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1989_, 0, v___x_1979_);
lean_ctor_set(v___x_1989_, 1, v___x_1988_);
v___x_1990_ = l_Lean_Syntax_node4(v___x_1979_, v___x_1980_, v___x_1983_, v___x_1987_, v___x_1989_, v___x_1977_);
v___x_1991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1991_, 0, v___x_1990_);
lean_ctor_set(v___x_1991_, 1, v_a_1968_);
return v___x_1991_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___boxed(lean_object* v_x_1992_, lean_object* v_a_1993_, lean_object* v_a_1994_){
_start:
{
lean_object* v_res_1995_; 
v_res_1995_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1(v_x_1992_, v_a_1993_, v_a_1994_);
lean_dec_ref(v_a_1993_);
return v_res_1995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnfR____1(lean_object* v_x_2014_, lean_object* v_a_2015_, lean_object* v_a_2016_){
_start:
{
lean_object* v___x_2017_; uint8_t v___x_2018_; 
v___x_2017_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23whnfR___00__closed__1));
lean_inc(v_x_2014_);
v___x_2018_ = l_Lean_Syntax_isOfKind(v_x_2014_, v___x_2017_);
if (v___x_2018_ == 0)
{
lean_object* v___x_2019_; lean_object* v___x_2020_; 
lean_dec(v_x_2014_);
v___x_2019_ = lean_box(1);
v___x_2020_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2020_, 0, v___x_2019_);
lean_ctor_set(v___x_2020_, 1, v_a_2016_);
return v___x_2020_;
}
else
{
lean_object* v_ref_2021_; lean_object* v___x_2022_; lean_object* v_tk_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; uint8_t v___x_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2031_; lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; lean_object* v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; 
v_ref_2021_ = lean_ctor_get(v_a_2015_, 5);
v___x_2022_ = lean_unsigned_to_nat(0u);
v_tk_2023_ = l_Lean_Syntax_getArg(v_x_2014_, v___x_2022_);
v___x_2024_ = lean_unsigned_to_nat(1u);
v___x_2025_ = l_Lean_Syntax_getArg(v_x_2014_, v___x_2024_);
lean_dec(v_x_2014_);
v___x_2026_ = 0;
v___x_2027_ = l_Lean_SourceInfo_fromRef(v_ref_2021_, v___x_2026_);
v___x_2028_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1));
v___x_2029_ = l_Lean_SourceInfo_fromRef(v_tk_2023_, v___x_2018_);
lean_dec(v_tk_2023_);
v___x_2030_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__0));
v___x_2031_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2031_, 0, v___x_2029_);
lean_ctor_set(v___x_2031_, 1, v___x_2030_);
v___x_2032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_withReducible___closed__1));
v___x_2033_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__withReducible__1___closed__1));
lean_inc_n(v___x_2027_, 8);
v___x_2034_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2034_, 0, v___x_2027_);
lean_ctor_set(v___x_2034_, 1, v___x_2033_);
v___x_2035_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__11));
v___x_2036_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convConvIn_____x3d_x3e____1___closed__3));
v___x_2037_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_2038_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__1));
v___x_2039_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__2));
v___x_2040_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2040_, 0, v___x_2027_);
lean_ctor_set(v___x_2040_, 1, v___x_2038_);
v___x_2041_ = l_Lean_Syntax_node1(v___x_2027_, v___x_2039_, v___x_2040_);
v___x_2042_ = l_Lean_Syntax_node1(v___x_2027_, v___x_2037_, v___x_2041_);
v___x_2043_ = l_Lean_Syntax_node1(v___x_2027_, v___x_2036_, v___x_2042_);
v___x_2044_ = l_Lean_Syntax_node1(v___x_2027_, v___x_2035_, v___x_2043_);
v___x_2045_ = l_Lean_Syntax_node2(v___x_2027_, v___x_2032_, v___x_2034_, v___x_2044_);
v___x_2046_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_2047_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2047_, 0, v___x_2027_);
lean_ctor_set(v___x_2047_, 1, v___x_2046_);
v___x_2048_ = l_Lean_Syntax_node4(v___x_2027_, v___x_2028_, v___x_2031_, v___x_2045_, v___x_2047_, v___x_2025_);
v___x_2049_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2049_, 0, v___x_2048_);
lean_ctor_set(v___x_2049_, 1, v_a_2016_);
return v___x_2049_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnfR____1___boxed(lean_object* v_x_2050_, lean_object* v_a_2051_, lean_object* v_a_2052_){
_start:
{
lean_object* v_res_2053_; 
v_res_2053_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnfR____1(v_x_2050_, v_a_2051_, v_a_2052_);
lean_dec_ref(v_a_2051_);
return v_res_2053_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__8(void){
_start:
{
lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; 
v___x_2074_ = l_Lean_Parser_Tactic_simpArgs;
v___x_2075_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__10));
v___x_2076_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2076_, 0, v___x_2075_);
lean_ctor_set(v___x_2076_, 1, v___x_2074_);
return v___x_2076_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__9(void){
_start:
{
lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; 
v___x_2077_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__8, &lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__8);
v___x_2078_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__7));
v___x_2079_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_2080_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2080_, 0, v___x_2079_);
lean_ctor_set(v___x_2080_, 1, v___x_2078_);
lean_ctor_set(v___x_2080_, 2, v___x_2077_);
return v___x_2080_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__13(void){
_start:
{
lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; 
v___x_2087_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__12));
v___x_2088_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__9, &lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__9);
v___x_2089_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_2090_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2090_, 0, v___x_2089_);
lean_ctor_set(v___x_2090_, 1, v___x_2088_);
lean_ctor_set(v___x_2090_, 2, v___x_2087_);
return v___x_2090_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__17(void){
_start:
{
lean_object* v___x_2096_; lean_object* v___x_2097_; lean_object* v___x_2098_; lean_object* v___x_2099_; 
v___x_2096_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__16));
v___x_2097_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__13, &lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__13);
v___x_2098_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_2099_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2099_, 0, v___x_2098_);
lean_ctor_set(v___x_2099_, 1, v___x_2097_);
lean_ctor_set(v___x_2099_, 2, v___x_2096_);
return v___x_2099_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__18(void){
_start:
{
lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; 
v___x_2100_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__25));
v___x_2101_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__17, &lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__17);
v___x_2102_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_convLHS___closed__6));
v___x_2103_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2103_, 0, v___x_2102_);
lean_ctor_set(v___x_2103_, 1, v___x_2101_);
lean_ctor_set(v___x_2103_, 2, v___x_2100_);
return v___x_2103_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__19(void){
_start:
{
lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; 
v___x_2104_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__18, &lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__18_once, _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__18);
v___x_2105_ = lean_unsigned_to_nat(1022u);
v___x_2106_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1));
v___x_2107_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_2107_, 0, v___x_2106_);
lean_ctor_set(v___x_2107_, 1, v___x_2105_);
lean_ctor_set(v___x_2107_, 2, v___x_2104_);
return v___x_2107_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e____(void){
_start:
{
lean_object* v___x_2108_; 
v___x_2108_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__19, &lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__19);
return v___x_2108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1(lean_object* v_x_2131_, lean_object* v_a_2132_, lean_object* v_a_2133_){
_start:
{
lean_object* v___y_2135_; lean_object* v___y_2136_; lean_object* v___y_2137_; lean_object* v___y_2138_; lean_object* v___y_2139_; lean_object* v___y_2140_; lean_object* v___y_2141_; lean_object* v___y_2142_; lean_object* v___y_2143_; lean_object* v___y_2144_; lean_object* v___y_2145_; lean_object* v___y_2146_; lean_object* v___y_2147_; lean_object* v___x_2155_; uint8_t v___x_2156_; 
v___x_2155_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e_____00__closed__1));
lean_inc(v_x_2131_);
v___x_2156_ = l_Lean_Syntax_isOfKind(v_x_2131_, v___x_2155_);
if (v___x_2156_ == 0)
{
lean_object* v___x_2157_; lean_object* v___x_2158_; 
lean_dec(v_x_2131_);
v___x_2157_ = lean_box(1);
v___x_2158_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2158_, 0, v___x_2157_);
lean_ctor_set(v___x_2158_, 1, v_a_2133_);
return v___x_2158_;
}
else
{
lean_object* v___x_2159_; lean_object* v___y_2161_; lean_object* v___y_2162_; lean_object* v___y_2163_; lean_object* v___y_2164_; lean_object* v___y_2165_; lean_object* v___y_2166_; lean_object* v___y_2167_; lean_object* v___y_2168_; lean_object* v___y_2169_; lean_object* v___y_2170_; lean_object* v___y_2171_; lean_object* v___y_2172_; lean_object* v___y_2173_; lean_object* v_tk_2185_; lean_object* v___y_2187_; lean_object* v___y_2188_; lean_object* v___y_2189_; lean_object* v___y_2190_; lean_object* v___x_2214_; lean_object* v___y_2216_; lean_object* v_args_2217_; lean_object* v___y_2218_; lean_object* v___y_2219_; lean_object* v_o_2227_; lean_object* v___y_2228_; lean_object* v___y_2229_; lean_object* v___x_2245_; uint8_t v___x_2246_; 
v___x_2159_ = lean_unsigned_to_nat(0u);
v_tk_2185_ = l_Lean_Syntax_getArg(v_x_2131_, v___x_2159_);
v___x_2214_ = lean_unsigned_to_nat(1u);
v___x_2245_ = l_Lean_Syntax_getArg(v_x_2131_, v___x_2214_);
v___x_2246_ = l_Lean_Syntax_isNone(v___x_2245_);
if (v___x_2246_ == 0)
{
uint8_t v___x_2247_; 
lean_inc(v___x_2245_);
v___x_2247_ = l_Lean_Syntax_matchesNull(v___x_2245_, v___x_2214_);
if (v___x_2247_ == 0)
{
lean_object* v___x_2248_; lean_object* v___x_2249_; 
lean_dec(v___x_2245_);
lean_dec(v_tk_2185_);
lean_dec(v_x_2131_);
v___x_2248_ = lean_box(1);
v___x_2249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2249_, 0, v___x_2248_);
lean_ctor_set(v___x_2249_, 1, v_a_2133_);
return v___x_2249_;
}
else
{
lean_object* v_o_2250_; lean_object* v___x_2251_; 
v_o_2250_ = l_Lean_Syntax_getArg(v___x_2245_, v___x_2159_);
lean_dec(v___x_2245_);
v___x_2251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2251_, 0, v_o_2250_);
v_o_2227_ = v___x_2251_;
v___y_2228_ = v_a_2132_;
v___y_2229_ = v_a_2133_;
goto v___jp_2226_;
}
}
else
{
lean_object* v___x_2252_; 
lean_dec(v___x_2245_);
v___x_2252_ = lean_box(0);
v_o_2227_ = v___x_2252_;
v___y_2228_ = v_a_2132_;
v___y_2229_ = v_a_2133_;
goto v___jp_2226_;
}
v___jp_2160_:
{
lean_object* v___x_2174_; lean_object* v___x_2175_; 
lean_inc_ref(v___y_2167_);
v___x_2174_ = l_Array_append___redArg(v___y_2167_, v___y_2173_);
lean_dec_ref(v___y_2173_);
lean_inc(v___y_2168_);
lean_inc(v___y_2162_);
v___x_2175_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2175_, 0, v___y_2162_);
lean_ctor_set(v___x_2175_, 1, v___y_2168_);
lean_ctor_set(v___x_2175_, 2, v___x_2174_);
if (lean_obj_tag(v___y_2169_) == 1)
{
lean_object* v_val_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; 
v_val_2176_ = lean_ctor_get(v___y_2169_, 0);
lean_inc(v_val_2176_);
lean_dec_ref_known(v___y_2169_, 1);
v___x_2177_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__0));
lean_inc_n(v___y_2162_, 3);
v___x_2178_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2178_, 0, v___y_2162_);
lean_ctor_set(v___x_2178_, 1, v___x_2177_);
lean_inc_ref(v___y_2167_);
v___x_2179_ = l_Array_append___redArg(v___y_2167_, v_val_2176_);
lean_dec(v_val_2176_);
lean_inc(v___y_2168_);
v___x_2180_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2180_, 0, v___y_2162_);
lean_ctor_set(v___x_2180_, 1, v___y_2168_);
lean_ctor_set(v___x_2180_, 2, v___x_2179_);
v___x_2181_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__1));
v___x_2182_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2182_, 0, v___y_2162_);
lean_ctor_set(v___x_2182_, 1, v___x_2181_);
v___x_2183_ = l_Array_mkArray3___redArg(v___x_2178_, v___x_2180_, v___x_2182_);
v___y_2135_ = v___x_2175_;
v___y_2136_ = v___y_2161_;
v___y_2137_ = v___y_2162_;
v___y_2138_ = v___y_2163_;
v___y_2139_ = v___y_2165_;
v___y_2140_ = v___y_2164_;
v___y_2141_ = v___y_2166_;
v___y_2142_ = v___y_2167_;
v___y_2143_ = v___y_2168_;
v___y_2144_ = v___y_2170_;
v___y_2145_ = v___y_2171_;
v___y_2146_ = v___y_2172_;
v___y_2147_ = v___x_2183_;
goto v___jp_2134_;
}
else
{
lean_object* v___x_2184_; 
lean_dec(v___y_2169_);
v___x_2184_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_2135_ = v___x_2175_;
v___y_2136_ = v___y_2161_;
v___y_2137_ = v___y_2162_;
v___y_2138_ = v___y_2163_;
v___y_2139_ = v___y_2165_;
v___y_2140_ = v___y_2164_;
v___y_2141_ = v___y_2166_;
v___y_2142_ = v___y_2167_;
v___y_2143_ = v___y_2168_;
v___y_2144_ = v___y_2170_;
v___y_2145_ = v___y_2171_;
v___y_2146_ = v___y_2172_;
v___y_2147_ = v___x_2184_;
goto v___jp_2134_;
}
}
v___jp_2186_:
{
lean_object* v_ref_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; uint8_t v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; 
v_ref_2191_ = lean_ctor_get(v___y_2189_, 5);
v___x_2192_ = lean_unsigned_to_nat(4u);
v___x_2193_ = l_Lean_Syntax_getArg(v_x_2131_, v___x_2192_);
lean_dec(v_x_2131_);
v___x_2194_ = 0;
v___x_2195_ = l_Lean_SourceInfo_fromRef(v_ref_2191_, v___x_2194_);
v___x_2196_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv_command_x23conv___x3d_x3e___00__closed__1));
v___x_2197_ = l_Lean_SourceInfo_fromRef(v_tk_2185_, v___x_2156_);
lean_dec(v_tk_2185_);
v___x_2198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23whnf____1___closed__0));
v___x_2199_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2199_, 0, v___x_2197_);
lean_ctor_set(v___x_2199_, 1, v___x_2198_);
v___x_2200_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__2));
v___x_2201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__3));
lean_inc_n(v___x_2195_, 3);
v___x_2202_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2202_, 0, v___x_2195_);
lean_ctor_set(v___x_2202_, 1, v___x_2200_);
v___x_2203_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__5));
v___x_2204_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__15));
v___x_2205_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16, &lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__16);
v___x_2206_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2206_, 0, v___x_2195_);
lean_ctor_set(v___x_2206_, 1, v___x_2204_);
lean_ctor_set(v___x_2206_, 2, v___x_2205_);
lean_inc_ref(v___x_2206_);
v___x_2207_ = l_Lean_Syntax_node1(v___x_2195_, v___x_2203_, v___x_2206_);
if (lean_obj_tag(v___y_2187_) == 1)
{
lean_object* v_val_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; 
v_val_2208_ = lean_ctor_get(v___y_2187_, 0);
lean_inc(v_val_2208_);
lean_dec_ref_known(v___y_2187_, 1);
v___x_2209_ = l_Lean_SourceInfo_fromRef(v_val_2208_, v___x_2156_);
lean_dec(v_val_2208_);
v___x_2210_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__6));
v___x_2211_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2211_, 0, v___x_2209_);
lean_ctor_set(v___x_2211_, 1, v___x_2210_);
v___x_2212_ = l_Array_mkArray1___redArg(v___x_2211_);
v___y_2161_ = v___x_2196_;
v___y_2162_ = v___x_2195_;
v___y_2163_ = v___x_2201_;
v___y_2164_ = v___x_2202_;
v___y_2165_ = v___y_2190_;
v___y_2166_ = v___x_2193_;
v___y_2167_ = v___x_2205_;
v___y_2168_ = v___x_2204_;
v___y_2169_ = v___y_2188_;
v___y_2170_ = v___x_2206_;
v___y_2171_ = v___x_2207_;
v___y_2172_ = v___x_2199_;
v___y_2173_ = v___x_2212_;
goto v___jp_2160_;
}
else
{
lean_object* v___x_2213_; 
lean_dec(v___y_2187_);
v___x_2213_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___lam__0___closed__0));
v___y_2161_ = v___x_2196_;
v___y_2162_ = v___x_2195_;
v___y_2163_ = v___x_2201_;
v___y_2164_ = v___x_2202_;
v___y_2165_ = v___y_2190_;
v___y_2166_ = v___x_2193_;
v___y_2167_ = v___x_2205_;
v___y_2168_ = v___x_2204_;
v___y_2169_ = v___y_2188_;
v___y_2170_ = v___x_2206_;
v___y_2171_ = v___x_2207_;
v___y_2172_ = v___x_2199_;
v___y_2173_ = v___x_2213_;
goto v___jp_2160_;
}
}
v___jp_2215_:
{
lean_object* v___x_2220_; lean_object* v___x_2221_; uint8_t v___x_2222_; 
v___x_2220_ = lean_unsigned_to_nat(3u);
v___x_2221_ = l_Lean_Syntax_getArg(v_x_2131_, v___x_2220_);
v___x_2222_ = l_Lean_Syntax_isNone(v___x_2221_);
if (v___x_2222_ == 0)
{
uint8_t v___x_2223_; 
v___x_2223_ = l_Lean_Syntax_matchesNull(v___x_2221_, v___x_2214_);
if (v___x_2223_ == 0)
{
lean_object* v___x_2224_; lean_object* v___x_2225_; 
lean_dec(v_args_2217_);
lean_dec(v___y_2216_);
lean_dec(v_tk_2185_);
lean_dec(v_x_2131_);
v___x_2224_ = lean_box(1);
v___x_2225_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2225_, 0, v___x_2224_);
lean_ctor_set(v___x_2225_, 1, v___y_2219_);
return v___x_2225_;
}
else
{
v___y_2187_ = v___y_2216_;
v___y_2188_ = v_args_2217_;
v___y_2189_ = v___y_2218_;
v___y_2190_ = v___y_2219_;
goto v___jp_2186_;
}
}
else
{
lean_dec(v___x_2221_);
v___y_2187_ = v___y_2216_;
v___y_2188_ = v_args_2217_;
v___y_2189_ = v___y_2218_;
v___y_2190_ = v___y_2219_;
goto v___jp_2186_;
}
}
v___jp_2226_:
{
lean_object* v___x_2230_; lean_object* v___x_2231_; uint8_t v___x_2232_; 
v___x_2230_ = lean_unsigned_to_nat(2u);
v___x_2231_ = l_Lean_Syntax_getArg(v_x_2131_, v___x_2230_);
v___x_2232_ = l_Lean_Syntax_isNone(v___x_2231_);
if (v___x_2232_ == 0)
{
uint8_t v___x_2233_; 
lean_inc(v___x_2231_);
v___x_2233_ = l_Lean_Syntax_matchesNull(v___x_2231_, v___x_2214_);
if (v___x_2233_ == 0)
{
lean_object* v___x_2234_; lean_object* v___x_2235_; 
lean_dec(v___x_2231_);
lean_dec(v_o_2227_);
lean_dec(v_tk_2185_);
lean_dec(v_x_2131_);
v___x_2234_ = lean_box(1);
v___x_2235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2235_, 0, v___x_2234_);
lean_ctor_set(v___x_2235_, 1, v___y_2229_);
return v___x_2235_;
}
else
{
lean_object* v___x_2236_; lean_object* v___x_2237_; uint8_t v___x_2238_; 
v___x_2236_ = l_Lean_Syntax_getArg(v___x_2231_, v___x_2159_);
lean_dec(v___x_2231_);
v___x_2237_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___closed__8));
lean_inc(v___x_2236_);
v___x_2238_ = l_Lean_Syntax_isOfKind(v___x_2236_, v___x_2237_);
if (v___x_2238_ == 0)
{
lean_object* v___x_2239_; lean_object* v___x_2240_; 
lean_dec(v___x_2236_);
lean_dec(v_o_2227_);
lean_dec(v_tk_2185_);
lean_dec(v_x_2131_);
v___x_2239_ = lean_box(1);
v___x_2240_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2240_, 0, v___x_2239_);
lean_ctor_set(v___x_2240_, 1, v___y_2229_);
return v___x_2240_;
}
else
{
lean_object* v___x_2241_; lean_object* v_args_2242_; lean_object* v___x_2243_; 
v___x_2241_ = l_Lean_Syntax_getArg(v___x_2236_, v___x_2214_);
lean_dec(v___x_2236_);
v_args_2242_ = l_Lean_Syntax_getArgs(v___x_2241_);
lean_dec(v___x_2241_);
v___x_2243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2243_, 0, v_args_2242_);
v___y_2216_ = v_o_2227_;
v_args_2217_ = v___x_2243_;
v___y_2218_ = v___y_2228_;
v___y_2219_ = v___y_2229_;
goto v___jp_2215_;
}
}
}
else
{
lean_object* v___x_2244_; 
lean_dec(v___x_2231_);
v___x_2244_ = lean_box(0);
v___y_2216_ = v_o_2227_;
v_args_2217_ = v___x_2244_;
v___y_2218_ = v___y_2228_;
v___y_2219_ = v___y_2229_;
goto v___jp_2215_;
}
}
}
v___jp_2134_:
{
lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; 
lean_inc_ref(v___y_2142_);
v___x_2148_ = l_Array_append___redArg(v___y_2142_, v___y_2147_);
lean_dec_ref(v___y_2147_);
lean_inc(v___y_2143_);
lean_inc_n(v___y_2137_, 3);
v___x_2149_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2149_, 0, v___y_2137_);
lean_ctor_set(v___x_2149_, 1, v___y_2143_);
lean_ctor_set(v___x_2149_, 2, v___x_2148_);
lean_inc(v___y_2138_);
v___x_2150_ = l_Lean_Syntax_node5(v___y_2137_, v___y_2138_, v___y_2140_, v___y_2145_, v___y_2144_, v___y_2135_, v___x_2149_);
v___x_2151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__convLHS__1___closed__0));
v___x_2152_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2152_, 0, v___y_2137_);
lean_ctor_set(v___x_2152_, 1, v___x_2151_);
lean_inc(v___y_2136_);
v___x_2153_ = l_Lean_Syntax_node4(v___y_2137_, v___y_2136_, v___y_2146_, v___x_2150_, v___x_2152_, v___y_2141_);
v___x_2154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2154_, 0, v___x_2153_);
lean_ctor_set(v___x_2154_, 1, v___y_2139_);
return v___x_2154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1___boxed(lean_object* v_x_2253_, lean_object* v_a_2254_, lean_object* v_a_2255_){
_start:
{
lean_object* v_res_2256_; 
v_res_2256_ = lp_mathlib_Mathlib_Tactic_Conv___aux__Mathlib__Tactic__Conv______macroRules__Mathlib__Tactic__Conv__command_x23simpOnly___x3d_x3e______1(v_x_2253_, v_a_2254_, v_a_2255_);
lean_dec_ref(v_a_2254_);
return v_res_2256_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_Conv_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Conv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Conv_convLHS = _init_lp_mathlib_Mathlib_Tactic_Conv_convLHS();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_convLHS);
lp_mathlib_Mathlib_Tactic_Conv_convRHS = _init_lp_mathlib_Mathlib_Tactic_Conv_convRHS();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_convRHS);
lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e__ = _init_lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_convConvIn_____x3d_x3e__);
lp_mathlib_Mathlib_Tactic_Conv_withReducible = _init_lp_mathlib_Mathlib_Tactic_Conv_withReducible();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_withReducible);
lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e____ = _init_lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e____();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Conv_command_x23simpOnly___x3d_x3e____);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Conv_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Conv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Conv(builtin);
}
#ifdef __cplusplus
}
#endif
