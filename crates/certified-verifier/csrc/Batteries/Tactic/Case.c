// Lean compiler output
// Module: Batteries.Tactic.Case
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.BuiltinTactic public meta import Lean.Elab.Tactic.RenameInaccessibles
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Elab_Tactic_renameInaccessibles(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getUnsolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t l_Lean_Name_isSuffixOf(lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_setGoals___redArg(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermWithHoles(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
uint64_t l_Lean_Expr_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getDelayedMVarAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Elab_Tactic_mkInitialTacticInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_closeUsingOrAdmit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_setTag___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_caseArg;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "casePattArg"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(164, 61, 27, 58, 97, 130, 120, 141)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__14_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePattArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__15;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePattArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePattArg___closed__16;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_casePattArg;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattTac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "casePattTac"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(91, 87, 28, 12, 45, 126, 105, 126)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattTac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattTac___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__4_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattTac___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__6_value),LEAN_SCALAR_PTR_LITERAL(247, 163, 83, 191, 48, 55, 64, 87)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattTac___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__9_value),LEAN_SCALAR_PTR_LITERAL(42, 158, 249, 156, 32, 69, 51, 99)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattTac___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__12_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__11_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__14_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattTac___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__0_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__1_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__17_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattTac___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__18_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_casePattTac = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__18_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "casePattExpr"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(217, 225, 146, 61, 130, 153, 26, 3)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__0_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__1_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattExpr___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__9_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_casePattExpr = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePattBody___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "casePattBody"};
static const lean_object* lp_batteries_Batteries_Tactic_casePattBody___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattBody___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattBody___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattBody___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__0_value),LEAN_SCALAR_PTR_LITERAL(95, 242, 102, 128, 135, 186, 55, 233)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattBody___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattBody___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__18_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattExpr___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattBody___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePattBody___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__0_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__1_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePattBody___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__3_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_casePattBody = (const lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePatt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "casePatt"};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 90, 42, 190, 129, 49, 179, 129)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePatt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "case "};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePatt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " | "};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__6;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__7;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_casePattBody___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__9;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt___closed__10;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_casePatt;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "case"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__9_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__12_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17_value;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__18_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_casePatt_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "casePatt'"};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt_x27___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt_x27___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt_x27___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__0_value),LEAN_SCALAR_PTR_LITERAL(121, 219, 96, 184, 93, 112, 67, 232)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_casePatt_x27___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "case' "};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_casePatt_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_casePatt_x27___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt_x27___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__4;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt_x27___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__5;
static lean_once_cell_t lp_batteries_Batteries_Tactic_casePatt_x27___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_casePatt_x27___closed__6;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_casePatt_x27;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0___closed__0 = (const lean_object*)&lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__0 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__1 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__1_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticRefine_lift_"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__2 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__2_value;
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value_aux_2),((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 164, 98, 111, 76, 231, 173, 69)}};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "refine_lift"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__4 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__4_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "show"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__5 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__5_value;
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value_aux_2),((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(78, 102, 233, 39, 129, 161, 235, 140)}};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fromTerm"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__7 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value_aux_2),((lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(64, 243, 96, 22, 30, 196, 76, 206)}};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8_value;
static const lean_string_object lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "from"};
static const lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__9 = (const lean_object*)&lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "No goals with tag "};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__1;
static const lean_string_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = " unify with the term "};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 66, .m_capacity = 66, .m_length = 65, .m_data = ", or too many names provided for renaming inaccessible variables."};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__5;
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_casePattTac___closed__6_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_processCasePattBody(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_processCasePattBody___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0 = (const lean_object*)&lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0_value;
static const lean_ctor_object lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__1 = (const lean_object*)&lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__0;
static lean_once_cell_t lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 154, 204, 143, 225, 235, 23, 70)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__0_value;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "'case' tactic failed, value"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__1_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__2;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "\ndepends on the main goal metavariable '"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__3 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__3_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__4;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__5 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__5_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_evalCase(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_evalCase___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9_spec__13(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "caseArg"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_casePattArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 119, 254, 229, 232, 21, 225, 201)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt_x27__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt_x27__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Batteries_Tactic_casePattArg___closed__15(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_30_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__14));
v___x_31_ = l_Lean_Parser_Tactic_caseArg;
v___x_32_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__5));
v___x_33_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___x_31_);
lean_ctor_set(v___x_33_, 2, v___x_30_);
return v___x_33_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePattArg___closed__16(void){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_34_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePattArg___closed__15, &lp_batteries_Batteries_Tactic_casePattArg___closed__15_once, _init_lp_batteries_Batteries_Tactic_casePattArg___closed__15);
v___x_35_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__3));
v___x_36_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__0));
v___x_37_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_37_, 0, v___x_36_);
lean_ctor_set(v___x_37_, 1, v___x_35_);
lean_ctor_set(v___x_37_, 2, v___x_34_);
return v___x_37_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePattArg(void){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePattArg___closed__16, &lp_batteries_Batteries_Tactic_casePattArg___closed__16_once, _init_lp_batteries_Batteries_Tactic_casePattArg___closed__16);
return v___x_38_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt___closed__6(void){
_start:
{
uint8_t v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_134_ = 0;
v___x_135_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__5));
v___x_136_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__4));
v___x_137_ = lp_batteries_Batteries_Tactic_casePattArg;
v___x_138_ = lean_alloc_ctor(11, 3, 1);
lean_ctor_set(v___x_138_, 0, v___x_137_);
lean_ctor_set(v___x_138_, 1, v___x_136_);
lean_ctor_set(v___x_138_, 2, v___x_135_);
lean_ctor_set_uint8(v___x_138_, sizeof(void*)*3, v___x_134_);
return v___x_138_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt___closed__7(void){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_139_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt___closed__6, &lp_batteries_Batteries_Tactic_casePatt___closed__6_once, _init_lp_batteries_Batteries_Tactic_casePatt___closed__6);
v___x_140_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__3));
v___x_141_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__5));
v___x_142_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v___x_140_);
lean_ctor_set(v___x_142_, 2, v___x_139_);
return v___x_142_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt___closed__9(void){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_146_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__8));
v___x_147_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt___closed__7, &lp_batteries_Batteries_Tactic_casePatt___closed__7_once, _init_lp_batteries_Batteries_Tactic_casePatt___closed__7);
v___x_148_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__5));
v___x_149_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___x_147_);
lean_ctor_set(v___x_149_, 2, v___x_146_);
return v___x_149_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt___closed__10(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_150_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt___closed__9, &lp_batteries_Batteries_Tactic_casePatt___closed__9_once, _init_lp_batteries_Batteries_Tactic_casePatt___closed__9);
v___x_151_ = lean_unsigned_to_nat(1022u);
v___x_152_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__1));
v___x_153_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v___x_151_);
lean_ctor_set(v___x_153_, 2, v___x_150_);
return v___x_153_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt(void){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt___closed__10, &lp_batteries_Batteries_Tactic_casePatt___closed__10_once, _init_lp_batteries_Batteries_Tactic_casePatt___closed__10);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(uint8_t v___x_155_, lean_object* v_as_156_, size_t v_i_157_, size_t v_stop_158_, lean_object* v_b_159_){
_start:
{
lean_object* v___y_161_; uint8_t v___x_165_; 
v___x_165_ = lean_usize_dec_eq(v_i_157_, v_stop_158_);
if (v___x_165_ == 0)
{
lean_object* v_fst_166_; uint8_t v___x_167_; 
v_fst_166_ = lean_ctor_get(v_b_159_, 0);
v___x_167_ = lean_unbox(v_fst_166_);
if (v___x_167_ == 0)
{
lean_object* v_snd_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_176_; 
v_snd_168_ = lean_ctor_get(v_b_159_, 1);
v_isSharedCheck_176_ = !lean_is_exclusive(v_b_159_);
if (v_isSharedCheck_176_ == 0)
{
lean_object* v_unused_177_; 
v_unused_177_ = lean_ctor_get(v_b_159_, 0);
lean_dec(v_unused_177_);
v___x_170_ = v_b_159_;
v_isShared_171_ = v_isSharedCheck_176_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_snd_168_);
lean_dec(v_b_159_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_176_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_172_; lean_object* v___x_174_; 
v___x_172_ = lean_box(v___x_155_);
if (v_isShared_171_ == 0)
{
lean_ctor_set(v___x_170_, 0, v___x_172_);
v___x_174_ = v___x_170_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v___x_172_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v_snd_168_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
v___y_161_ = v___x_174_;
goto v___jp_160_;
}
}
}
else
{
lean_object* v_snd_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_188_; 
v_snd_178_ = lean_ctor_get(v_b_159_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v_b_159_);
if (v_isSharedCheck_188_ == 0)
{
lean_object* v_unused_189_; 
v_unused_189_ = lean_ctor_get(v_b_159_, 0);
lean_dec(v_unused_189_);
v___x_180_ = v_b_159_;
v_isShared_181_ = v_isSharedCheck_188_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_snd_178_);
lean_dec(v_b_159_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_188_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_186_; 
v___x_182_ = lean_array_uget_borrowed(v_as_156_, v_i_157_);
lean_inc(v___x_182_);
v___x_183_ = lean_array_push(v_snd_178_, v___x_182_);
v___x_184_ = lean_box(v___x_165_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 1, v___x_183_);
lean_ctor_set(v___x_180_, 0, v___x_184_);
v___x_186_ = v___x_180_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_184_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v___x_183_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
v___y_161_ = v___x_186_;
goto v___jp_160_;
}
}
}
}
else
{
return v_b_159_;
}
v___jp_160_:
{
size_t v___x_162_; size_t v___x_163_; 
v___x_162_ = ((size_t)1ULL);
v___x_163_ = lean_usize_add(v_i_157_, v___x_162_);
v_i_157_ = v___x_163_;
v_b_159_ = v___y_161_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2___boxed(lean_object* v___x_190_, lean_object* v_as_191_, lean_object* v_i_192_, lean_object* v_stop_193_, lean_object* v_b_194_){
_start:
{
uint8_t v___x_5789__boxed_195_; size_t v_i_boxed_196_; size_t v_stop_boxed_197_; lean_object* v_res_198_; 
v___x_5789__boxed_195_ = lean_unbox(v___x_190_);
v_i_boxed_196_ = lean_unbox_usize(v_i_192_);
lean_dec(v_i_192_);
v_stop_boxed_197_ = lean_unbox_usize(v_stop_193_);
lean_dec(v_stop_193_);
v_res_198_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_5789__boxed_195_, v_as_191_, v_i_boxed_196_, v_stop_boxed_197_, v_b_194_);
lean_dec_ref(v_as_191_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1(size_t v_sz_199_, size_t v_i_200_, lean_object* v_bs_201_){
_start:
{
uint8_t v___x_202_; 
v___x_202_ = lean_usize_dec_lt(v_i_200_, v_sz_199_);
if (v___x_202_ == 0)
{
return v_bs_201_;
}
else
{
lean_object* v_v_203_; lean_object* v___x_204_; lean_object* v_bs_x27_205_; size_t v___x_206_; size_t v___x_207_; lean_object* v___x_208_; 
v_v_203_ = lean_array_uget(v_bs_201_, v_i_200_);
v___x_204_ = lean_unsigned_to_nat(0u);
v_bs_x27_205_ = lean_array_uset(v_bs_201_, v_i_200_, v___x_204_);
v___x_206_ = ((size_t)1ULL);
v___x_207_ = lean_usize_add(v_i_200_, v___x_206_);
v___x_208_ = lean_array_uset(v_bs_x27_205_, v_i_200_, v_v_203_);
v_i_200_ = v___x_207_;
v_bs_201_ = v___x_208_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1___boxed(lean_object* v_sz_210_, lean_object* v_i_211_, lean_object* v_bs_212_){
_start:
{
size_t v_sz_boxed_213_; size_t v_i_boxed_214_; lean_object* v_res_215_; 
v_sz_boxed_213_ = lean_unbox_usize(v_sz_210_);
lean_dec(v_sz_210_);
v_i_boxed_214_ = lean_unbox_usize(v_i_211_);
lean_dec(v_i_211_);
v_res_215_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1(v_sz_boxed_213_, v_i_boxed_214_, v_bs_212_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__0(size_t v_sz_216_, size_t v_i_217_, lean_object* v_bs_218_){
_start:
{
uint8_t v___x_219_; 
v___x_219_ = lean_usize_dec_lt(v_i_217_, v_sz_216_);
if (v___x_219_ == 0)
{
lean_object* v___x_220_; 
v___x_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_220_, 0, v_bs_218_);
return v___x_220_;
}
else
{
lean_object* v_v_221_; lean_object* v___x_222_; uint8_t v___x_223_; 
v_v_221_ = lean_array_uget(v_bs_218_, v_i_217_);
v___x_222_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__3));
lean_inc(v_v_221_);
v___x_223_ = l_Lean_Syntax_isOfKind(v_v_221_, v___x_222_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; 
lean_dec(v_v_221_);
lean_dec_ref(v_bs_218_);
v___x_224_ = lean_box(0);
return v___x_224_;
}
else
{
lean_object* v___x_225_; lean_object* v_bs_x27_226_; size_t v___x_227_; size_t v___x_228_; lean_object* v___x_229_; 
v___x_225_ = lean_unsigned_to_nat(0u);
v_bs_x27_226_ = lean_array_uset(v_bs_218_, v_i_217_, v___x_225_);
v___x_227_ = ((size_t)1ULL);
v___x_228_ = lean_usize_add(v_i_217_, v___x_227_);
v___x_229_ = lean_array_uset(v_bs_x27_226_, v_i_217_, v_v_221_);
v_i_217_ = v___x_228_;
v_bs_218_ = v___x_229_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__0___boxed(lean_object* v_sz_231_, lean_object* v_i_232_, lean_object* v_bs_233_){
_start:
{
size_t v_sz_boxed_234_; size_t v_i_boxed_235_; lean_object* v_res_236_; 
v_sz_boxed_234_ = lean_unbox_usize(v_sz_231_);
lean_dec(v_sz_231_);
v_i_boxed_235_ = lean_unbox_usize(v_i_232_);
lean_dec(v_i_232_);
v_res_236_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__0(v_sz_boxed_234_, v_i_boxed_235_, v_bs_233_);
return v_res_236_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3(void){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = l_Array_mkArray0(lean_box(0));
return v___x_241_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5(void){
_start:
{
lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_243_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__4));
v___x_244_ = l_Lean_mkAtom(v___x_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1(lean_object* v_x_275_, lean_object* v_a_276_, lean_object* v_a_277_){
_start:
{
lean_object* v___x_278_; uint8_t v___x_279_; 
v___x_278_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__1));
lean_inc(v_x_275_);
v___x_279_ = l_Lean_Syntax_isOfKind(v_x_275_, v___x_278_);
if (v___x_279_ == 0)
{
lean_object* v___x_280_; lean_object* v___x_281_; 
lean_dec(v_x_275_);
v___x_280_ = lean_box(1);
v___x_281_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_280_);
lean_ctor_set(v___x_281_, 1, v_a_277_);
return v___x_281_;
}
else
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___y_285_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; uint8_t v___x_370_; 
v___x_282_ = lean_unsigned_to_nat(0u);
v___x_283_ = lean_unsigned_to_nat(1u);
v___x_366_ = l_Lean_Syntax_getArg(v_x_275_, v___x_283_);
v___x_367_ = l_Lean_Syntax_getArgs(v___x_366_);
lean_dec(v___x_366_);
v___x_368_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__18));
v___x_369_ = lean_array_get_size(v___x_367_);
v___x_370_ = lean_nat_dec_lt(v___x_282_, v___x_369_);
if (v___x_370_ == 0)
{
lean_dec_ref(v___x_367_);
v___y_285_ = v___x_368_;
goto v___jp_284_;
}
else
{
lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_371_ = lean_box(v___x_279_);
v___x_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_368_);
v___x_373_ = lean_nat_dec_le(v___x_369_, v___x_369_);
if (v___x_373_ == 0)
{
if (v___x_370_ == 0)
{
lean_dec_ref_known(v___x_372_, 2);
lean_dec_ref(v___x_367_);
v___y_285_ = v___x_368_;
goto v___jp_284_;
}
else
{
size_t v___x_374_; size_t v___x_375_; lean_object* v___x_376_; lean_object* v_snd_377_; 
v___x_374_ = ((size_t)0ULL);
v___x_375_ = lean_usize_of_nat(v___x_369_);
v___x_376_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_279_, v___x_367_, v___x_374_, v___x_375_, v___x_372_);
lean_dec_ref(v___x_367_);
v_snd_377_ = lean_ctor_get(v___x_376_, 1);
lean_inc(v_snd_377_);
lean_dec_ref(v___x_376_);
v___y_285_ = v_snd_377_;
goto v___jp_284_;
}
}
else
{
size_t v___x_378_; size_t v___x_379_; lean_object* v___x_380_; lean_object* v_snd_381_; 
v___x_378_ = ((size_t)0ULL);
v___x_379_ = lean_usize_of_nat(v___x_369_);
v___x_380_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_279_, v___x_367_, v___x_378_, v___x_379_, v___x_372_);
lean_dec_ref(v___x_367_);
v_snd_381_ = lean_ctor_get(v___x_380_, 1);
lean_inc(v_snd_381_);
lean_dec_ref(v___x_380_);
v___y_285_ = v_snd_381_;
goto v___jp_284_;
}
}
v___jp_284_:
{
size_t v_sz_286_; size_t v___x_287_; lean_object* v___x_288_; 
v_sz_286_ = lean_array_size(v___y_285_);
v___x_287_ = ((size_t)0ULL);
v___x_288_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__0(v_sz_286_, v___x_287_, v___y_285_);
if (lean_obj_tag(v___x_288_) == 0)
{
lean_object* v___x_289_; lean_object* v___x_290_; 
lean_dec(v_x_275_);
v___x_289_ = lean_box(1);
v___x_290_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_289_);
lean_ctor_set(v___x_290_, 1, v_a_277_);
return v___x_290_;
}
else
{
lean_object* v_val_291_; lean_object* v___x_292_; lean_object* v___x_293_; uint8_t v___x_294_; 
v_val_291_ = lean_ctor_get(v___x_288_, 0);
lean_inc(v_val_291_);
lean_dec_ref_known(v___x_288_, 1);
v___x_292_ = lean_unsigned_to_nat(2u);
v___x_293_ = l_Lean_Syntax_getArg(v_x_275_, v___x_292_);
lean_dec(v_x_275_);
lean_inc(v___x_293_);
v___x_294_ = l_Lean_Syntax_matchesNull(v___x_293_, v___x_283_);
if (v___x_294_ == 0)
{
uint8_t v___x_295_; 
v___x_295_ = l_Lean_Syntax_matchesNull(v___x_293_, v___x_282_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; 
lean_dec(v_val_291_);
v___x_296_ = lean_box(1);
v___x_297_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v_a_277_);
return v___x_297_;
}
else
{
lean_object* v_ref_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; size_t v_sz_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; 
v_ref_298_ = lean_ctor_get(v_a_276_, 5);
v___x_299_ = l_Lean_SourceInfo_fromRef(v_ref_298_, v___x_294_);
v___x_300_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__0));
lean_inc_n(v___x_299_, 9);
v___x_301_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_299_);
lean_ctor_set(v___x_301_, 1, v___x_300_);
v___x_302_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__2));
v___x_303_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3);
v_sz_304_ = lean_array_size(v_val_291_);
v___x_305_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1(v_sz_304_, v___x_287_, v_val_291_);
v___x_306_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5);
v___x_307_ = l_Lean_mkSepArray(v___x_305_, v___x_306_);
lean_dec_ref(v___x_305_);
v___x_308_ = l_Array_append___redArg(v___x_303_, v___x_307_);
lean_dec_ref(v___x_307_);
v___x_309_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_309_, 0, v___x_299_);
lean_ctor_set(v___x_309_, 1, v___x_302_);
lean_ctor_set(v___x_309_, 2, v___x_308_);
v___x_310_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattBody___closed__1));
v___x_311_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattTac___closed__1));
v___x_312_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__6));
v___x_313_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_299_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
v___x_314_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10));
v___x_315_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__11));
v___x_316_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_299_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12));
v___x_318_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_318_, 0, v___x_299_);
lean_ctor_set(v___x_318_, 1, v___x_317_);
v___x_319_ = l_Lean_Syntax_node2(v___x_299_, v___x_314_, v___x_316_, v___x_318_);
v___x_320_ = l_Lean_Syntax_node2(v___x_299_, v___x_311_, v___x_313_, v___x_319_);
v___x_321_ = l_Lean_Syntax_node1(v___x_299_, v___x_310_, v___x_320_);
v___x_322_ = l_Lean_Syntax_node1(v___x_299_, v___x_302_, v___x_321_);
v___x_323_ = l_Lean_Syntax_node3(v___x_299_, v___x_278_, v___x_301_, v___x_309_, v___x_322_);
v___x_324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_323_);
lean_ctor_set(v___x_324_, 1, v_a_277_);
return v___x_324_;
}
}
else
{
lean_object* v___x_325_; lean_object* v___x_326_; uint8_t v___x_327_; 
v___x_325_ = l_Lean_Syntax_getArg(v___x_293_, v___x_282_);
lean_dec(v___x_293_);
v___x_326_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattBody___closed__1));
lean_inc(v___x_325_);
v___x_327_ = l_Lean_Syntax_isOfKind(v___x_325_, v___x_326_);
if (v___x_327_ == 0)
{
lean_object* v___x_328_; lean_object* v___x_329_; 
lean_dec(v___x_325_);
lean_dec(v_val_291_);
v___x_328_ = lean_box(1);
v___x_329_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
lean_ctor_set(v___x_329_, 1, v_a_277_);
return v___x_329_;
}
else
{
lean_object* v___x_330_; lean_object* v___x_331_; uint8_t v___x_332_; 
v___x_330_ = l_Lean_Syntax_getArg(v___x_325_, v___x_282_);
lean_dec(v___x_325_);
v___x_331_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattExpr___closed__1));
lean_inc(v___x_330_);
v___x_332_ = l_Lean_Syntax_isOfKind(v___x_330_, v___x_331_);
if (v___x_332_ == 0)
{
lean_object* v___x_333_; lean_object* v___x_334_; 
lean_dec(v___x_330_);
lean_dec(v_val_291_);
v___x_333_ = lean_box(1);
v___x_334_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_333_);
lean_ctor_set(v___x_334_, 1, v_a_277_);
return v___x_334_;
}
else
{
lean_object* v_ref_335_; lean_object* v___x_336_; uint8_t v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; size_t v_sz_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v_ref_335_ = lean_ctor_get(v_a_276_, 5);
v___x_336_ = l_Lean_Syntax_getArg(v___x_330_, v___x_283_);
lean_dec(v___x_330_);
v___x_337_ = 0;
v___x_338_ = l_Lean_SourceInfo_fromRef(v_ref_335_, v___x_337_);
v___x_339_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__0));
lean_inc_n(v___x_338_, 11);
v___x_340_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_340_, 0, v___x_338_);
lean_ctor_set(v___x_340_, 1, v___x_339_);
v___x_341_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__2));
v___x_342_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__3);
v_sz_343_ = lean_array_size(v_val_291_);
v___x_344_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__1(v_sz_343_, v___x_287_, v_val_291_);
v___x_345_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__5);
v___x_346_ = l_Lean_mkSepArray(v___x_344_, v___x_345_);
lean_dec_ref(v___x_344_);
v___x_347_ = l_Array_append___redArg(v___x_342_, v___x_346_);
lean_dec_ref(v___x_346_);
v___x_348_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_348_, 0, v___x_338_);
lean_ctor_set(v___x_348_, 1, v___x_341_);
lean_ctor_set(v___x_348_, 2, v___x_347_);
v___x_349_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattTac___closed__1));
v___x_350_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__6));
v___x_351_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_338_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
v___x_352_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13));
v___x_353_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__15));
v___x_354_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__16));
v___x_355_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__17));
v___x_356_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_338_);
lean_ctor_set(v___x_356_, 1, v___x_354_);
v___x_357_ = l_Lean_Syntax_node2(v___x_338_, v___x_355_, v___x_356_, v___x_336_);
v___x_358_ = l_Lean_Syntax_node1(v___x_338_, v___x_341_, v___x_357_);
v___x_359_ = l_Lean_Syntax_node1(v___x_338_, v___x_353_, v___x_358_);
v___x_360_ = l_Lean_Syntax_node1(v___x_338_, v___x_352_, v___x_359_);
v___x_361_ = l_Lean_Syntax_node2(v___x_338_, v___x_349_, v___x_351_, v___x_360_);
v___x_362_ = l_Lean_Syntax_node1(v___x_338_, v___x_326_, v___x_361_);
v___x_363_ = l_Lean_Syntax_node1(v___x_338_, v___x_341_, v___x_362_);
v___x_364_ = l_Lean_Syntax_node3(v___x_338_, v___x_278_, v___x_340_, v___x_348_, v___x_363_);
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v___x_364_);
lean_ctor_set(v___x_365_, 1, v_a_277_);
return v___x_365_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___boxed(lean_object* v_x_382_, lean_object* v_a_383_, lean_object* v_a_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1(v_x_382_, v_a_383_, v_a_384_);
lean_dec_ref(v_a_383_);
return v_res_385_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt_x27___closed__4(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
v___x_395_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt___closed__6, &lp_batteries_Batteries_Tactic_casePatt___closed__6_once, _init_lp_batteries_Batteries_Tactic_casePatt___closed__6);
v___x_396_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt_x27___closed__3));
v___x_397_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__5));
v___x_398_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set(v___x_398_, 1, v___x_396_);
lean_ctor_set(v___x_398_, 2, v___x_395_);
return v___x_398_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt_x27___closed__5(void){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_399_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattTac));
v___x_400_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt_x27___closed__4, &lp_batteries_Batteries_Tactic_casePatt_x27___closed__4_once, _init_lp_batteries_Batteries_Tactic_casePatt_x27___closed__4);
v___x_401_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__5));
v___x_402_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
lean_ctor_set(v___x_402_, 1, v___x_400_);
lean_ctor_set(v___x_402_, 2, v___x_399_);
return v___x_402_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt_x27___closed__6(void){
_start:
{
lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_403_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt_x27___closed__5, &lp_batteries_Batteries_Tactic_casePatt_x27___closed__5_once, _init_lp_batteries_Batteries_Tactic_casePatt_x27___closed__5);
v___x_404_ = lean_unsigned_to_nat(1022u);
v___x_405_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt_x27___closed__1));
v___x_406_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_406_, 0, v___x_405_);
lean_ctor_set(v___x_406_, 1, v___x_404_);
lean_ctor_set(v___x_406_, 2, v___x_403_);
return v___x_406_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_casePatt_x27(void){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lean_obj_once(&lp_batteries_Batteries_Tactic_casePatt_x27___closed__6, &lp_batteries_Batteries_Tactic_casePatt_x27___closed__6_once, _init_lp_batteries_Batteries_Tactic_casePatt_x27___closed__6);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1_spec__2___redArg(lean_object* v_xs_408_, lean_object* v_j_409_){
_start:
{
lean_object* v_zero_410_; uint8_t v_isZero_411_; 
v_zero_410_ = lean_unsigned_to_nat(0u);
v_isZero_411_ = lean_nat_dec_eq(v_j_409_, v_zero_410_);
if (v_isZero_411_ == 1)
{
lean_dec(v_j_409_);
return v_xs_408_;
}
else
{
lean_object* v___x_412_; lean_object* v_fst_413_; lean_object* v_one_414_; lean_object* v_n_415_; lean_object* v___x_416_; lean_object* v_fst_417_; uint8_t v___x_418_; 
v___x_412_ = lean_array_fget_borrowed(v_xs_408_, v_j_409_);
v_fst_413_ = lean_ctor_get(v___x_412_, 0);
v_one_414_ = lean_unsigned_to_nat(1u);
v_n_415_ = lean_nat_sub(v_j_409_, v_one_414_);
v___x_416_ = lean_array_fget_borrowed(v_xs_408_, v_n_415_);
v_fst_417_ = lean_ctor_get(v___x_416_, 0);
v___x_418_ = lean_nat_dec_lt(v_fst_413_, v_fst_417_);
if (v___x_418_ == 0)
{
lean_dec(v_n_415_);
lean_dec(v_j_409_);
return v_xs_408_;
}
else
{
lean_object* v___x_419_; 
v___x_419_ = lean_array_fswap(v_xs_408_, v_j_409_, v_n_415_);
lean_dec(v_j_409_);
v_xs_408_ = v___x_419_;
v_j_409_ = v_n_415_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1(lean_object* v_xs_421_, lean_object* v_i_422_, lean_object* v_fuel_423_){
_start:
{
lean_object* v_zero_424_; uint8_t v_isZero_425_; 
v_zero_424_ = lean_unsigned_to_nat(0u);
v_isZero_425_ = lean_nat_dec_eq(v_fuel_423_, v_zero_424_);
if (v_isZero_425_ == 1)
{
lean_dec(v_fuel_423_);
lean_dec(v_i_422_);
return v_xs_421_;
}
else
{
lean_object* v___x_426_; uint8_t v___x_427_; 
v___x_426_ = lean_array_get_size(v_xs_421_);
v___x_427_ = lean_nat_dec_lt(v_i_422_, v___x_426_);
if (v___x_427_ == 0)
{
lean_dec(v_fuel_423_);
lean_dec(v_i_422_);
return v_xs_421_;
}
else
{
lean_object* v_one_428_; lean_object* v_n_429_; lean_object* v___x_430_; lean_object* v___x_431_; 
v_one_428_ = lean_unsigned_to_nat(1u);
v_n_429_ = lean_nat_sub(v_fuel_423_, v_one_428_);
lean_dec(v_fuel_423_);
lean_inc(v_i_422_);
v___x_430_ = lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1_spec__2___redArg(v_xs_421_, v_i_422_);
v___x_431_ = lean_nat_add(v_i_422_, v_one_428_);
lean_dec(v_i_422_);
v_xs_421_ = v___x_430_;
v_i_422_ = v___x_431_;
v_fuel_423_ = v_n_429_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__2(size_t v_sz_433_, size_t v_i_434_, lean_object* v_bs_435_){
_start:
{
uint8_t v___x_436_; 
v___x_436_ = lean_usize_dec_lt(v_i_434_, v_sz_433_);
if (v___x_436_ == 0)
{
return v_bs_435_;
}
else
{
lean_object* v_v_437_; lean_object* v_snd_438_; lean_object* v___x_439_; lean_object* v_bs_x27_440_; size_t v___x_441_; size_t v___x_442_; lean_object* v___x_443_; 
v_v_437_ = lean_array_uget_borrowed(v_bs_435_, v_i_434_);
v_snd_438_ = lean_ctor_get(v_v_437_, 1);
lean_inc(v_snd_438_);
v___x_439_ = lean_unsigned_to_nat(0u);
v_bs_x27_440_ = lean_array_uset(v_bs_435_, v_i_434_, v___x_439_);
v___x_441_ = ((size_t)1ULL);
v___x_442_ = lean_usize_add(v_i_434_, v___x_441_);
v___x_443_ = lean_array_uset(v_bs_x27_440_, v_i_434_, v_snd_438_);
v_i_434_ = v___x_442_;
v_bs_435_ = v___x_443_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__2___boxed(lean_object* v_sz_445_, lean_object* v_i_446_, lean_object* v_bs_447_){
_start:
{
size_t v_sz_boxed_448_; size_t v_i_boxed_449_; lean_object* v_res_450_; 
v_sz_boxed_448_ = lean_unbox_usize(v_sz_445_);
lean_dec(v_sz_445_);
v_i_boxed_449_ = lean_unbox_usize(v_i_446_);
lean_dec(v_i_446_);
v_res_450_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__2(v_sz_boxed_448_, v_i_boxed_449_, v_bs_447_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg(lean_object* v_tag_451_, lean_object* v_as_452_, size_t v_i_453_, size_t v_stop_454_, lean_object* v_b_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v_a_462_; lean_object* v_val_467_; uint8_t v___x_469_; 
v___x_469_ = lean_usize_dec_eq(v_i_453_, v_stop_454_);
if (v___x_469_ == 0)
{
lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_470_ = lean_array_uget_borrowed(v_as_452_, v_i_453_);
lean_inc(v___x_470_);
v___x_471_ = l_Lean_MVarId_getDecl(v___x_470_, v___y_456_, v___y_457_, v___y_458_, v___y_459_);
if (lean_obj_tag(v___x_471_) == 0)
{
lean_object* v_a_472_; lean_object* v_userName_473_; uint8_t v___x_474_; 
v_a_472_ = lean_ctor_get(v___x_471_, 0);
lean_inc(v_a_472_);
lean_dec_ref_known(v___x_471_, 1);
v_userName_473_ = lean_ctor_get(v_a_472_, 0);
lean_inc(v_userName_473_);
lean_dec(v_a_472_);
v___x_474_ = lean_name_eq(v_tag_451_, v_userName_473_);
if (v___x_474_ == 0)
{
uint8_t v___x_475_; 
v___x_475_ = l_Lean_Name_isSuffixOf(v_tag_451_, v_userName_473_);
if (v___x_475_ == 0)
{
uint8_t v___x_476_; 
v___x_476_ = l_Lean_Name_isPrefixOf(v_tag_451_, v_userName_473_);
lean_dec(v_userName_473_);
if (v___x_476_ == 0)
{
v_a_462_ = v_b_455_;
goto v___jp_461_;
}
else
{
lean_object* v___x_477_; lean_object* v___x_478_; 
v___x_477_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_470_);
v___x_478_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_477_);
lean_ctor_set(v___x_478_, 1, v___x_470_);
v_val_467_ = v___x_478_;
goto v___jp_466_;
}
}
else
{
lean_object* v___x_479_; lean_object* v___x_480_; 
lean_dec(v_userName_473_);
v___x_479_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_470_);
v___x_480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_480_, 0, v___x_479_);
lean_ctor_set(v___x_480_, 1, v___x_470_);
v_val_467_ = v___x_480_;
goto v___jp_466_;
}
}
else
{
lean_object* v___x_481_; lean_object* v___x_482_; 
lean_dec(v_userName_473_);
v___x_481_ = lean_unsigned_to_nat(0u);
lean_inc(v___x_470_);
v___x_482_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_482_, 0, v___x_481_);
lean_ctor_set(v___x_482_, 1, v___x_470_);
v_val_467_ = v___x_482_;
goto v___jp_466_;
}
}
else
{
lean_object* v_a_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_490_; 
lean_dec_ref(v_b_455_);
v_a_483_ = lean_ctor_get(v___x_471_, 0);
v_isSharedCheck_490_ = !lean_is_exclusive(v___x_471_);
if (v_isSharedCheck_490_ == 0)
{
v___x_485_ = v___x_471_;
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_a_483_);
lean_dec(v___x_471_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_490_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___x_488_; 
if (v_isShared_486_ == 0)
{
v___x_488_ = v___x_485_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v_a_483_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
}
else
{
lean_object* v___x_491_; 
v___x_491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_491_, 0, v_b_455_);
return v___x_491_;
}
v___jp_461_:
{
size_t v___x_463_; size_t v___x_464_; 
v___x_463_ = ((size_t)1ULL);
v___x_464_ = lean_usize_add(v_i_453_, v___x_463_);
v_i_453_ = v___x_464_;
v_b_455_ = v_a_462_;
goto _start;
}
v___jp_466_:
{
lean_object* v___x_468_; 
v___x_468_ = lean_array_push(v_b_455_, v_val_467_);
v_a_462_ = v___x_468_;
goto v___jp_461_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg___boxed(lean_object* v_tag_492_, lean_object* v_as_493_, lean_object* v_i_494_, lean_object* v_stop_495_, lean_object* v_b_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_){
_start:
{
size_t v_i_boxed_502_; size_t v_stop_boxed_503_; lean_object* v_res_504_; 
v_i_boxed_502_ = lean_unbox_usize(v_i_494_);
lean_dec(v_i_494_);
v_stop_boxed_503_ = lean_unbox_usize(v_stop_495_);
lean_dec(v_stop_495_);
v_res_504_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg(v_tag_492_, v_as_493_, v_i_boxed_502_, v_stop_boxed_503_, v_b_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
lean_dec(v___y_500_);
lean_dec_ref(v___y_499_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
lean_dec_ref(v_as_493_);
lean_dec(v_tag_492_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0(lean_object* v_tag_507_, lean_object* v_as_508_, lean_object* v_start_509_, lean_object* v_stop_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_){
_start:
{
lean_object* v___x_520_; uint8_t v___x_521_; 
v___x_520_ = ((lean_object*)(lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0___closed__0));
v___x_521_ = lean_nat_dec_lt(v_start_509_, v_stop_510_);
if (v___x_521_ == 0)
{
lean_object* v___x_522_; 
v___x_522_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_522_, 0, v___x_520_);
return v___x_522_;
}
else
{
lean_object* v___x_523_; uint8_t v___x_524_; 
v___x_523_ = lean_array_get_size(v_as_508_);
v___x_524_ = lean_nat_dec_le(v_stop_510_, v___x_523_);
if (v___x_524_ == 0)
{
uint8_t v___x_525_; 
v___x_525_ = lean_nat_dec_lt(v_start_509_, v___x_523_);
if (v___x_525_ == 0)
{
lean_object* v___x_526_; 
v___x_526_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_526_, 0, v___x_520_);
return v___x_526_;
}
else
{
size_t v___x_527_; size_t v___x_528_; lean_object* v___x_529_; 
v___x_527_ = lean_usize_of_nat(v_start_509_);
v___x_528_ = lean_usize_of_nat(v___x_523_);
v___x_529_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg(v_tag_507_, v_as_508_, v___x_527_, v___x_528_, v___x_520_, v___y_515_, v___y_516_, v___y_517_, v___y_518_);
return v___x_529_;
}
}
else
{
size_t v___x_530_; size_t v___x_531_; lean_object* v___x_532_; 
v___x_530_ = lean_usize_of_nat(v_start_509_);
v___x_531_ = lean_usize_of_nat(v_stop_510_);
v___x_532_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg(v_tag_507_, v_as_508_, v___x_530_, v___x_531_, v___x_520_, v___y_515_, v___y_516_, v___y_517_, v___y_518_);
return v___x_532_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0___boxed(lean_object* v_tag_533_, lean_object* v_as_534_, lean_object* v_start_535_, lean_object* v_stop_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0(v_tag_533_, v_as_534_, v_start_535_, v_stop_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
lean_dec(v___y_540_);
lean_dec_ref(v___y_539_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
lean_dec(v_stop_536_);
lean_dec(v_start_535_);
lean_dec_ref(v_as_534_);
lean_dec(v_tag_533_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag(lean_object* v_mvarIds_547_, lean_object* v_tag_548_, lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_, lean_object* v_a_555_, lean_object* v_a_556_){
_start:
{
lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_558_ = lean_array_mk(v_mvarIds_547_);
v___x_559_ = lean_unsigned_to_nat(0u);
v___x_560_ = lean_array_get_size(v___x_558_);
v___x_561_ = lp_batteries_Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0(v_tag_548_, v___x_558_, v___x_559_, v___x_560_, v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_, v_a_554_, v_a_555_, v_a_556_);
lean_dec_ref(v___x_558_);
if (lean_obj_tag(v___x_561_) == 0)
{
lean_object* v_a_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_575_; 
v_a_562_ = lean_ctor_get(v___x_561_, 0);
v_isSharedCheck_575_ = !lean_is_exclusive(v___x_561_);
if (v_isSharedCheck_575_ == 0)
{
v___x_564_ = v___x_561_;
v_isShared_565_ = v_isSharedCheck_575_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_a_562_);
lean_dec(v___x_561_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_575_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
lean_object* v___x_566_; lean_object* v___x_567_; size_t v_sz_568_; size_t v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_573_; 
v___x_566_ = lean_array_get_size(v_a_562_);
v___x_567_ = lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1(v_a_562_, v___x_559_, v___x_566_);
v_sz_568_ = lean_array_size(v___x_567_);
v___x_569_ = ((size_t)0ULL);
v___x_570_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__2(v_sz_568_, v___x_569_, v___x_567_);
v___x_571_ = lean_array_to_list(v___x_570_);
if (v_isShared_565_ == 0)
{
lean_ctor_set(v___x_564_, 0, v___x_571_);
v___x_573_ = v___x_564_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v___x_571_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
return v___x_573_;
}
}
}
else
{
lean_object* v_a_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_583_; 
v_a_576_ = lean_ctor_get(v___x_561_, 0);
v_isSharedCheck_583_ = !lean_is_exclusive(v___x_561_);
if (v_isSharedCheck_583_ == 0)
{
v___x_578_ = v___x_561_;
v_isShared_579_ = v_isSharedCheck_583_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_a_576_);
lean_dec(v___x_561_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_583_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
lean_object* v___x_581_; 
if (v_isShared_579_ == 0)
{
v___x_581_ = v___x_578_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v_a_576_);
v___x_581_ = v_reuseFailAlloc_582_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
return v___x_581_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag___boxed(lean_object* v_mvarIds_584_, lean_object* v_tag_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_, lean_object* v_a_593_, lean_object* v_a_594_){
_start:
{
lean_object* v_res_595_; 
v_res_595_ = lp_batteries___private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag(v_mvarIds_584_, v_tag_585_, v_a_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v_a_591_, v_a_592_, v_a_593_);
lean_dec(v_a_593_);
lean_dec_ref(v_a_592_);
lean_dec(v_a_591_);
lean_dec_ref(v_a_590_);
lean_dec(v_a_589_);
lean_dec_ref(v_a_588_);
lean_dec(v_a_587_);
lean_dec_ref(v_a_586_);
lean_dec(v_tag_585_);
return v_res_595_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0(lean_object* v_tag_596_, lean_object* v_as_597_, size_t v_i_598_, size_t v_stop_599_, lean_object* v_b_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___redArg(v_tag_596_, v_as_597_, v_i_598_, v_stop_599_, v_b_600_, v___y_605_, v___y_606_, v___y_607_, v___y_608_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0___boxed(lean_object* v_tag_611_, lean_object* v_as_612_, lean_object* v_i_613_, lean_object* v_stop_614_, lean_object* v_b_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_){
_start:
{
size_t v_i_boxed_625_; size_t v_stop_boxed_626_; lean_object* v_res_627_; 
v_i_boxed_625_ = lean_unbox_usize(v_i_613_);
lean_dec(v_i_613_);
v_stop_boxed_626_ = lean_unbox_usize(v_stop_614_);
lean_dec(v_stop_614_);
v_res_627_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__0_spec__0(v_tag_611_, v_as_612_, v_i_boxed_625_, v_stop_boxed_626_, v_b_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_, v___y_620_, v___y_621_, v___y_622_, v___y_623_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
lean_dec(v___y_621_);
lean_dec_ref(v___y_620_);
lean_dec(v___y_619_);
lean_dec_ref(v___y_618_);
lean_dec(v___y_617_);
lean_dec_ref(v___y_616_);
lean_dec_ref(v_as_612_);
lean_dec(v_tag_611_);
return v_res_627_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1_spec__2(lean_object* v_xs_628_, lean_object* v_j_629_, lean_object* v_h_630_){
_start:
{
lean_object* v___x_631_; 
v___x_631_ = lp_batteries___private_Init_Data_Array_InsertionSort_0__Array_insertionSort_swapLoop___at___00__private_Init_Data_Array_InsertionSort_0__Array_insertionSort_traverse___at___00__private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag_spec__1_spec__2___redArg(v_xs_628_, v_j_629_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___lam__0(lean_object* v_x_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
lean_object* v___x_642_; 
lean_inc(v___y_636_);
lean_inc_ref(v___y_635_);
lean_inc(v___y_634_);
lean_inc_ref(v___y_633_);
v___x_642_ = lean_apply_9(v_x_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_, v___y_640_, lean_box(0));
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___lam__0___boxed(lean_object* v_x_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___lam__0(v_x_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec(v___y_645_);
lean_dec_ref(v___y_644_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg(lean_object* v_mvarId_654_, lean_object* v_x_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_){
_start:
{
lean_object* v___f_665_; lean_object* v___x_666_; 
lean_inc(v___y_659_);
lean_inc_ref(v___y_658_);
lean_inc(v___y_657_);
lean_inc_ref(v___y_656_);
v___f_665_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_665_, 0, v_x_655_);
lean_closure_set(v___f_665_, 1, v___y_656_);
lean_closure_set(v___f_665_, 2, v___y_657_);
lean_closure_set(v___f_665_, 3, v___y_658_);
lean_closure_set(v___f_665_, 4, v___y_659_);
v___x_666_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_654_, v___f_665_, v___y_660_, v___y_661_, v___y_662_, v___y_663_);
if (lean_obj_tag(v___x_666_) == 0)
{
return v___x_666_;
}
else
{
lean_object* v_a_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_674_; 
v_a_667_ = lean_ctor_get(v___x_666_, 0);
v_isSharedCheck_674_ = !lean_is_exclusive(v___x_666_);
if (v_isSharedCheck_674_ == 0)
{
v___x_669_ = v___x_666_;
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_a_667_);
lean_dec(v___x_666_);
v___x_669_ = lean_box(0);
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
v_resetjp_668_:
{
lean_object* v___x_672_; 
if (v_isShared_670_ == 0)
{
v___x_672_ = v___x_669_;
goto v_reusejp_671_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v_a_667_);
v___x_672_ = v_reuseFailAlloc_673_;
goto v_reusejp_671_;
}
v_reusejp_671_:
{
return v___x_672_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg___boxed(lean_object* v_mvarId_675_, lean_object* v_x_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
lean_object* v_res_686_; 
v_res_686_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg(v_mvarId_675_, v_x_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
return v_res_686_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1(lean_object* v_00_u03b1_687_, lean_object* v_mvarId_688_, lean_object* v_x_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_){
_start:
{
lean_object* v___x_699_; 
v___x_699_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg(v_mvarId_688_, v_x_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_, v___y_696_, v___y_697_);
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___boxed(lean_object* v_00_u03b1_700_, lean_object* v_mvarId_701_, lean_object* v_x_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1(v_00_u03b1_700_, v_mvarId_701_, v_x_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
lean_dec(v___y_706_);
lean_dec_ref(v___y_705_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg(lean_object* v_a_713_, lean_object* v___y_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v___x_723_; lean_object* v___x_724_; 
lean_inc(v___y_715_);
lean_inc_ref(v___y_714_);
v___x_723_ = lean_apply_2(v_a_713_, v___y_714_, v___y_715_);
v___x_724_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___x_723_, v___y_716_, v___y_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg___boxed(lean_object* v_a_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_){
_start:
{
lean_object* v_res_735_; 
v_res_735_ = lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg(v_a_725_, v___y_726_, v___y_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_, v___y_732_, v___y_733_);
lean_dec(v___y_733_);
lean_dec_ref(v___y_732_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
lean_dec(v___y_729_);
lean_dec_ref(v___y_728_);
lean_dec(v___y_727_);
lean_dec_ref(v___y_726_);
return v_res_735_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5(lean_object* v_00_u03b1_736_, lean_object* v_a_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg(v_a_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___boxed(lean_object* v_00_u03b1_748_, lean_object* v_a_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_){
_start:
{
lean_object* v_res_759_; 
v_res_759_ = lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5(v_00_u03b1_748_, v_a_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_);
lean_dec(v___y_757_);
lean_dec_ref(v___y_756_);
lean_dec(v___y_755_);
lean_dec_ref(v___y_754_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v___y_751_);
lean_dec_ref(v___y_750_);
return v_res_759_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4_spec__6(lean_object* v_msgData_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_){
_start:
{
lean_object* v___x_766_; lean_object* v_env_767_; lean_object* v___x_768_; lean_object* v_mctx_769_; lean_object* v_lctx_770_; lean_object* v_options_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_766_ = lean_st_ref_get(v___y_764_);
v_env_767_ = lean_ctor_get(v___x_766_, 0);
lean_inc_ref(v_env_767_);
lean_dec(v___x_766_);
v___x_768_ = lean_st_ref_get(v___y_762_);
v_mctx_769_ = lean_ctor_get(v___x_768_, 0);
lean_inc_ref(v_mctx_769_);
lean_dec(v___x_768_);
v_lctx_770_ = lean_ctor_get(v___y_761_, 2);
v_options_771_ = lean_ctor_get(v___y_763_, 2);
lean_inc_ref(v_options_771_);
lean_inc_ref(v_lctx_770_);
v___x_772_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_772_, 0, v_env_767_);
lean_ctor_set(v___x_772_, 1, v_mctx_769_);
lean_ctor_set(v___x_772_, 2, v_lctx_770_);
lean_ctor_set(v___x_772_, 3, v_options_771_);
v___x_773_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_773_, 0, v___x_772_);
lean_ctor_set(v___x_773_, 1, v_msgData_760_);
v___x_774_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4_spec__6___boxed(lean_object* v_msgData_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4_spec__6(v_msgData_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
lean_dec(v___y_777_);
lean_dec_ref(v___y_776_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg(lean_object* v_msg_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_){
_start:
{
lean_object* v_ref_788_; lean_object* v___x_789_; lean_object* v_a_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_798_; 
v_ref_788_ = lean_ctor_get(v___y_785_, 5);
v___x_789_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4_spec__6(v_msg_782_, v___y_783_, v___y_784_, v___y_785_, v___y_786_);
v_a_790_ = lean_ctor_get(v___x_789_, 0);
v_isSharedCheck_798_ = !lean_is_exclusive(v___x_789_);
if (v_isSharedCheck_798_ == 0)
{
v___x_792_ = v___x_789_;
v_isShared_793_ = v_isSharedCheck_798_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_a_790_);
lean_dec(v___x_789_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_798_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___x_794_; lean_object* v___x_796_; 
lean_inc(v_ref_788_);
v___x_794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_794_, 0, v_ref_788_);
lean_ctor_set(v___x_794_, 1, v_a_790_);
if (v_isShared_793_ == 0)
{
lean_ctor_set_tag(v___x_792_, 1);
lean_ctor_set(v___x_792_, 0, v___x_794_);
v___x_796_ = v___x_792_;
goto v_reusejp_795_;
}
else
{
lean_object* v_reuseFailAlloc_797_; 
v_reuseFailAlloc_797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_797_, 0, v___x_794_);
v___x_796_ = v_reuseFailAlloc_797_;
goto v_reusejp_795_;
}
v_reusejp_795_:
{
return v___x_796_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg___boxed(lean_object* v_msg_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_){
_start:
{
lean_object* v_res_805_; 
v_res_805_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg(v_msg_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_);
lean_dec(v___y_803_);
lean_dec_ref(v___y_802_);
lean_dec(v___y_801_);
lean_dec_ref(v___y_800_);
return v_res_805_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9_spec__10___redArg(lean_object* v_x_806_, lean_object* v_x_807_, lean_object* v_x_808_, lean_object* v_x_809_){
_start:
{
lean_object* v_ks_810_; lean_object* v_vs_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_835_; 
v_ks_810_ = lean_ctor_get(v_x_806_, 0);
v_vs_811_ = lean_ctor_get(v_x_806_, 1);
v_isSharedCheck_835_ = !lean_is_exclusive(v_x_806_);
if (v_isSharedCheck_835_ == 0)
{
v___x_813_ = v_x_806_;
v_isShared_814_ = v_isSharedCheck_835_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_vs_811_);
lean_inc(v_ks_810_);
lean_dec(v_x_806_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_835_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v___x_815_; uint8_t v___x_816_; 
v___x_815_ = lean_array_get_size(v_ks_810_);
v___x_816_ = lean_nat_dec_lt(v_x_807_, v___x_815_);
if (v___x_816_ == 0)
{
lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_820_; 
lean_dec(v_x_807_);
v___x_817_ = lean_array_push(v_ks_810_, v_x_808_);
v___x_818_ = lean_array_push(v_vs_811_, v_x_809_);
if (v_isShared_814_ == 0)
{
lean_ctor_set(v___x_813_, 1, v___x_818_);
lean_ctor_set(v___x_813_, 0, v___x_817_);
v___x_820_ = v___x_813_;
goto v_reusejp_819_;
}
else
{
lean_object* v_reuseFailAlloc_821_; 
v_reuseFailAlloc_821_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_821_, 0, v___x_817_);
lean_ctor_set(v_reuseFailAlloc_821_, 1, v___x_818_);
v___x_820_ = v_reuseFailAlloc_821_;
goto v_reusejp_819_;
}
v_reusejp_819_:
{
return v___x_820_;
}
}
else
{
lean_object* v_k_x27_822_; uint8_t v___x_823_; 
v_k_x27_822_ = lean_array_fget_borrowed(v_ks_810_, v_x_807_);
v___x_823_ = l_Lean_instBEqMVarId_beq(v_x_808_, v_k_x27_822_);
if (v___x_823_ == 0)
{
lean_object* v___x_825_; 
if (v_isShared_814_ == 0)
{
v___x_825_ = v___x_813_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_829_; 
v_reuseFailAlloc_829_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_829_, 0, v_ks_810_);
lean_ctor_set(v_reuseFailAlloc_829_, 1, v_vs_811_);
v___x_825_ = v_reuseFailAlloc_829_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_826_ = lean_unsigned_to_nat(1u);
v___x_827_ = lean_nat_add(v_x_807_, v___x_826_);
lean_dec(v_x_807_);
v_x_806_ = v___x_825_;
v_x_807_ = v___x_827_;
goto _start;
}
}
else
{
lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_833_; 
v___x_830_ = lean_array_fset(v_ks_810_, v_x_807_, v_x_808_);
v___x_831_ = lean_array_fset(v_vs_811_, v_x_807_, v_x_809_);
lean_dec(v_x_807_);
if (v_isShared_814_ == 0)
{
lean_ctor_set(v___x_813_, 1, v___x_831_);
lean_ctor_set(v___x_813_, 0, v___x_830_);
v___x_833_ = v___x_813_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_830_);
lean_ctor_set(v_reuseFailAlloc_834_, 1, v___x_831_);
v___x_833_ = v_reuseFailAlloc_834_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
return v___x_833_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9___redArg(lean_object* v_n_836_, lean_object* v_k_837_, lean_object* v_v_838_){
_start:
{
lean_object* v___x_839_; lean_object* v___x_840_; 
v___x_839_ = lean_unsigned_to_nat(0u);
v___x_840_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9_spec__10___redArg(v_n_836_, v___x_839_, v_k_837_, v_v_838_);
return v___x_840_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_841_; 
v___x_841_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(lean_object* v_x_842_, size_t v_x_843_, size_t v_x_844_, lean_object* v_x_845_, lean_object* v_x_846_){
_start:
{
if (lean_obj_tag(v_x_842_) == 0)
{
lean_object* v_es_847_; size_t v___x_848_; size_t v___x_849_; lean_object* v_j_850_; lean_object* v___x_851_; uint8_t v___x_852_; 
v_es_847_ = lean_ctor_get(v_x_842_, 0);
v___x_848_ = ((size_t)31ULL);
v___x_849_ = lean_usize_land(v_x_843_, v___x_848_);
v_j_850_ = lean_usize_to_nat(v___x_849_);
v___x_851_ = lean_array_get_size(v_es_847_);
v___x_852_ = lean_nat_dec_lt(v_j_850_, v___x_851_);
if (v___x_852_ == 0)
{
lean_dec(v_j_850_);
lean_dec(v_x_846_);
lean_dec(v_x_845_);
return v_x_842_;
}
else
{
lean_object* v___x_854_; uint8_t v_isShared_855_; uint8_t v_isSharedCheck_891_; 
lean_inc_ref(v_es_847_);
v_isSharedCheck_891_ = !lean_is_exclusive(v_x_842_);
if (v_isSharedCheck_891_ == 0)
{
lean_object* v_unused_892_; 
v_unused_892_ = lean_ctor_get(v_x_842_, 0);
lean_dec(v_unused_892_);
v___x_854_ = v_x_842_;
v_isShared_855_ = v_isSharedCheck_891_;
goto v_resetjp_853_;
}
else
{
lean_dec(v_x_842_);
v___x_854_ = lean_box(0);
v_isShared_855_ = v_isSharedCheck_891_;
goto v_resetjp_853_;
}
v_resetjp_853_:
{
lean_object* v_v_856_; lean_object* v___x_857_; lean_object* v_xs_x27_858_; lean_object* v___y_860_; 
v_v_856_ = lean_array_fget(v_es_847_, v_j_850_);
v___x_857_ = lean_box(0);
v_xs_x27_858_ = lean_array_fset(v_es_847_, v_j_850_, v___x_857_);
switch(lean_obj_tag(v_v_856_))
{
case 0:
{
lean_object* v_key_865_; lean_object* v_val_866_; lean_object* v___x_868_; uint8_t v_isShared_869_; uint8_t v_isSharedCheck_876_; 
v_key_865_ = lean_ctor_get(v_v_856_, 0);
v_val_866_ = lean_ctor_get(v_v_856_, 1);
v_isSharedCheck_876_ = !lean_is_exclusive(v_v_856_);
if (v_isSharedCheck_876_ == 0)
{
v___x_868_ = v_v_856_;
v_isShared_869_ = v_isSharedCheck_876_;
goto v_resetjp_867_;
}
else
{
lean_inc(v_val_866_);
lean_inc(v_key_865_);
lean_dec(v_v_856_);
v___x_868_ = lean_box(0);
v_isShared_869_ = v_isSharedCheck_876_;
goto v_resetjp_867_;
}
v_resetjp_867_:
{
uint8_t v___x_870_; 
v___x_870_ = l_Lean_instBEqMVarId_beq(v_x_845_, v_key_865_);
if (v___x_870_ == 0)
{
lean_object* v___x_871_; lean_object* v___x_872_; 
lean_del_object(v___x_868_);
v___x_871_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_865_, v_val_866_, v_x_845_, v_x_846_);
v___x_872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_872_, 0, v___x_871_);
v___y_860_ = v___x_872_;
goto v___jp_859_;
}
else
{
lean_object* v___x_874_; 
lean_dec(v_val_866_);
lean_dec(v_key_865_);
if (v_isShared_869_ == 0)
{
lean_ctor_set(v___x_868_, 1, v_x_846_);
lean_ctor_set(v___x_868_, 0, v_x_845_);
v___x_874_ = v___x_868_;
goto v_reusejp_873_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v_x_845_);
lean_ctor_set(v_reuseFailAlloc_875_, 1, v_x_846_);
v___x_874_ = v_reuseFailAlloc_875_;
goto v_reusejp_873_;
}
v_reusejp_873_:
{
v___y_860_ = v___x_874_;
goto v___jp_859_;
}
}
}
}
case 1:
{
lean_object* v_node_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_889_; 
v_node_877_ = lean_ctor_get(v_v_856_, 0);
v_isSharedCheck_889_ = !lean_is_exclusive(v_v_856_);
if (v_isSharedCheck_889_ == 0)
{
v___x_879_ = v_v_856_;
v_isShared_880_ = v_isSharedCheck_889_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_node_877_);
lean_dec(v_v_856_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_889_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
size_t v___x_881_; size_t v___x_882_; size_t v___x_883_; size_t v___x_884_; lean_object* v___x_885_; lean_object* v___x_887_; 
v___x_881_ = ((size_t)5ULL);
v___x_882_ = lean_usize_shift_right(v_x_843_, v___x_881_);
v___x_883_ = ((size_t)1ULL);
v___x_884_ = lean_usize_add(v_x_844_, v___x_883_);
v___x_885_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(v_node_877_, v___x_882_, v___x_884_, v_x_845_, v_x_846_);
if (v_isShared_880_ == 0)
{
lean_ctor_set(v___x_879_, 0, v___x_885_);
v___x_887_ = v___x_879_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_888_; 
v_reuseFailAlloc_888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_888_, 0, v___x_885_);
v___x_887_ = v_reuseFailAlloc_888_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
v___y_860_ = v___x_887_;
goto v___jp_859_;
}
}
}
default: 
{
lean_object* v___x_890_; 
v___x_890_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_890_, 0, v_x_845_);
lean_ctor_set(v___x_890_, 1, v_x_846_);
v___y_860_ = v___x_890_;
goto v___jp_859_;
}
}
v___jp_859_:
{
lean_object* v___x_861_; lean_object* v___x_863_; 
v___x_861_ = lean_array_fset(v_xs_x27_858_, v_j_850_, v___y_860_);
lean_dec(v_j_850_);
if (v_isShared_855_ == 0)
{
lean_ctor_set(v___x_854_, 0, v___x_861_);
v___x_863_ = v___x_854_;
goto v_reusejp_862_;
}
else
{
lean_object* v_reuseFailAlloc_864_; 
v_reuseFailAlloc_864_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_864_, 0, v___x_861_);
v___x_863_ = v_reuseFailAlloc_864_;
goto v_reusejp_862_;
}
v_reusejp_862_:
{
return v___x_863_;
}
}
}
}
}
else
{
lean_object* v_ks_893_; lean_object* v_vs_894_; lean_object* v___x_896_; uint8_t v_isShared_897_; uint8_t v_isSharedCheck_914_; 
v_ks_893_ = lean_ctor_get(v_x_842_, 0);
v_vs_894_ = lean_ctor_get(v_x_842_, 1);
v_isSharedCheck_914_ = !lean_is_exclusive(v_x_842_);
if (v_isSharedCheck_914_ == 0)
{
v___x_896_ = v_x_842_;
v_isShared_897_ = v_isSharedCheck_914_;
goto v_resetjp_895_;
}
else
{
lean_inc(v_vs_894_);
lean_inc(v_ks_893_);
lean_dec(v_x_842_);
v___x_896_ = lean_box(0);
v_isShared_897_ = v_isSharedCheck_914_;
goto v_resetjp_895_;
}
v_resetjp_895_:
{
lean_object* v___x_899_; 
if (v_isShared_897_ == 0)
{
v___x_899_ = v___x_896_;
goto v_reusejp_898_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v_ks_893_);
lean_ctor_set(v_reuseFailAlloc_913_, 1, v_vs_894_);
v___x_899_ = v_reuseFailAlloc_913_;
goto v_reusejp_898_;
}
v_reusejp_898_:
{
lean_object* v_newNode_900_; uint8_t v___y_902_; size_t v___x_908_; uint8_t v___x_909_; 
v_newNode_900_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9___redArg(v___x_899_, v_x_845_, v_x_846_);
v___x_908_ = ((size_t)7ULL);
v___x_909_ = lean_usize_dec_le(v___x_908_, v_x_844_);
if (v___x_909_ == 0)
{
lean_object* v___x_910_; lean_object* v___x_911_; uint8_t v___x_912_; 
v___x_910_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_900_);
v___x_911_ = lean_unsigned_to_nat(4u);
v___x_912_ = lean_nat_dec_lt(v___x_910_, v___x_911_);
lean_dec(v___x_910_);
v___y_902_ = v___x_912_;
goto v___jp_901_;
}
else
{
v___y_902_ = v___x_909_;
goto v___jp_901_;
}
v___jp_901_:
{
if (v___y_902_ == 0)
{
lean_object* v_ks_903_; lean_object* v_vs_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; 
v_ks_903_ = lean_ctor_get(v_newNode_900_, 0);
lean_inc_ref(v_ks_903_);
v_vs_904_ = lean_ctor_get(v_newNode_900_, 1);
lean_inc_ref(v_vs_904_);
lean_dec_ref(v_newNode_900_);
v___x_905_ = lean_unsigned_to_nat(0u);
v___x_906_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___closed__0);
v___x_907_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg(v_x_844_, v_ks_903_, v_vs_904_, v___x_905_, v___x_906_);
lean_dec_ref(v_vs_904_);
lean_dec_ref(v_ks_903_);
return v___x_907_;
}
else
{
return v_newNode_900_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg(size_t v_depth_915_, lean_object* v_keys_916_, lean_object* v_vals_917_, lean_object* v_i_918_, lean_object* v_entries_919_){
_start:
{
lean_object* v___x_920_; uint8_t v___x_921_; 
v___x_920_ = lean_array_get_size(v_keys_916_);
v___x_921_ = lean_nat_dec_lt(v_i_918_, v___x_920_);
if (v___x_921_ == 0)
{
lean_dec(v_i_918_);
return v_entries_919_;
}
else
{
lean_object* v_k_922_; lean_object* v_v_923_; uint64_t v___x_924_; size_t v_h_925_; size_t v___x_926_; lean_object* v___x_927_; size_t v___x_928_; size_t v___x_929_; size_t v___x_930_; size_t v_h_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v_k_922_ = lean_array_fget_borrowed(v_keys_916_, v_i_918_);
v_v_923_ = lean_array_fget_borrowed(v_vals_917_, v_i_918_);
v___x_924_ = l_Lean_instHashableMVarId_hash(v_k_922_);
v_h_925_ = lean_uint64_to_usize(v___x_924_);
v___x_926_ = ((size_t)5ULL);
v___x_927_ = lean_unsigned_to_nat(1u);
v___x_928_ = ((size_t)1ULL);
v___x_929_ = lean_usize_sub(v_depth_915_, v___x_928_);
v___x_930_ = lean_usize_mul(v___x_926_, v___x_929_);
v_h_931_ = lean_usize_shift_right(v_h_925_, v___x_930_);
v___x_932_ = lean_nat_add(v_i_918_, v___x_927_);
lean_dec(v_i_918_);
lean_inc(v_v_923_);
lean_inc(v_k_922_);
v___x_933_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(v_entries_919_, v_h_931_, v_depth_915_, v_k_922_, v_v_923_);
v_i_918_ = v___x_932_;
v_entries_919_ = v___x_933_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg___boxed(lean_object* v_depth_935_, lean_object* v_keys_936_, lean_object* v_vals_937_, lean_object* v_i_938_, lean_object* v_entries_939_){
_start:
{
size_t v_depth_boxed_940_; lean_object* v_res_941_; 
v_depth_boxed_940_ = lean_unbox_usize(v_depth_935_);
lean_dec(v_depth_935_);
v_res_941_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg(v_depth_boxed_940_, v_keys_936_, v_vals_937_, v_i_938_, v_entries_939_);
lean_dec_ref(v_vals_937_);
lean_dec_ref(v_keys_936_);
return v_res_941_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_x_942_, lean_object* v_x_943_, lean_object* v_x_944_, lean_object* v_x_945_, lean_object* v_x_946_){
_start:
{
size_t v_x_18863__boxed_947_; size_t v_x_18864__boxed_948_; lean_object* v_res_949_; 
v_x_18863__boxed_947_ = lean_unbox_usize(v_x_943_);
lean_dec(v_x_943_);
v_x_18864__boxed_948_ = lean_unbox_usize(v_x_944_);
lean_dec(v_x_944_);
v_res_949_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(v_x_942_, v_x_18863__boxed_947_, v_x_18864__boxed_948_, v_x_945_, v_x_946_);
return v_res_949_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3___redArg(lean_object* v_x_950_, lean_object* v_x_951_, lean_object* v_x_952_){
_start:
{
uint64_t v___x_953_; size_t v___x_954_; size_t v___x_955_; lean_object* v___x_956_; 
v___x_953_ = l_Lean_instHashableMVarId_hash(v_x_951_);
v___x_954_ = lean_uint64_to_usize(v___x_953_);
v___x_955_ = ((size_t)1ULL);
v___x_956_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(v_x_950_, v___x_954_, v___x_955_, v_x_951_, v_x_952_);
return v___x_956_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(lean_object* v_mvarId_957_, lean_object* v_val_958_, lean_object* v___y_959_){
_start:
{
lean_object* v___x_961_; lean_object* v_mctx_962_; lean_object* v_cache_963_; lean_object* v_zetaDeltaFVarIds_964_; lean_object* v_postponed_965_; lean_object* v_diag_966_; lean_object* v___x_968_; uint8_t v_isShared_969_; uint8_t v_isSharedCheck_994_; 
v___x_961_ = lean_st_ref_take(v___y_959_);
v_mctx_962_ = lean_ctor_get(v___x_961_, 0);
v_cache_963_ = lean_ctor_get(v___x_961_, 1);
v_zetaDeltaFVarIds_964_ = lean_ctor_get(v___x_961_, 2);
v_postponed_965_ = lean_ctor_get(v___x_961_, 3);
v_diag_966_ = lean_ctor_get(v___x_961_, 4);
v_isSharedCheck_994_ = !lean_is_exclusive(v___x_961_);
if (v_isSharedCheck_994_ == 0)
{
v___x_968_ = v___x_961_;
v_isShared_969_ = v_isSharedCheck_994_;
goto v_resetjp_967_;
}
else
{
lean_inc(v_diag_966_);
lean_inc(v_postponed_965_);
lean_inc(v_zetaDeltaFVarIds_964_);
lean_inc(v_cache_963_);
lean_inc(v_mctx_962_);
lean_dec(v___x_961_);
v___x_968_ = lean_box(0);
v_isShared_969_ = v_isSharedCheck_994_;
goto v_resetjp_967_;
}
v_resetjp_967_:
{
lean_object* v_depth_970_; lean_object* v_levelAssignDepth_971_; lean_object* v_lmvarCounter_972_; lean_object* v_mvarCounter_973_; lean_object* v_lDecls_974_; lean_object* v_decls_975_; lean_object* v_userNames_976_; lean_object* v_lAssignment_977_; lean_object* v_eAssignment_978_; lean_object* v_dAssignment_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_993_; 
v_depth_970_ = lean_ctor_get(v_mctx_962_, 0);
v_levelAssignDepth_971_ = lean_ctor_get(v_mctx_962_, 1);
v_lmvarCounter_972_ = lean_ctor_get(v_mctx_962_, 2);
v_mvarCounter_973_ = lean_ctor_get(v_mctx_962_, 3);
v_lDecls_974_ = lean_ctor_get(v_mctx_962_, 4);
v_decls_975_ = lean_ctor_get(v_mctx_962_, 5);
v_userNames_976_ = lean_ctor_get(v_mctx_962_, 6);
v_lAssignment_977_ = lean_ctor_get(v_mctx_962_, 7);
v_eAssignment_978_ = lean_ctor_get(v_mctx_962_, 8);
v_dAssignment_979_ = lean_ctor_get(v_mctx_962_, 9);
v_isSharedCheck_993_ = !lean_is_exclusive(v_mctx_962_);
if (v_isSharedCheck_993_ == 0)
{
v___x_981_ = v_mctx_962_;
v_isShared_982_ = v_isSharedCheck_993_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_dAssignment_979_);
lean_inc(v_eAssignment_978_);
lean_inc(v_lAssignment_977_);
lean_inc(v_userNames_976_);
lean_inc(v_decls_975_);
lean_inc(v_lDecls_974_);
lean_inc(v_mvarCounter_973_);
lean_inc(v_lmvarCounter_972_);
lean_inc(v_levelAssignDepth_971_);
lean_inc(v_depth_970_);
lean_dec(v_mctx_962_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_993_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v___x_983_; lean_object* v___x_985_; 
v___x_983_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3___redArg(v_eAssignment_978_, v_mvarId_957_, v_val_958_);
if (v_isShared_982_ == 0)
{
lean_ctor_set(v___x_981_, 8, v___x_983_);
v___x_985_ = v___x_981_;
goto v_reusejp_984_;
}
else
{
lean_object* v_reuseFailAlloc_992_; 
v_reuseFailAlloc_992_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_992_, 0, v_depth_970_);
lean_ctor_set(v_reuseFailAlloc_992_, 1, v_levelAssignDepth_971_);
lean_ctor_set(v_reuseFailAlloc_992_, 2, v_lmvarCounter_972_);
lean_ctor_set(v_reuseFailAlloc_992_, 3, v_mvarCounter_973_);
lean_ctor_set(v_reuseFailAlloc_992_, 4, v_lDecls_974_);
lean_ctor_set(v_reuseFailAlloc_992_, 5, v_decls_975_);
lean_ctor_set(v_reuseFailAlloc_992_, 6, v_userNames_976_);
lean_ctor_set(v_reuseFailAlloc_992_, 7, v_lAssignment_977_);
lean_ctor_set(v_reuseFailAlloc_992_, 8, v___x_983_);
lean_ctor_set(v_reuseFailAlloc_992_, 9, v_dAssignment_979_);
v___x_985_ = v_reuseFailAlloc_992_;
goto v_reusejp_984_;
}
v_reusejp_984_:
{
lean_object* v___x_987_; 
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 0, v___x_985_);
v___x_987_ = v___x_968_;
goto v_reusejp_986_;
}
else
{
lean_object* v_reuseFailAlloc_991_; 
v_reuseFailAlloc_991_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_991_, 0, v___x_985_);
lean_ctor_set(v_reuseFailAlloc_991_, 1, v_cache_963_);
lean_ctor_set(v_reuseFailAlloc_991_, 2, v_zetaDeltaFVarIds_964_);
lean_ctor_set(v_reuseFailAlloc_991_, 3, v_postponed_965_);
lean_ctor_set(v_reuseFailAlloc_991_, 4, v_diag_966_);
v___x_987_ = v_reuseFailAlloc_991_;
goto v_reusejp_986_;
}
v_reusejp_986_:
{
lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_988_ = lean_st_ref_set(v___y_959_, v___x_987_);
v___x_989_ = lean_box(0);
v___x_990_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_990_, 0, v___x_989_);
return v___x_990_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg___boxed(lean_object* v_mvarId_995_, lean_object* v_val_996_, lean_object* v___y_997_, lean_object* v___y_998_){
_start:
{
lean_object* v_res_999_; 
v_res_999_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(v_mvarId_995_, v_val_996_, v___y_997_);
lean_dec(v___y_997_);
return v_res_999_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__0(lean_object* v_a_1000_, lean_object* v_a_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_){
_start:
{
lean_object* v___x_1011_; 
v___x_1011_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_a_1000_, v_a_1001_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_);
return v___x_1011_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__0___boxed(lean_object* v_a_1012_, lean_object* v_a_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
lean_object* v_res_1023_; 
v_res_1023_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__0(v_a_1012_, v_a_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_);
lean_dec(v___y_1021_);
lean_dec_ref(v___y_1020_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
return v_res_1023_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2(lean_object* v_tail_1024_, lean_object* v___x_1025_, lean_object* v_head_1026_, lean_object* v_____r_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v___x_1037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1037_, 0, v_tail_1024_);
lean_ctor_set(v___x_1037_, 1, v___x_1025_);
v___x_1038_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1038_, 0, v_head_1026_);
lean_ctor_set(v___x_1038_, 1, v___x_1037_);
v___x_1039_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1039_, 0, v___x_1038_);
v___x_1040_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1040_, 0, v___x_1039_);
return v___x_1040_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2___boxed(lean_object* v_tail_1041_, lean_object* v___x_1042_, lean_object* v_head_1043_, lean_object* v_____r_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_){
_start:
{
lean_object* v_res_1054_; 
v_res_1054_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2(v_tail_1041_, v___x_1042_, v_head_1043_, v_____r_1044_, v___y_1045_, v___y_1046_, v___y_1047_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_);
lean_dec(v___y_1052_);
lean_dec_ref(v___y_1051_);
lean_dec(v___y_1050_);
lean_dec_ref(v___y_1049_);
lean_dec(v___y_1048_);
lean_dec_ref(v___y_1047_);
lean_dec(v___y_1046_);
lean_dec_ref(v___y_1045_);
return v_res_1054_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__1(uint8_t v___x_1055_, lean_object* v_a_1056_, lean_object* v_a_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_){
_start:
{
lean_object* v_keyedConfig_1067_; uint8_t v_trackZetaDelta_1068_; lean_object* v_zetaDeltaSet_1069_; lean_object* v_lctx_1070_; lean_object* v_localInstances_1071_; lean_object* v_defEqCtx_x3f_1072_; lean_object* v_synthPendingDepth_1073_; lean_object* v_customCanUnfoldPredicate_x3f_1074_; uint8_t v_univApprox_1075_; uint8_t v_inTypeClassResolution_1076_; uint8_t v_cacheInferType_1077_; lean_object* v___x_1079_; uint8_t v_isShared_1080_; uint8_t v_isSharedCheck_1094_; 
v_keyedConfig_1067_ = lean_ctor_get(v___y_1062_, 0);
v_trackZetaDelta_1068_ = lean_ctor_get_uint8(v___y_1062_, sizeof(void*)*7);
v_zetaDeltaSet_1069_ = lean_ctor_get(v___y_1062_, 1);
v_lctx_1070_ = lean_ctor_get(v___y_1062_, 2);
v_localInstances_1071_ = lean_ctor_get(v___y_1062_, 3);
v_defEqCtx_x3f_1072_ = lean_ctor_get(v___y_1062_, 4);
v_synthPendingDepth_1073_ = lean_ctor_get(v___y_1062_, 5);
v_customCanUnfoldPredicate_x3f_1074_ = lean_ctor_get(v___y_1062_, 6);
v_univApprox_1075_ = lean_ctor_get_uint8(v___y_1062_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1076_ = lean_ctor_get_uint8(v___y_1062_, sizeof(void*)*7 + 2);
v_cacheInferType_1077_ = lean_ctor_get_uint8(v___y_1062_, sizeof(void*)*7 + 3);
v_isSharedCheck_1094_ = !lean_is_exclusive(v___y_1062_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1079_ = v___y_1062_;
v_isShared_1080_ = v_isSharedCheck_1094_;
goto v_resetjp_1078_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1074_);
lean_inc(v_synthPendingDepth_1073_);
lean_inc(v_defEqCtx_x3f_1072_);
lean_inc(v_localInstances_1071_);
lean_inc(v_lctx_1070_);
lean_inc(v_zetaDeltaSet_1069_);
lean_inc(v_keyedConfig_1067_);
lean_dec(v___y_1062_);
v___x_1079_ = lean_box(0);
v_isShared_1080_ = v_isSharedCheck_1094_;
goto v_resetjp_1078_;
}
v_resetjp_1078_:
{
lean_object* v___x_1081_; lean_object* v___x_1083_; 
v___x_1081_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1055_, v_keyedConfig_1067_);
if (v_isShared_1080_ == 0)
{
lean_ctor_set(v___x_1079_, 0, v___x_1081_);
v___x_1083_ = v___x_1079_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1093_; 
v_reuseFailAlloc_1093_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1093_, 0, v___x_1081_);
lean_ctor_set(v_reuseFailAlloc_1093_, 1, v_zetaDeltaSet_1069_);
lean_ctor_set(v_reuseFailAlloc_1093_, 2, v_lctx_1070_);
lean_ctor_set(v_reuseFailAlloc_1093_, 3, v_localInstances_1071_);
lean_ctor_set(v_reuseFailAlloc_1093_, 4, v_defEqCtx_x3f_1072_);
lean_ctor_set(v_reuseFailAlloc_1093_, 5, v_synthPendingDepth_1073_);
lean_ctor_set(v_reuseFailAlloc_1093_, 6, v_customCanUnfoldPredicate_x3f_1074_);
lean_ctor_set_uint8(v_reuseFailAlloc_1093_, sizeof(void*)*7, v_trackZetaDelta_1068_);
lean_ctor_set_uint8(v_reuseFailAlloc_1093_, sizeof(void*)*7 + 1, v_univApprox_1075_);
lean_ctor_set_uint8(v_reuseFailAlloc_1093_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1076_);
lean_ctor_set_uint8(v_reuseFailAlloc_1093_, sizeof(void*)*7 + 3, v_cacheInferType_1077_);
v___x_1083_ = v_reuseFailAlloc_1093_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
lean_object* v___x_1084_; 
v___x_1084_ = l_Lean_Meta_isExprDefEq(v_a_1056_, v_a_1057_, v___x_1083_, v___y_1063_, v___y_1064_, v___y_1065_);
lean_dec_ref(v___x_1083_);
if (lean_obj_tag(v___x_1084_) == 0)
{
lean_object* v_a_1085_; lean_object* v___x_1087_; uint8_t v_isShared_1088_; uint8_t v_isSharedCheck_1092_; 
v_a_1085_ = lean_ctor_get(v___x_1084_, 0);
v_isSharedCheck_1092_ = !lean_is_exclusive(v___x_1084_);
if (v_isSharedCheck_1092_ == 0)
{
v___x_1087_ = v___x_1084_;
v_isShared_1088_ = v_isSharedCheck_1092_;
goto v_resetjp_1086_;
}
else
{
lean_inc(v_a_1085_);
lean_dec(v___x_1084_);
v___x_1087_ = lean_box(0);
v_isShared_1088_ = v_isSharedCheck_1092_;
goto v_resetjp_1086_;
}
v_resetjp_1086_:
{
lean_object* v___x_1090_; 
if (v_isShared_1088_ == 0)
{
v___x_1090_ = v___x_1087_;
goto v_reusejp_1089_;
}
else
{
lean_object* v_reuseFailAlloc_1091_; 
v_reuseFailAlloc_1091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1091_, 0, v_a_1085_);
v___x_1090_ = v_reuseFailAlloc_1091_;
goto v_reusejp_1089_;
}
v_reusejp_1089_:
{
return v___x_1090_;
}
}
}
else
{
return v___x_1084_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__1___boxed(lean_object* v___x_1095_, lean_object* v_a_1096_, lean_object* v_a_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_){
_start:
{
uint8_t v___x_19158__boxed_1107_; lean_object* v_res_1108_; 
v___x_19158__boxed_1107_ = lean_unbox(v___x_1095_);
v_res_1108_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__1(v___x_19158__boxed_1107_, v_a_1096_, v_a_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_, v___y_1105_);
lean_dec(v___y_1105_);
lean_dec_ref(v___y_1104_);
lean_dec(v___y_1103_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
return v_res_1108_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0_spec__0(lean_object* v_as_1109_, size_t v_i_1110_, size_t v_stop_1111_, lean_object* v_b_1112_){
_start:
{
uint8_t v___x_1113_; 
v___x_1113_ = lean_usize_dec_eq(v_i_1110_, v_stop_1111_);
if (v___x_1113_ == 0)
{
size_t v___x_1114_; size_t v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; 
v___x_1114_ = ((size_t)1ULL);
v___x_1115_ = lean_usize_sub(v_i_1110_, v___x_1114_);
v___x_1116_ = lean_array_uget_borrowed(v_as_1109_, v___x_1115_);
lean_inc(v___x_1116_);
v___x_1117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1117_, 0, v___x_1116_);
lean_ctor_set(v___x_1117_, 1, v_b_1112_);
v_i_1110_ = v___x_1115_;
v_b_1112_ = v___x_1117_;
goto _start;
}
else
{
return v_b_1112_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0_spec__0___boxed(lean_object* v_as_1119_, lean_object* v_i_1120_, lean_object* v_stop_1121_, lean_object* v_b_1122_){
_start:
{
size_t v_i_boxed_1123_; size_t v_stop_boxed_1124_; lean_object* v_res_1125_; 
v_i_boxed_1123_ = lean_unbox_usize(v_i_1120_);
lean_dec(v_i_1120_);
v_stop_boxed_1124_ = lean_unbox_usize(v_stop_1121_);
lean_dec(v_stop_1121_);
v_res_1125_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0_spec__0(v_as_1119_, v_i_boxed_1123_, v_stop_boxed_1124_, v_b_1122_);
lean_dec_ref(v_as_1119_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0(lean_object* v_l_1126_, lean_object* v_a_1127_, lean_object* v_a_1128_, lean_object* v_a_1129_){
_start:
{
if (lean_obj_tag(v_a_1128_) == 0)
{
lean_dec_ref(v_a_1129_);
lean_inc(v_l_1126_);
return v_l_1126_;
}
else
{
lean_object* v_head_1130_; lean_object* v_tail_1131_; uint8_t v___x_1132_; 
v_head_1130_ = lean_ctor_get(v_a_1128_, 0);
lean_inc(v_head_1130_);
v_tail_1131_ = lean_ctor_get(v_a_1128_, 1);
lean_inc(v_tail_1131_);
lean_dec_ref_known(v_a_1128_, 2);
v___x_1132_ = l_Lean_instBEqMVarId_beq(v_head_1130_, v_a_1127_);
if (v___x_1132_ == 0)
{
lean_object* v___x_1133_; 
v___x_1133_ = lean_array_push(v_a_1129_, v_head_1130_);
v_a_1128_ = v_tail_1131_;
v_a_1129_ = v___x_1133_;
goto _start;
}
else
{
lean_object* v___x_1135_; lean_object* v___x_1136_; uint8_t v___x_1137_; 
lean_dec(v_head_1130_);
v___x_1135_ = lean_array_get_size(v_a_1129_);
v___x_1136_ = lean_unsigned_to_nat(0u);
v___x_1137_ = lean_nat_dec_lt(v___x_1136_, v___x_1135_);
if (v___x_1137_ == 0)
{
lean_dec_ref(v_a_1129_);
return v_tail_1131_;
}
else
{
size_t v___x_1138_; size_t v___x_1139_; lean_object* v___x_1140_; 
v___x_1138_ = lean_usize_of_nat(v___x_1135_);
v___x_1139_ = ((size_t)0ULL);
v___x_1140_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0_spec__0(v_a_1129_, v___x_1138_, v___x_1139_, v_tail_1131_);
lean_dec_ref(v_a_1129_);
return v___x_1140_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0___boxed(lean_object* v_l_1141_, lean_object* v_a_1142_, lean_object* v_a_1143_, lean_object* v_a_1144_){
_start:
{
lean_object* v_res_1145_; 
v_res_1145_ = lp_batteries___private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0(v_l_1141_, v_a_1142_, v_a_1143_, v_a_1144_);
lean_dec(v_a_1142_);
lean_dec(v_l_1141_);
return v_res_1145_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg(lean_object* v_gs_1171_, lean_object* v_patt_x3f_1172_, lean_object* v_renameI_1173_, lean_object* v_as_x27_1174_, lean_object* v_b_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_){
_start:
{
if (lean_obj_tag(v_as_x27_1174_) == 0)
{
lean_object* v___x_1185_; 
lean_dec_ref(v_renameI_1173_);
lean_dec(v_patt_x3f_1172_);
lean_dec(v_gs_1171_);
v___x_1185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1185_, 0, v_b_1175_);
return v___x_1185_;
}
else
{
lean_object* v_head_1186_; lean_object* v_tail_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; 
lean_dec_ref(v_b_1175_);
v_head_1186_ = lean_ctor_get(v_as_x27_1174_, 0);
v_tail_1187_ = lean_ctor_get(v_as_x27_1174_, 1);
v___x_1188_ = lean_box(0);
v___x_1189_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__0));
lean_inc(v_gs_1171_);
v___x_1190_ = lp_batteries___private_Init_Data_List_Impl_0__List_eraseTR_go___at___00Batteries_Tactic_findGoalOfPatt_spec__0(v_gs_1171_, v_head_1186_, v_gs_1171_, v___x_1189_);
if (lean_obj_tag(v_patt_x3f_1172_) == 1)
{
lean_object* v_val_1191_; lean_object* v___x_1192_; 
v_val_1191_ = lean_ctor_get(v_patt_x3f_1172_, 0);
v___x_1192_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1177_, v___y_1179_, v___y_1181_, v___y_1183_);
if (lean_obj_tag(v___x_1192_) == 0)
{
lean_object* v_a_1193_; lean_object* v___x_1194_; 
v_a_1193_ = lean_ctor_get(v___x_1192_, 0);
lean_inc(v_a_1193_);
lean_dec_ref_known(v___x_1192_, 1);
v___x_1194_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1177_, v___y_1179_, v___y_1181_, v___y_1183_);
if (lean_obj_tag(v___x_1194_) == 0)
{
lean_object* v_a_1195_; lean_object* v___x_1197_; uint8_t v_isShared_1198_; uint8_t v_isSharedCheck_1315_; 
v_a_1195_ = lean_ctor_get(v___x_1194_, 0);
v_isSharedCheck_1315_ = !lean_is_exclusive(v___x_1194_);
if (v_isSharedCheck_1315_ == 0)
{
v___x_1197_ = v___x_1194_;
v_isShared_1198_ = v_isSharedCheck_1315_;
goto v_resetjp_1196_;
}
else
{
lean_inc(v_a_1195_);
lean_dec(v___x_1194_);
v___x_1197_ = lean_box(0);
v_isShared_1198_ = v_isSharedCheck_1315_;
goto v_resetjp_1196_;
}
v_resetjp_1196_:
{
lean_object* v___x_1199_; lean_object* v___y_1201_; uint8_t v___y_1202_; lean_object* v_a_1226_; lean_object* v___y_1230_; lean_object* v___x_1250_; 
v___x_1199_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__1));
lean_inc_ref(v_renameI_1173_);
lean_inc(v_head_1186_);
v___x_1250_ = l_Lean_Elab_Tactic_renameInaccessibles(v_head_1186_, v_renameI_1173_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1250_) == 0)
{
lean_object* v_a_1251_; lean_object* v___x_1252_; 
v_a_1251_ = lean_ctor_get(v___x_1250_, 0);
lean_inc_n(v_a_1251_, 2);
lean_dec_ref_known(v___x_1250_, 1);
v___x_1252_ = l_Lean_MVarId_getType(v_a_1251_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1252_) == 0)
{
lean_object* v_a_1253_; lean_object* v___x_1254_; 
v_a_1253_ = lean_ctor_get(v___x_1252_, 0);
lean_inc(v_a_1253_);
lean_dec_ref_known(v___x_1252_, 1);
lean_inc(v_a_1251_);
v___x_1254_ = l_Lean_MVarId_getTag(v_a_1251_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1254_) == 0)
{
lean_object* v_a_1255_; lean_object* v___f_1256_; lean_object* v___x_1257_; 
v_a_1255_ = lean_ctor_get(v___x_1254_, 0);
lean_inc(v_a_1255_);
lean_dec_ref_known(v___x_1254_, 1);
v___f_1256_ = lean_alloc_closure((void*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__0___boxed), 11, 2);
lean_closure_set(v___f_1256_, 0, v_a_1253_);
lean_closure_set(v___f_1256_, 1, v_a_1255_);
lean_inc(v_a_1251_);
v___x_1257_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg(v_a_1251_, v___f_1256_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1257_) == 0)
{
lean_object* v_a_1258_; lean_object* v_ref_1259_; uint8_t v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; 
v_a_1258_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_a_1258_);
lean_dec_ref_known(v___x_1257_, 1);
v_ref_1259_ = lean_ctor_get(v___y_1182_, 5);
v___x_1260_ = 0;
v___x_1261_ = l_Lean_SourceInfo_fromRef(v_ref_1259_, v___x_1260_);
v___x_1262_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__3));
v___x_1263_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__4));
lean_inc_n(v___x_1261_, 8);
v___x_1264_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1261_);
lean_ctor_set(v___x_1264_, 1, v___x_1263_);
v___x_1265_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__5));
v___x_1266_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__6));
v___x_1267_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1267_, 0, v___x_1261_);
lean_ctor_set(v___x_1267_, 1, v___x_1265_);
v___x_1268_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__8));
v___x_1269_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__9));
v___x_1270_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1261_);
lean_ctor_set(v___x_1270_, 1, v___x_1269_);
v___x_1271_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10));
v___x_1272_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__11));
v___x_1273_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1273_, 0, v___x_1261_);
lean_ctor_set(v___x_1273_, 1, v___x_1272_);
v___x_1274_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12));
v___x_1275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1275_, 0, v___x_1261_);
lean_ctor_set(v___x_1275_, 1, v___x_1274_);
v___x_1276_ = l_Lean_Syntax_node2(v___x_1261_, v___x_1271_, v___x_1273_, v___x_1275_);
v___x_1277_ = l_Lean_Syntax_node2(v___x_1261_, v___x_1268_, v___x_1270_, v___x_1276_);
lean_inc(v_val_1191_);
v___x_1278_ = l_Lean_Syntax_node3(v___x_1261_, v___x_1266_, v___x_1267_, v_val_1191_, v___x_1277_);
v___x_1279_ = l_Lean_Syntax_node2(v___x_1261_, v___x_1262_, v___x_1264_, v___x_1278_);
v___x_1280_ = l_Lean_Expr_mvarId_x21(v_a_1258_);
v___x_1281_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_1281_, 0, v___x_1279_);
v___x_1282_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_1282_, 0, lean_box(0));
lean_closure_set(v___x_1282_, 1, v___x_1281_);
v___x_1283_ = l_Lean_Elab_Tactic_run(v___x_1280_, v___x_1282_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1283_) == 0)
{
lean_object* v_a_1284_; 
v_a_1284_ = lean_ctor_get(v___x_1283_, 0);
lean_inc(v_a_1284_);
lean_dec_ref_known(v___x_1283_, 1);
if (lean_obj_tag(v_a_1284_) == 1)
{
lean_object* v_head_1285_; lean_object* v_tail_1286_; lean_object* v___x_1287_; 
v_head_1285_ = lean_ctor_get(v_a_1284_, 0);
lean_inc(v_head_1285_);
v_tail_1286_ = lean_ctor_get(v_a_1284_, 1);
lean_inc(v_tail_1286_);
lean_dec_ref_known(v_a_1284_, 2);
lean_inc(v_a_1251_);
v___x_1287_ = l_Lean_MVarId_getType(v_a_1251_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1287_) == 0)
{
lean_object* v_a_1288_; lean_object* v___x_1289_; 
v_a_1288_ = lean_ctor_get(v___x_1287_, 0);
lean_inc(v_a_1288_);
lean_dec_ref_known(v___x_1287_, 1);
lean_inc(v_head_1285_);
v___x_1289_ = l_Lean_MVarId_getType(v_head_1285_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1289_) == 0)
{
lean_object* v_a_1290_; uint8_t v___x_1291_; lean_object* v___x_1292_; lean_object* v___f_1293_; lean_object* v___x_1294_; 
v_a_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1290_);
lean_dec_ref_known(v___x_1289_, 1);
v___x_1291_ = 2;
v___x_1292_ = lean_box(v___x_1291_);
v___f_1293_ = lean_alloc_closure((void*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__1___boxed), 12, 3);
lean_closure_set(v___f_1293_, 0, v___x_1292_);
lean_closure_set(v___f_1293_, 1, v_a_1288_);
lean_closure_set(v___f_1293_, 2, v_a_1290_);
lean_inc(v_a_1251_);
v___x_1294_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic_findGoalOfPatt_spec__1___redArg(v_a_1251_, v___f_1293_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1294_) == 0)
{
lean_object* v_a_1295_; uint8_t v___x_1296_; 
lean_del_object(v___x_1197_);
lean_dec(v_a_1195_);
lean_dec(v_a_1193_);
v_a_1295_ = lean_ctor_get(v___x_1294_, 0);
lean_inc(v_a_1295_);
lean_dec_ref_known(v___x_1294_, 1);
v___x_1296_ = lean_unbox(v_a_1295_);
lean_dec(v_a_1295_);
if (v___x_1296_ == 0)
{
lean_object* v___x_1297_; lean_object* v_a_1298_; lean_object* v___x_1299_; 
v___x_1297_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(v_a_1251_, v_a_1258_, v___y_1181_);
v_a_1298_ = lean_ctor_get(v___x_1297_, 0);
lean_inc(v_a_1298_);
lean_dec_ref(v___x_1297_);
v___x_1299_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2(v_tail_1286_, v___x_1190_, v_head_1285_, v_a_1298_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
v___y_1230_ = v___x_1299_;
goto v___jp_1229_;
}
else
{
lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v_a_1302_; lean_object* v___x_1303_; 
lean_dec(v_a_1258_);
lean_inc(v_head_1285_);
v___x_1300_ = l_Lean_Expr_mvar___override(v_head_1285_);
v___x_1301_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(v_a_1251_, v___x_1300_, v___y_1181_);
v_a_1302_ = lean_ctor_get(v___x_1301_, 0);
lean_inc(v_a_1302_);
lean_dec_ref(v___x_1301_);
v___x_1303_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___lam__2(v_tail_1286_, v___x_1190_, v_head_1285_, v_a_1302_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
v___y_1230_ = v___x_1303_;
goto v___jp_1229_;
}
}
else
{
lean_object* v_a_1304_; 
lean_dec(v_tail_1286_);
lean_dec(v_head_1285_);
lean_dec(v_a_1258_);
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1304_ = lean_ctor_get(v___x_1294_, 0);
lean_inc(v_a_1304_);
lean_dec_ref_known(v___x_1294_, 1);
v_a_1226_ = v_a_1304_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1305_; 
lean_dec(v_a_1288_);
lean_dec(v_tail_1286_);
lean_dec(v_head_1285_);
lean_dec(v_a_1258_);
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1305_ = lean_ctor_get(v___x_1289_, 0);
lean_inc(v_a_1305_);
lean_dec_ref_known(v___x_1289_, 1);
v_a_1226_ = v_a_1305_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1306_; 
lean_dec(v_tail_1286_);
lean_dec(v_head_1285_);
lean_dec(v_a_1258_);
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1306_ = lean_ctor_get(v___x_1287_, 0);
lean_inc(v_a_1306_);
lean_dec_ref_known(v___x_1287_, 1);
v_a_1226_ = v_a_1306_;
goto v___jp_1225_;
}
}
else
{
lean_object* v___x_1307_; 
lean_dec(v_a_1284_);
lean_dec(v_a_1258_);
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v___x_1307_ = l_Lean_Elab_Tactic_throwNoGoalsToBeSolved___redArg(v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1307_) == 0)
{
lean_dec_ref_known(v___x_1307_, 1);
lean_del_object(v___x_1197_);
lean_dec(v_a_1195_);
lean_dec(v_a_1193_);
v_as_x27_1174_ = v_tail_1187_;
v_b_1175_ = v___x_1199_;
goto _start;
}
else
{
lean_object* v_a_1309_; 
v_a_1309_ = lean_ctor_get(v___x_1307_, 0);
lean_inc(v_a_1309_);
lean_dec_ref_known(v___x_1307_, 1);
v_a_1226_ = v_a_1309_;
goto v___jp_1225_;
}
}
}
else
{
lean_object* v_a_1310_; 
lean_dec(v_a_1258_);
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1310_ = lean_ctor_get(v___x_1283_, 0);
lean_inc(v_a_1310_);
lean_dec_ref_known(v___x_1283_, 1);
v_a_1226_ = v_a_1310_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1311_; 
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1311_ = lean_ctor_get(v___x_1257_, 0);
lean_inc(v_a_1311_);
lean_dec_ref_known(v___x_1257_, 1);
v_a_1226_ = v_a_1311_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1312_; 
lean_dec(v_a_1253_);
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1312_ = lean_ctor_get(v___x_1254_, 0);
lean_inc(v_a_1312_);
lean_dec_ref_known(v___x_1254_, 1);
v_a_1226_ = v_a_1312_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1313_; 
lean_dec(v_a_1251_);
lean_dec(v___x_1190_);
v_a_1313_ = lean_ctor_get(v___x_1252_, 0);
lean_inc(v_a_1313_);
lean_dec_ref_known(v___x_1252_, 1);
v_a_1226_ = v_a_1313_;
goto v___jp_1225_;
}
}
else
{
lean_object* v_a_1314_; 
lean_dec(v___x_1190_);
v_a_1314_ = lean_ctor_get(v___x_1250_, 0);
lean_inc(v_a_1314_);
lean_dec_ref_known(v___x_1250_, 1);
v_a_1226_ = v_a_1314_;
goto v___jp_1225_;
}
v___jp_1200_:
{
if (v___y_1202_ == 0)
{
lean_object* v___x_1203_; 
lean_dec_ref(v___y_1201_);
lean_del_object(v___x_1197_);
v___x_1203_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1195_, v___y_1202_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1203_) == 0)
{
lean_object* v___x_1204_; 
lean_dec_ref_known(v___x_1203_, 1);
v___x_1204_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1193_, v___y_1202_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1204_) == 0)
{
lean_dec_ref_known(v___x_1204_, 1);
v_as_x27_1174_ = v_tail_1187_;
v_b_1175_ = v___x_1199_;
goto _start;
}
else
{
lean_object* v_a_1206_; lean_object* v___x_1208_; uint8_t v_isShared_1209_; uint8_t v_isSharedCheck_1213_; 
lean_dec_ref_known(v_patt_x3f_1172_, 1);
lean_dec_ref(v_renameI_1173_);
lean_dec(v_gs_1171_);
v_a_1206_ = lean_ctor_get(v___x_1204_, 0);
v_isSharedCheck_1213_ = !lean_is_exclusive(v___x_1204_);
if (v_isSharedCheck_1213_ == 0)
{
v___x_1208_ = v___x_1204_;
v_isShared_1209_ = v_isSharedCheck_1213_;
goto v_resetjp_1207_;
}
else
{
lean_inc(v_a_1206_);
lean_dec(v___x_1204_);
v___x_1208_ = lean_box(0);
v_isShared_1209_ = v_isSharedCheck_1213_;
goto v_resetjp_1207_;
}
v_resetjp_1207_:
{
lean_object* v___x_1211_; 
if (v_isShared_1209_ == 0)
{
v___x_1211_ = v___x_1208_;
goto v_reusejp_1210_;
}
else
{
lean_object* v_reuseFailAlloc_1212_; 
v_reuseFailAlloc_1212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1212_, 0, v_a_1206_);
v___x_1211_ = v_reuseFailAlloc_1212_;
goto v_reusejp_1210_;
}
v_reusejp_1210_:
{
return v___x_1211_;
}
}
}
}
else
{
lean_object* v_a_1214_; lean_object* v___x_1216_; uint8_t v_isShared_1217_; uint8_t v_isSharedCheck_1221_; 
lean_dec(v_a_1193_);
lean_dec_ref_known(v_patt_x3f_1172_, 1);
lean_dec_ref(v_renameI_1173_);
lean_dec(v_gs_1171_);
v_a_1214_ = lean_ctor_get(v___x_1203_, 0);
v_isSharedCheck_1221_ = !lean_is_exclusive(v___x_1203_);
if (v_isSharedCheck_1221_ == 0)
{
v___x_1216_ = v___x_1203_;
v_isShared_1217_ = v_isSharedCheck_1221_;
goto v_resetjp_1215_;
}
else
{
lean_inc(v_a_1214_);
lean_dec(v___x_1203_);
v___x_1216_ = lean_box(0);
v_isShared_1217_ = v_isSharedCheck_1221_;
goto v_resetjp_1215_;
}
v_resetjp_1215_:
{
lean_object* v___x_1219_; 
if (v_isShared_1217_ == 0)
{
v___x_1219_ = v___x_1216_;
goto v_reusejp_1218_;
}
else
{
lean_object* v_reuseFailAlloc_1220_; 
v_reuseFailAlloc_1220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1220_, 0, v_a_1214_);
v___x_1219_ = v_reuseFailAlloc_1220_;
goto v_reusejp_1218_;
}
v_reusejp_1218_:
{
return v___x_1219_;
}
}
}
}
else
{
lean_object* v___x_1223_; 
lean_dec(v_a_1195_);
lean_dec(v_a_1193_);
lean_dec_ref_known(v_patt_x3f_1172_, 1);
lean_dec_ref(v_renameI_1173_);
lean_dec(v_gs_1171_);
if (v_isShared_1198_ == 0)
{
lean_ctor_set_tag(v___x_1197_, 1);
lean_ctor_set(v___x_1197_, 0, v___y_1201_);
v___x_1223_ = v___x_1197_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v___y_1201_);
v___x_1223_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
return v___x_1223_;
}
}
}
v___jp_1225_:
{
uint8_t v___x_1227_; 
v___x_1227_ = l_Lean_Exception_isInterrupt(v_a_1226_);
if (v___x_1227_ == 0)
{
uint8_t v___x_1228_; 
lean_inc_ref(v_a_1226_);
v___x_1228_ = l_Lean_Exception_isRuntime(v_a_1226_);
v___y_1201_ = v_a_1226_;
v___y_1202_ = v___x_1228_;
goto v___jp_1200_;
}
else
{
v___y_1201_ = v_a_1226_;
v___y_1202_ = v___x_1227_;
goto v___jp_1200_;
}
}
v___jp_1229_:
{
lean_object* v_a_1231_; lean_object* v___x_1233_; uint8_t v_isShared_1234_; uint8_t v_isSharedCheck_1249_; 
v_a_1231_ = lean_ctor_get(v___y_1230_, 0);
v_isSharedCheck_1249_ = !lean_is_exclusive(v___y_1230_);
if (v_isSharedCheck_1249_ == 0)
{
v___x_1233_ = v___y_1230_;
v_isShared_1234_ = v_isSharedCheck_1249_;
goto v_resetjp_1232_;
}
else
{
lean_inc(v_a_1231_);
lean_dec(v___y_1230_);
v___x_1233_ = lean_box(0);
v_isShared_1234_ = v_isSharedCheck_1249_;
goto v_resetjp_1232_;
}
v_resetjp_1232_:
{
if (lean_obj_tag(v_a_1231_) == 0)
{
lean_object* v___x_1236_; uint8_t v_isShared_1237_; uint8_t v_isSharedCheck_1246_; 
lean_dec_ref(v_renameI_1173_);
lean_dec(v_gs_1171_);
v_isSharedCheck_1246_ = !lean_is_exclusive(v_patt_x3f_1172_);
if (v_isSharedCheck_1246_ == 0)
{
lean_object* v_unused_1247_; 
v_unused_1247_ = lean_ctor_get(v_patt_x3f_1172_, 0);
lean_dec(v_unused_1247_);
v___x_1236_ = v_patt_x3f_1172_;
v_isShared_1237_ = v_isSharedCheck_1246_;
goto v_resetjp_1235_;
}
else
{
lean_dec(v_patt_x3f_1172_);
v___x_1236_ = lean_box(0);
v_isShared_1237_ = v_isSharedCheck_1246_;
goto v_resetjp_1235_;
}
v_resetjp_1235_:
{
lean_object* v_a_1238_; lean_object* v___x_1240_; 
v_a_1238_ = lean_ctor_get(v_a_1231_, 0);
lean_inc(v_a_1238_);
lean_dec_ref_known(v_a_1231_, 1);
if (v_isShared_1237_ == 0)
{
lean_ctor_set(v___x_1236_, 0, v_a_1238_);
v___x_1240_ = v___x_1236_;
goto v_reusejp_1239_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v_a_1238_);
v___x_1240_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1239_;
}
v_reusejp_1239_:
{
lean_object* v___x_1241_; lean_object* v___x_1243_; 
v___x_1241_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1241_, 0, v___x_1240_);
lean_ctor_set(v___x_1241_, 1, v___x_1188_);
if (v_isShared_1234_ == 0)
{
lean_ctor_set(v___x_1233_, 0, v___x_1241_);
v___x_1243_ = v___x_1233_;
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
}
else
{
lean_dec_ref_known(v_a_1231_, 1);
lean_del_object(v___x_1233_);
v_as_x27_1174_ = v_tail_1187_;
v_b_1175_ = v___x_1199_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_1316_; lean_object* v___x_1318_; uint8_t v_isShared_1319_; uint8_t v_isSharedCheck_1323_; 
lean_dec(v_a_1193_);
lean_dec_ref_known(v_patt_x3f_1172_, 1);
lean_dec(v___x_1190_);
lean_dec_ref(v_renameI_1173_);
lean_dec(v_gs_1171_);
v_a_1316_ = lean_ctor_get(v___x_1194_, 0);
v_isSharedCheck_1323_ = !lean_is_exclusive(v___x_1194_);
if (v_isSharedCheck_1323_ == 0)
{
v___x_1318_ = v___x_1194_;
v_isShared_1319_ = v_isSharedCheck_1323_;
goto v_resetjp_1317_;
}
else
{
lean_inc(v_a_1316_);
lean_dec(v___x_1194_);
v___x_1318_ = lean_box(0);
v_isShared_1319_ = v_isSharedCheck_1323_;
goto v_resetjp_1317_;
}
v_resetjp_1317_:
{
lean_object* v___x_1321_; 
if (v_isShared_1319_ == 0)
{
v___x_1321_ = v___x_1318_;
goto v_reusejp_1320_;
}
else
{
lean_object* v_reuseFailAlloc_1322_; 
v_reuseFailAlloc_1322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1322_, 0, v_a_1316_);
v___x_1321_ = v_reuseFailAlloc_1322_;
goto v_reusejp_1320_;
}
v_reusejp_1320_:
{
return v___x_1321_;
}
}
}
}
else
{
lean_object* v_a_1324_; lean_object* v___x_1326_; uint8_t v_isShared_1327_; uint8_t v_isSharedCheck_1331_; 
lean_dec_ref_known(v_patt_x3f_1172_, 1);
lean_dec(v___x_1190_);
lean_dec_ref(v_renameI_1173_);
lean_dec(v_gs_1171_);
v_a_1324_ = lean_ctor_get(v___x_1192_, 0);
v_isSharedCheck_1331_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1331_ == 0)
{
v___x_1326_ = v___x_1192_;
v_isShared_1327_ = v_isSharedCheck_1331_;
goto v_resetjp_1325_;
}
else
{
lean_inc(v_a_1324_);
lean_dec(v___x_1192_);
v___x_1326_ = lean_box(0);
v_isShared_1327_ = v_isSharedCheck_1331_;
goto v_resetjp_1325_;
}
v_resetjp_1325_:
{
lean_object* v___x_1329_; 
if (v_isShared_1327_ == 0)
{
v___x_1329_ = v___x_1326_;
goto v_reusejp_1328_;
}
else
{
lean_object* v_reuseFailAlloc_1330_; 
v_reuseFailAlloc_1330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1330_, 0, v_a_1324_);
v___x_1329_ = v_reuseFailAlloc_1330_;
goto v_reusejp_1328_;
}
v_reusejp_1328_:
{
return v___x_1329_;
}
}
}
}
else
{
lean_object* v___x_1332_; 
lean_dec(v_patt_x3f_1172_);
lean_dec(v_gs_1171_);
lean_inc(v_head_1186_);
v___x_1332_ = l_Lean_Elab_Tactic_renameInaccessibles(v_head_1186_, v_renameI_1173_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
if (lean_obj_tag(v___x_1332_) == 0)
{
lean_object* v_a_1333_; lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1345_; 
v_a_1333_ = lean_ctor_get(v___x_1332_, 0);
v_isSharedCheck_1345_ = !lean_is_exclusive(v___x_1332_);
if (v_isSharedCheck_1345_ == 0)
{
v___x_1335_ = v___x_1332_;
v_isShared_1336_ = v_isSharedCheck_1345_;
goto v_resetjp_1334_;
}
else
{
lean_inc(v_a_1333_);
lean_dec(v___x_1332_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1345_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1343_; 
v___x_1337_ = lean_box(0);
v___x_1338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1338_, 0, v___x_1337_);
lean_ctor_set(v___x_1338_, 1, v___x_1190_);
v___x_1339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1339_, 0, v_a_1333_);
lean_ctor_set(v___x_1339_, 1, v___x_1338_);
v___x_1340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1340_, 0, v___x_1339_);
v___x_1341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1341_, 0, v___x_1340_);
lean_ctor_set(v___x_1341_, 1, v___x_1188_);
if (v_isShared_1336_ == 0)
{
lean_ctor_set(v___x_1335_, 0, v___x_1341_);
v___x_1343_ = v___x_1335_;
goto v_reusejp_1342_;
}
else
{
lean_object* v_reuseFailAlloc_1344_; 
v_reuseFailAlloc_1344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1344_, 0, v___x_1341_);
v___x_1343_ = v_reuseFailAlloc_1344_;
goto v_reusejp_1342_;
}
v_reusejp_1342_:
{
return v___x_1343_;
}
}
}
else
{
lean_object* v_a_1346_; lean_object* v___x_1348_; uint8_t v_isShared_1349_; uint8_t v_isSharedCheck_1353_; 
lean_dec(v___x_1190_);
v_a_1346_ = lean_ctor_get(v___x_1332_, 0);
v_isSharedCheck_1353_ = !lean_is_exclusive(v___x_1332_);
if (v_isSharedCheck_1353_ == 0)
{
v___x_1348_ = v___x_1332_;
v_isShared_1349_ = v_isSharedCheck_1353_;
goto v_resetjp_1347_;
}
else
{
lean_inc(v_a_1346_);
lean_dec(v___x_1332_);
v___x_1348_ = lean_box(0);
v_isShared_1349_ = v_isSharedCheck_1353_;
goto v_resetjp_1347_;
}
v_resetjp_1347_:
{
lean_object* v___x_1351_; 
if (v_isShared_1349_ == 0)
{
v___x_1351_ = v___x_1348_;
goto v_reusejp_1350_;
}
else
{
lean_object* v_reuseFailAlloc_1352_; 
v_reuseFailAlloc_1352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1352_, 0, v_a_1346_);
v___x_1351_ = v_reuseFailAlloc_1352_;
goto v_reusejp_1350_;
}
v_reusejp_1350_:
{
return v___x_1351_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___boxed(lean_object* v_gs_1354_, lean_object* v_patt_x3f_1355_, lean_object* v_renameI_1356_, lean_object* v_as_x27_1357_, lean_object* v_b_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_){
_start:
{
lean_object* v_res_1368_; 
v_res_1368_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg(v_gs_1354_, v_patt_x3f_1355_, v_renameI_1356_, v_as_x27_1357_, v_b_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_, v___y_1366_);
lean_dec(v___y_1366_);
lean_dec_ref(v___y_1365_);
lean_dec(v___y_1364_);
lean_dec_ref(v___y_1363_);
lean_dec(v___y_1362_);
lean_dec_ref(v___y_1361_);
lean_dec(v___y_1360_);
lean_dec_ref(v___y_1359_);
lean_dec(v_as_x27_1357_);
return v_res_1368_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; 
v___x_1370_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__0));
v___x_1371_ = l_Lean_stringToMessageData(v___x_1370_);
return v___x_1371_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__3(void){
_start:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1373_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__2));
v___x_1374_ = l_Lean_stringToMessageData(v___x_1373_);
return v___x_1374_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; 
v___x_1376_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__4));
v___x_1377_ = l_Lean_stringToMessageData(v___x_1376_);
return v___x_1377_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0(lean_object* v_gs_1383_, lean_object* v_patt_x3f_1384_, lean_object* v_renameI_1385_, lean_object* v_tag_1386_, lean_object* v_fgs_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = ((lean_object*)(lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg___closed__1));
lean_inc(v_patt_x3f_1384_);
v___x_1398_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg(v_gs_1383_, v_patt_x3f_1384_, v_renameI_1385_, v_fgs_1387_, v___x_1397_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_);
if (lean_obj_tag(v___x_1398_) == 0)
{
lean_object* v_a_1399_; lean_object* v___x_1401_; uint8_t v_isShared_1402_; uint8_t v_isSharedCheck_1435_; 
v_a_1399_ = lean_ctor_get(v___x_1398_, 0);
v_isSharedCheck_1435_ = !lean_is_exclusive(v___x_1398_);
if (v_isSharedCheck_1435_ == 0)
{
v___x_1401_ = v___x_1398_;
v_isShared_1402_ = v_isSharedCheck_1435_;
goto v_resetjp_1400_;
}
else
{
lean_inc(v_a_1399_);
lean_dec(v___x_1398_);
v___x_1401_ = lean_box(0);
v_isShared_1402_ = v_isSharedCheck_1435_;
goto v_resetjp_1400_;
}
v_resetjp_1400_:
{
lean_object* v_fst_1403_; lean_object* v___x_1405_; uint8_t v_isShared_1406_; uint8_t v_isSharedCheck_1433_; 
v_fst_1403_ = lean_ctor_get(v_a_1399_, 0);
v_isSharedCheck_1433_ = !lean_is_exclusive(v_a_1399_);
if (v_isSharedCheck_1433_ == 0)
{
lean_object* v_unused_1434_; 
v_unused_1434_ = lean_ctor_get(v_a_1399_, 1);
lean_dec(v_unused_1434_);
v___x_1405_ = v_a_1399_;
v_isShared_1406_ = v_isSharedCheck_1433_;
goto v_resetjp_1404_;
}
else
{
lean_inc(v_fst_1403_);
lean_dec(v_a_1399_);
v___x_1405_ = lean_box(0);
v_isShared_1406_ = v_isSharedCheck_1433_;
goto v_resetjp_1404_;
}
v_resetjp_1404_:
{
if (lean_obj_tag(v_fst_1403_) == 0)
{
lean_object* v_ref_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1411_; 
lean_del_object(v___x_1401_);
v_ref_1407_ = lean_ctor_get(v___y_1394_, 5);
v___x_1408_ = lean_obj_once(&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__1, &lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__1_once, _init_lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__1);
v___x_1409_ = l_Lean_MessageData_ofSyntax(v_tag_1386_);
if (v_isShared_1406_ == 0)
{
lean_ctor_set_tag(v___x_1405_, 7);
lean_ctor_set(v___x_1405_, 1, v___x_1409_);
lean_ctor_set(v___x_1405_, 0, v___x_1408_);
v___x_1411_ = v___x_1405_;
goto v_reusejp_1410_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v___x_1408_);
lean_ctor_set(v_reuseFailAlloc_1428_, 1, v___x_1409_);
v___x_1411_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1410_;
}
v_reusejp_1410_:
{
lean_object* v___x_1412_; lean_object* v___x_1413_; lean_object* v___y_1415_; 
v___x_1412_ = lean_obj_once(&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__3, &lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__3_once, _init_lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__3);
v___x_1413_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1413_, 0, v___x_1411_);
lean_ctor_set(v___x_1413_, 1, v___x_1412_);
if (lean_obj_tag(v_patt_x3f_1384_) == 0)
{
uint8_t v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; 
v___x_1421_ = 0;
v___x_1422_ = l_Lean_SourceInfo_fromRef(v_ref_1407_, v___x_1421_);
v___x_1423_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12));
lean_inc(v___x_1422_);
v___x_1424_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1424_, 0, v___x_1422_);
lean_ctor_set(v___x_1424_, 1, v___x_1423_);
v___x_1425_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6));
v___x_1426_ = l_Lean_Syntax_node1(v___x_1422_, v___x_1425_, v___x_1424_);
v___y_1415_ = v___x_1426_;
goto v___jp_1414_;
}
else
{
lean_object* v_val_1427_; 
v_val_1427_ = lean_ctor_get(v_patt_x3f_1384_, 0);
lean_inc(v_val_1427_);
lean_dec_ref_known(v_patt_x3f_1384_, 1);
v___y_1415_ = v_val_1427_;
goto v___jp_1414_;
}
v___jp_1414_:
{
lean_object* v___x_1416_; lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; 
v___x_1416_ = l_Lean_MessageData_ofSyntax(v___y_1415_);
v___x_1417_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1417_, 0, v___x_1413_);
lean_ctor_set(v___x_1417_, 1, v___x_1416_);
v___x_1418_ = lean_obj_once(&lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__5, &lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__5_once, _init_lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__5);
v___x_1419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1419_, 0, v___x_1417_);
lean_ctor_set(v___x_1419_, 1, v___x_1418_);
v___x_1420_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg(v___x_1419_, v___y_1392_, v___y_1393_, v___y_1394_, v___y_1395_);
return v___x_1420_;
}
}
}
else
{
lean_object* v_val_1429_; lean_object* v___x_1431_; 
lean_del_object(v___x_1405_);
lean_dec(v_tag_1386_);
lean_dec(v_patt_x3f_1384_);
v_val_1429_ = lean_ctor_get(v_fst_1403_, 0);
lean_inc(v_val_1429_);
lean_dec_ref_known(v_fst_1403_, 1);
if (v_isShared_1402_ == 0)
{
lean_ctor_set(v___x_1401_, 0, v_val_1429_);
v___x_1431_ = v___x_1401_;
goto v_reusejp_1430_;
}
else
{
lean_object* v_reuseFailAlloc_1432_; 
v_reuseFailAlloc_1432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1432_, 0, v_val_1429_);
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
}
else
{
lean_object* v_a_1436_; lean_object* v___x_1438_; uint8_t v_isShared_1439_; uint8_t v_isSharedCheck_1443_; 
lean_dec(v_tag_1386_);
lean_dec(v_patt_x3f_1384_);
v_a_1436_ = lean_ctor_get(v___x_1398_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1398_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1438_ = v___x_1398_;
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
else
{
lean_inc(v_a_1436_);
lean_dec(v___x_1398_);
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
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___boxed(lean_object* v_gs_1444_, lean_object* v_patt_x3f_1445_, lean_object* v_renameI_1446_, lean_object* v_tag_1447_, lean_object* v_fgs_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_){
_start:
{
lean_object* v_res_1458_; 
v_res_1458_ = lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0(v_gs_1444_, v_patt_x3f_1445_, v_renameI_1446_, v_tag_1447_, v_fgs_1448_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_);
lean_dec(v___y_1456_);
lean_dec_ref(v___y_1455_);
lean_dec(v___y_1454_);
lean_dec_ref(v___y_1453_);
lean_dec(v___y_1452_);
lean_dec_ref(v___y_1451_);
lean_dec(v___y_1450_);
lean_dec_ref(v___y_1449_);
lean_dec(v_fgs_1448_);
return v_res_1458_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1(uint8_t v___x_1462_, lean_object* v___f_1463_, lean_object* v_gs_1464_, lean_object* v_tag_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
if (v___x_1462_ == 0)
{
lean_object* v___x_1475_; 
lean_inc(v___y_1473_);
lean_inc_ref(v___y_1472_);
lean_inc(v___y_1471_);
lean_inc_ref(v___y_1470_);
lean_inc(v___y_1469_);
lean_inc_ref(v___y_1468_);
lean_inc(v___y_1467_);
lean_inc_ref(v___y_1466_);
v___x_1475_ = lean_apply_10(v___f_1463_, v_gs_1464_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_, lean_box(0));
return v___x_1475_;
}
else
{
lean_object* v___x_1476_; lean_object* v_tag_1477_; lean_object* v___x_1478_; uint8_t v___x_1479_; 
v___x_1476_ = lean_unsigned_to_nat(0u);
v_tag_1477_ = l_Lean_Syntax_getArg(v_tag_1465_, v___x_1476_);
v___x_1478_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__1));
lean_inc(v_tag_1477_);
v___x_1479_ = l_Lean_Syntax_isOfKind(v_tag_1477_, v___x_1478_);
if (v___x_1479_ == 0)
{
lean_object* v___x_1480_; 
lean_dec(v_tag_1477_);
lean_inc(v___y_1473_);
lean_inc_ref(v___y_1472_);
lean_inc(v___y_1471_);
lean_inc_ref(v___y_1470_);
lean_inc(v___y_1469_);
lean_inc_ref(v___y_1468_);
lean_inc(v___y_1467_);
lean_inc_ref(v___y_1466_);
v___x_1480_ = lean_apply_10(v___f_1463_, v_gs_1464_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_, lean_box(0));
return v___x_1480_;
}
else
{
lean_object* v___x_1481_; lean_object* v___x_1482_; 
v___x_1481_ = l_Lean_TSyntax_getId(v_tag_1477_);
lean_dec(v_tag_1477_);
v___x_1482_ = lp_batteries___private_Batteries_Tactic_Case_0__Batteries_Tactic_filterTag(v_gs_1464_, v___x_1481_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec(v___x_1481_);
if (lean_obj_tag(v___x_1482_) == 0)
{
lean_object* v_a_1483_; lean_object* v___x_1484_; 
v_a_1483_ = lean_ctor_get(v___x_1482_, 0);
lean_inc(v_a_1483_);
lean_dec_ref_known(v___x_1482_, 1);
lean_inc(v___y_1473_);
lean_inc_ref(v___y_1472_);
lean_inc(v___y_1471_);
lean_inc_ref(v___y_1470_);
lean_inc(v___y_1469_);
lean_inc_ref(v___y_1468_);
lean_inc(v___y_1467_);
lean_inc_ref(v___y_1466_);
v___x_1484_ = lean_apply_10(v___f_1463_, v_a_1483_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_, lean_box(0));
return v___x_1484_;
}
else
{
lean_object* v_a_1485_; lean_object* v___x_1487_; uint8_t v_isShared_1488_; uint8_t v_isSharedCheck_1492_; 
lean_dec_ref(v___f_1463_);
v_a_1485_ = lean_ctor_get(v___x_1482_, 0);
v_isSharedCheck_1492_ = !lean_is_exclusive(v___x_1482_);
if (v_isSharedCheck_1492_ == 0)
{
v___x_1487_ = v___x_1482_;
v_isShared_1488_ = v_isSharedCheck_1492_;
goto v_resetjp_1486_;
}
else
{
lean_inc(v_a_1485_);
lean_dec(v___x_1482_);
v___x_1487_ = lean_box(0);
v_isShared_1488_ = v_isSharedCheck_1492_;
goto v_resetjp_1486_;
}
v_resetjp_1486_:
{
lean_object* v___x_1490_; 
if (v_isShared_1488_ == 0)
{
v___x_1490_ = v___x_1487_;
goto v_reusejp_1489_;
}
else
{
lean_object* v_reuseFailAlloc_1491_; 
v_reuseFailAlloc_1491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1491_, 0, v_a_1485_);
v___x_1490_ = v_reuseFailAlloc_1491_;
goto v_reusejp_1489_;
}
v_reusejp_1489_:
{
return v___x_1490_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___boxed(lean_object* v___x_1493_, lean_object* v___f_1494_, lean_object* v_gs_1495_, lean_object* v_tag_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_, lean_object* v___y_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_){
_start:
{
uint8_t v___x_19899__boxed_1506_; lean_object* v_res_1507_; 
v___x_19899__boxed_1506_ = lean_unbox(v___x_1493_);
v_res_1507_ = lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1(v___x_19899__boxed_1506_, v___f_1494_, v_gs_1495_, v_tag_1496_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_, v___y_1503_, v___y_1504_);
lean_dec(v___y_1504_);
lean_dec_ref(v___y_1503_);
lean_dec(v___y_1502_);
lean_dec_ref(v___y_1501_);
lean_dec(v___y_1500_);
lean_dec_ref(v___y_1499_);
lean_dec(v___y_1498_);
lean_dec_ref(v___y_1497_);
lean_dec(v_tag_1496_);
return v_res_1507_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt(lean_object* v_gs_1512_, lean_object* v_tag_1513_, lean_object* v_patt_x3f_1514_, lean_object* v_renameI_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_){
_start:
{
lean_object* v___f_1525_; lean_object* v___x_1526_; uint8_t v___x_1527_; lean_object* v___x_1528_; lean_object* v___y_1529_; lean_object* v___x_1530_; 
lean_inc_n(v_tag_1513_, 2);
lean_inc(v_gs_1512_);
v___f_1525_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___boxed), 14, 4);
lean_closure_set(v___f_1525_, 0, v_gs_1512_);
lean_closure_set(v___f_1525_, 1, v_patt_x3f_1514_);
lean_closure_set(v___f_1525_, 2, v_renameI_1515_);
lean_closure_set(v___f_1525_, 3, v_tag_1513_);
v___x_1526_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1));
v___x_1527_ = l_Lean_Syntax_isOfKind(v_tag_1513_, v___x_1526_);
v___x_1528_ = lean_box(v___x_1527_);
v___y_1529_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___boxed), 13, 4);
lean_closure_set(v___y_1529_, 0, v___x_1528_);
lean_closure_set(v___y_1529_, 1, v___f_1525_);
lean_closure_set(v___y_1529_, 2, v_gs_1512_);
lean_closure_set(v___y_1529_, 3, v_tag_1513_);
v___x_1530_ = lp_batteries_Lean_Elab_Term_withoutErrToSorry___at___00Batteries_Tactic_findGoalOfPatt_spec__5___redArg(v___y_1529_, v_a_1516_, v_a_1517_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_);
return v___x_1530_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_findGoalOfPatt___boxed(lean_object* v_gs_1531_, lean_object* v_tag_1532_, lean_object* v_patt_x3f_1533_, lean_object* v_renameI_1534_, lean_object* v_a_1535_, lean_object* v_a_1536_, lean_object* v_a_1537_, lean_object* v_a_1538_, lean_object* v_a_1539_, lean_object* v_a_1540_, lean_object* v_a_1541_, lean_object* v_a_1542_, lean_object* v_a_1543_){
_start:
{
lean_object* v_res_1544_; 
v_res_1544_ = lp_batteries_Batteries_Tactic_findGoalOfPatt(v_gs_1531_, v_tag_1532_, v_patt_x3f_1533_, v_renameI_1534_, v_a_1535_, v_a_1536_, v_a_1537_, v_a_1538_, v_a_1539_, v_a_1540_, v_a_1541_, v_a_1542_);
lean_dec(v_a_1542_);
lean_dec_ref(v_a_1541_);
lean_dec(v_a_1540_);
lean_dec_ref(v_a_1539_);
lean_dec(v_a_1538_);
lean_dec_ref(v_a_1537_);
lean_dec(v_a_1536_);
lean_dec_ref(v_a_1535_);
return v_res_1544_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2(lean_object* v_mvarId_1545_, lean_object* v_val_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_){
_start:
{
lean_object* v___x_1556_; 
v___x_1556_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(v_mvarId_1545_, v_val_1546_, v___y_1552_);
return v___x_1556_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___boxed(lean_object* v_mvarId_1557_, lean_object* v_val_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_){
_start:
{
lean_object* v_res_1568_; 
v_res_1568_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2(v_mvarId_1557_, v_val_1558_, v___y_1559_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_);
lean_dec(v___y_1566_);
lean_dec_ref(v___y_1565_);
lean_dec(v___y_1564_);
lean_dec_ref(v___y_1563_);
lean_dec(v___y_1562_);
lean_dec_ref(v___y_1561_);
lean_dec(v___y_1560_);
lean_dec_ref(v___y_1559_);
return v_res_1568_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3(lean_object* v_gs_1569_, lean_object* v_patt_x3f_1570_, lean_object* v_renameI_1571_, lean_object* v_as_1572_, lean_object* v_as_x27_1573_, lean_object* v_b_1574_, lean_object* v_a_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_){
_start:
{
lean_object* v___x_1585_; 
v___x_1585_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___redArg(v_gs_1569_, v_patt_x3f_1570_, v_renameI_1571_, v_as_x27_1573_, v_b_1574_, v___y_1576_, v___y_1577_, v___y_1578_, v___y_1579_, v___y_1580_, v___y_1581_, v___y_1582_, v___y_1583_);
return v___x_1585_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3___boxed(lean_object* v_gs_1586_, lean_object* v_patt_x3f_1587_, lean_object* v_renameI_1588_, lean_object* v_as_1589_, lean_object* v_as_x27_1590_, lean_object* v_b_1591_, lean_object* v_a_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_){
_start:
{
lean_object* v_res_1602_; 
v_res_1602_ = lp_batteries_List_forIn_x27_loop___at___00Batteries_Tactic_findGoalOfPatt_spec__3(v_gs_1586_, v_patt_x3f_1587_, v_renameI_1588_, v_as_1589_, v_as_x27_1590_, v_b_1591_, v_a_1592_, v___y_1593_, v___y_1594_, v___y_1595_, v___y_1596_, v___y_1597_, v___y_1598_, v___y_1599_, v___y_1600_);
lean_dec(v___y_1600_);
lean_dec_ref(v___y_1599_);
lean_dec(v___y_1598_);
lean_dec_ref(v___y_1597_);
lean_dec(v___y_1596_);
lean_dec_ref(v___y_1595_);
lean_dec(v___y_1594_);
lean_dec_ref(v___y_1593_);
lean_dec(v_as_x27_1590_);
lean_dec(v_as_1589_);
return v_res_1602_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4(lean_object* v_00_u03b1_1603_, lean_object* v_msg_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
lean_object* v___x_1614_; 
v___x_1614_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg(v_msg_1604_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_);
return v___x_1614_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___boxed(lean_object* v_00_u03b1_1615_, lean_object* v_msg_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v_res_1626_; 
v_res_1626_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4(v_00_u03b1_1615_, v_msg_1616_, v___y_1617_, v___y_1618_, v___y_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_);
lean_dec(v___y_1624_);
lean_dec_ref(v___y_1623_);
lean_dec(v___y_1622_);
lean_dec_ref(v___y_1621_);
lean_dec(v___y_1620_);
lean_dec_ref(v___y_1619_);
lean_dec(v___y_1618_);
lean_dec_ref(v___y_1617_);
return v_res_1626_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3(lean_object* v_00_u03b2_1627_, lean_object* v_x_1628_, lean_object* v_x_1629_, lean_object* v_x_1630_){
_start:
{
lean_object* v___x_1631_; 
v___x_1631_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3___redArg(v_x_1628_, v_x_1629_, v_x_1630_);
return v___x_1631_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_1632_, lean_object* v_x_1633_, size_t v_x_1634_, size_t v_x_1635_, lean_object* v_x_1636_, lean_object* v_x_1637_){
_start:
{
lean_object* v___x_1638_; 
v___x_1638_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___redArg(v_x_1633_, v_x_1634_, v_x_1635_, v_x_1636_, v_x_1637_);
return v___x_1638_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1639_, lean_object* v_x_1640_, lean_object* v_x_1641_, lean_object* v_x_1642_, lean_object* v_x_1643_, lean_object* v_x_1644_){
_start:
{
size_t v_x_20096__boxed_1645_; size_t v_x_20097__boxed_1646_; lean_object* v_res_1647_; 
v_x_20096__boxed_1645_ = lean_unbox_usize(v_x_1641_);
lean_dec(v_x_1641_);
v_x_20097__boxed_1646_ = lean_unbox_usize(v_x_1642_);
lean_dec(v_x_1642_);
v_res_1647_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5(v_00_u03b2_1639_, v_x_1640_, v_x_20096__boxed_1645_, v_x_20097__boxed_1646_, v_x_1643_, v_x_1644_);
return v_res_1647_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9(lean_object* v_00_u03b2_1648_, lean_object* v_n_1649_, lean_object* v_k_1650_, lean_object* v_v_1651_){
_start:
{
lean_object* v___x_1652_; 
v___x_1652_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9___redArg(v_n_1649_, v_k_1650_, v_v_1651_);
return v___x_1652_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10(lean_object* v_00_u03b2_1653_, size_t v_depth_1654_, lean_object* v_keys_1655_, lean_object* v_vals_1656_, lean_object* v_heq_1657_, lean_object* v_i_1658_, lean_object* v_entries_1659_){
_start:
{
lean_object* v___x_1660_; 
v___x_1660_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___redArg(v_depth_1654_, v_keys_1655_, v_vals_1656_, v_i_1658_, v_entries_1659_);
return v___x_1660_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10___boxed(lean_object* v_00_u03b2_1661_, lean_object* v_depth_1662_, lean_object* v_keys_1663_, lean_object* v_vals_1664_, lean_object* v_heq_1665_, lean_object* v_i_1666_, lean_object* v_entries_1667_){
_start:
{
size_t v_depth_boxed_1668_; lean_object* v_res_1669_; 
v_depth_boxed_1668_ = lean_unbox_usize(v_depth_1662_);
lean_dec(v_depth_1662_);
v_res_1669_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__10(v_00_u03b2_1661_, v_depth_boxed_1668_, v_keys_1663_, v_vals_1664_, v_heq_1665_, v_i_1666_, v_entries_1667_);
lean_dec_ref(v_vals_1664_);
lean_dec_ref(v_keys_1663_);
return v_res_1669_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9_spec__10(lean_object* v_00_u03b2_1670_, lean_object* v_x_1671_, lean_object* v_x_1672_, lean_object* v_x_1673_, lean_object* v_x_1674_){
_start:
{
lean_object* v___x_1675_; 
v___x_1675_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2_spec__3_spec__5_spec__9_spec__10___redArg(v_x_1671_, v_x_1672_, v_x_1673_, v_x_1674_);
return v___x_1675_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; 
v___x_1676_ = lean_box(0);
v___x_1677_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1678_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1678_, 0, v___x_1677_);
lean_ctor_set(v___x_1678_, 1, v___x_1676_);
return v___x_1678_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg(){
_start:
{
lean_object* v___x_1680_; lean_object* v___x_1681_; 
v___x_1680_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___closed__0);
v___x_1681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1681_, 0, v___x_1680_);
return v___x_1681_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg___boxed(lean_object* v___y_1682_){
_start:
{
lean_object* v_res_1683_; 
v_res_1683_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v_res_1683_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0(lean_object* v_00_u03b1_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_){
_start:
{
lean_object* v___x_1694_; 
v___x_1694_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_1694_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___boxed(lean_object* v_00_u03b1_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_){
_start:
{
lean_object* v_res_1705_; 
v_res_1705_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0(v_00_u03b1_1695_, v___y_1696_, v___y_1697_, v___y_1698_, v___y_1699_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_);
lean_dec(v___y_1703_);
lean_dec_ref(v___y_1702_);
lean_dec(v___y_1701_);
lean_dec_ref(v___y_1700_);
lean_dec(v___y_1699_);
lean_dec_ref(v___y_1698_);
lean_dec(v___y_1697_);
lean_dec_ref(v___y_1696_);
return v_res_1705_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_processCasePattBody(lean_object* v_stx_1706_, lean_object* v_a_1707_, lean_object* v_a_1708_, lean_object* v_a_1709_, lean_object* v_a_1710_, lean_object* v_a_1711_, lean_object* v_a_1712_, lean_object* v_a_1713_, lean_object* v_a_1714_){
_start:
{
lean_object* v___x_1716_; uint8_t v___x_1717_; 
v___x_1716_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattTac___closed__1));
lean_inc(v_stx_1706_);
v___x_1717_ = l_Lean_Syntax_isOfKind(v_stx_1706_, v___x_1716_);
if (v___x_1717_ == 0)
{
lean_object* v___x_1718_; 
lean_dec(v_stx_1706_);
v___x_1718_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_1718_;
}
else
{
lean_object* v___x_1719_; lean_object* v_tac_1720_; lean_object* v___x_1721_; uint8_t v___x_1722_; 
v___x_1719_ = lean_unsigned_to_nat(1u);
v_tac_1720_ = l_Lean_Syntax_getArg(v_stx_1706_, v___x_1719_);
v___x_1721_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__0___closed__6));
lean_inc(v_tac_1720_);
v___x_1722_ = l_Lean_Syntax_isOfKind(v_tac_1720_, v___x_1721_);
if (v___x_1722_ == 0)
{
lean_object* v___x_1723_; uint8_t v___x_1724_; 
v___x_1723_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10));
lean_inc(v_tac_1720_);
v___x_1724_ = l_Lean_Syntax_isOfKind(v_tac_1720_, v___x_1723_);
if (v___x_1724_ == 0)
{
lean_object* v___x_1725_; uint8_t v___x_1726_; 
v___x_1725_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__13));
lean_inc(v_tac_1720_);
v___x_1726_ = l_Lean_Syntax_isOfKind(v_tac_1720_, v___x_1725_);
if (v___x_1726_ == 0)
{
lean_object* v___x_1727_; 
lean_dec(v_tac_1720_);
lean_dec(v_stx_1706_);
v___x_1727_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_1727_;
}
else
{
lean_object* v___x_1728_; lean_object* v_arr_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; 
v___x_1728_ = lean_unsigned_to_nat(0u);
v_arr_1729_ = l_Lean_Syntax_getArg(v_stx_1706_, v___x_1728_);
lean_dec(v_stx_1706_);
v___x_1730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1730_, 0, v_arr_1729_);
lean_ctor_set(v___x_1730_, 1, v_tac_1720_);
v___x_1731_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1731_, 0, v___x_1730_);
v___x_1732_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1732_, 0, v___x_1731_);
return v___x_1732_;
}
}
else
{
lean_object* v___x_1733_; lean_object* v___x_1734_; 
lean_dec(v_stx_1706_);
v___x_1733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1733_, 0, v_tac_1720_);
v___x_1734_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1734_, 0, v___x_1733_);
return v___x_1734_;
}
}
else
{
lean_object* v_ref_1735_; lean_object* v_ref_1736_; uint8_t v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; 
lean_dec(v_stx_1706_);
v_ref_1735_ = lean_ctor_get(v_a_1713_, 5);
v_ref_1736_ = l_Lean_replaceRef(v_tac_1720_, v_ref_1735_);
lean_dec(v_tac_1720_);
v___x_1737_ = 0;
v___x_1738_ = l_Lean_SourceInfo_fromRef(v_ref_1736_, v___x_1737_);
lean_dec(v_ref_1736_);
v___x_1739_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__10));
v___x_1740_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__11));
lean_inc_n(v___x_1738_, 2);
v___x_1741_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1741_, 0, v___x_1738_);
lean_ctor_set(v___x_1741_, 1, v___x_1740_);
v___x_1742_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__12));
v___x_1743_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1743_, 0, v___x_1738_);
lean_ctor_set(v___x_1743_, 1, v___x_1742_);
v___x_1744_ = l_Lean_Syntax_node2(v___x_1738_, v___x_1739_, v___x_1741_, v___x_1743_);
v___x_1745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1745_, 0, v___x_1744_);
v___x_1746_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1746_, 0, v___x_1745_);
return v___x_1746_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_processCasePattBody___boxed(lean_object* v_stx_1747_, lean_object* v_a_1748_, lean_object* v_a_1749_, lean_object* v_a_1750_, lean_object* v_a_1751_, lean_object* v_a_1752_, lean_object* v_a_1753_, lean_object* v_a_1754_, lean_object* v_a_1755_, lean_object* v_a_1756_){
_start:
{
lean_object* v_res_1757_; 
v_res_1757_ = lp_batteries_Batteries_Tactic_processCasePattBody(v_stx_1747_, v_a_1748_, v_a_1749_, v_a_1750_, v_a_1751_, v_a_1752_, v_a_1753_, v_a_1754_, v_a_1755_);
lean_dec(v_a_1755_);
lean_dec_ref(v_a_1754_);
lean_dec(v_a_1753_);
lean_dec_ref(v_a_1752_);
lean_dec(v_a_1751_);
lean_dec_ref(v_a_1750_);
lean_dec(v_a_1749_);
lean_dec_ref(v_a_1748_);
return v_res_1757_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___redArg(lean_object* v_arrow_1758_, lean_object* v_body_1759_, lean_object* v_x_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_){
_start:
{
lean_object* v_fileName_1770_; lean_object* v_fileMap_1771_; lean_object* v_options_1772_; lean_object* v_currRecDepth_1773_; lean_object* v_maxRecDepth_1774_; lean_object* v_ref_1775_; lean_object* v_currNamespace_1776_; lean_object* v_openDecls_1777_; lean_object* v_initHeartbeats_1778_; lean_object* v_maxHeartbeats_1779_; lean_object* v_quotContext_1780_; lean_object* v_currMacroScope_1781_; uint8_t v_diag_1782_; lean_object* v_cancelTk_x3f_1783_; uint8_t v_suppressElabErrors_1784_; lean_object* v_inheritedTraceOptions_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v_ref_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; 
v_fileName_1770_ = lean_ctor_get(v___y_1767_, 0);
v_fileMap_1771_ = lean_ctor_get(v___y_1767_, 1);
v_options_1772_ = lean_ctor_get(v___y_1767_, 2);
v_currRecDepth_1773_ = lean_ctor_get(v___y_1767_, 3);
v_maxRecDepth_1774_ = lean_ctor_get(v___y_1767_, 4);
v_ref_1775_ = lean_ctor_get(v___y_1767_, 5);
v_currNamespace_1776_ = lean_ctor_get(v___y_1767_, 6);
v_openDecls_1777_ = lean_ctor_get(v___y_1767_, 7);
v_initHeartbeats_1778_ = lean_ctor_get(v___y_1767_, 8);
v_maxHeartbeats_1779_ = lean_ctor_get(v___y_1767_, 9);
v_quotContext_1780_ = lean_ctor_get(v___y_1767_, 10);
v_currMacroScope_1781_ = lean_ctor_get(v___y_1767_, 11);
v_diag_1782_ = lean_ctor_get_uint8(v___y_1767_, sizeof(void*)*14);
v_cancelTk_x3f_1783_ = lean_ctor_get(v___y_1767_, 12);
v_suppressElabErrors_1784_ = lean_ctor_get_uint8(v___y_1767_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1785_ = lean_ctor_get(v___y_1767_, 13);
v___x_1786_ = lean_unsigned_to_nat(2u);
v___x_1787_ = lean_mk_empty_array_with_capacity(v___x_1786_);
v___x_1788_ = lean_array_push(v___x_1787_, v_arrow_1758_);
v___x_1789_ = lean_array_push(v___x_1788_, v_body_1759_);
v___x_1790_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__2));
v___x_1791_ = lean_box(2);
v___x_1792_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1792_, 0, v___x_1791_);
lean_ctor_set(v___x_1792_, 1, v___x_1790_);
lean_ctor_set(v___x_1792_, 2, v___x_1789_);
v_ref_1793_ = l_Lean_replaceRef(v___x_1792_, v_ref_1775_);
lean_dec_ref_known(v___x_1792_, 3);
lean_inc_ref(v_inheritedTraceOptions_1785_);
lean_inc(v_cancelTk_x3f_1783_);
lean_inc(v_currMacroScope_1781_);
lean_inc(v_quotContext_1780_);
lean_inc(v_maxHeartbeats_1779_);
lean_inc(v_initHeartbeats_1778_);
lean_inc(v_openDecls_1777_);
lean_inc(v_currNamespace_1776_);
lean_inc(v_maxRecDepth_1774_);
lean_inc(v_currRecDepth_1773_);
lean_inc_ref(v_options_1772_);
lean_inc_ref(v_fileMap_1771_);
lean_inc_ref(v_fileName_1770_);
v___x_1794_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1794_, 0, v_fileName_1770_);
lean_ctor_set(v___x_1794_, 1, v_fileMap_1771_);
lean_ctor_set(v___x_1794_, 2, v_options_1772_);
lean_ctor_set(v___x_1794_, 3, v_currRecDepth_1773_);
lean_ctor_set(v___x_1794_, 4, v_maxRecDepth_1774_);
lean_ctor_set(v___x_1794_, 5, v_ref_1793_);
lean_ctor_set(v___x_1794_, 6, v_currNamespace_1776_);
lean_ctor_set(v___x_1794_, 7, v_openDecls_1777_);
lean_ctor_set(v___x_1794_, 8, v_initHeartbeats_1778_);
lean_ctor_set(v___x_1794_, 9, v_maxHeartbeats_1779_);
lean_ctor_set(v___x_1794_, 10, v_quotContext_1780_);
lean_ctor_set(v___x_1794_, 11, v_currMacroScope_1781_);
lean_ctor_set(v___x_1794_, 12, v_cancelTk_x3f_1783_);
lean_ctor_set(v___x_1794_, 13, v_inheritedTraceOptions_1785_);
lean_ctor_set_uint8(v___x_1794_, sizeof(void*)*14, v_diag_1782_);
lean_ctor_set_uint8(v___x_1794_, sizeof(void*)*14 + 1, v_suppressElabErrors_1784_);
lean_inc(v___y_1768_);
lean_inc(v___y_1766_);
lean_inc_ref(v___y_1765_);
lean_inc(v___y_1764_);
lean_inc_ref(v___y_1763_);
lean_inc(v___y_1762_);
lean_inc_ref(v___y_1761_);
v___x_1795_ = lean_apply_9(v_x_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_, v___y_1765_, v___y_1766_, v___x_1794_, v___y_1768_, lean_box(0));
return v___x_1795_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___redArg___boxed(lean_object* v_arrow_1796_, lean_object* v_body_1797_, lean_object* v_x_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_){
_start:
{
lean_object* v_res_1808_; 
v_res_1808_ = lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___redArg(v_arrow_1796_, v_body_1797_, v_x_1798_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
lean_dec(v___y_1804_);
lean_dec_ref(v___y_1803_);
lean_dec(v___y_1802_);
lean_dec_ref(v___y_1801_);
lean_dec(v___y_1800_);
lean_dec_ref(v___y_1799_);
return v_res_1808_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2(lean_object* v_00_u03b1_1809_, lean_object* v_arrow_1810_, lean_object* v_body_1811_, lean_object* v_x_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_){
_start:
{
lean_object* v___x_1822_; 
v___x_1822_ = lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___redArg(v_arrow_1810_, v_body_1811_, v_x_1812_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_, v___y_1820_);
return v___x_1822_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___boxed(lean_object* v_00_u03b1_1823_, lean_object* v_arrow_1824_, lean_object* v_body_1825_, lean_object* v_x_1826_, lean_object* v___y_1827_, lean_object* v___y_1828_, lean_object* v___y_1829_, lean_object* v___y_1830_, lean_object* v___y_1831_, lean_object* v___y_1832_, lean_object* v___y_1833_, lean_object* v___y_1834_, lean_object* v___y_1835_){
_start:
{
lean_object* v_res_1836_; 
v_res_1836_ = lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2(v_00_u03b1_1823_, v_arrow_1824_, v_body_1825_, v_x_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_, v___y_1831_, v___y_1832_, v___y_1833_, v___y_1834_);
lean_dec(v___y_1834_);
lean_dec_ref(v___y_1833_);
lean_dec(v___y_1832_);
lean_dec_ref(v___y_1831_);
lean_dec(v___y_1830_);
lean_dec_ref(v___y_1829_);
lean_dec(v___y_1828_);
lean_dec_ref(v___y_1827_);
return v_res_1836_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0(lean_object* v___y_1837_, lean_object* v_mkInfoTree_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_, lean_object* v_a_1846_, lean_object* v_a_x3f_1847_){
_start:
{
lean_object* v___x_1849_; lean_object* v_infoState_1850_; lean_object* v_trees_1851_; lean_object* v___x_1852_; 
v___x_1849_ = lean_st_ref_get(v___y_1837_);
v_infoState_1850_ = lean_ctor_get(v___x_1849_, 7);
lean_inc_ref(v_infoState_1850_);
lean_dec(v___x_1849_);
v_trees_1851_ = lean_ctor_get(v_infoState_1850_, 2);
lean_inc_ref(v_trees_1851_);
lean_dec_ref(v_infoState_1850_);
lean_inc(v___y_1837_);
lean_inc_ref(v___y_1845_);
lean_inc(v___y_1844_);
lean_inc_ref(v___y_1843_);
lean_inc(v___y_1842_);
lean_inc_ref(v___y_1841_);
lean_inc(v___y_1840_);
lean_inc_ref(v___y_1839_);
v___x_1852_ = lean_apply_10(v_mkInfoTree_1838_, v_trees_1851_, v___y_1839_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_, v___y_1845_, v___y_1837_, lean_box(0));
if (lean_obj_tag(v___x_1852_) == 0)
{
lean_object* v_a_1853_; lean_object* v___x_1855_; uint8_t v_isShared_1856_; uint8_t v_isSharedCheck_1891_; 
v_a_1853_ = lean_ctor_get(v___x_1852_, 0);
v_isSharedCheck_1891_ = !lean_is_exclusive(v___x_1852_);
if (v_isSharedCheck_1891_ == 0)
{
v___x_1855_ = v___x_1852_;
v_isShared_1856_ = v_isSharedCheck_1891_;
goto v_resetjp_1854_;
}
else
{
lean_inc(v_a_1853_);
lean_dec(v___x_1852_);
v___x_1855_ = lean_box(0);
v_isShared_1856_ = v_isSharedCheck_1891_;
goto v_resetjp_1854_;
}
v_resetjp_1854_:
{
lean_object* v___x_1857_; lean_object* v_infoState_1858_; lean_object* v_env_1859_; lean_object* v_nextMacroScope_1860_; lean_object* v_ngen_1861_; lean_object* v_auxDeclNGen_1862_; lean_object* v_traceState_1863_; lean_object* v_cache_1864_; lean_object* v_messages_1865_; lean_object* v_snapshotTasks_1866_; lean_object* v___x_1868_; uint8_t v_isShared_1869_; uint8_t v_isSharedCheck_1890_; 
v___x_1857_ = lean_st_ref_take(v___y_1837_);
v_infoState_1858_ = lean_ctor_get(v___x_1857_, 7);
v_env_1859_ = lean_ctor_get(v___x_1857_, 0);
v_nextMacroScope_1860_ = lean_ctor_get(v___x_1857_, 1);
v_ngen_1861_ = lean_ctor_get(v___x_1857_, 2);
v_auxDeclNGen_1862_ = lean_ctor_get(v___x_1857_, 3);
v_traceState_1863_ = lean_ctor_get(v___x_1857_, 4);
v_cache_1864_ = lean_ctor_get(v___x_1857_, 5);
v_messages_1865_ = lean_ctor_get(v___x_1857_, 6);
v_snapshotTasks_1866_ = lean_ctor_get(v___x_1857_, 8);
v_isSharedCheck_1890_ = !lean_is_exclusive(v___x_1857_);
if (v_isSharedCheck_1890_ == 0)
{
v___x_1868_ = v___x_1857_;
v_isShared_1869_ = v_isSharedCheck_1890_;
goto v_resetjp_1867_;
}
else
{
lean_inc(v_snapshotTasks_1866_);
lean_inc(v_infoState_1858_);
lean_inc(v_messages_1865_);
lean_inc(v_cache_1864_);
lean_inc(v_traceState_1863_);
lean_inc(v_auxDeclNGen_1862_);
lean_inc(v_ngen_1861_);
lean_inc(v_nextMacroScope_1860_);
lean_inc(v_env_1859_);
lean_dec(v___x_1857_);
v___x_1868_ = lean_box(0);
v_isShared_1869_ = v_isSharedCheck_1890_;
goto v_resetjp_1867_;
}
v_resetjp_1867_:
{
uint8_t v_enabled_1870_; lean_object* v_assignment_1871_; lean_object* v_lazyAssignment_1872_; lean_object* v___x_1874_; uint8_t v_isShared_1875_; uint8_t v_isSharedCheck_1888_; 
v_enabled_1870_ = lean_ctor_get_uint8(v_infoState_1858_, sizeof(void*)*3);
v_assignment_1871_ = lean_ctor_get(v_infoState_1858_, 0);
v_lazyAssignment_1872_ = lean_ctor_get(v_infoState_1858_, 1);
v_isSharedCheck_1888_ = !lean_is_exclusive(v_infoState_1858_);
if (v_isSharedCheck_1888_ == 0)
{
lean_object* v_unused_1889_; 
v_unused_1889_ = lean_ctor_get(v_infoState_1858_, 2);
lean_dec(v_unused_1889_);
v___x_1874_ = v_infoState_1858_;
v_isShared_1875_ = v_isSharedCheck_1888_;
goto v_resetjp_1873_;
}
else
{
lean_inc(v_lazyAssignment_1872_);
lean_inc(v_assignment_1871_);
lean_dec(v_infoState_1858_);
v___x_1874_ = lean_box(0);
v_isShared_1875_ = v_isSharedCheck_1888_;
goto v_resetjp_1873_;
}
v_resetjp_1873_:
{
lean_object* v___x_1876_; lean_object* v___x_1878_; 
v___x_1876_ = l_Lean_PersistentArray_push___redArg(v_a_1846_, v_a_1853_);
if (v_isShared_1875_ == 0)
{
lean_ctor_set(v___x_1874_, 2, v___x_1876_);
v___x_1878_ = v___x_1874_;
goto v_reusejp_1877_;
}
else
{
lean_object* v_reuseFailAlloc_1887_; 
v_reuseFailAlloc_1887_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_1887_, 0, v_assignment_1871_);
lean_ctor_set(v_reuseFailAlloc_1887_, 1, v_lazyAssignment_1872_);
lean_ctor_set(v_reuseFailAlloc_1887_, 2, v___x_1876_);
lean_ctor_set_uint8(v_reuseFailAlloc_1887_, sizeof(void*)*3, v_enabled_1870_);
v___x_1878_ = v_reuseFailAlloc_1887_;
goto v_reusejp_1877_;
}
v_reusejp_1877_:
{
lean_object* v___x_1880_; 
if (v_isShared_1869_ == 0)
{
lean_ctor_set(v___x_1868_, 7, v___x_1878_);
v___x_1880_ = v___x_1868_;
goto v_reusejp_1879_;
}
else
{
lean_object* v_reuseFailAlloc_1886_; 
v_reuseFailAlloc_1886_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1886_, 0, v_env_1859_);
lean_ctor_set(v_reuseFailAlloc_1886_, 1, v_nextMacroScope_1860_);
lean_ctor_set(v_reuseFailAlloc_1886_, 2, v_ngen_1861_);
lean_ctor_set(v_reuseFailAlloc_1886_, 3, v_auxDeclNGen_1862_);
lean_ctor_set(v_reuseFailAlloc_1886_, 4, v_traceState_1863_);
lean_ctor_set(v_reuseFailAlloc_1886_, 5, v_cache_1864_);
lean_ctor_set(v_reuseFailAlloc_1886_, 6, v_messages_1865_);
lean_ctor_set(v_reuseFailAlloc_1886_, 7, v___x_1878_);
lean_ctor_set(v_reuseFailAlloc_1886_, 8, v_snapshotTasks_1866_);
v___x_1880_ = v_reuseFailAlloc_1886_;
goto v_reusejp_1879_;
}
v_reusejp_1879_:
{
lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1884_; 
v___x_1881_ = lean_st_ref_set(v___y_1837_, v___x_1880_);
v___x_1882_ = lean_box(0);
if (v_isShared_1856_ == 0)
{
lean_ctor_set(v___x_1855_, 0, v___x_1882_);
v___x_1884_ = v___x_1855_;
goto v_reusejp_1883_;
}
else
{
lean_object* v_reuseFailAlloc_1885_; 
v_reuseFailAlloc_1885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1885_, 0, v___x_1882_);
v___x_1884_ = v_reuseFailAlloc_1885_;
goto v_reusejp_1883_;
}
v_reusejp_1883_:
{
return v___x_1884_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1892_; lean_object* v___x_1894_; uint8_t v_isShared_1895_; uint8_t v_isSharedCheck_1899_; 
lean_dec_ref(v_a_1846_);
v_a_1892_ = lean_ctor_get(v___x_1852_, 0);
v_isSharedCheck_1899_ = !lean_is_exclusive(v___x_1852_);
if (v_isSharedCheck_1899_ == 0)
{
v___x_1894_ = v___x_1852_;
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
else
{
lean_inc(v_a_1892_);
lean_dec(v___x_1852_);
v___x_1894_ = lean_box(0);
v_isShared_1895_ = v_isSharedCheck_1899_;
goto v_resetjp_1893_;
}
v_resetjp_1893_:
{
lean_object* v___x_1897_; 
if (v_isShared_1895_ == 0)
{
v___x_1897_ = v___x_1894_;
goto v_reusejp_1896_;
}
else
{
lean_object* v_reuseFailAlloc_1898_; 
v_reuseFailAlloc_1898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1898_, 0, v_a_1892_);
v___x_1897_ = v_reuseFailAlloc_1898_;
goto v_reusejp_1896_;
}
v_reusejp_1896_:
{
return v___x_1897_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0___boxed(lean_object* v___y_1900_, lean_object* v_mkInfoTree_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v_a_1909_, lean_object* v_a_x3f_1910_, lean_object* v___y_1911_){
_start:
{
lean_object* v_res_1912_; 
v_res_1912_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0(v___y_1900_, v_mkInfoTree_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_, v_a_1909_, v_a_x3f_1910_);
lean_dec(v_a_x3f_1910_);
lean_dec_ref(v___y_1908_);
lean_dec(v___y_1907_);
lean_dec_ref(v___y_1906_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v___y_1900_);
return v_res_1912_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; 
v___x_1913_ = lean_unsigned_to_nat(32u);
v___x_1914_ = lean_mk_empty_array_with_capacity(v___x_1913_);
v___x_1915_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1915_, 0, v___x_1914_);
return v___x_1915_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__1(void){
_start:
{
size_t v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; 
v___x_1916_ = ((size_t)5ULL);
v___x_1917_ = lean_unsigned_to_nat(0u);
v___x_1918_ = lean_unsigned_to_nat(32u);
v___x_1919_ = lean_mk_empty_array_with_capacity(v___x_1918_);
v___x_1920_ = lean_obj_once(&lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__0, &lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__0);
v___x_1921_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1921_, 0, v___x_1920_);
lean_ctor_set(v___x_1921_, 1, v___x_1919_);
lean_ctor_set(v___x_1921_, 2, v___x_1917_);
lean_ctor_set(v___x_1921_, 3, v___x_1917_);
lean_ctor_set_usize(v___x_1921_, 4, v___x_1916_);
return v___x_1921_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg(lean_object* v___y_1922_){
_start:
{
lean_object* v___x_1924_; lean_object* v_infoState_1925_; lean_object* v_trees_1926_; lean_object* v___x_1927_; lean_object* v_infoState_1928_; lean_object* v_env_1929_; lean_object* v_nextMacroScope_1930_; lean_object* v_ngen_1931_; lean_object* v_auxDeclNGen_1932_; lean_object* v_traceState_1933_; lean_object* v_cache_1934_; lean_object* v_messages_1935_; lean_object* v_snapshotTasks_1936_; lean_object* v___x_1938_; uint8_t v_isShared_1939_; uint8_t v_isSharedCheck_1957_; 
v___x_1924_ = lean_st_ref_get(v___y_1922_);
v_infoState_1925_ = lean_ctor_get(v___x_1924_, 7);
lean_inc_ref(v_infoState_1925_);
lean_dec(v___x_1924_);
v_trees_1926_ = lean_ctor_get(v_infoState_1925_, 2);
lean_inc_ref(v_trees_1926_);
lean_dec_ref(v_infoState_1925_);
v___x_1927_ = lean_st_ref_take(v___y_1922_);
v_infoState_1928_ = lean_ctor_get(v___x_1927_, 7);
v_env_1929_ = lean_ctor_get(v___x_1927_, 0);
v_nextMacroScope_1930_ = lean_ctor_get(v___x_1927_, 1);
v_ngen_1931_ = lean_ctor_get(v___x_1927_, 2);
v_auxDeclNGen_1932_ = lean_ctor_get(v___x_1927_, 3);
v_traceState_1933_ = lean_ctor_get(v___x_1927_, 4);
v_cache_1934_ = lean_ctor_get(v___x_1927_, 5);
v_messages_1935_ = lean_ctor_get(v___x_1927_, 6);
v_snapshotTasks_1936_ = lean_ctor_get(v___x_1927_, 8);
v_isSharedCheck_1957_ = !lean_is_exclusive(v___x_1927_);
if (v_isSharedCheck_1957_ == 0)
{
v___x_1938_ = v___x_1927_;
v_isShared_1939_ = v_isSharedCheck_1957_;
goto v_resetjp_1937_;
}
else
{
lean_inc(v_snapshotTasks_1936_);
lean_inc(v_infoState_1928_);
lean_inc(v_messages_1935_);
lean_inc(v_cache_1934_);
lean_inc(v_traceState_1933_);
lean_inc(v_auxDeclNGen_1932_);
lean_inc(v_ngen_1931_);
lean_inc(v_nextMacroScope_1930_);
lean_inc(v_env_1929_);
lean_dec(v___x_1927_);
v___x_1938_ = lean_box(0);
v_isShared_1939_ = v_isSharedCheck_1957_;
goto v_resetjp_1937_;
}
v_resetjp_1937_:
{
uint8_t v_enabled_1940_; lean_object* v_assignment_1941_; lean_object* v_lazyAssignment_1942_; lean_object* v___x_1944_; uint8_t v_isShared_1945_; uint8_t v_isSharedCheck_1955_; 
v_enabled_1940_ = lean_ctor_get_uint8(v_infoState_1928_, sizeof(void*)*3);
v_assignment_1941_ = lean_ctor_get(v_infoState_1928_, 0);
v_lazyAssignment_1942_ = lean_ctor_get(v_infoState_1928_, 1);
v_isSharedCheck_1955_ = !lean_is_exclusive(v_infoState_1928_);
if (v_isSharedCheck_1955_ == 0)
{
lean_object* v_unused_1956_; 
v_unused_1956_ = lean_ctor_get(v_infoState_1928_, 2);
lean_dec(v_unused_1956_);
v___x_1944_ = v_infoState_1928_;
v_isShared_1945_ = v_isSharedCheck_1955_;
goto v_resetjp_1943_;
}
else
{
lean_inc(v_lazyAssignment_1942_);
lean_inc(v_assignment_1941_);
lean_dec(v_infoState_1928_);
v___x_1944_ = lean_box(0);
v_isShared_1945_ = v_isSharedCheck_1955_;
goto v_resetjp_1943_;
}
v_resetjp_1943_:
{
lean_object* v___x_1946_; lean_object* v___x_1948_; 
v___x_1946_ = lean_obj_once(&lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__1, &lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__1_once, _init_lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___closed__1);
if (v_isShared_1945_ == 0)
{
lean_ctor_set(v___x_1944_, 2, v___x_1946_);
v___x_1948_ = v___x_1944_;
goto v_reusejp_1947_;
}
else
{
lean_object* v_reuseFailAlloc_1954_; 
v_reuseFailAlloc_1954_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_reuseFailAlloc_1954_, 0, v_assignment_1941_);
lean_ctor_set(v_reuseFailAlloc_1954_, 1, v_lazyAssignment_1942_);
lean_ctor_set(v_reuseFailAlloc_1954_, 2, v___x_1946_);
lean_ctor_set_uint8(v_reuseFailAlloc_1954_, sizeof(void*)*3, v_enabled_1940_);
v___x_1948_ = v_reuseFailAlloc_1954_;
goto v_reusejp_1947_;
}
v_reusejp_1947_:
{
lean_object* v___x_1950_; 
if (v_isShared_1939_ == 0)
{
lean_ctor_set(v___x_1938_, 7, v___x_1948_);
v___x_1950_ = v___x_1938_;
goto v_reusejp_1949_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v_env_1929_);
lean_ctor_set(v_reuseFailAlloc_1953_, 1, v_nextMacroScope_1930_);
lean_ctor_set(v_reuseFailAlloc_1953_, 2, v_ngen_1931_);
lean_ctor_set(v_reuseFailAlloc_1953_, 3, v_auxDeclNGen_1932_);
lean_ctor_set(v_reuseFailAlloc_1953_, 4, v_traceState_1933_);
lean_ctor_set(v_reuseFailAlloc_1953_, 5, v_cache_1934_);
lean_ctor_set(v_reuseFailAlloc_1953_, 6, v_messages_1935_);
lean_ctor_set(v_reuseFailAlloc_1953_, 7, v___x_1948_);
lean_ctor_set(v_reuseFailAlloc_1953_, 8, v_snapshotTasks_1936_);
v___x_1950_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1949_;
}
v_reusejp_1949_:
{
lean_object* v___x_1951_; lean_object* v___x_1952_; 
v___x_1951_ = lean_st_ref_set(v___y_1922_, v___x_1950_);
v___x_1952_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1952_, 0, v_trees_1926_);
return v___x_1952_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg___boxed(lean_object* v___y_1958_, lean_object* v___y_1959_){
_start:
{
lean_object* v_res_1960_; 
v_res_1960_ = lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg(v___y_1958_);
lean_dec(v___y_1958_);
return v_res_1960_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg(lean_object* v_x_1961_, lean_object* v_mkInfoTree_1962_, lean_object* v___y_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_){
_start:
{
lean_object* v___x_1972_; lean_object* v_infoState_1973_; uint8_t v_enabled_1974_; 
v___x_1972_ = lean_st_ref_get(v___y_1970_);
v_infoState_1973_ = lean_ctor_get(v___x_1972_, 7);
lean_inc_ref(v_infoState_1973_);
lean_dec(v___x_1972_);
v_enabled_1974_ = lean_ctor_get_uint8(v_infoState_1973_, sizeof(void*)*3);
lean_dec_ref(v_infoState_1973_);
if (v_enabled_1974_ == 0)
{
lean_object* v___x_1975_; 
lean_dec_ref(v_mkInfoTree_1962_);
lean_inc(v___y_1970_);
lean_inc_ref(v___y_1969_);
lean_inc(v___y_1968_);
lean_inc_ref(v___y_1967_);
lean_inc(v___y_1966_);
lean_inc_ref(v___y_1965_);
lean_inc(v___y_1964_);
lean_inc_ref(v___y_1963_);
v___x_1975_ = lean_apply_9(v_x_1961_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_, lean_box(0));
return v___x_1975_;
}
else
{
lean_object* v___x_1976_; lean_object* v_a_1977_; lean_object* v_r_1978_; 
v___x_1976_ = lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg(v___y_1970_);
v_a_1977_ = lean_ctor_get(v___x_1976_, 0);
lean_inc(v_a_1977_);
lean_dec_ref(v___x_1976_);
lean_inc(v___y_1970_);
lean_inc_ref(v___y_1969_);
lean_inc(v___y_1968_);
lean_inc_ref(v___y_1967_);
lean_inc(v___y_1966_);
lean_inc_ref(v___y_1965_);
lean_inc(v___y_1964_);
lean_inc_ref(v___y_1963_);
v_r_1978_ = lean_apply_9(v_x_1961_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_, v___y_1970_, lean_box(0));
if (lean_obj_tag(v_r_1978_) == 0)
{
lean_object* v_a_1979_; lean_object* v___x_1981_; uint8_t v_isShared_1982_; uint8_t v_isSharedCheck_2003_; 
v_a_1979_ = lean_ctor_get(v_r_1978_, 0);
v_isSharedCheck_2003_ = !lean_is_exclusive(v_r_1978_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1981_ = v_r_1978_;
v_isShared_1982_ = v_isSharedCheck_2003_;
goto v_resetjp_1980_;
}
else
{
lean_inc(v_a_1979_);
lean_dec(v_r_1978_);
v___x_1981_ = lean_box(0);
v_isShared_1982_ = v_isSharedCheck_2003_;
goto v_resetjp_1980_;
}
v_resetjp_1980_:
{
lean_object* v___x_1984_; 
lean_inc(v_a_1979_);
if (v_isShared_1982_ == 0)
{
lean_ctor_set_tag(v___x_1981_, 1);
v___x_1984_ = v___x_1981_;
goto v_reusejp_1983_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v_a_1979_);
v___x_1984_ = v_reuseFailAlloc_2002_;
goto v_reusejp_1983_;
}
v_reusejp_1983_:
{
lean_object* v___x_1985_; 
v___x_1985_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0(v___y_1970_, v_mkInfoTree_1962_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_, v_a_1977_, v___x_1984_);
lean_dec_ref(v___x_1984_);
if (lean_obj_tag(v___x_1985_) == 0)
{
lean_object* v___x_1987_; uint8_t v_isShared_1988_; uint8_t v_isSharedCheck_1992_; 
v_isSharedCheck_1992_ = !lean_is_exclusive(v___x_1985_);
if (v_isSharedCheck_1992_ == 0)
{
lean_object* v_unused_1993_; 
v_unused_1993_ = lean_ctor_get(v___x_1985_, 0);
lean_dec(v_unused_1993_);
v___x_1987_ = v___x_1985_;
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
else
{
lean_dec(v___x_1985_);
v___x_1987_ = lean_box(0);
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
v_resetjp_1986_:
{
lean_object* v___x_1990_; 
if (v_isShared_1988_ == 0)
{
lean_ctor_set(v___x_1987_, 0, v_a_1979_);
v___x_1990_ = v___x_1987_;
goto v_reusejp_1989_;
}
else
{
lean_object* v_reuseFailAlloc_1991_; 
v_reuseFailAlloc_1991_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1991_, 0, v_a_1979_);
v___x_1990_ = v_reuseFailAlloc_1991_;
goto v_reusejp_1989_;
}
v_reusejp_1989_:
{
return v___x_1990_;
}
}
}
else
{
lean_object* v_a_1994_; lean_object* v___x_1996_; uint8_t v_isShared_1997_; uint8_t v_isSharedCheck_2001_; 
lean_dec(v_a_1979_);
v_a_1994_ = lean_ctor_get(v___x_1985_, 0);
v_isSharedCheck_2001_ = !lean_is_exclusive(v___x_1985_);
if (v_isSharedCheck_2001_ == 0)
{
v___x_1996_ = v___x_1985_;
v_isShared_1997_ = v_isSharedCheck_2001_;
goto v_resetjp_1995_;
}
else
{
lean_inc(v_a_1994_);
lean_dec(v___x_1985_);
v___x_1996_ = lean_box(0);
v_isShared_1997_ = v_isSharedCheck_2001_;
goto v_resetjp_1995_;
}
v_resetjp_1995_:
{
lean_object* v___x_1999_; 
if (v_isShared_1997_ == 0)
{
v___x_1999_ = v___x_1996_;
goto v_reusejp_1998_;
}
else
{
lean_object* v_reuseFailAlloc_2000_; 
v_reuseFailAlloc_2000_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2000_, 0, v_a_1994_);
v___x_1999_ = v_reuseFailAlloc_2000_;
goto v_reusejp_1998_;
}
v_reusejp_1998_:
{
return v___x_1999_;
}
}
}
}
}
}
else
{
lean_object* v_a_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; 
v_a_2004_ = lean_ctor_get(v_r_1978_, 0);
lean_inc(v_a_2004_);
lean_dec_ref_known(v_r_1978_, 1);
v___x_2005_ = lean_box(0);
v___x_2006_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___lam__0(v___y_1970_, v_mkInfoTree_1962_, v___y_1963_, v___y_1964_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_, v_a_1977_, v___x_2005_);
if (lean_obj_tag(v___x_2006_) == 0)
{
lean_object* v___x_2008_; uint8_t v_isShared_2009_; uint8_t v_isSharedCheck_2013_; 
v_isSharedCheck_2013_ = !lean_is_exclusive(v___x_2006_);
if (v_isSharedCheck_2013_ == 0)
{
lean_object* v_unused_2014_; 
v_unused_2014_ = lean_ctor_get(v___x_2006_, 0);
lean_dec(v_unused_2014_);
v___x_2008_ = v___x_2006_;
v_isShared_2009_ = v_isSharedCheck_2013_;
goto v_resetjp_2007_;
}
else
{
lean_dec(v___x_2006_);
v___x_2008_ = lean_box(0);
v_isShared_2009_ = v_isSharedCheck_2013_;
goto v_resetjp_2007_;
}
v_resetjp_2007_:
{
lean_object* v___x_2011_; 
if (v_isShared_2009_ == 0)
{
lean_ctor_set_tag(v___x_2008_, 1);
lean_ctor_set(v___x_2008_, 0, v_a_2004_);
v___x_2011_ = v___x_2008_;
goto v_reusejp_2010_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v_a_2004_);
v___x_2011_ = v_reuseFailAlloc_2012_;
goto v_reusejp_2010_;
}
v_reusejp_2010_:
{
return v___x_2011_;
}
}
}
else
{
lean_object* v_a_2015_; lean_object* v___x_2017_; uint8_t v_isShared_2018_; uint8_t v_isSharedCheck_2022_; 
lean_dec(v_a_2004_);
v_a_2015_ = lean_ctor_get(v___x_2006_, 0);
v_isSharedCheck_2022_ = !lean_is_exclusive(v___x_2006_);
if (v_isSharedCheck_2022_ == 0)
{
v___x_2017_ = v___x_2006_;
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
else
{
lean_inc(v_a_2015_);
lean_dec(v___x_2006_);
v___x_2017_ = lean_box(0);
v_isShared_2018_ = v_isSharedCheck_2022_;
goto v_resetjp_2016_;
}
v_resetjp_2016_:
{
lean_object* v___x_2020_; 
if (v_isShared_2018_ == 0)
{
v___x_2020_ = v___x_2017_;
goto v_reusejp_2019_;
}
else
{
lean_object* v_reuseFailAlloc_2021_; 
v_reuseFailAlloc_2021_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2021_, 0, v_a_2015_);
v___x_2020_ = v_reuseFailAlloc_2021_;
goto v_reusejp_2019_;
}
v_reusejp_2019_:
{
return v___x_2020_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg___boxed(lean_object* v_x_2023_, lean_object* v_mkInfoTree_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_, lean_object* v___y_2031_, lean_object* v___y_2032_, lean_object* v___y_2033_){
_start:
{
lean_object* v_res_2034_; 
v_res_2034_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg(v_x_2023_, v_mkInfoTree_2024_, v___y_2025_, v___y_2026_, v___y_2027_, v___y_2028_, v___y_2029_, v___y_2030_, v___y_2031_, v___y_2032_);
lean_dec(v___y_2032_);
lean_dec_ref(v___y_2031_);
lean_dec(v___y_2030_);
lean_dec_ref(v___y_2029_);
lean_dec(v___y_2028_);
lean_dec_ref(v___y_2027_);
lean_dec(v___y_2026_);
lean_dec_ref(v___y_2025_);
return v_res_2034_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__1(lean_object* v_a_2035_, lean_object* v_trees_2036_, lean_object* v___y_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_){
_start:
{
lean_object* v___x_2046_; 
lean_inc(v___y_2044_);
lean_inc_ref(v___y_2043_);
lean_inc(v___y_2042_);
lean_inc_ref(v___y_2041_);
lean_inc(v___y_2040_);
lean_inc_ref(v___y_2039_);
lean_inc(v___y_2038_);
lean_inc_ref(v___y_2037_);
v___x_2046_ = lean_apply_9(v_a_2035_, v___y_2037_, v___y_2038_, v___y_2039_, v___y_2040_, v___y_2041_, v___y_2042_, v___y_2043_, v___y_2044_, lean_box(0));
if (lean_obj_tag(v___x_2046_) == 0)
{
lean_object* v_a_2047_; lean_object* v___x_2049_; uint8_t v_isShared_2050_; uint8_t v_isSharedCheck_2055_; 
v_a_2047_ = lean_ctor_get(v___x_2046_, 0);
v_isSharedCheck_2055_ = !lean_is_exclusive(v___x_2046_);
if (v_isSharedCheck_2055_ == 0)
{
v___x_2049_ = v___x_2046_;
v_isShared_2050_ = v_isSharedCheck_2055_;
goto v_resetjp_2048_;
}
else
{
lean_inc(v_a_2047_);
lean_dec(v___x_2046_);
v___x_2049_ = lean_box(0);
v_isShared_2050_ = v_isSharedCheck_2055_;
goto v_resetjp_2048_;
}
v_resetjp_2048_:
{
lean_object* v___x_2051_; lean_object* v___x_2053_; 
v___x_2051_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2051_, 0, v_a_2047_);
lean_ctor_set(v___x_2051_, 1, v_trees_2036_);
if (v_isShared_2050_ == 0)
{
lean_ctor_set(v___x_2049_, 0, v___x_2051_);
v___x_2053_ = v___x_2049_;
goto v_reusejp_2052_;
}
else
{
lean_object* v_reuseFailAlloc_2054_; 
v_reuseFailAlloc_2054_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2054_, 0, v___x_2051_);
v___x_2053_ = v_reuseFailAlloc_2054_;
goto v_reusejp_2052_;
}
v_reusejp_2052_:
{
return v___x_2053_;
}
}
}
else
{
lean_object* v_a_2056_; lean_object* v___x_2058_; uint8_t v_isShared_2059_; uint8_t v_isSharedCheck_2063_; 
lean_dec_ref(v_trees_2036_);
v_a_2056_ = lean_ctor_get(v___x_2046_, 0);
v_isSharedCheck_2063_ = !lean_is_exclusive(v___x_2046_);
if (v_isSharedCheck_2063_ == 0)
{
v___x_2058_ = v___x_2046_;
v_isShared_2059_ = v_isSharedCheck_2063_;
goto v_resetjp_2057_;
}
else
{
lean_inc(v_a_2056_);
lean_dec(v___x_2046_);
v___x_2058_ = lean_box(0);
v_isShared_2059_ = v_isSharedCheck_2063_;
goto v_resetjp_2057_;
}
v_resetjp_2057_:
{
lean_object* v___x_2061_; 
if (v_isShared_2059_ == 0)
{
v___x_2061_ = v___x_2058_;
goto v_reusejp_2060_;
}
else
{
lean_object* v_reuseFailAlloc_2062_; 
v_reuseFailAlloc_2062_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2062_, 0, v_a_2056_);
v___x_2061_ = v_reuseFailAlloc_2062_;
goto v_reusejp_2060_;
}
v_reusejp_2060_:
{
return v___x_2061_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__1___boxed(lean_object* v_a_2064_, lean_object* v_trees_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_){
_start:
{
lean_object* v_res_2075_; 
v_res_2075_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__1(v_a_2064_, v_trees_2065_, v___y_2066_, v___y_2067_, v___y_2068_, v___y_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_);
lean_dec(v___y_2073_);
lean_dec_ref(v___y_2072_);
lean_dec(v___y_2071_);
lean_dec_ref(v___y_2070_);
lean_dec(v___y_2069_);
lean_dec_ref(v___y_2068_);
lean_dec(v___y_2067_);
lean_dec_ref(v___y_2066_);
return v_res_2075_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__2(lean_object* v_stx_2076_, lean_object* v___x_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_){
_start:
{
lean_object* v___x_2087_; 
v___x_2087_ = l_Lean_Elab_Tactic_mkInitialTacticInfo(v_stx_2076_, v___y_2078_, v___y_2079_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_, v___y_2084_, v___y_2085_);
if (lean_obj_tag(v___x_2087_) == 0)
{
lean_object* v_a_2088_; lean_object* v___f_2089_; lean_object* v___x_2090_; 
v_a_2088_ = lean_ctor_get(v___x_2087_, 0);
lean_inc(v_a_2088_);
lean_dec_ref_known(v___x_2087_, 1);
v___f_2089_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__1___boxed), 11, 1);
lean_closure_set(v___f_2089_, 0, v_a_2088_);
v___x_2090_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg(v___x_2077_, v___f_2089_, v___y_2078_, v___y_2079_, v___y_2080_, v___y_2081_, v___y_2082_, v___y_2083_, v___y_2084_, v___y_2085_);
return v___x_2090_;
}
else
{
lean_object* v_a_2091_; lean_object* v___x_2093_; uint8_t v_isShared_2094_; uint8_t v_isSharedCheck_2098_; 
lean_dec_ref(v___x_2077_);
v_a_2091_ = lean_ctor_get(v___x_2087_, 0);
v_isSharedCheck_2098_ = !lean_is_exclusive(v___x_2087_);
if (v_isSharedCheck_2098_ == 0)
{
v___x_2093_ = v___x_2087_;
v_isShared_2094_ = v_isSharedCheck_2098_;
goto v_resetjp_2092_;
}
else
{
lean_inc(v_a_2091_);
lean_dec(v___x_2087_);
v___x_2093_ = lean_box(0);
v_isShared_2094_ = v_isSharedCheck_2098_;
goto v_resetjp_2092_;
}
v_resetjp_2092_:
{
lean_object* v___x_2096_; 
if (v_isShared_2094_ == 0)
{
v___x_2096_ = v___x_2093_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2097_; 
v_reuseFailAlloc_2097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2097_, 0, v_a_2091_);
v___x_2096_ = v_reuseFailAlloc_2097_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
return v___x_2096_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__2___boxed(lean_object* v_stx_2099_, lean_object* v___x_2100_, lean_object* v___y_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_){
_start:
{
lean_object* v_res_2110_; 
v_res_2110_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__2(v_stx_2099_, v___x_2100_, v___y_2101_, v___y_2102_, v___y_2103_, v___y_2104_, v___y_2105_, v___y_2106_, v___y_2107_, v___y_2108_);
lean_dec(v___y_2108_);
lean_dec_ref(v___y_2107_);
lean_dec(v___y_2106_);
lean_dec_ref(v___y_2105_);
lean_dec(v___y_2104_);
lean_dec_ref(v___y_2103_);
lean_dec(v___y_2102_);
lean_dec_ref(v___y_2101_);
return v_res_2110_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9_spec__13___redArg(lean_object* v_x_2111_, lean_object* v_x_2112_){
_start:
{
if (lean_obj_tag(v_x_2112_) == 0)
{
return v_x_2111_;
}
else
{
lean_object* v_key_2113_; lean_object* v_value_2114_; lean_object* v_tail_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2138_; 
v_key_2113_ = lean_ctor_get(v_x_2112_, 0);
v_value_2114_ = lean_ctor_get(v_x_2112_, 1);
v_tail_2115_ = lean_ctor_get(v_x_2112_, 2);
v_isSharedCheck_2138_ = !lean_is_exclusive(v_x_2112_);
if (v_isSharedCheck_2138_ == 0)
{
v___x_2117_ = v_x_2112_;
v_isShared_2118_ = v_isSharedCheck_2138_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_tail_2115_);
lean_inc(v_value_2114_);
lean_inc(v_key_2113_);
lean_dec(v_x_2112_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2138_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
lean_object* v___x_2119_; uint64_t v___x_2120_; uint64_t v___x_2121_; uint64_t v___x_2122_; uint64_t v_fold_2123_; uint64_t v___x_2124_; uint64_t v___x_2125_; uint64_t v___x_2126_; size_t v___x_2127_; size_t v___x_2128_; size_t v___x_2129_; size_t v___x_2130_; size_t v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2134_; 
v___x_2119_ = lean_array_get_size(v_x_2111_);
v___x_2120_ = l_Lean_Expr_hash(v_key_2113_);
v___x_2121_ = 32ULL;
v___x_2122_ = lean_uint64_shift_right(v___x_2120_, v___x_2121_);
v_fold_2123_ = lean_uint64_xor(v___x_2120_, v___x_2122_);
v___x_2124_ = 16ULL;
v___x_2125_ = lean_uint64_shift_right(v_fold_2123_, v___x_2124_);
v___x_2126_ = lean_uint64_xor(v_fold_2123_, v___x_2125_);
v___x_2127_ = lean_uint64_to_usize(v___x_2126_);
v___x_2128_ = lean_usize_of_nat(v___x_2119_);
v___x_2129_ = ((size_t)1ULL);
v___x_2130_ = lean_usize_sub(v___x_2128_, v___x_2129_);
v___x_2131_ = lean_usize_land(v___x_2127_, v___x_2130_);
v___x_2132_ = lean_array_uget_borrowed(v_x_2111_, v___x_2131_);
lean_inc(v___x_2132_);
if (v_isShared_2118_ == 0)
{
lean_ctor_set(v___x_2117_, 2, v___x_2132_);
v___x_2134_ = v___x_2117_;
goto v_reusejp_2133_;
}
else
{
lean_object* v_reuseFailAlloc_2137_; 
v_reuseFailAlloc_2137_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2137_, 0, v_key_2113_);
lean_ctor_set(v_reuseFailAlloc_2137_, 1, v_value_2114_);
lean_ctor_set(v_reuseFailAlloc_2137_, 2, v___x_2132_);
v___x_2134_ = v_reuseFailAlloc_2137_;
goto v_reusejp_2133_;
}
v_reusejp_2133_:
{
lean_object* v___x_2135_; 
v___x_2135_ = lean_array_uset(v_x_2111_, v___x_2131_, v___x_2134_);
v_x_2111_ = v___x_2135_;
v_x_2112_ = v_tail_2115_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9___redArg(lean_object* v_i_2139_, lean_object* v_source_2140_, lean_object* v_target_2141_){
_start:
{
lean_object* v___x_2142_; uint8_t v___x_2143_; 
v___x_2142_ = lean_array_get_size(v_source_2140_);
v___x_2143_ = lean_nat_dec_lt(v_i_2139_, v___x_2142_);
if (v___x_2143_ == 0)
{
lean_dec_ref(v_source_2140_);
lean_dec(v_i_2139_);
return v_target_2141_;
}
else
{
lean_object* v_es_2144_; lean_object* v___x_2145_; lean_object* v_source_2146_; lean_object* v_target_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; 
v_es_2144_ = lean_array_fget(v_source_2140_, v_i_2139_);
v___x_2145_ = lean_box(0);
v_source_2146_ = lean_array_fset(v_source_2140_, v_i_2139_, v___x_2145_);
v_target_2147_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9_spec__13___redArg(v_target_2141_, v_es_2144_);
v___x_2148_ = lean_unsigned_to_nat(1u);
v___x_2149_ = lean_nat_add(v_i_2139_, v___x_2148_);
lean_dec(v_i_2139_);
v_i_2139_ = v___x_2149_;
v_source_2140_ = v_source_2146_;
v_target_2141_ = v_target_2147_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8___redArg(lean_object* v_data_2151_){
_start:
{
lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v_nbuckets_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___x_2158_; 
v___x_2152_ = lean_array_get_size(v_data_2151_);
v___x_2153_ = lean_unsigned_to_nat(2u);
v_nbuckets_2154_ = lean_nat_mul(v___x_2152_, v___x_2153_);
v___x_2155_ = lean_unsigned_to_nat(0u);
v___x_2156_ = lean_box(0);
v___x_2157_ = lean_mk_array(v_nbuckets_2154_, v___x_2156_);
v___x_2158_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9___redArg(v___x_2155_, v_data_2151_, v___x_2157_);
return v___x_2158_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg(lean_object* v_a_2159_, lean_object* v_x_2160_){
_start:
{
if (lean_obj_tag(v_x_2160_) == 0)
{
uint8_t v___x_2161_; 
v___x_2161_ = 0;
return v___x_2161_;
}
else
{
lean_object* v_key_2162_; lean_object* v_tail_2163_; uint8_t v___x_2164_; 
v_key_2162_ = lean_ctor_get(v_x_2160_, 0);
v_tail_2163_ = lean_ctor_get(v_x_2160_, 2);
v___x_2164_ = lean_expr_eqv(v_key_2162_, v_a_2159_);
if (v___x_2164_ == 0)
{
v_x_2160_ = v_tail_2163_;
goto _start;
}
else
{
return v___x_2164_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg___boxed(lean_object* v_a_2166_, lean_object* v_x_2167_){
_start:
{
uint8_t v_res_2168_; lean_object* v_r_2169_; 
v_res_2168_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg(v_a_2166_, v_x_2167_);
lean_dec(v_x_2167_);
lean_dec_ref(v_a_2166_);
v_r_2169_ = lean_box(v_res_2168_);
return v_r_2169_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3___redArg(lean_object* v_m_2170_, lean_object* v_a_2171_, lean_object* v_b_2172_){
_start:
{
lean_object* v_size_2173_; lean_object* v_buckets_2174_; lean_object* v___x_2175_; uint64_t v___x_2176_; uint64_t v___x_2177_; uint64_t v___x_2178_; uint64_t v_fold_2179_; uint64_t v___x_2180_; uint64_t v___x_2181_; uint64_t v___x_2182_; size_t v___x_2183_; size_t v___x_2184_; size_t v___x_2185_; size_t v___x_2186_; size_t v___x_2187_; lean_object* v_bkt_2188_; uint8_t v___x_2189_; 
v_size_2173_ = lean_ctor_get(v_m_2170_, 0);
v_buckets_2174_ = lean_ctor_get(v_m_2170_, 1);
v___x_2175_ = lean_array_get_size(v_buckets_2174_);
v___x_2176_ = l_Lean_Expr_hash(v_a_2171_);
v___x_2177_ = 32ULL;
v___x_2178_ = lean_uint64_shift_right(v___x_2176_, v___x_2177_);
v_fold_2179_ = lean_uint64_xor(v___x_2176_, v___x_2178_);
v___x_2180_ = 16ULL;
v___x_2181_ = lean_uint64_shift_right(v_fold_2179_, v___x_2180_);
v___x_2182_ = lean_uint64_xor(v_fold_2179_, v___x_2181_);
v___x_2183_ = lean_uint64_to_usize(v___x_2182_);
v___x_2184_ = lean_usize_of_nat(v___x_2175_);
v___x_2185_ = ((size_t)1ULL);
v___x_2186_ = lean_usize_sub(v___x_2184_, v___x_2185_);
v___x_2187_ = lean_usize_land(v___x_2183_, v___x_2186_);
v_bkt_2188_ = lean_array_uget_borrowed(v_buckets_2174_, v___x_2187_);
v___x_2189_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg(v_a_2171_, v_bkt_2188_);
if (v___x_2189_ == 0)
{
lean_object* v___x_2191_; uint8_t v_isShared_2192_; uint8_t v_isSharedCheck_2210_; 
lean_inc_ref(v_buckets_2174_);
lean_inc(v_size_2173_);
v_isSharedCheck_2210_ = !lean_is_exclusive(v_m_2170_);
if (v_isSharedCheck_2210_ == 0)
{
lean_object* v_unused_2211_; lean_object* v_unused_2212_; 
v_unused_2211_ = lean_ctor_get(v_m_2170_, 1);
lean_dec(v_unused_2211_);
v_unused_2212_ = lean_ctor_get(v_m_2170_, 0);
lean_dec(v_unused_2212_);
v___x_2191_ = v_m_2170_;
v_isShared_2192_ = v_isSharedCheck_2210_;
goto v_resetjp_2190_;
}
else
{
lean_dec(v_m_2170_);
v___x_2191_ = lean_box(0);
v_isShared_2192_ = v_isSharedCheck_2210_;
goto v_resetjp_2190_;
}
v_resetjp_2190_:
{
lean_object* v___x_2193_; lean_object* v_size_x27_2194_; lean_object* v___x_2195_; lean_object* v_buckets_x27_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; uint8_t v___x_2202_; 
v___x_2193_ = lean_unsigned_to_nat(1u);
v_size_x27_2194_ = lean_nat_add(v_size_2173_, v___x_2193_);
lean_dec(v_size_2173_);
lean_inc(v_bkt_2188_);
v___x_2195_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2195_, 0, v_a_2171_);
lean_ctor_set(v___x_2195_, 1, v_b_2172_);
lean_ctor_set(v___x_2195_, 2, v_bkt_2188_);
v_buckets_x27_2196_ = lean_array_uset(v_buckets_2174_, v___x_2187_, v___x_2195_);
v___x_2197_ = lean_unsigned_to_nat(4u);
v___x_2198_ = lean_nat_mul(v_size_x27_2194_, v___x_2197_);
v___x_2199_ = lean_unsigned_to_nat(3u);
v___x_2200_ = lean_nat_div(v___x_2198_, v___x_2199_);
lean_dec(v___x_2198_);
v___x_2201_ = lean_array_get_size(v_buckets_x27_2196_);
v___x_2202_ = lean_nat_dec_le(v___x_2200_, v___x_2201_);
lean_dec(v___x_2200_);
if (v___x_2202_ == 0)
{
lean_object* v_val_2203_; lean_object* v___x_2205_; 
v_val_2203_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8___redArg(v_buckets_x27_2196_);
if (v_isShared_2192_ == 0)
{
lean_ctor_set(v___x_2191_, 1, v_val_2203_);
lean_ctor_set(v___x_2191_, 0, v_size_x27_2194_);
v___x_2205_ = v___x_2191_;
goto v_reusejp_2204_;
}
else
{
lean_object* v_reuseFailAlloc_2206_; 
v_reuseFailAlloc_2206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2206_, 0, v_size_x27_2194_);
lean_ctor_set(v_reuseFailAlloc_2206_, 1, v_val_2203_);
v___x_2205_ = v_reuseFailAlloc_2206_;
goto v_reusejp_2204_;
}
v_reusejp_2204_:
{
return v___x_2205_;
}
}
else
{
lean_object* v___x_2208_; 
if (v_isShared_2192_ == 0)
{
lean_ctor_set(v___x_2191_, 1, v_buckets_x27_2196_);
lean_ctor_set(v___x_2191_, 0, v_size_x27_2194_);
v___x_2208_ = v___x_2191_;
goto v_reusejp_2207_;
}
else
{
lean_object* v_reuseFailAlloc_2209_; 
v_reuseFailAlloc_2209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2209_, 0, v_size_x27_2194_);
lean_ctor_set(v_reuseFailAlloc_2209_, 1, v_buckets_x27_2196_);
v___x_2208_ = v_reuseFailAlloc_2209_;
goto v_reusejp_2207_;
}
v_reusejp_2207_:
{
return v___x_2208_;
}
}
}
}
else
{
lean_dec(v_b_2172_);
lean_dec_ref(v_a_2171_);
return v_m_2170_;
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg(lean_object* v_m_2213_, lean_object* v_a_2214_){
_start:
{
lean_object* v_buckets_2215_; lean_object* v___x_2216_; uint64_t v___x_2217_; uint64_t v___x_2218_; uint64_t v___x_2219_; uint64_t v_fold_2220_; uint64_t v___x_2221_; uint64_t v___x_2222_; uint64_t v___x_2223_; size_t v___x_2224_; size_t v___x_2225_; size_t v___x_2226_; size_t v___x_2227_; size_t v___x_2228_; lean_object* v___x_2229_; uint8_t v___x_2230_; 
v_buckets_2215_ = lean_ctor_get(v_m_2213_, 1);
v___x_2216_ = lean_array_get_size(v_buckets_2215_);
v___x_2217_ = l_Lean_Expr_hash(v_a_2214_);
v___x_2218_ = 32ULL;
v___x_2219_ = lean_uint64_shift_right(v___x_2217_, v___x_2218_);
v_fold_2220_ = lean_uint64_xor(v___x_2217_, v___x_2219_);
v___x_2221_ = 16ULL;
v___x_2222_ = lean_uint64_shift_right(v_fold_2220_, v___x_2221_);
v___x_2223_ = lean_uint64_xor(v_fold_2220_, v___x_2222_);
v___x_2224_ = lean_uint64_to_usize(v___x_2223_);
v___x_2225_ = lean_usize_of_nat(v___x_2216_);
v___x_2226_ = ((size_t)1ULL);
v___x_2227_ = lean_usize_sub(v___x_2225_, v___x_2226_);
v___x_2228_ = lean_usize_land(v___x_2224_, v___x_2227_);
v___x_2229_ = lean_array_uget_borrowed(v_buckets_2215_, v___x_2228_);
v___x_2230_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg(v_a_2214_, v___x_2229_);
return v___x_2230_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_m_2231_, lean_object* v_a_2232_){
_start:
{
uint8_t v_res_2233_; lean_object* v_r_2234_; 
v_res_2233_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg(v_m_2231_, v_a_2232_);
lean_dec_ref(v_a_2232_);
lean_dec_ref(v_m_2231_);
v_r_2234_ = lean_box(v_res_2233_);
return v_r_2234_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg(lean_object* v_mvarId_2235_, lean_object* v___y_2236_, lean_object* v___y_2237_){
_start:
{
lean_object* v___x_2239_; lean_object* v_mctx_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; 
v___x_2239_ = lean_st_ref_get(v___y_2237_);
v_mctx_2240_ = lean_ctor_get(v___x_2239_, 0);
lean_inc_ref(v_mctx_2240_);
lean_dec(v___x_2239_);
v___x_2241_ = l_Lean_MetavarContext_getDelayedMVarAssignmentCore_x3f(v_mctx_2240_, v_mvarId_2235_);
lean_dec_ref(v_mctx_2240_);
v___x_2242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2242_, 0, v___x_2241_);
v___x_2243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2242_);
lean_ctor_set(v___x_2243_, 1, v___y_2236_);
v___x_2244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2244_, 0, v___x_2243_);
return v___x_2244_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg___boxed(lean_object* v_mvarId_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_){
_start:
{
lean_object* v_res_2249_; 
v_res_2249_ = lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg(v_mvarId_2245_, v___y_2246_, v___y_2247_);
lean_dec(v___y_2247_);
lean_dec(v_mvarId_2245_);
return v_res_2249_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg(lean_object* v_mvarId_2250_, lean_object* v___y_2251_, lean_object* v___y_2252_){
_start:
{
lean_object* v___x_2254_; lean_object* v_mctx_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; 
v___x_2254_ = lean_st_ref_get(v___y_2252_);
v_mctx_2255_ = lean_ctor_get(v___x_2254_, 0);
lean_inc_ref(v_mctx_2255_);
lean_dec(v___x_2254_);
v___x_2256_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_2255_, v_mvarId_2250_);
lean_dec_ref(v_mctx_2255_);
v___x_2257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2257_, 0, v___x_2256_);
v___x_2258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2258_, 0, v___x_2257_);
lean_ctor_set(v___x_2258_, 1, v___y_2251_);
v___x_2259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2259_, 0, v___x_2258_);
return v___x_2259_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg___boxed(lean_object* v_mvarId_2260_, lean_object* v___y_2261_, lean_object* v___y_2262_, lean_object* v___y_2263_){
_start:
{
lean_object* v_res_2264_; 
v_res_2264_ = lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg(v_mvarId_2260_, v___y_2261_, v___y_2262_);
lean_dec(v___y_2262_);
lean_dec(v_mvarId_2260_);
return v_res_2264_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(lean_object* v_mvarId_2269_, lean_object* v_e_2270_, lean_object* v_a_2271_, lean_object* v___y_2272_, lean_object* v___y_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_){
_start:
{
lean_object* v_d_2282_; lean_object* v_b_2283_; lean_object* v___y_2284_; uint8_t v___x_2290_; 
v___x_2290_ = l_Lean_Expr_hasExprMVar(v_e_2270_);
if (v___x_2290_ == 0)
{
lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; 
lean_dec_ref(v_e_2270_);
v___x_2291_ = ((lean_object*)(lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0));
v___x_2292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2292_, 0, v___x_2291_);
lean_ctor_set(v___x_2292_, 1, v_a_2271_);
v___x_2293_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2293_, 0, v___x_2292_);
return v___x_2293_;
}
else
{
uint8_t v___x_2294_; 
v___x_2294_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg(v_a_2271_, v_e_2270_);
if (v___x_2294_ == 0)
{
lean_object* v___x_2295_; lean_object* v___x_2296_; 
v___x_2295_ = lean_box(0);
lean_inc_ref(v_e_2270_);
v___x_2296_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3___redArg(v_a_2271_, v_e_2270_, v___x_2295_);
switch(lean_obj_tag(v_e_2270_))
{
case 11:
{
lean_object* v_struct_2297_; 
v_struct_2297_ = lean_ctor_get(v_e_2270_, 2);
lean_inc_ref(v_struct_2297_);
lean_dec_ref_known(v_e_2270_, 3);
v_e_2270_ = v_struct_2297_;
v_a_2271_ = v___x_2296_;
goto _start;
}
case 7:
{
lean_object* v_binderType_2299_; lean_object* v_body_2300_; 
v_binderType_2299_ = lean_ctor_get(v_e_2270_, 1);
lean_inc_ref(v_binderType_2299_);
v_body_2300_ = lean_ctor_get(v_e_2270_, 2);
lean_inc_ref(v_body_2300_);
lean_dec_ref_known(v_e_2270_, 3);
v_d_2282_ = v_binderType_2299_;
v_b_2283_ = v_body_2300_;
v___y_2284_ = v___x_2296_;
goto v___jp_2281_;
}
case 6:
{
lean_object* v_binderType_2301_; lean_object* v_body_2302_; 
v_binderType_2301_ = lean_ctor_get(v_e_2270_, 1);
lean_inc_ref(v_binderType_2301_);
v_body_2302_ = lean_ctor_get(v_e_2270_, 2);
lean_inc_ref(v_body_2302_);
lean_dec_ref_known(v_e_2270_, 3);
v_d_2282_ = v_binderType_2301_;
v_b_2283_ = v_body_2302_;
v___y_2284_ = v___x_2296_;
goto v___jp_2281_;
}
case 8:
{
lean_object* v_type_2303_; lean_object* v_value_2304_; lean_object* v_body_2305_; lean_object* v___x_2306_; 
v_type_2303_ = lean_ctor_get(v_e_2270_, 1);
lean_inc_ref(v_type_2303_);
v_value_2304_ = lean_ctor_get(v_e_2270_, 2);
lean_inc_ref(v_value_2304_);
v_body_2305_ = lean_ctor_get(v_e_2270_, 3);
lean_inc_ref(v_body_2305_);
lean_dec_ref_known(v_e_2270_, 4);
v___x_2306_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2269_, v_type_2303_, v___x_2296_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
if (lean_obj_tag(v___x_2306_) == 0)
{
lean_object* v_a_2307_; lean_object* v_fst_2308_; 
v_a_2307_ = lean_ctor_get(v___x_2306_, 0);
lean_inc(v_a_2307_);
v_fst_2308_ = lean_ctor_get(v_a_2307_, 0);
if (lean_obj_tag(v_fst_2308_) == 0)
{
lean_dec(v_a_2307_);
lean_dec_ref(v_body_2305_);
lean_dec_ref(v_value_2304_);
return v___x_2306_;
}
else
{
lean_object* v_snd_2309_; lean_object* v___x_2310_; 
lean_dec_ref_known(v___x_2306_, 1);
v_snd_2309_ = lean_ctor_get(v_a_2307_, 1);
lean_inc(v_snd_2309_);
lean_dec(v_a_2307_);
v___x_2310_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2269_, v_value_2304_, v_snd_2309_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
if (lean_obj_tag(v___x_2310_) == 0)
{
lean_object* v_a_2311_; lean_object* v_fst_2312_; 
v_a_2311_ = lean_ctor_get(v___x_2310_, 0);
lean_inc(v_a_2311_);
v_fst_2312_ = lean_ctor_get(v_a_2311_, 0);
if (lean_obj_tag(v_fst_2312_) == 0)
{
lean_dec(v_a_2311_);
lean_dec_ref(v_body_2305_);
return v___x_2310_;
}
else
{
lean_object* v_snd_2313_; 
lean_dec_ref_known(v___x_2310_, 1);
v_snd_2313_ = lean_ctor_get(v_a_2311_, 1);
lean_inc(v_snd_2313_);
lean_dec(v_a_2311_);
v_e_2270_ = v_body_2305_;
v_a_2271_ = v_snd_2313_;
goto _start;
}
}
else
{
lean_dec_ref(v_body_2305_);
return v___x_2310_;
}
}
}
else
{
lean_dec_ref(v_body_2305_);
lean_dec_ref(v_value_2304_);
return v___x_2306_;
}
}
case 10:
{
lean_object* v_expr_2315_; 
v_expr_2315_ = lean_ctor_get(v_e_2270_, 1);
lean_inc_ref(v_expr_2315_);
lean_dec_ref_known(v_e_2270_, 2);
v_e_2270_ = v_expr_2315_;
v_a_2271_ = v___x_2296_;
goto _start;
}
case 5:
{
lean_object* v_fn_2317_; lean_object* v_arg_2318_; lean_object* v___x_2319_; 
v_fn_2317_ = lean_ctor_get(v_e_2270_, 0);
lean_inc_ref(v_fn_2317_);
v_arg_2318_ = lean_ctor_get(v_e_2270_, 1);
lean_inc_ref(v_arg_2318_);
lean_dec_ref_known(v_e_2270_, 2);
v___x_2319_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2269_, v_fn_2317_, v___x_2296_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
if (lean_obj_tag(v___x_2319_) == 0)
{
lean_object* v_a_2320_; lean_object* v_fst_2321_; 
v_a_2320_ = lean_ctor_get(v___x_2319_, 0);
lean_inc(v_a_2320_);
v_fst_2321_ = lean_ctor_get(v_a_2320_, 0);
if (lean_obj_tag(v_fst_2321_) == 0)
{
lean_dec(v_a_2320_);
lean_dec_ref(v_arg_2318_);
return v___x_2319_;
}
else
{
lean_object* v_snd_2322_; 
lean_dec_ref_known(v___x_2319_, 1);
v_snd_2322_ = lean_ctor_get(v_a_2320_, 1);
lean_inc(v_snd_2322_);
lean_dec(v_a_2320_);
v_e_2270_ = v_arg_2318_;
v_a_2271_ = v_snd_2322_;
goto _start;
}
}
else
{
lean_dec_ref(v_arg_2318_);
return v___x_2319_;
}
}
case 2:
{
lean_object* v_mvarId_2324_; lean_object* v___x_2325_; 
v_mvarId_2324_ = lean_ctor_get(v_e_2270_, 0);
lean_inc(v_mvarId_2324_);
lean_dec_ref_known(v_e_2270_, 1);
v___x_2325_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4(v_mvarId_2269_, v_mvarId_2324_, v___x_2296_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
return v___x_2325_;
}
default: 
{
lean_object* v___x_2326_; lean_object* v___x_2327_; lean_object* v___x_2328_; 
lean_dec_ref(v_e_2270_);
v___x_2326_ = ((lean_object*)(lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0));
v___x_2327_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2327_, 0, v___x_2326_);
lean_ctor_set(v___x_2327_, 1, v___x_2296_);
v___x_2328_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2328_, 0, v___x_2327_);
return v___x_2328_;
}
}
}
else
{
lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; 
lean_dec_ref(v_e_2270_);
v___x_2329_ = ((lean_object*)(lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0));
v___x_2330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2330_, 0, v___x_2329_);
lean_ctor_set(v___x_2330_, 1, v_a_2271_);
v___x_2331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2331_, 0, v___x_2330_);
return v___x_2331_;
}
}
v___jp_2281_:
{
lean_object* v___x_2285_; 
v___x_2285_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2269_, v_d_2282_, v___y_2284_, v___y_2272_, v___y_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
if (lean_obj_tag(v___x_2285_) == 0)
{
lean_object* v_a_2286_; lean_object* v_fst_2287_; 
v_a_2286_ = lean_ctor_get(v___x_2285_, 0);
lean_inc(v_a_2286_);
v_fst_2287_ = lean_ctor_get(v_a_2286_, 0);
if (lean_obj_tag(v_fst_2287_) == 0)
{
lean_dec(v_a_2286_);
lean_dec_ref(v_b_2283_);
return v___x_2285_;
}
else
{
lean_object* v_snd_2288_; 
lean_dec_ref_known(v___x_2285_, 1);
v_snd_2288_ = lean_ctor_get(v_a_2286_, 1);
lean_inc(v_snd_2288_);
lean_dec(v_a_2286_);
v_e_2270_ = v_b_2283_;
v_a_2271_ = v_snd_2288_;
goto _start;
}
}
else
{
lean_dec_ref(v_b_2283_);
return v___x_2285_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4(lean_object* v_mvarId_2332_, lean_object* v_mvarId_x27_2333_, lean_object* v_a_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_, lean_object* v___y_2341_, lean_object* v___y_2342_){
_start:
{
uint8_t v___x_2344_; 
v___x_2344_ = l_Lean_instBEqMVarId_beq(v_mvarId_2332_, v_mvarId_x27_2333_);
if (v___x_2344_ == 0)
{
lean_object* v___x_2345_; 
v___x_2345_ = lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg(v_mvarId_x27_2333_, v_a_2334_, v___y_2340_);
if (lean_obj_tag(v___x_2345_) == 0)
{
lean_object* v_a_2346_; lean_object* v___x_2348_; uint8_t v_isShared_2349_; uint8_t v_isSharedCheck_2429_; 
v_a_2346_ = lean_ctor_get(v___x_2345_, 0);
v_isSharedCheck_2429_ = !lean_is_exclusive(v___x_2345_);
if (v_isSharedCheck_2429_ == 0)
{
v___x_2348_ = v___x_2345_;
v_isShared_2349_ = v_isSharedCheck_2429_;
goto v_resetjp_2347_;
}
else
{
lean_inc(v_a_2346_);
lean_dec(v___x_2345_);
v___x_2348_ = lean_box(0);
v_isShared_2349_ = v_isSharedCheck_2429_;
goto v_resetjp_2347_;
}
v_resetjp_2347_:
{
lean_object* v_fst_2350_; 
v_fst_2350_ = lean_ctor_get(v_a_2346_, 0);
lean_inc(v_fst_2350_);
if (lean_obj_tag(v_fst_2350_) == 0)
{
lean_object* v_snd_2351_; lean_object* v___x_2353_; uint8_t v_isShared_2354_; uint8_t v_isSharedCheck_2369_; 
lean_dec(v_mvarId_x27_2333_);
v_snd_2351_ = lean_ctor_get(v_a_2346_, 1);
v_isSharedCheck_2369_ = !lean_is_exclusive(v_a_2346_);
if (v_isSharedCheck_2369_ == 0)
{
lean_object* v_unused_2370_; 
v_unused_2370_ = lean_ctor_get(v_a_2346_, 0);
lean_dec(v_unused_2370_);
v___x_2353_ = v_a_2346_;
v_isShared_2354_ = v_isSharedCheck_2369_;
goto v_resetjp_2352_;
}
else
{
lean_inc(v_snd_2351_);
lean_dec(v_a_2346_);
v___x_2353_ = lean_box(0);
v_isShared_2354_ = v_isSharedCheck_2369_;
goto v_resetjp_2352_;
}
v_resetjp_2352_:
{
lean_object* v_a_2355_; lean_object* v___x_2357_; uint8_t v_isShared_2358_; uint8_t v_isSharedCheck_2368_; 
v_a_2355_ = lean_ctor_get(v_fst_2350_, 0);
v_isSharedCheck_2368_ = !lean_is_exclusive(v_fst_2350_);
if (v_isSharedCheck_2368_ == 0)
{
v___x_2357_ = v_fst_2350_;
v_isShared_2358_ = v_isSharedCheck_2368_;
goto v_resetjp_2356_;
}
else
{
lean_inc(v_a_2355_);
lean_dec(v_fst_2350_);
v___x_2357_ = lean_box(0);
v_isShared_2358_ = v_isSharedCheck_2368_;
goto v_resetjp_2356_;
}
v_resetjp_2356_:
{
lean_object* v___x_2360_; 
if (v_isShared_2358_ == 0)
{
v___x_2360_ = v___x_2357_;
goto v_reusejp_2359_;
}
else
{
lean_object* v_reuseFailAlloc_2367_; 
v_reuseFailAlloc_2367_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2367_, 0, v_a_2355_);
v___x_2360_ = v_reuseFailAlloc_2367_;
goto v_reusejp_2359_;
}
v_reusejp_2359_:
{
lean_object* v___x_2362_; 
if (v_isShared_2354_ == 0)
{
lean_ctor_set(v___x_2353_, 0, v___x_2360_);
v___x_2362_ = v___x_2353_;
goto v_reusejp_2361_;
}
else
{
lean_object* v_reuseFailAlloc_2366_; 
v_reuseFailAlloc_2366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2366_, 0, v___x_2360_);
lean_ctor_set(v_reuseFailAlloc_2366_, 1, v_snd_2351_);
v___x_2362_ = v_reuseFailAlloc_2366_;
goto v_reusejp_2361_;
}
v_reusejp_2361_:
{
lean_object* v___x_2364_; 
if (v_isShared_2349_ == 0)
{
lean_ctor_set(v___x_2348_, 0, v___x_2362_);
v___x_2364_ = v___x_2348_;
goto v_reusejp_2363_;
}
else
{
lean_object* v_reuseFailAlloc_2365_; 
v_reuseFailAlloc_2365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2365_, 0, v___x_2362_);
v___x_2364_ = v_reuseFailAlloc_2365_;
goto v_reusejp_2363_;
}
v_reusejp_2363_:
{
return v___x_2364_;
}
}
}
}
}
}
else
{
lean_object* v_a_2371_; 
lean_del_object(v___x_2348_);
v_a_2371_ = lean_ctor_get(v_fst_2350_, 0);
lean_inc(v_a_2371_);
lean_dec_ref_known(v_fst_2350_, 1);
if (lean_obj_tag(v_a_2371_) == 0)
{
lean_object* v_snd_2372_; lean_object* v___x_2373_; 
v_snd_2372_ = lean_ctor_get(v_a_2346_, 1);
lean_inc(v_snd_2372_);
lean_dec(v_a_2346_);
v___x_2373_ = lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg(v_mvarId_x27_2333_, v_snd_2372_, v___y_2340_);
lean_dec(v_mvarId_x27_2333_);
if (lean_obj_tag(v___x_2373_) == 0)
{
lean_object* v_a_2374_; lean_object* v___x_2376_; uint8_t v_isShared_2377_; uint8_t v_isSharedCheck_2417_; 
v_a_2374_ = lean_ctor_get(v___x_2373_, 0);
v_isSharedCheck_2417_ = !lean_is_exclusive(v___x_2373_);
if (v_isSharedCheck_2417_ == 0)
{
v___x_2376_ = v___x_2373_;
v_isShared_2377_ = v_isSharedCheck_2417_;
goto v_resetjp_2375_;
}
else
{
lean_inc(v_a_2374_);
lean_dec(v___x_2373_);
v___x_2376_ = lean_box(0);
v_isShared_2377_ = v_isSharedCheck_2417_;
goto v_resetjp_2375_;
}
v_resetjp_2375_:
{
lean_object* v_fst_2378_; 
v_fst_2378_ = lean_ctor_get(v_a_2374_, 0);
lean_inc(v_fst_2378_);
if (lean_obj_tag(v_fst_2378_) == 0)
{
lean_object* v_snd_2379_; lean_object* v___x_2381_; uint8_t v_isShared_2382_; uint8_t v_isSharedCheck_2397_; 
v_snd_2379_ = lean_ctor_get(v_a_2374_, 1);
v_isSharedCheck_2397_ = !lean_is_exclusive(v_a_2374_);
if (v_isSharedCheck_2397_ == 0)
{
lean_object* v_unused_2398_; 
v_unused_2398_ = lean_ctor_get(v_a_2374_, 0);
lean_dec(v_unused_2398_);
v___x_2381_ = v_a_2374_;
v_isShared_2382_ = v_isSharedCheck_2397_;
goto v_resetjp_2380_;
}
else
{
lean_inc(v_snd_2379_);
lean_dec(v_a_2374_);
v___x_2381_ = lean_box(0);
v_isShared_2382_ = v_isSharedCheck_2397_;
goto v_resetjp_2380_;
}
v_resetjp_2380_:
{
lean_object* v_a_2383_; lean_object* v___x_2385_; uint8_t v_isShared_2386_; uint8_t v_isSharedCheck_2396_; 
v_a_2383_ = lean_ctor_get(v_fst_2378_, 0);
v_isSharedCheck_2396_ = !lean_is_exclusive(v_fst_2378_);
if (v_isSharedCheck_2396_ == 0)
{
v___x_2385_ = v_fst_2378_;
v_isShared_2386_ = v_isSharedCheck_2396_;
goto v_resetjp_2384_;
}
else
{
lean_inc(v_a_2383_);
lean_dec(v_fst_2378_);
v___x_2385_ = lean_box(0);
v_isShared_2386_ = v_isSharedCheck_2396_;
goto v_resetjp_2384_;
}
v_resetjp_2384_:
{
lean_object* v___x_2388_; 
if (v_isShared_2386_ == 0)
{
v___x_2388_ = v___x_2385_;
goto v_reusejp_2387_;
}
else
{
lean_object* v_reuseFailAlloc_2395_; 
v_reuseFailAlloc_2395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2395_, 0, v_a_2383_);
v___x_2388_ = v_reuseFailAlloc_2395_;
goto v_reusejp_2387_;
}
v_reusejp_2387_:
{
lean_object* v___x_2390_; 
if (v_isShared_2382_ == 0)
{
lean_ctor_set(v___x_2381_, 0, v___x_2388_);
v___x_2390_ = v___x_2381_;
goto v_reusejp_2389_;
}
else
{
lean_object* v_reuseFailAlloc_2394_; 
v_reuseFailAlloc_2394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2394_, 0, v___x_2388_);
lean_ctor_set(v_reuseFailAlloc_2394_, 1, v_snd_2379_);
v___x_2390_ = v_reuseFailAlloc_2394_;
goto v_reusejp_2389_;
}
v_reusejp_2389_:
{
lean_object* v___x_2392_; 
if (v_isShared_2377_ == 0)
{
lean_ctor_set(v___x_2376_, 0, v___x_2390_);
v___x_2392_ = v___x_2376_;
goto v_reusejp_2391_;
}
else
{
lean_object* v_reuseFailAlloc_2393_; 
v_reuseFailAlloc_2393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2393_, 0, v___x_2390_);
v___x_2392_ = v_reuseFailAlloc_2393_;
goto v_reusejp_2391_;
}
v_reusejp_2391_:
{
return v___x_2392_;
}
}
}
}
}
}
else
{
lean_object* v_a_2399_; 
v_a_2399_ = lean_ctor_get(v_fst_2378_, 0);
lean_inc(v_a_2399_);
lean_dec_ref_known(v_fst_2378_, 1);
if (lean_obj_tag(v_a_2399_) == 0)
{
lean_object* v_snd_2400_; lean_object* v___x_2402_; uint8_t v_isShared_2403_; uint8_t v_isSharedCheck_2411_; 
v_snd_2400_ = lean_ctor_get(v_a_2374_, 1);
v_isSharedCheck_2411_ = !lean_is_exclusive(v_a_2374_);
if (v_isSharedCheck_2411_ == 0)
{
lean_object* v_unused_2412_; 
v_unused_2412_ = lean_ctor_get(v_a_2374_, 0);
lean_dec(v_unused_2412_);
v___x_2402_ = v_a_2374_;
v_isShared_2403_ = v_isSharedCheck_2411_;
goto v_resetjp_2401_;
}
else
{
lean_inc(v_snd_2400_);
lean_dec(v_a_2374_);
v___x_2402_ = lean_box(0);
v_isShared_2403_ = v_isSharedCheck_2411_;
goto v_resetjp_2401_;
}
v_resetjp_2401_:
{
lean_object* v___x_2404_; lean_object* v___x_2406_; 
v___x_2404_ = ((lean_object*)(lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__0));
if (v_isShared_2403_ == 0)
{
lean_ctor_set(v___x_2402_, 0, v___x_2404_);
v___x_2406_ = v___x_2402_;
goto v_reusejp_2405_;
}
else
{
lean_object* v_reuseFailAlloc_2410_; 
v_reuseFailAlloc_2410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2410_, 0, v___x_2404_);
lean_ctor_set(v_reuseFailAlloc_2410_, 1, v_snd_2400_);
v___x_2406_ = v_reuseFailAlloc_2410_;
goto v_reusejp_2405_;
}
v_reusejp_2405_:
{
lean_object* v___x_2408_; 
if (v_isShared_2377_ == 0)
{
lean_ctor_set(v___x_2376_, 0, v___x_2406_);
v___x_2408_ = v___x_2376_;
goto v_reusejp_2407_;
}
else
{
lean_object* v_reuseFailAlloc_2409_; 
v_reuseFailAlloc_2409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2409_, 0, v___x_2406_);
v___x_2408_ = v_reuseFailAlloc_2409_;
goto v_reusejp_2407_;
}
v_reusejp_2407_:
{
return v___x_2408_;
}
}
}
}
else
{
lean_object* v_val_2413_; lean_object* v_snd_2414_; lean_object* v_mvarIdPending_2415_; 
lean_del_object(v___x_2376_);
v_val_2413_ = lean_ctor_get(v_a_2399_, 0);
lean_inc(v_val_2413_);
lean_dec_ref_known(v_a_2399_, 1);
v_snd_2414_ = lean_ctor_get(v_a_2374_, 1);
lean_inc(v_snd_2414_);
lean_dec(v_a_2374_);
v_mvarIdPending_2415_ = lean_ctor_get(v_val_2413_, 1);
lean_inc(v_mvarIdPending_2415_);
lean_dec(v_val_2413_);
v_mvarId_x27_2333_ = v_mvarIdPending_2415_;
v_a_2334_ = v_snd_2414_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_2418_; lean_object* v___x_2420_; uint8_t v_isShared_2421_; uint8_t v_isSharedCheck_2425_; 
v_a_2418_ = lean_ctor_get(v___x_2373_, 0);
v_isSharedCheck_2425_ = !lean_is_exclusive(v___x_2373_);
if (v_isSharedCheck_2425_ == 0)
{
v___x_2420_ = v___x_2373_;
v_isShared_2421_ = v_isSharedCheck_2425_;
goto v_resetjp_2419_;
}
else
{
lean_inc(v_a_2418_);
lean_dec(v___x_2373_);
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
else
{
lean_object* v_snd_2426_; lean_object* v_val_2427_; lean_object* v___x_2428_; 
lean_dec(v_mvarId_x27_2333_);
v_snd_2426_ = lean_ctor_get(v_a_2346_, 1);
lean_inc(v_snd_2426_);
lean_dec(v_a_2346_);
v_val_2427_ = lean_ctor_get(v_a_2371_, 0);
lean_inc(v_val_2427_);
lean_dec_ref_known(v_a_2371_, 1);
v___x_2428_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2332_, v_val_2427_, v_snd_2426_, v___y_2335_, v___y_2336_, v___y_2337_, v___y_2338_, v___y_2339_, v___y_2340_, v___y_2341_, v___y_2342_);
return v___x_2428_;
}
}
}
}
else
{
lean_object* v_a_2430_; lean_object* v___x_2432_; uint8_t v_isShared_2433_; uint8_t v_isSharedCheck_2437_; 
lean_dec(v_mvarId_x27_2333_);
v_a_2430_ = lean_ctor_get(v___x_2345_, 0);
v_isSharedCheck_2437_ = !lean_is_exclusive(v___x_2345_);
if (v_isSharedCheck_2437_ == 0)
{
v___x_2432_ = v___x_2345_;
v_isShared_2433_ = v_isSharedCheck_2437_;
goto v_resetjp_2431_;
}
else
{
lean_inc(v_a_2430_);
lean_dec(v___x_2345_);
v___x_2432_ = lean_box(0);
v_isShared_2433_ = v_isSharedCheck_2437_;
goto v_resetjp_2431_;
}
v_resetjp_2431_:
{
lean_object* v___x_2435_; 
if (v_isShared_2433_ == 0)
{
v___x_2435_ = v___x_2432_;
goto v_reusejp_2434_;
}
else
{
lean_object* v_reuseFailAlloc_2436_; 
v_reuseFailAlloc_2436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2436_, 0, v_a_2430_);
v___x_2435_ = v_reuseFailAlloc_2436_;
goto v_reusejp_2434_;
}
v_reusejp_2434_:
{
return v___x_2435_;
}
}
}
}
else
{
lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; 
lean_dec(v_mvarId_x27_2333_);
v___x_2438_ = ((lean_object*)(lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___closed__1));
v___x_2439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2439_, 0, v___x_2438_);
lean_ctor_set(v___x_2439_, 1, v_a_2334_);
v___x_2440_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2440_, 0, v___x_2439_);
return v___x_2440_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4___boxed(lean_object* v_mvarId_2441_, lean_object* v_mvarId_x27_2442_, lean_object* v_a_2443_, lean_object* v___y_2444_, lean_object* v___y_2445_, lean_object* v___y_2446_, lean_object* v___y_2447_, lean_object* v___y_2448_, lean_object* v___y_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_){
_start:
{
lean_object* v_res_2453_; 
v_res_2453_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4(v_mvarId_2441_, v_mvarId_x27_2442_, v_a_2443_, v___y_2444_, v___y_2445_, v___y_2446_, v___y_2447_, v___y_2448_, v___y_2449_, v___y_2450_, v___y_2451_);
lean_dec(v___y_2451_);
lean_dec_ref(v___y_2450_);
lean_dec(v___y_2449_);
lean_dec_ref(v___y_2448_);
lean_dec(v___y_2447_);
lean_dec_ref(v___y_2446_);
lean_dec(v___y_2445_);
lean_dec_ref(v___y_2444_);
lean_dec(v_mvarId_2441_);
return v_res_2453_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0___boxed(lean_object* v_mvarId_2454_, lean_object* v_e_2455_, lean_object* v_a_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_){
_start:
{
lean_object* v_res_2466_; 
v_res_2466_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2454_, v_e_2455_, v_a_2456_, v___y_2457_, v___y_2458_, v___y_2459_, v___y_2460_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_);
lean_dec(v___y_2464_);
lean_dec_ref(v___y_2463_);
lean_dec(v___y_2462_);
lean_dec_ref(v___y_2461_);
lean_dec(v___y_2460_);
lean_dec_ref(v___y_2459_);
lean_dec(v___y_2458_);
lean_dec_ref(v___y_2457_);
lean_dec(v_mvarId_2454_);
return v_res_2466_;
}
}
static lean_object* _init_lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__0(void){
_start:
{
lean_object* v___x_2467_; lean_object* v___x_2468_; lean_object* v___x_2469_; 
v___x_2467_ = lean_box(0);
v___x_2468_ = lean_unsigned_to_nat(16u);
v___x_2469_ = lean_mk_array(v___x_2468_, v___x_2467_);
return v___x_2469_;
}
}
static lean_object* _init_lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; 
v___x_2470_ = lean_obj_once(&lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__0, &lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__0_once, _init_lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__0);
v___x_2471_ = lean_unsigned_to_nat(0u);
v___x_2472_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2472_, 0, v___x_2471_);
lean_ctor_set(v___x_2472_, 1, v___x_2470_);
return v___x_2472_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0(lean_object* v_mvarId_2473_, lean_object* v_e_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_, lean_object* v___y_2477_, lean_object* v___y_2478_, lean_object* v___y_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_){
_start:
{
uint8_t v___x_2484_; 
v___x_2484_ = l_Lean_Expr_hasExprMVar(v_e_2474_);
if (v___x_2484_ == 0)
{
uint8_t v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; 
lean_dec_ref(v_e_2474_);
v___x_2485_ = 1;
v___x_2486_ = lean_box(v___x_2485_);
v___x_2487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2487_, 0, v___x_2486_);
return v___x_2487_;
}
else
{
lean_object* v___x_2488_; lean_object* v___x_2489_; 
v___x_2488_ = lean_obj_once(&lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__1, &lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__1_once, _init_lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___closed__1);
v___x_2489_ = lp_batteries___private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0(v_mvarId_2473_, v_e_2474_, v___x_2488_, v___y_2475_, v___y_2476_, v___y_2477_, v___y_2478_, v___y_2479_, v___y_2480_, v___y_2481_, v___y_2482_);
if (lean_obj_tag(v___x_2489_) == 0)
{
lean_object* v_a_2490_; lean_object* v___x_2492_; uint8_t v_isShared_2493_; uint8_t v_isSharedCheck_2504_; 
v_a_2490_ = lean_ctor_get(v___x_2489_, 0);
v_isSharedCheck_2504_ = !lean_is_exclusive(v___x_2489_);
if (v_isSharedCheck_2504_ == 0)
{
v___x_2492_ = v___x_2489_;
v_isShared_2493_ = v_isSharedCheck_2504_;
goto v_resetjp_2491_;
}
else
{
lean_inc(v_a_2490_);
lean_dec(v___x_2489_);
v___x_2492_ = lean_box(0);
v_isShared_2493_ = v_isSharedCheck_2504_;
goto v_resetjp_2491_;
}
v_resetjp_2491_:
{
lean_object* v_fst_2494_; 
v_fst_2494_ = lean_ctor_get(v_a_2490_, 0);
lean_inc(v_fst_2494_);
lean_dec(v_a_2490_);
if (lean_obj_tag(v_fst_2494_) == 0)
{
uint8_t v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2498_; 
lean_dec_ref_known(v_fst_2494_, 1);
v___x_2495_ = 0;
v___x_2496_ = lean_box(v___x_2495_);
if (v_isShared_2493_ == 0)
{
lean_ctor_set(v___x_2492_, 0, v___x_2496_);
v___x_2498_ = v___x_2492_;
goto v_reusejp_2497_;
}
else
{
lean_object* v_reuseFailAlloc_2499_; 
v_reuseFailAlloc_2499_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2499_, 0, v___x_2496_);
v___x_2498_ = v_reuseFailAlloc_2499_;
goto v_reusejp_2497_;
}
v_reusejp_2497_:
{
return v___x_2498_;
}
}
else
{
lean_object* v___x_2500_; lean_object* v___x_2502_; 
lean_dec_ref_known(v_fst_2494_, 1);
v___x_2500_ = lean_box(v___x_2484_);
if (v_isShared_2493_ == 0)
{
lean_ctor_set(v___x_2492_, 0, v___x_2500_);
v___x_2502_ = v___x_2492_;
goto v_reusejp_2501_;
}
else
{
lean_object* v_reuseFailAlloc_2503_; 
v_reuseFailAlloc_2503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2503_, 0, v___x_2500_);
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
else
{
lean_object* v_a_2505_; lean_object* v___x_2507_; uint8_t v_isShared_2508_; uint8_t v_isSharedCheck_2512_; 
v_a_2505_ = lean_ctor_get(v___x_2489_, 0);
v_isSharedCheck_2512_ = !lean_is_exclusive(v___x_2489_);
if (v_isSharedCheck_2512_ == 0)
{
v___x_2507_ = v___x_2489_;
v_isShared_2508_ = v_isSharedCheck_2512_;
goto v_resetjp_2506_;
}
else
{
lean_inc(v_a_2505_);
lean_dec(v___x_2489_);
v___x_2507_ = lean_box(0);
v_isShared_2508_ = v_isSharedCheck_2512_;
goto v_resetjp_2506_;
}
v_resetjp_2506_:
{
lean_object* v___x_2510_; 
if (v_isShared_2508_ == 0)
{
v___x_2510_ = v___x_2507_;
goto v_reusejp_2509_;
}
else
{
lean_object* v_reuseFailAlloc_2511_; 
v_reuseFailAlloc_2511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2511_, 0, v_a_2505_);
v___x_2510_ = v_reuseFailAlloc_2511_;
goto v_reusejp_2509_;
}
v_reusejp_2509_:
{
return v___x_2510_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0___boxed(lean_object* v_mvarId_2513_, lean_object* v_e_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_, lean_object* v___y_2519_, lean_object* v___y_2520_, lean_object* v___y_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_){
_start:
{
lean_object* v_res_2524_; 
v_res_2524_ = lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0(v_mvarId_2513_, v_e_2514_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_, v___y_2519_, v___y_2520_, v___y_2521_, v___y_2522_);
lean_dec(v___y_2522_);
lean_dec_ref(v___y_2521_);
lean_dec(v___y_2520_);
lean_dec_ref(v___y_2519_);
lean_dec(v___y_2518_);
lean_dec_ref(v___y_2517_);
lean_dec(v___y_2516_);
lean_dec_ref(v___y_2515_);
lean_dec(v_mvarId_2513_);
return v_res_2524_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__2(void){
_start:
{
lean_object* v___x_2528_; lean_object* v___x_2529_; 
v___x_2528_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__1));
v___x_2529_ = l_Lean_stringToMessageData(v___x_2528_);
return v___x_2529_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__4(void){
_start:
{
lean_object* v___x_2531_; lean_object* v___x_2532_; 
v___x_2531_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__3));
v___x_2532_ = l_Lean_stringToMessageData(v___x_2531_);
return v___x_2532_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__6(void){
_start:
{
lean_object* v___x_2534_; lean_object* v___x_2535_; 
v___x_2534_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__5));
v___x_2535_ = l_Lean_stringToMessageData(v___x_2534_);
return v___x_2535_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0(lean_object* v_val_2536_, lean_object* v_fst_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_, lean_object* v___y_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_){
_start:
{
lean_object* v_fileName_2547_; lean_object* v_fileMap_2548_; lean_object* v_options_2549_; lean_object* v_currRecDepth_2550_; lean_object* v_maxRecDepth_2551_; lean_object* v_ref_2552_; lean_object* v_currNamespace_2553_; lean_object* v_openDecls_2554_; lean_object* v_initHeartbeats_2555_; lean_object* v_maxHeartbeats_2556_; lean_object* v_quotContext_2557_; lean_object* v_currMacroScope_2558_; uint8_t v_diag_2559_; lean_object* v_cancelTk_x3f_2560_; uint8_t v_suppressElabErrors_2561_; lean_object* v_inheritedTraceOptions_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2629_; 
v_fileName_2547_ = lean_ctor_get(v___y_2544_, 0);
v_fileMap_2548_ = lean_ctor_get(v___y_2544_, 1);
v_options_2549_ = lean_ctor_get(v___y_2544_, 2);
v_currRecDepth_2550_ = lean_ctor_get(v___y_2544_, 3);
v_maxRecDepth_2551_ = lean_ctor_get(v___y_2544_, 4);
v_ref_2552_ = lean_ctor_get(v___y_2544_, 5);
v_currNamespace_2553_ = lean_ctor_get(v___y_2544_, 6);
v_openDecls_2554_ = lean_ctor_get(v___y_2544_, 7);
v_initHeartbeats_2555_ = lean_ctor_get(v___y_2544_, 8);
v_maxHeartbeats_2556_ = lean_ctor_get(v___y_2544_, 9);
v_quotContext_2557_ = lean_ctor_get(v___y_2544_, 10);
v_currMacroScope_2558_ = lean_ctor_get(v___y_2544_, 11);
v_diag_2559_ = lean_ctor_get_uint8(v___y_2544_, sizeof(void*)*14);
v_cancelTk_x3f_2560_ = lean_ctor_get(v___y_2544_, 12);
v_suppressElabErrors_2561_ = lean_ctor_get_uint8(v___y_2544_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2562_ = lean_ctor_get(v___y_2544_, 13);
v_isSharedCheck_2629_ = !lean_is_exclusive(v___y_2544_);
if (v_isSharedCheck_2629_ == 0)
{
v___x_2564_ = v___y_2544_;
v_isShared_2565_ = v_isSharedCheck_2629_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_inheritedTraceOptions_2562_);
lean_inc(v_cancelTk_x3f_2560_);
lean_inc(v_currMacroScope_2558_);
lean_inc(v_quotContext_2557_);
lean_inc(v_maxHeartbeats_2556_);
lean_inc(v_initHeartbeats_2555_);
lean_inc(v_openDecls_2554_);
lean_inc(v_currNamespace_2553_);
lean_inc(v_ref_2552_);
lean_inc(v_maxRecDepth_2551_);
lean_inc(v_currRecDepth_2550_);
lean_inc(v_options_2549_);
lean_inc(v_fileMap_2548_);
lean_inc(v_fileName_2547_);
lean_dec(v___y_2544_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2629_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v_ref_2566_; lean_object* v___x_2568_; 
v_ref_2566_ = l_Lean_replaceRef(v_val_2536_, v_ref_2552_);
lean_dec(v_ref_2552_);
if (v_isShared_2565_ == 0)
{
lean_ctor_set(v___x_2564_, 5, v_ref_2566_);
v___x_2568_ = v___x_2564_;
goto v_reusejp_2567_;
}
else
{
lean_object* v_reuseFailAlloc_2628_; 
v_reuseFailAlloc_2628_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_2628_, 0, v_fileName_2547_);
lean_ctor_set(v_reuseFailAlloc_2628_, 1, v_fileMap_2548_);
lean_ctor_set(v_reuseFailAlloc_2628_, 2, v_options_2549_);
lean_ctor_set(v_reuseFailAlloc_2628_, 3, v_currRecDepth_2550_);
lean_ctor_set(v_reuseFailAlloc_2628_, 4, v_maxRecDepth_2551_);
lean_ctor_set(v_reuseFailAlloc_2628_, 5, v_ref_2566_);
lean_ctor_set(v_reuseFailAlloc_2628_, 6, v_currNamespace_2553_);
lean_ctor_set(v_reuseFailAlloc_2628_, 7, v_openDecls_2554_);
lean_ctor_set(v_reuseFailAlloc_2628_, 8, v_initHeartbeats_2555_);
lean_ctor_set(v_reuseFailAlloc_2628_, 9, v_maxHeartbeats_2556_);
lean_ctor_set(v_reuseFailAlloc_2628_, 10, v_quotContext_2557_);
lean_ctor_set(v_reuseFailAlloc_2628_, 11, v_currMacroScope_2558_);
lean_ctor_set(v_reuseFailAlloc_2628_, 12, v_cancelTk_x3f_2560_);
lean_ctor_set(v_reuseFailAlloc_2628_, 13, v_inheritedTraceOptions_2562_);
lean_ctor_set_uint8(v_reuseFailAlloc_2628_, sizeof(void*)*14, v_diag_2559_);
lean_ctor_set_uint8(v_reuseFailAlloc_2628_, sizeof(void*)*14 + 1, v_suppressElabErrors_2561_);
v___x_2568_ = v_reuseFailAlloc_2628_;
goto v_reusejp_2567_;
}
v_reusejp_2567_:
{
lean_object* v___x_2569_; 
v___x_2569_ = l_Lean_Elab_Tactic_getMainTarget(v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_, v___y_2542_, v___y_2543_, v___x_2568_, v___y_2545_);
if (lean_obj_tag(v___x_2569_) == 0)
{
lean_object* v_a_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; uint8_t v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; 
v_a_2570_ = lean_ctor_get(v___x_2569_, 0);
lean_inc(v_a_2570_);
lean_dec_ref_known(v___x_2569_, 1);
v___x_2571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2571_, 0, v_a_2570_);
v___x_2572_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__0));
v___x_2573_ = 0;
v___x_2574_ = lean_box(0);
v___x_2575_ = l_Lean_Elab_Tactic_elabTermWithHoles(v_val_2536_, v___x_2571_, v___x_2572_, v___x_2573_, v___x_2574_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_, v___y_2542_, v___y_2543_, v___x_2568_, v___y_2545_);
if (lean_obj_tag(v___x_2575_) == 0)
{
lean_object* v_a_2576_; lean_object* v_fst_2577_; lean_object* v_snd_2578_; lean_object* v___x_2580_; uint8_t v_isShared_2581_; uint8_t v_isSharedCheck_2611_; 
v_a_2576_ = lean_ctor_get(v___x_2575_, 0);
lean_inc(v_a_2576_);
lean_dec_ref_known(v___x_2575_, 1);
v_fst_2577_ = lean_ctor_get(v_a_2576_, 0);
v_snd_2578_ = lean_ctor_get(v_a_2576_, 1);
v_isSharedCheck_2611_ = !lean_is_exclusive(v_a_2576_);
if (v_isSharedCheck_2611_ == 0)
{
v___x_2580_ = v_a_2576_;
v_isShared_2581_ = v_isSharedCheck_2611_;
goto v_resetjp_2579_;
}
else
{
lean_inc(v_snd_2578_);
lean_inc(v_fst_2577_);
lean_dec(v_a_2576_);
v___x_2580_ = lean_box(0);
v_isShared_2581_ = v_isSharedCheck_2611_;
goto v_resetjp_2579_;
}
v_resetjp_2579_:
{
lean_object* v___y_2583_; lean_object* v___y_2584_; lean_object* v___x_2587_; 
lean_inc(v_fst_2577_);
v___x_2587_ = lp_batteries_Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0(v_fst_2537_, v_fst_2577_, v___y_2538_, v___y_2539_, v___y_2540_, v___y_2541_, v___y_2542_, v___y_2543_, v___x_2568_, v___y_2545_);
if (lean_obj_tag(v___x_2587_) == 0)
{
lean_object* v_a_2588_; uint8_t v___x_2589_; 
v_a_2588_ = lean_ctor_get(v___x_2587_, 0);
lean_inc(v_a_2588_);
lean_dec_ref_known(v___x_2587_, 1);
v___x_2589_ = lean_unbox(v_a_2588_);
lean_dec(v_a_2588_);
if (v___x_2589_ == 0)
{
lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2593_; 
v___x_2590_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__2, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__2_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__2);
lean_inc(v_fst_2577_);
v___x_2591_ = l_Lean_indentExpr(v_fst_2577_);
if (v_isShared_2581_ == 0)
{
lean_ctor_set_tag(v___x_2580_, 7);
lean_ctor_set(v___x_2580_, 1, v___x_2591_);
lean_ctor_set(v___x_2580_, 0, v___x_2590_);
v___x_2593_ = v___x_2580_;
goto v_reusejp_2592_;
}
else
{
lean_object* v_reuseFailAlloc_2602_; 
v_reuseFailAlloc_2602_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2602_, 0, v___x_2590_);
lean_ctor_set(v_reuseFailAlloc_2602_, 1, v___x_2591_);
v___x_2593_ = v_reuseFailAlloc_2602_;
goto v_reusejp_2592_;
}
v_reusejp_2592_:
{
lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2599_; lean_object* v___x_2600_; lean_object* v___x_2601_; 
v___x_2594_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__4, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__4_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__4);
v___x_2595_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2595_, 0, v___x_2593_);
lean_ctor_set(v___x_2595_, 1, v___x_2594_);
lean_inc(v_fst_2537_);
v___x_2596_ = l_Lean_Expr_mvar___override(v_fst_2537_);
v___x_2597_ = l_Lean_MessageData_ofExpr(v___x_2596_);
v___x_2598_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2598_, 0, v___x_2595_);
lean_ctor_set(v___x_2598_, 1, v___x_2597_);
v___x_2599_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__6, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__6_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___closed__6);
v___x_2600_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2600_, 0, v___x_2598_);
lean_ctor_set(v___x_2600_, 1, v___x_2599_);
v___x_2601_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic_findGoalOfPatt_spec__4___redArg(v___x_2600_, v___y_2542_, v___y_2543_, v___x_2568_, v___y_2545_);
lean_dec_ref(v___x_2568_);
if (lean_obj_tag(v___x_2601_) == 0)
{
lean_dec_ref_known(v___x_2601_, 1);
v___y_2583_ = v___y_2539_;
v___y_2584_ = v___y_2543_;
goto v___jp_2582_;
}
else
{
lean_dec(v_snd_2578_);
lean_dec(v_fst_2577_);
lean_dec(v_fst_2537_);
return v___x_2601_;
}
}
}
else
{
lean_del_object(v___x_2580_);
lean_dec_ref(v___x_2568_);
v___y_2583_ = v___y_2539_;
v___y_2584_ = v___y_2543_;
goto v___jp_2582_;
}
}
else
{
lean_object* v_a_2603_; lean_object* v___x_2605_; uint8_t v_isShared_2606_; uint8_t v_isSharedCheck_2610_; 
lean_del_object(v___x_2580_);
lean_dec(v_snd_2578_);
lean_dec(v_fst_2577_);
lean_dec_ref(v___x_2568_);
lean_dec(v_fst_2537_);
v_a_2603_ = lean_ctor_get(v___x_2587_, 0);
v_isSharedCheck_2610_ = !lean_is_exclusive(v___x_2587_);
if (v_isSharedCheck_2610_ == 0)
{
v___x_2605_ = v___x_2587_;
v_isShared_2606_ = v_isSharedCheck_2610_;
goto v_resetjp_2604_;
}
else
{
lean_inc(v_a_2603_);
lean_dec(v___x_2587_);
v___x_2605_ = lean_box(0);
v_isShared_2606_ = v_isSharedCheck_2610_;
goto v_resetjp_2604_;
}
v_resetjp_2604_:
{
lean_object* v___x_2608_; 
if (v_isShared_2606_ == 0)
{
v___x_2608_ = v___x_2605_;
goto v_reusejp_2607_;
}
else
{
lean_object* v_reuseFailAlloc_2609_; 
v_reuseFailAlloc_2609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2609_, 0, v_a_2603_);
v___x_2608_ = v_reuseFailAlloc_2609_;
goto v_reusejp_2607_;
}
v_reusejp_2607_:
{
return v___x_2608_;
}
}
}
v___jp_2582_:
{
lean_object* v___x_2585_; 
v___x_2585_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic_findGoalOfPatt_spec__2___redArg(v_fst_2537_, v_fst_2577_, v___y_2584_);
if (lean_obj_tag(v___x_2585_) == 0)
{
lean_object* v___x_2586_; 
lean_dec_ref_known(v___x_2585_, 1);
v___x_2586_ = l_Lean_Elab_Tactic_setGoals___redArg(v_snd_2578_, v___y_2583_);
return v___x_2586_;
}
else
{
lean_dec(v_snd_2578_);
return v___x_2585_;
}
}
}
}
else
{
lean_object* v_a_2612_; lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2619_; 
lean_dec_ref(v___x_2568_);
lean_dec(v_fst_2537_);
v_a_2612_ = lean_ctor_get(v___x_2575_, 0);
v_isSharedCheck_2619_ = !lean_is_exclusive(v___x_2575_);
if (v_isSharedCheck_2619_ == 0)
{
v___x_2614_ = v___x_2575_;
v_isShared_2615_ = v_isSharedCheck_2619_;
goto v_resetjp_2613_;
}
else
{
lean_inc(v_a_2612_);
lean_dec(v___x_2575_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2619_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v___x_2617_; 
if (v_isShared_2615_ == 0)
{
v___x_2617_ = v___x_2614_;
goto v_reusejp_2616_;
}
else
{
lean_object* v_reuseFailAlloc_2618_; 
v_reuseFailAlloc_2618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2618_, 0, v_a_2612_);
v___x_2617_ = v_reuseFailAlloc_2618_;
goto v_reusejp_2616_;
}
v_reusejp_2616_:
{
return v___x_2617_;
}
}
}
}
else
{
lean_object* v_a_2620_; lean_object* v___x_2622_; uint8_t v_isShared_2623_; uint8_t v_isSharedCheck_2627_; 
lean_dec_ref(v___x_2568_);
lean_dec(v_fst_2537_);
lean_dec(v_val_2536_);
v_a_2620_ = lean_ctor_get(v___x_2569_, 0);
v_isSharedCheck_2627_ = !lean_is_exclusive(v___x_2569_);
if (v_isSharedCheck_2627_ == 0)
{
v___x_2622_ = v___x_2569_;
v_isShared_2623_ = v_isSharedCheck_2627_;
goto v_resetjp_2621_;
}
else
{
lean_inc(v_a_2620_);
lean_dec(v___x_2569_);
v___x_2622_ = lean_box(0);
v_isShared_2623_ = v_isSharedCheck_2627_;
goto v_resetjp_2621_;
}
v_resetjp_2621_:
{
lean_object* v___x_2625_; 
if (v_isShared_2623_ == 0)
{
v___x_2625_ = v___x_2622_;
goto v_reusejp_2624_;
}
else
{
lean_object* v_reuseFailAlloc_2626_; 
v_reuseFailAlloc_2626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2626_, 0, v_a_2620_);
v___x_2625_ = v_reuseFailAlloc_2626_;
goto v_reusejp_2624_;
}
v_reusejp_2624_:
{
return v___x_2625_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___boxed(lean_object* v_val_2630_, lean_object* v_fst_2631_, lean_object* v___y_2632_, lean_object* v___y_2633_, lean_object* v___y_2634_, lean_object* v___y_2635_, lean_object* v___y_2636_, lean_object* v___y_2637_, lean_object* v___y_2638_, lean_object* v___y_2639_, lean_object* v___y_2640_){
_start:
{
lean_object* v_res_2641_; 
v_res_2641_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0(v_val_2630_, v_fst_2631_, v___y_2632_, v___y_2633_, v___y_2634_, v___y_2635_, v___y_2636_, v___y_2637_, v___y_2638_, v___y_2639_);
lean_dec(v___y_2639_);
lean_dec(v___y_2637_);
lean_dec_ref(v___y_2636_);
lean_dec(v___y_2635_);
lean_dec_ref(v___y_2634_);
lean_dec(v___y_2633_);
lean_dec_ref(v___y_2632_);
return v_res_2641_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3(lean_object* v_a_2642_, lean_object* v_stx_2643_, uint8_t v_close_2644_, lean_object* v_as_2645_, size_t v_sz_2646_, size_t v_i_2647_, lean_object* v_b_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_, lean_object* v___y_2655_, lean_object* v___y_2656_){
_start:
{
lean_object* v_a_2659_; uint8_t v___x_2663_; 
v___x_2663_ = lean_usize_dec_lt(v_i_2647_, v_sz_2646_);
if (v___x_2663_ == 0)
{
lean_object* v___x_2664_; 
lean_dec(v_stx_2643_);
lean_dec_ref(v_a_2642_);
v___x_2664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2664_, 0, v_b_2648_);
return v___x_2664_;
}
else
{
lean_object* v_snd_2665_; lean_object* v_snd_2666_; lean_object* v_snd_2667_; lean_object* v_fst_2668_; lean_object* v___x_2670_; uint8_t v_isShared_2671_; uint8_t v_isSharedCheck_2899_; 
v_snd_2665_ = lean_ctor_get(v_b_2648_, 1);
lean_inc(v_snd_2665_);
v_snd_2666_ = lean_ctor_get(v_snd_2665_, 1);
lean_inc(v_snd_2666_);
v_snd_2667_ = lean_ctor_get(v_snd_2666_, 1);
lean_inc(v_snd_2667_);
v_fst_2668_ = lean_ctor_get(v_b_2648_, 0);
v_isSharedCheck_2899_ = !lean_is_exclusive(v_b_2648_);
if (v_isSharedCheck_2899_ == 0)
{
lean_object* v_unused_2900_; 
v_unused_2900_ = lean_ctor_get(v_b_2648_, 1);
lean_dec(v_unused_2900_);
v___x_2670_ = v_b_2648_;
v_isShared_2671_ = v_isSharedCheck_2899_;
goto v_resetjp_2669_;
}
else
{
lean_inc(v_fst_2668_);
lean_dec(v_b_2648_);
v___x_2670_ = lean_box(0);
v_isShared_2671_ = v_isSharedCheck_2899_;
goto v_resetjp_2669_;
}
v_resetjp_2669_:
{
lean_object* v_fst_2672_; lean_object* v___x_2674_; uint8_t v_isShared_2675_; uint8_t v_isSharedCheck_2897_; 
v_fst_2672_ = lean_ctor_get(v_snd_2665_, 0);
v_isSharedCheck_2897_ = !lean_is_exclusive(v_snd_2665_);
if (v_isSharedCheck_2897_ == 0)
{
lean_object* v_unused_2898_; 
v_unused_2898_ = lean_ctor_get(v_snd_2665_, 1);
lean_dec(v_unused_2898_);
v___x_2674_ = v_snd_2665_;
v_isShared_2675_ = v_isSharedCheck_2897_;
goto v_resetjp_2673_;
}
else
{
lean_inc(v_fst_2672_);
lean_dec(v_snd_2665_);
v___x_2674_ = lean_box(0);
v_isShared_2675_ = v_isSharedCheck_2897_;
goto v_resetjp_2673_;
}
v_resetjp_2673_:
{
lean_object* v_fst_2676_; lean_object* v___x_2678_; uint8_t v_isShared_2679_; uint8_t v_isSharedCheck_2895_; 
v_fst_2676_ = lean_ctor_get(v_snd_2666_, 0);
v_isSharedCheck_2895_ = !lean_is_exclusive(v_snd_2666_);
if (v_isSharedCheck_2895_ == 0)
{
lean_object* v_unused_2896_; 
v_unused_2896_ = lean_ctor_get(v_snd_2666_, 1);
lean_dec(v_unused_2896_);
v___x_2678_ = v_snd_2666_;
v_isShared_2679_ = v_isSharedCheck_2895_;
goto v_resetjp_2677_;
}
else
{
lean_inc(v_fst_2676_);
lean_dec(v_snd_2666_);
v___x_2678_ = lean_box(0);
v_isShared_2679_ = v_isSharedCheck_2895_;
goto v_resetjp_2677_;
}
v_resetjp_2677_:
{
lean_object* v_array_2680_; lean_object* v_start_2681_; lean_object* v_stop_2682_; uint8_t v___x_2683_; 
v_array_2680_ = lean_ctor_get(v_snd_2667_, 0);
v_start_2681_ = lean_ctor_get(v_snd_2667_, 1);
v_stop_2682_ = lean_ctor_get(v_snd_2667_, 2);
v___x_2683_ = lean_nat_dec_lt(v_start_2681_, v_stop_2682_);
if (v___x_2683_ == 0)
{
lean_object* v___x_2685_; 
lean_dec(v_stx_2643_);
lean_dec_ref(v_a_2642_);
if (v_isShared_2679_ == 0)
{
v___x_2685_ = v___x_2678_;
goto v_reusejp_2684_;
}
else
{
lean_object* v_reuseFailAlloc_2693_; 
v_reuseFailAlloc_2693_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2693_, 0, v_fst_2676_);
lean_ctor_set(v_reuseFailAlloc_2693_, 1, v_snd_2667_);
v___x_2685_ = v_reuseFailAlloc_2693_;
goto v_reusejp_2684_;
}
v_reusejp_2684_:
{
lean_object* v___x_2687_; 
if (v_isShared_2675_ == 0)
{
lean_ctor_set(v___x_2674_, 1, v___x_2685_);
v___x_2687_ = v___x_2674_;
goto v_reusejp_2686_;
}
else
{
lean_object* v_reuseFailAlloc_2692_; 
v_reuseFailAlloc_2692_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2692_, 0, v_fst_2672_);
lean_ctor_set(v_reuseFailAlloc_2692_, 1, v___x_2685_);
v___x_2687_ = v_reuseFailAlloc_2692_;
goto v_reusejp_2686_;
}
v_reusejp_2686_:
{
lean_object* v___x_2689_; 
if (v_isShared_2671_ == 0)
{
lean_ctor_set(v___x_2670_, 1, v___x_2687_);
v___x_2689_ = v___x_2670_;
goto v_reusejp_2688_;
}
else
{
lean_object* v_reuseFailAlloc_2691_; 
v_reuseFailAlloc_2691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2691_, 0, v_fst_2668_);
lean_ctor_set(v_reuseFailAlloc_2691_, 1, v___x_2687_);
v___x_2689_ = v_reuseFailAlloc_2691_;
goto v_reusejp_2688_;
}
v_reusejp_2688_:
{
lean_object* v___x_2690_; 
v___x_2690_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2690_, 0, v___x_2689_);
return v___x_2690_;
}
}
}
}
else
{
lean_object* v___x_2695_; uint8_t v_isShared_2696_; uint8_t v_isSharedCheck_2891_; 
lean_inc(v_stop_2682_);
lean_inc(v_start_2681_);
lean_inc_ref(v_array_2680_);
v_isSharedCheck_2891_ = !lean_is_exclusive(v_snd_2667_);
if (v_isSharedCheck_2891_ == 0)
{
lean_object* v_unused_2892_; lean_object* v_unused_2893_; lean_object* v_unused_2894_; 
v_unused_2892_ = lean_ctor_get(v_snd_2667_, 2);
lean_dec(v_unused_2892_);
v_unused_2893_ = lean_ctor_get(v_snd_2667_, 1);
lean_dec(v_unused_2893_);
v_unused_2894_ = lean_ctor_get(v_snd_2667_, 0);
lean_dec(v_unused_2894_);
v___x_2695_ = v_snd_2667_;
v_isShared_2696_ = v_isSharedCheck_2891_;
goto v_resetjp_2694_;
}
else
{
lean_dec(v_snd_2667_);
v___x_2695_ = lean_box(0);
v_isShared_2696_ = v_isSharedCheck_2891_;
goto v_resetjp_2694_;
}
v_resetjp_2694_:
{
lean_object* v_array_2697_; lean_object* v_start_2698_; lean_object* v_stop_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2704_; 
v_array_2697_ = lean_ctor_get(v_fst_2676_, 0);
v_start_2698_ = lean_ctor_get(v_fst_2676_, 1);
v_stop_2699_ = lean_ctor_get(v_fst_2676_, 2);
v___x_2700_ = lean_array_fget(v_array_2680_, v_start_2681_);
v___x_2701_ = lean_unsigned_to_nat(1u);
v___x_2702_ = lean_nat_add(v_start_2681_, v___x_2701_);
lean_dec(v_start_2681_);
if (v_isShared_2696_ == 0)
{
lean_ctor_set(v___x_2695_, 1, v___x_2702_);
v___x_2704_ = v___x_2695_;
goto v_reusejp_2703_;
}
else
{
lean_object* v_reuseFailAlloc_2890_; 
v_reuseFailAlloc_2890_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2890_, 0, v_array_2680_);
lean_ctor_set(v_reuseFailAlloc_2890_, 1, v___x_2702_);
lean_ctor_set(v_reuseFailAlloc_2890_, 2, v_stop_2682_);
v___x_2704_ = v_reuseFailAlloc_2890_;
goto v_reusejp_2703_;
}
v_reusejp_2703_:
{
uint8_t v___x_2705_; 
v___x_2705_ = lean_nat_dec_lt(v_start_2698_, v_stop_2699_);
if (v___x_2705_ == 0)
{
lean_object* v___x_2707_; 
lean_dec(v___x_2700_);
lean_dec(v_stx_2643_);
lean_dec_ref(v_a_2642_);
if (v_isShared_2679_ == 0)
{
lean_ctor_set(v___x_2678_, 1, v___x_2704_);
v___x_2707_ = v___x_2678_;
goto v_reusejp_2706_;
}
else
{
lean_object* v_reuseFailAlloc_2715_; 
v_reuseFailAlloc_2715_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2715_, 0, v_fst_2676_);
lean_ctor_set(v_reuseFailAlloc_2715_, 1, v___x_2704_);
v___x_2707_ = v_reuseFailAlloc_2715_;
goto v_reusejp_2706_;
}
v_reusejp_2706_:
{
lean_object* v___x_2709_; 
if (v_isShared_2675_ == 0)
{
lean_ctor_set(v___x_2674_, 1, v___x_2707_);
v___x_2709_ = v___x_2674_;
goto v_reusejp_2708_;
}
else
{
lean_object* v_reuseFailAlloc_2714_; 
v_reuseFailAlloc_2714_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2714_, 0, v_fst_2672_);
lean_ctor_set(v_reuseFailAlloc_2714_, 1, v___x_2707_);
v___x_2709_ = v_reuseFailAlloc_2714_;
goto v_reusejp_2708_;
}
v_reusejp_2708_:
{
lean_object* v___x_2711_; 
if (v_isShared_2671_ == 0)
{
lean_ctor_set(v___x_2670_, 1, v___x_2709_);
v___x_2711_ = v___x_2670_;
goto v_reusejp_2710_;
}
else
{
lean_object* v_reuseFailAlloc_2713_; 
v_reuseFailAlloc_2713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2713_, 0, v_fst_2668_);
lean_ctor_set(v_reuseFailAlloc_2713_, 1, v___x_2709_);
v___x_2711_ = v_reuseFailAlloc_2713_;
goto v_reusejp_2710_;
}
v_reusejp_2710_:
{
lean_object* v___x_2712_; 
v___x_2712_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2712_, 0, v___x_2711_);
return v___x_2712_;
}
}
}
}
else
{
lean_object* v___x_2717_; uint8_t v_isShared_2718_; uint8_t v_isSharedCheck_2886_; 
lean_inc(v_stop_2699_);
lean_inc(v_start_2698_);
lean_inc_ref(v_array_2697_);
v_isSharedCheck_2886_ = !lean_is_exclusive(v_fst_2676_);
if (v_isSharedCheck_2886_ == 0)
{
lean_object* v_unused_2887_; lean_object* v_unused_2888_; lean_object* v_unused_2889_; 
v_unused_2887_ = lean_ctor_get(v_fst_2676_, 2);
lean_dec(v_unused_2887_);
v_unused_2888_ = lean_ctor_get(v_fst_2676_, 1);
lean_dec(v_unused_2888_);
v_unused_2889_ = lean_ctor_get(v_fst_2676_, 0);
lean_dec(v_unused_2889_);
v___x_2717_ = v_fst_2676_;
v_isShared_2718_ = v_isSharedCheck_2886_;
goto v_resetjp_2716_;
}
else
{
lean_dec(v_fst_2676_);
v___x_2717_ = lean_box(0);
v_isShared_2718_ = v_isSharedCheck_2886_;
goto v_resetjp_2716_;
}
v_resetjp_2716_:
{
lean_object* v___x_2719_; 
v___x_2719_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2719_) == 0)
{
lean_object* v_a_2720_; lean_object* v_a_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; 
v_a_2720_ = lean_ctor_get(v___x_2719_, 0);
lean_inc(v_a_2720_);
lean_dec_ref_known(v___x_2719_, 1);
v_a_2721_ = lean_array_uget_borrowed(v_as_2645_, v_i_2647_);
v___x_2722_ = lean_array_fget_borrowed(v_array_2697_, v_start_2698_);
lean_inc(v___x_2722_);
lean_inc(v_a_2721_);
v___x_2723_ = lp_batteries_Batteries_Tactic_findGoalOfPatt(v_a_2720_, v_a_2721_, v___x_2700_, v___x_2722_, v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2723_) == 0)
{
lean_object* v_a_2724_; lean_object* v_snd_2725_; lean_object* v_fst_2726_; lean_object* v___x_2728_; uint8_t v_isShared_2729_; uint8_t v_isSharedCheck_2869_; 
v_a_2724_ = lean_ctor_get(v___x_2723_, 0);
lean_inc(v_a_2724_);
lean_dec_ref_known(v___x_2723_, 1);
v_snd_2725_ = lean_ctor_get(v_a_2724_, 1);
v_fst_2726_ = lean_ctor_get(v_a_2724_, 0);
v_isSharedCheck_2869_ = !lean_is_exclusive(v_a_2724_);
if (v_isSharedCheck_2869_ == 0)
{
v___x_2728_ = v_a_2724_;
v_isShared_2729_ = v_isSharedCheck_2869_;
goto v_resetjp_2727_;
}
else
{
lean_inc(v_snd_2725_);
lean_inc(v_fst_2726_);
lean_dec(v_a_2724_);
v___x_2728_ = lean_box(0);
v_isShared_2729_ = v_isSharedCheck_2869_;
goto v_resetjp_2727_;
}
v_resetjp_2727_:
{
lean_object* v_fst_2730_; lean_object* v_snd_2731_; lean_object* v___x_2733_; uint8_t v_isShared_2734_; uint8_t v_isSharedCheck_2868_; 
v_fst_2730_ = lean_ctor_get(v_snd_2725_, 0);
v_snd_2731_ = lean_ctor_get(v_snd_2725_, 1);
v_isSharedCheck_2868_ = !lean_is_exclusive(v_snd_2725_);
if (v_isSharedCheck_2868_ == 0)
{
v___x_2733_ = v_snd_2725_;
v_isShared_2734_ = v_isSharedCheck_2868_;
goto v_resetjp_2732_;
}
else
{
lean_inc(v_snd_2731_);
lean_inc(v_fst_2730_);
lean_dec(v_snd_2725_);
v___x_2733_ = lean_box(0);
v_isShared_2734_ = v_isSharedCheck_2868_;
goto v_resetjp_2732_;
}
v_resetjp_2732_:
{
lean_object* v___x_2735_; 
v___x_2735_ = l_Lean_Elab_Tactic_setGoals___redArg(v_snd_2731_, v___y_2650_);
if (lean_obj_tag(v___x_2735_) == 0)
{
lean_object* v___x_2736_; lean_object* v___x_2738_; 
lean_dec_ref_known(v___x_2735_, 1);
v___x_2736_ = lean_nat_add(v_start_2698_, v___x_2701_);
lean_dec(v_start_2698_);
if (v_isShared_2718_ == 0)
{
lean_ctor_set(v___x_2717_, 1, v___x_2736_);
v___x_2738_ = v___x_2717_;
goto v_reusejp_2737_;
}
else
{
lean_object* v_reuseFailAlloc_2859_; 
v_reuseFailAlloc_2859_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2859_, 0, v_array_2697_);
lean_ctor_set(v_reuseFailAlloc_2859_, 1, v___x_2736_);
lean_ctor_set(v_reuseFailAlloc_2859_, 2, v_stop_2699_);
v___x_2738_ = v_reuseFailAlloc_2859_;
goto v_reusejp_2737_;
}
v_reusejp_2737_:
{
lean_object* v___x_2739_; 
v___x_2739_ = l_List_appendTR___redArg(v_fst_2672_, v_fst_2730_);
if (lean_obj_tag(v_a_2642_) == 0)
{
lean_object* v_val_2740_; lean_object* v___f_2741_; lean_object* v___x_2742_; 
lean_del_object(v___x_2674_);
lean_del_object(v___x_2670_);
v_val_2740_ = lean_ctor_get(v_a_2642_, 0);
lean_inc(v_fst_2726_);
lean_inc(v_val_2740_);
v___f_2741_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__0___boxed), 11, 2);
lean_closure_set(v___f_2741_, 0, v_val_2740_);
lean_closure_set(v___f_2741_, 1, v_fst_2726_);
v___x_2742_ = l_Lean_Elab_Tactic_run(v_fst_2726_, v___f_2741_, v___y_2651_, v___y_2652_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2742_) == 0)
{
lean_object* v_a_2743_; lean_object* v___x_2744_; lean_object* v___x_2746_; 
v_a_2743_ = lean_ctor_get(v___x_2742_, 0);
lean_inc(v_a_2743_);
lean_dec_ref_known(v___x_2742_, 1);
v___x_2744_ = l_List_appendTR___redArg(v_fst_2668_, v_a_2743_);
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 1, v___x_2704_);
lean_ctor_set(v___x_2733_, 0, v___x_2738_);
v___x_2746_ = v___x_2733_;
goto v_reusejp_2745_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v___x_2738_);
lean_ctor_set(v_reuseFailAlloc_2753_, 1, v___x_2704_);
v___x_2746_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2745_;
}
v_reusejp_2745_:
{
lean_object* v___x_2748_; 
if (v_isShared_2729_ == 0)
{
lean_ctor_set(v___x_2728_, 1, v___x_2746_);
lean_ctor_set(v___x_2728_, 0, v___x_2739_);
v___x_2748_ = v___x_2728_;
goto v_reusejp_2747_;
}
else
{
lean_object* v_reuseFailAlloc_2752_; 
v_reuseFailAlloc_2752_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2752_, 0, v___x_2739_);
lean_ctor_set(v_reuseFailAlloc_2752_, 1, v___x_2746_);
v___x_2748_ = v_reuseFailAlloc_2752_;
goto v_reusejp_2747_;
}
v_reusejp_2747_:
{
lean_object* v___x_2750_; 
if (v_isShared_2679_ == 0)
{
lean_ctor_set(v___x_2678_, 1, v___x_2748_);
lean_ctor_set(v___x_2678_, 0, v___x_2744_);
v___x_2750_ = v___x_2678_;
goto v_reusejp_2749_;
}
else
{
lean_object* v_reuseFailAlloc_2751_; 
v_reuseFailAlloc_2751_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2751_, 0, v___x_2744_);
lean_ctor_set(v_reuseFailAlloc_2751_, 1, v___x_2748_);
v___x_2750_ = v_reuseFailAlloc_2751_;
goto v_reusejp_2749_;
}
v_reusejp_2749_:
{
v_a_2659_ = v___x_2750_;
goto v___jp_2658_;
}
}
}
}
else
{
lean_object* v_a_2754_; lean_object* v___x_2756_; uint8_t v_isShared_2757_; uint8_t v_isSharedCheck_2761_; 
lean_dec_ref_known(v_a_2642_, 1);
lean_dec(v___x_2739_);
lean_dec_ref(v___x_2738_);
lean_del_object(v___x_2733_);
lean_del_object(v___x_2728_);
lean_dec_ref(v___x_2704_);
lean_del_object(v___x_2678_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
v_a_2754_ = lean_ctor_get(v___x_2742_, 0);
v_isSharedCheck_2761_ = !lean_is_exclusive(v___x_2742_);
if (v_isSharedCheck_2761_ == 0)
{
v___x_2756_ = v___x_2742_;
v_isShared_2757_ = v_isSharedCheck_2761_;
goto v_resetjp_2755_;
}
else
{
lean_inc(v_a_2754_);
lean_dec(v___x_2742_);
v___x_2756_ = lean_box(0);
v_isShared_2757_ = v_isSharedCheck_2761_;
goto v_resetjp_2755_;
}
v_resetjp_2755_:
{
lean_object* v___x_2759_; 
if (v_isShared_2757_ == 0)
{
v___x_2759_ = v___x_2756_;
goto v_reusejp_2758_;
}
else
{
lean_object* v_reuseFailAlloc_2760_; 
v_reuseFailAlloc_2760_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2760_, 0, v_a_2754_);
v___x_2759_ = v_reuseFailAlloc_2760_;
goto v_reusejp_2758_;
}
v_reusejp_2758_:
{
return v___x_2759_;
}
}
}
}
else
{
lean_object* v_val_2762_; lean_object* v_fst_2763_; lean_object* v_snd_2764_; lean_object* v___x_2766_; uint8_t v_isShared_2767_; uint8_t v_isSharedCheck_2858_; 
v_val_2762_ = lean_ctor_get(v_a_2642_, 0);
lean_inc(v_val_2762_);
v_fst_2763_ = lean_ctor_get(v_val_2762_, 0);
v_snd_2764_ = lean_ctor_get(v_val_2762_, 1);
v_isSharedCheck_2858_ = !lean_is_exclusive(v_val_2762_);
if (v_isSharedCheck_2858_ == 0)
{
v___x_2766_ = v_val_2762_;
v_isShared_2767_ = v_isSharedCheck_2858_;
goto v_resetjp_2765_;
}
else
{
lean_inc(v_snd_2764_);
lean_inc(v_fst_2763_);
lean_dec(v_val_2762_);
v___x_2766_ = lean_box(0);
v_isShared_2767_ = v_isSharedCheck_2858_;
goto v_resetjp_2765_;
}
v_resetjp_2765_:
{
lean_object* v___y_2769_; lean_object* v___y_2770_; lean_object* v___y_2771_; lean_object* v___y_2772_; lean_object* v___y_2773_; lean_object* v___y_2774_; 
if (v_close_2644_ == 0)
{
lean_object* v___x_2808_; 
lean_del_object(v___x_2766_);
lean_del_object(v___x_2733_);
lean_del_object(v___x_2728_);
lean_inc(v_fst_2726_);
v___x_2808_ = l_Lean_MVarId_getTag(v_fst_2726_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2808_) == 0)
{
lean_object* v_a_2809_; lean_object* v___x_2810_; lean_object* v___x_2811_; lean_object* v___x_2812_; 
v_a_2809_ = lean_ctor_get(v___x_2808_, 0);
lean_inc(v_a_2809_);
lean_dec_ref_known(v___x_2808_, 1);
lean_inc(v_snd_2764_);
v___x_2810_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_2810_, 0, v_snd_2764_);
v___x_2811_ = lean_alloc_closure((void*)(lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___boxed), 13, 4);
lean_closure_set(v___x_2811_, 0, lean_box(0));
lean_closure_set(v___x_2811_, 1, v_fst_2763_);
lean_closure_set(v___x_2811_, 2, v_snd_2764_);
lean_closure_set(v___x_2811_, 3, v___x_2810_);
v___x_2812_ = l_Lean_Elab_Tactic_run(v_fst_2726_, v___x_2811_, v___y_2651_, v___y_2652_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2812_) == 0)
{
lean_object* v_a_2813_; 
v_a_2813_ = lean_ctor_get(v___x_2812_, 0);
lean_inc(v_a_2813_);
lean_dec_ref_known(v___x_2812_, 1);
if (lean_obj_tag(v_a_2813_) == 1)
{
lean_object* v_tail_2825_; 
v_tail_2825_ = lean_ctor_get(v_a_2813_, 1);
if (lean_obj_tag(v_tail_2825_) == 0)
{
lean_object* v_head_2826_; lean_object* v___x_2827_; 
v_head_2826_ = lean_ctor_get(v_a_2813_, 0);
lean_inc(v_head_2826_);
v___x_2827_ = l_Lean_MVarId_setTag___redArg(v_head_2826_, v_a_2809_, v___y_2654_);
if (lean_obj_tag(v___x_2827_) == 0)
{
lean_dec_ref_known(v___x_2827_, 1);
goto v___jp_2814_;
}
else
{
lean_object* v_a_2828_; lean_object* v___x_2830_; uint8_t v_isShared_2831_; uint8_t v_isSharedCheck_2835_; 
lean_dec_ref_known(v_a_2813_, 2);
lean_dec_ref_known(v_a_2642_, 1);
lean_dec(v___x_2739_);
lean_dec_ref(v___x_2738_);
lean_dec_ref(v___x_2704_);
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_del_object(v___x_2670_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
v_a_2828_ = lean_ctor_get(v___x_2827_, 0);
v_isSharedCheck_2835_ = !lean_is_exclusive(v___x_2827_);
if (v_isSharedCheck_2835_ == 0)
{
v___x_2830_ = v___x_2827_;
v_isShared_2831_ = v_isSharedCheck_2835_;
goto v_resetjp_2829_;
}
else
{
lean_inc(v_a_2828_);
lean_dec(v___x_2827_);
v___x_2830_ = lean_box(0);
v_isShared_2831_ = v_isSharedCheck_2835_;
goto v_resetjp_2829_;
}
v_resetjp_2829_:
{
lean_object* v___x_2833_; 
if (v_isShared_2831_ == 0)
{
v___x_2833_ = v___x_2830_;
goto v_reusejp_2832_;
}
else
{
lean_object* v_reuseFailAlloc_2834_; 
v_reuseFailAlloc_2834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2834_, 0, v_a_2828_);
v___x_2833_ = v_reuseFailAlloc_2834_;
goto v_reusejp_2832_;
}
v_reusejp_2832_:
{
return v___x_2833_;
}
}
}
}
else
{
lean_dec(v_a_2809_);
goto v___jp_2814_;
}
}
else
{
lean_dec(v_a_2809_);
goto v___jp_2814_;
}
v___jp_2814_:
{
lean_object* v___x_2815_; lean_object* v___x_2817_; 
v___x_2815_ = l_List_appendTR___redArg(v_fst_2668_, v_a_2813_);
if (v_isShared_2679_ == 0)
{
lean_ctor_set(v___x_2678_, 1, v___x_2704_);
lean_ctor_set(v___x_2678_, 0, v___x_2738_);
v___x_2817_ = v___x_2678_;
goto v_reusejp_2816_;
}
else
{
lean_object* v_reuseFailAlloc_2824_; 
v_reuseFailAlloc_2824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2824_, 0, v___x_2738_);
lean_ctor_set(v_reuseFailAlloc_2824_, 1, v___x_2704_);
v___x_2817_ = v_reuseFailAlloc_2824_;
goto v_reusejp_2816_;
}
v_reusejp_2816_:
{
lean_object* v___x_2819_; 
if (v_isShared_2675_ == 0)
{
lean_ctor_set(v___x_2674_, 1, v___x_2817_);
lean_ctor_set(v___x_2674_, 0, v___x_2739_);
v___x_2819_ = v___x_2674_;
goto v_reusejp_2818_;
}
else
{
lean_object* v_reuseFailAlloc_2823_; 
v_reuseFailAlloc_2823_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2823_, 0, v___x_2739_);
lean_ctor_set(v_reuseFailAlloc_2823_, 1, v___x_2817_);
v___x_2819_ = v_reuseFailAlloc_2823_;
goto v_reusejp_2818_;
}
v_reusejp_2818_:
{
lean_object* v___x_2821_; 
if (v_isShared_2671_ == 0)
{
lean_ctor_set(v___x_2670_, 1, v___x_2819_);
lean_ctor_set(v___x_2670_, 0, v___x_2815_);
v___x_2821_ = v___x_2670_;
goto v_reusejp_2820_;
}
else
{
lean_object* v_reuseFailAlloc_2822_; 
v_reuseFailAlloc_2822_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2822_, 0, v___x_2815_);
lean_ctor_set(v_reuseFailAlloc_2822_, 1, v___x_2819_);
v___x_2821_ = v_reuseFailAlloc_2822_;
goto v_reusejp_2820_;
}
v_reusejp_2820_:
{
v_a_2659_ = v___x_2821_;
goto v___jp_2658_;
}
}
}
}
}
else
{
lean_object* v_a_2836_; lean_object* v___x_2838_; uint8_t v_isShared_2839_; uint8_t v_isSharedCheck_2843_; 
lean_dec(v_a_2809_);
lean_dec_ref_known(v_a_2642_, 1);
lean_dec(v___x_2739_);
lean_dec_ref(v___x_2738_);
lean_dec_ref(v___x_2704_);
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_del_object(v___x_2670_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
v_a_2836_ = lean_ctor_get(v___x_2812_, 0);
v_isSharedCheck_2843_ = !lean_is_exclusive(v___x_2812_);
if (v_isSharedCheck_2843_ == 0)
{
v___x_2838_ = v___x_2812_;
v_isShared_2839_ = v_isSharedCheck_2843_;
goto v_resetjp_2837_;
}
else
{
lean_inc(v_a_2836_);
lean_dec(v___x_2812_);
v___x_2838_ = lean_box(0);
v_isShared_2839_ = v_isSharedCheck_2843_;
goto v_resetjp_2837_;
}
v_resetjp_2837_:
{
lean_object* v___x_2841_; 
if (v_isShared_2839_ == 0)
{
v___x_2841_ = v___x_2838_;
goto v_reusejp_2840_;
}
else
{
lean_object* v_reuseFailAlloc_2842_; 
v_reuseFailAlloc_2842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2842_, 0, v_a_2836_);
v___x_2841_ = v_reuseFailAlloc_2842_;
goto v_reusejp_2840_;
}
v_reusejp_2840_:
{
return v___x_2841_;
}
}
}
}
else
{
lean_object* v_a_2844_; lean_object* v___x_2846_; uint8_t v_isShared_2847_; uint8_t v_isSharedCheck_2851_; 
lean_dec(v_snd_2764_);
lean_dec(v_fst_2763_);
lean_dec_ref_known(v_a_2642_, 1);
lean_dec(v___x_2739_);
lean_dec_ref(v___x_2738_);
lean_dec(v_fst_2726_);
lean_dec_ref(v___x_2704_);
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_del_object(v___x_2670_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
v_a_2844_ = lean_ctor_get(v___x_2808_, 0);
v_isSharedCheck_2851_ = !lean_is_exclusive(v___x_2808_);
if (v_isSharedCheck_2851_ == 0)
{
v___x_2846_ = v___x_2808_;
v_isShared_2847_ = v_isSharedCheck_2851_;
goto v_resetjp_2845_;
}
else
{
lean_inc(v_a_2844_);
lean_dec(v___x_2808_);
v___x_2846_ = lean_box(0);
v_isShared_2847_ = v_isSharedCheck_2851_;
goto v_resetjp_2845_;
}
v_resetjp_2845_:
{
lean_object* v___x_2849_; 
if (v_isShared_2847_ == 0)
{
v___x_2849_ = v___x_2846_;
goto v_reusejp_2848_;
}
else
{
lean_object* v_reuseFailAlloc_2850_; 
v_reuseFailAlloc_2850_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2850_, 0, v_a_2844_);
v___x_2849_ = v_reuseFailAlloc_2850_;
goto v_reusejp_2848_;
}
v_reusejp_2848_:
{
return v___x_2849_;
}
}
}
}
else
{
lean_object* v___x_2852_; uint8_t v___x_2853_; 
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_del_object(v___x_2670_);
v___x_2852_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___closed__1));
lean_inc(v_a_2721_);
v___x_2853_ = l_Lean_Syntax_isOfKind(v_a_2721_, v___x_2852_);
if (v___x_2853_ == 0)
{
v___y_2769_ = v___y_2651_;
v___y_2770_ = v___y_2652_;
v___y_2771_ = v___y_2653_;
v___y_2772_ = v___y_2654_;
v___y_2773_ = v___y_2655_;
v___y_2774_ = v___y_2656_;
goto v___jp_2768_;
}
else
{
lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; uint8_t v___x_2857_; 
v___x_2854_ = lean_unsigned_to_nat(0u);
v___x_2855_ = l_Lean_Syntax_getArg(v_a_2721_, v___x_2854_);
v___x_2856_ = ((lean_object*)(lp_batteries_Batteries_Tactic_findGoalOfPatt___lam__1___closed__1));
v___x_2857_ = l_Lean_Syntax_isOfKind(v___x_2855_, v___x_2856_);
if (v___x_2857_ == 0)
{
if (v___x_2857_ == 0)
{
v___y_2769_ = v___y_2651_;
v___y_2770_ = v___y_2652_;
v___y_2771_ = v___y_2653_;
v___y_2772_ = v___y_2654_;
v___y_2773_ = v___y_2655_;
v___y_2774_ = v___y_2656_;
goto v___jp_2768_;
}
else
{
goto v___jp_2797_;
}
}
else
{
goto v___jp_2797_;
}
}
}
v___jp_2768_:
{
lean_object* v___x_2775_; lean_object* v___f_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2779_; 
lean_inc(v_snd_2764_);
v___x_2775_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_2775_, 0, v_snd_2764_);
lean_inc(v_stx_2643_);
v___f_2776_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___lam__2___boxed), 11, 2);
lean_closure_set(v___f_2776_, 0, v_stx_2643_);
lean_closure_set(v___f_2776_, 1, v___x_2775_);
v___x_2777_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_closeUsingOrAdmit___boxed), 10, 1);
lean_closure_set(v___x_2777_, 0, v___f_2776_);
v___x_2778_ = lean_alloc_closure((void*)(lp_batteries_Lean_Elab_Tactic_withCaseRef___at___00Batteries_Tactic_evalCase_spec__2___boxed), 13, 4);
lean_closure_set(v___x_2778_, 0, lean_box(0));
lean_closure_set(v___x_2778_, 1, v_fst_2763_);
lean_closure_set(v___x_2778_, 2, v_snd_2764_);
lean_closure_set(v___x_2778_, 3, v___x_2777_);
v___x_2779_ = l_Lean_Elab_Tactic_run(v_fst_2726_, v___x_2778_, v___y_2769_, v___y_2770_, v___y_2771_, v___y_2772_, v___y_2773_, v___y_2774_);
if (lean_obj_tag(v___x_2779_) == 0)
{
lean_object* v___x_2781_; 
lean_dec_ref_known(v___x_2779_, 1);
if (v_isShared_2767_ == 0)
{
lean_ctor_set(v___x_2766_, 1, v___x_2704_);
lean_ctor_set(v___x_2766_, 0, v___x_2738_);
v___x_2781_ = v___x_2766_;
goto v_reusejp_2780_;
}
else
{
lean_object* v_reuseFailAlloc_2788_; 
v_reuseFailAlloc_2788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2788_, 0, v___x_2738_);
lean_ctor_set(v_reuseFailAlloc_2788_, 1, v___x_2704_);
v___x_2781_ = v_reuseFailAlloc_2788_;
goto v_reusejp_2780_;
}
v_reusejp_2780_:
{
lean_object* v___x_2783_; 
if (v_isShared_2734_ == 0)
{
lean_ctor_set(v___x_2733_, 1, v___x_2781_);
lean_ctor_set(v___x_2733_, 0, v___x_2739_);
v___x_2783_ = v___x_2733_;
goto v_reusejp_2782_;
}
else
{
lean_object* v_reuseFailAlloc_2787_; 
v_reuseFailAlloc_2787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2787_, 0, v___x_2739_);
lean_ctor_set(v_reuseFailAlloc_2787_, 1, v___x_2781_);
v___x_2783_ = v_reuseFailAlloc_2787_;
goto v_reusejp_2782_;
}
v_reusejp_2782_:
{
lean_object* v___x_2785_; 
if (v_isShared_2729_ == 0)
{
lean_ctor_set(v___x_2728_, 1, v___x_2783_);
lean_ctor_set(v___x_2728_, 0, v_fst_2668_);
v___x_2785_ = v___x_2728_;
goto v_reusejp_2784_;
}
else
{
lean_object* v_reuseFailAlloc_2786_; 
v_reuseFailAlloc_2786_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2786_, 0, v_fst_2668_);
lean_ctor_set(v_reuseFailAlloc_2786_, 1, v___x_2783_);
v___x_2785_ = v_reuseFailAlloc_2786_;
goto v_reusejp_2784_;
}
v_reusejp_2784_:
{
v_a_2659_ = v___x_2785_;
goto v___jp_2658_;
}
}
}
}
else
{
lean_object* v_a_2789_; lean_object* v___x_2791_; uint8_t v_isShared_2792_; uint8_t v_isSharedCheck_2796_; 
lean_del_object(v___x_2766_);
lean_dec_ref_known(v_a_2642_, 1);
lean_dec(v___x_2739_);
lean_dec_ref(v___x_2738_);
lean_del_object(v___x_2733_);
lean_del_object(v___x_2728_);
lean_dec_ref(v___x_2704_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
v_a_2789_ = lean_ctor_get(v___x_2779_, 0);
v_isSharedCheck_2796_ = !lean_is_exclusive(v___x_2779_);
if (v_isSharedCheck_2796_ == 0)
{
v___x_2791_ = v___x_2779_;
v_isShared_2792_ = v_isSharedCheck_2796_;
goto v_resetjp_2790_;
}
else
{
lean_inc(v_a_2789_);
lean_dec(v___x_2779_);
v___x_2791_ = lean_box(0);
v_isShared_2792_ = v_isSharedCheck_2796_;
goto v_resetjp_2790_;
}
v_resetjp_2790_:
{
lean_object* v___x_2794_; 
if (v_isShared_2792_ == 0)
{
v___x_2794_ = v___x_2791_;
goto v_reusejp_2793_;
}
else
{
lean_object* v_reuseFailAlloc_2795_; 
v_reuseFailAlloc_2795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2795_, 0, v_a_2789_);
v___x_2794_ = v_reuseFailAlloc_2795_;
goto v_reusejp_2793_;
}
v_reusejp_2793_:
{
return v___x_2794_;
}
}
}
}
v___jp_2797_:
{
lean_object* v___x_2798_; lean_object* v___x_2799_; 
v___x_2798_ = lean_box(0);
lean_inc(v_fst_2726_);
v___x_2799_ = l_Lean_MVarId_setTag___redArg(v_fst_2726_, v___x_2798_, v___y_2654_);
if (lean_obj_tag(v___x_2799_) == 0)
{
lean_dec_ref_known(v___x_2799_, 1);
v___y_2769_ = v___y_2651_;
v___y_2770_ = v___y_2652_;
v___y_2771_ = v___y_2653_;
v___y_2772_ = v___y_2654_;
v___y_2773_ = v___y_2655_;
v___y_2774_ = v___y_2656_;
goto v___jp_2768_;
}
else
{
lean_object* v_a_2800_; lean_object* v___x_2802_; uint8_t v_isShared_2803_; uint8_t v_isSharedCheck_2807_; 
lean_del_object(v___x_2766_);
lean_dec(v_snd_2764_);
lean_dec(v_fst_2763_);
lean_dec_ref_known(v_a_2642_, 1);
lean_dec(v___x_2739_);
lean_dec_ref(v___x_2738_);
lean_del_object(v___x_2733_);
lean_del_object(v___x_2728_);
lean_dec(v_fst_2726_);
lean_dec_ref(v___x_2704_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
v_a_2800_ = lean_ctor_get(v___x_2799_, 0);
v_isSharedCheck_2807_ = !lean_is_exclusive(v___x_2799_);
if (v_isSharedCheck_2807_ == 0)
{
v___x_2802_ = v___x_2799_;
v_isShared_2803_ = v_isSharedCheck_2807_;
goto v_resetjp_2801_;
}
else
{
lean_inc(v_a_2800_);
lean_dec(v___x_2799_);
v___x_2802_ = lean_box(0);
v_isShared_2803_ = v_isSharedCheck_2807_;
goto v_resetjp_2801_;
}
v_resetjp_2801_:
{
lean_object* v___x_2805_; 
if (v_isShared_2803_ == 0)
{
v___x_2805_ = v___x_2802_;
goto v_reusejp_2804_;
}
else
{
lean_object* v_reuseFailAlloc_2806_; 
v_reuseFailAlloc_2806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2806_, 0, v_a_2800_);
v___x_2805_ = v_reuseFailAlloc_2806_;
goto v_reusejp_2804_;
}
v_reusejp_2804_:
{
return v___x_2805_;
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
lean_object* v_a_2860_; lean_object* v___x_2862_; uint8_t v_isShared_2863_; uint8_t v_isSharedCheck_2867_; 
lean_del_object(v___x_2733_);
lean_dec(v_fst_2730_);
lean_del_object(v___x_2728_);
lean_dec(v_fst_2726_);
lean_del_object(v___x_2717_);
lean_dec_ref(v___x_2704_);
lean_dec(v_stop_2699_);
lean_dec(v_start_2698_);
lean_dec_ref(v_array_2697_);
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_dec(v_fst_2672_);
lean_del_object(v___x_2670_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
lean_dec_ref(v_a_2642_);
v_a_2860_ = lean_ctor_get(v___x_2735_, 0);
v_isSharedCheck_2867_ = !lean_is_exclusive(v___x_2735_);
if (v_isSharedCheck_2867_ == 0)
{
v___x_2862_ = v___x_2735_;
v_isShared_2863_ = v_isSharedCheck_2867_;
goto v_resetjp_2861_;
}
else
{
lean_inc(v_a_2860_);
lean_dec(v___x_2735_);
v___x_2862_ = lean_box(0);
v_isShared_2863_ = v_isSharedCheck_2867_;
goto v_resetjp_2861_;
}
v_resetjp_2861_:
{
lean_object* v___x_2865_; 
if (v_isShared_2863_ == 0)
{
v___x_2865_ = v___x_2862_;
goto v_reusejp_2864_;
}
else
{
lean_object* v_reuseFailAlloc_2866_; 
v_reuseFailAlloc_2866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2866_, 0, v_a_2860_);
v___x_2865_ = v_reuseFailAlloc_2866_;
goto v_reusejp_2864_;
}
v_reusejp_2864_:
{
return v___x_2865_;
}
}
}
}
}
}
else
{
lean_object* v_a_2870_; lean_object* v___x_2872_; uint8_t v_isShared_2873_; uint8_t v_isSharedCheck_2877_; 
lean_del_object(v___x_2717_);
lean_dec_ref(v___x_2704_);
lean_dec(v_stop_2699_);
lean_dec(v_start_2698_);
lean_dec_ref(v_array_2697_);
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_dec(v_fst_2672_);
lean_del_object(v___x_2670_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
lean_dec_ref(v_a_2642_);
v_a_2870_ = lean_ctor_get(v___x_2723_, 0);
v_isSharedCheck_2877_ = !lean_is_exclusive(v___x_2723_);
if (v_isSharedCheck_2877_ == 0)
{
v___x_2872_ = v___x_2723_;
v_isShared_2873_ = v_isSharedCheck_2877_;
goto v_resetjp_2871_;
}
else
{
lean_inc(v_a_2870_);
lean_dec(v___x_2723_);
v___x_2872_ = lean_box(0);
v_isShared_2873_ = v_isSharedCheck_2877_;
goto v_resetjp_2871_;
}
v_resetjp_2871_:
{
lean_object* v___x_2875_; 
if (v_isShared_2873_ == 0)
{
v___x_2875_ = v___x_2872_;
goto v_reusejp_2874_;
}
else
{
lean_object* v_reuseFailAlloc_2876_; 
v_reuseFailAlloc_2876_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2876_, 0, v_a_2870_);
v___x_2875_ = v_reuseFailAlloc_2876_;
goto v_reusejp_2874_;
}
v_reusejp_2874_:
{
return v___x_2875_;
}
}
}
}
else
{
lean_object* v_a_2878_; lean_object* v___x_2880_; uint8_t v_isShared_2881_; uint8_t v_isSharedCheck_2885_; 
lean_del_object(v___x_2717_);
lean_dec_ref(v___x_2704_);
lean_dec(v___x_2700_);
lean_dec(v_stop_2699_);
lean_dec(v_start_2698_);
lean_dec_ref(v_array_2697_);
lean_del_object(v___x_2678_);
lean_del_object(v___x_2674_);
lean_dec(v_fst_2672_);
lean_del_object(v___x_2670_);
lean_dec(v_fst_2668_);
lean_dec(v_stx_2643_);
lean_dec_ref(v_a_2642_);
v_a_2878_ = lean_ctor_get(v___x_2719_, 0);
v_isSharedCheck_2885_ = !lean_is_exclusive(v___x_2719_);
if (v_isSharedCheck_2885_ == 0)
{
v___x_2880_ = v___x_2719_;
v_isShared_2881_ = v_isSharedCheck_2885_;
goto v_resetjp_2879_;
}
else
{
lean_inc(v_a_2878_);
lean_dec(v___x_2719_);
v___x_2880_ = lean_box(0);
v_isShared_2881_ = v_isSharedCheck_2885_;
goto v_resetjp_2879_;
}
v_resetjp_2879_:
{
lean_object* v___x_2883_; 
if (v_isShared_2881_ == 0)
{
v___x_2883_ = v___x_2880_;
goto v_reusejp_2882_;
}
else
{
lean_object* v_reuseFailAlloc_2884_; 
v_reuseFailAlloc_2884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2884_, 0, v_a_2878_);
v___x_2883_ = v_reuseFailAlloc_2884_;
goto v_reusejp_2882_;
}
v_reusejp_2882_:
{
return v___x_2883_;
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
v___jp_2658_:
{
size_t v___x_2660_; size_t v___x_2661_; 
v___x_2660_ = ((size_t)1ULL);
v___x_2661_ = lean_usize_add(v_i_2647_, v___x_2660_);
v_i_2647_ = v___x_2661_;
v_b_2648_ = v_a_2659_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3___boxed(lean_object* v_a_2901_, lean_object* v_stx_2902_, lean_object* v_close_2903_, lean_object* v_as_2904_, lean_object* v_sz_2905_, lean_object* v_i_2906_, lean_object* v_b_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_, lean_object* v___y_2912_, lean_object* v___y_2913_, lean_object* v___y_2914_, lean_object* v___y_2915_, lean_object* v___y_2916_){
_start:
{
uint8_t v_close_boxed_2917_; size_t v_sz_boxed_2918_; size_t v_i_boxed_2919_; lean_object* v_res_2920_; 
v_close_boxed_2917_ = lean_unbox(v_close_2903_);
v_sz_boxed_2918_ = lean_unbox_usize(v_sz_2905_);
lean_dec(v_sz_2905_);
v_i_boxed_2919_ = lean_unbox_usize(v_i_2906_);
lean_dec(v_i_2906_);
v_res_2920_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3(v_a_2901_, v_stx_2902_, v_close_boxed_2917_, v_as_2904_, v_sz_boxed_2918_, v_i_boxed_2919_, v_b_2907_, v___y_2908_, v___y_2909_, v___y_2910_, v___y_2911_, v___y_2912_, v___y_2913_, v___y_2914_, v___y_2915_);
lean_dec(v___y_2915_);
lean_dec_ref(v___y_2914_);
lean_dec(v___y_2913_);
lean_dec_ref(v___y_2912_);
lean_dec(v___y_2911_);
lean_dec_ref(v___y_2910_);
lean_dec(v___y_2909_);
lean_dec_ref(v___y_2908_);
lean_dec_ref(v_as_2904_);
return v_res_2920_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_evalCase(uint8_t v_close_2921_, lean_object* v_stx_2922_, lean_object* v_tags_2923_, lean_object* v_hss_2924_, lean_object* v_patts_x3f_2925_, lean_object* v_caseBody_2926_, lean_object* v_a_2927_, lean_object* v_a_2928_, lean_object* v_a_2929_, lean_object* v_a_2930_, lean_object* v_a_2931_, lean_object* v_a_2932_, lean_object* v_a_2933_, lean_object* v_a_2934_){
_start:
{
lean_object* v___x_2936_; 
v___x_2936_ = lp_batteries_Batteries_Tactic_processCasePattBody(v_caseBody_2926_, v_a_2927_, v_a_2928_, v_a_2929_, v_a_2930_, v_a_2931_, v_a_2932_, v_a_2933_, v_a_2934_);
if (lean_obj_tag(v___x_2936_) == 0)
{
lean_object* v_a_2937_; lean_object* v___x_2938_; lean_object* v___x_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; size_t v_sz_2947_; size_t v___x_2948_; lean_object* v___x_2949_; 
v_a_2937_ = lean_ctor_get(v___x_2936_, 0);
lean_inc(v_a_2937_);
lean_dec_ref_known(v___x_2936_, 1);
v___x_2938_ = lean_box(0);
v___x_2939_ = lean_unsigned_to_nat(0u);
v___x_2940_ = lean_array_get_size(v_hss_2924_);
v___x_2941_ = l_Array_toSubarray___redArg(v_hss_2924_, v___x_2939_, v___x_2940_);
v___x_2942_ = lean_array_get_size(v_patts_x3f_2925_);
v___x_2943_ = l_Array_toSubarray___redArg(v_patts_x3f_2925_, v___x_2939_, v___x_2942_);
v___x_2944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2944_, 0, v___x_2941_);
lean_ctor_set(v___x_2944_, 1, v___x_2943_);
v___x_2945_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2945_, 0, v___x_2938_);
lean_ctor_set(v___x_2945_, 1, v___x_2944_);
v___x_2946_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2946_, 0, v___x_2938_);
lean_ctor_set(v___x_2946_, 1, v___x_2945_);
v_sz_2947_ = lean_array_size(v_tags_2923_);
v___x_2948_ = ((size_t)0ULL);
v___x_2949_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic_evalCase_spec__3(v_a_2937_, v_stx_2922_, v_close_2921_, v_tags_2923_, v_sz_2947_, v___x_2948_, v___x_2946_, v_a_2927_, v_a_2928_, v_a_2929_, v_a_2930_, v_a_2931_, v_a_2932_, v_a_2933_, v_a_2934_);
if (lean_obj_tag(v___x_2949_) == 0)
{
lean_object* v_a_2950_; lean_object* v___x_2951_; 
v_a_2950_ = lean_ctor_get(v___x_2949_, 0);
lean_inc(v_a_2950_);
lean_dec_ref_known(v___x_2949_, 1);
v___x_2951_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v_a_2927_, v_a_2928_, v_a_2929_, v_a_2930_, v_a_2931_, v_a_2932_, v_a_2933_, v_a_2934_);
if (lean_obj_tag(v___x_2951_) == 0)
{
lean_object* v_snd_2952_; lean_object* v_a_2953_; lean_object* v_fst_2954_; lean_object* v_fst_2955_; lean_object* v___x_2956_; lean_object* v___x_2957_; lean_object* v___x_2958_; 
v_snd_2952_ = lean_ctor_get(v_a_2950_, 1);
lean_inc(v_snd_2952_);
v_a_2953_ = lean_ctor_get(v___x_2951_, 0);
lean_inc(v_a_2953_);
lean_dec_ref_known(v___x_2951_, 1);
v_fst_2954_ = lean_ctor_get(v_a_2950_, 0);
lean_inc(v_fst_2954_);
lean_dec(v_a_2950_);
v_fst_2955_ = lean_ctor_get(v_snd_2952_, 0);
lean_inc(v_fst_2955_);
lean_dec(v_snd_2952_);
v___x_2956_ = l_List_appendTR___redArg(v_fst_2954_, v_fst_2955_);
v___x_2957_ = l_List_appendTR___redArg(v___x_2956_, v_a_2953_);
v___x_2958_ = l_Lean_Elab_Tactic_setGoals___redArg(v___x_2957_, v_a_2928_);
return v___x_2958_;
}
else
{
lean_object* v_a_2959_; lean_object* v___x_2961_; uint8_t v_isShared_2962_; uint8_t v_isSharedCheck_2966_; 
lean_dec(v_a_2950_);
v_a_2959_ = lean_ctor_get(v___x_2951_, 0);
v_isSharedCheck_2966_ = !lean_is_exclusive(v___x_2951_);
if (v_isSharedCheck_2966_ == 0)
{
v___x_2961_ = v___x_2951_;
v_isShared_2962_ = v_isSharedCheck_2966_;
goto v_resetjp_2960_;
}
else
{
lean_inc(v_a_2959_);
lean_dec(v___x_2951_);
v___x_2961_ = lean_box(0);
v_isShared_2962_ = v_isSharedCheck_2966_;
goto v_resetjp_2960_;
}
v_resetjp_2960_:
{
lean_object* v___x_2964_; 
if (v_isShared_2962_ == 0)
{
v___x_2964_ = v___x_2961_;
goto v_reusejp_2963_;
}
else
{
lean_object* v_reuseFailAlloc_2965_; 
v_reuseFailAlloc_2965_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2965_, 0, v_a_2959_);
v___x_2964_ = v_reuseFailAlloc_2965_;
goto v_reusejp_2963_;
}
v_reusejp_2963_:
{
return v___x_2964_;
}
}
}
}
else
{
lean_object* v_a_2967_; lean_object* v___x_2969_; uint8_t v_isShared_2970_; uint8_t v_isSharedCheck_2974_; 
v_a_2967_ = lean_ctor_get(v___x_2949_, 0);
v_isSharedCheck_2974_ = !lean_is_exclusive(v___x_2949_);
if (v_isSharedCheck_2974_ == 0)
{
v___x_2969_ = v___x_2949_;
v_isShared_2970_ = v_isSharedCheck_2974_;
goto v_resetjp_2968_;
}
else
{
lean_inc(v_a_2967_);
lean_dec(v___x_2949_);
v___x_2969_ = lean_box(0);
v_isShared_2970_ = v_isSharedCheck_2974_;
goto v_resetjp_2968_;
}
v_resetjp_2968_:
{
lean_object* v___x_2972_; 
if (v_isShared_2970_ == 0)
{
v___x_2972_ = v___x_2969_;
goto v_reusejp_2971_;
}
else
{
lean_object* v_reuseFailAlloc_2973_; 
v_reuseFailAlloc_2973_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2973_, 0, v_a_2967_);
v___x_2972_ = v_reuseFailAlloc_2973_;
goto v_reusejp_2971_;
}
v_reusejp_2971_:
{
return v___x_2972_;
}
}
}
}
else
{
lean_object* v_a_2975_; lean_object* v___x_2977_; uint8_t v_isShared_2978_; uint8_t v_isSharedCheck_2982_; 
lean_dec_ref(v_patts_x3f_2925_);
lean_dec_ref(v_hss_2924_);
lean_dec(v_stx_2922_);
v_a_2975_ = lean_ctor_get(v___x_2936_, 0);
v_isSharedCheck_2982_ = !lean_is_exclusive(v___x_2936_);
if (v_isSharedCheck_2982_ == 0)
{
v___x_2977_ = v___x_2936_;
v_isShared_2978_ = v_isSharedCheck_2982_;
goto v_resetjp_2976_;
}
else
{
lean_inc(v_a_2975_);
lean_dec(v___x_2936_);
v___x_2977_ = lean_box(0);
v_isShared_2978_ = v_isSharedCheck_2982_;
goto v_resetjp_2976_;
}
v_resetjp_2976_:
{
lean_object* v___x_2980_; 
if (v_isShared_2978_ == 0)
{
v___x_2980_ = v___x_2977_;
goto v_reusejp_2979_;
}
else
{
lean_object* v_reuseFailAlloc_2981_; 
v_reuseFailAlloc_2981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2981_, 0, v_a_2975_);
v___x_2980_ = v_reuseFailAlloc_2981_;
goto v_reusejp_2979_;
}
v_reusejp_2979_:
{
return v___x_2980_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_evalCase___boxed(lean_object* v_close_2983_, lean_object* v_stx_2984_, lean_object* v_tags_2985_, lean_object* v_hss_2986_, lean_object* v_patts_x3f_2987_, lean_object* v_caseBody_2988_, lean_object* v_a_2989_, lean_object* v_a_2990_, lean_object* v_a_2991_, lean_object* v_a_2992_, lean_object* v_a_2993_, lean_object* v_a_2994_, lean_object* v_a_2995_, lean_object* v_a_2996_, lean_object* v_a_2997_){
_start:
{
uint8_t v_close_boxed_2998_; lean_object* v_res_2999_; 
v_close_boxed_2998_ = lean_unbox(v_close_2983_);
v_res_2999_ = lp_batteries_Batteries_Tactic_evalCase(v_close_boxed_2998_, v_stx_2984_, v_tags_2985_, v_hss_2986_, v_patts_x3f_2987_, v_caseBody_2988_, v_a_2989_, v_a_2990_, v_a_2991_, v_a_2992_, v_a_2993_, v_a_2994_, v_a_2995_, v_a_2996_);
lean_dec(v_a_2996_);
lean_dec_ref(v_a_2995_);
lean_dec(v_a_2994_);
lean_dec_ref(v_a_2993_);
lean_dec(v_a_2992_);
lean_dec_ref(v_a_2991_);
lean_dec(v_a_2990_);
lean_dec_ref(v_a_2989_);
lean_dec_ref(v_tags_2985_);
return v_res_2999_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2(lean_object* v___y_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_){
_start:
{
lean_object* v___x_3009_; 
v___x_3009_ = lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___redArg(v___y_3007_);
return v___x_3009_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2___boxed(lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_){
_start:
{
lean_object* v_res_3019_; 
v_res_3019_ = lp_batteries_Lean_Elab_getResetInfoTrees___at___00Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1_spec__2(v___y_3010_, v___y_3011_, v___y_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_, v___y_3017_);
lean_dec(v___y_3017_);
lean_dec_ref(v___y_3016_);
lean_dec(v___y_3015_);
lean_dec_ref(v___y_3014_);
lean_dec(v___y_3013_);
lean_dec_ref(v___y_3012_);
lean_dec(v___y_3011_);
lean_dec_ref(v___y_3010_);
return v_res_3019_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1(lean_object* v_00_u03b1_3020_, lean_object* v_x_3021_, lean_object* v_mkInfoTree_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_){
_start:
{
lean_object* v___x_3032_; 
v___x_3032_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___redArg(v_x_3021_, v_mkInfoTree_3022_, v___y_3023_, v___y_3024_, v___y_3025_, v___y_3026_, v___y_3027_, v___y_3028_, v___y_3029_, v___y_3030_);
return v___x_3032_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1___boxed(lean_object* v_00_u03b1_3033_, lean_object* v_x_3034_, lean_object* v_mkInfoTree_3035_, lean_object* v___y_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_, lean_object* v___y_3044_){
_start:
{
lean_object* v_res_3045_; 
v_res_3045_ = lp_batteries_Lean_Elab_withInfoTreeContext___at___00Batteries_Tactic_evalCase_spec__1(v_00_u03b1_3033_, v_x_3034_, v_mkInfoTree_3035_, v___y_3036_, v___y_3037_, v___y_3038_, v___y_3039_, v___y_3040_, v___y_3041_, v___y_3042_, v___y_3043_);
lean_dec(v___y_3043_);
lean_dec_ref(v___y_3042_);
lean_dec(v___y_3041_);
lean_dec_ref(v___y_3040_);
lean_dec(v___y_3039_);
lean_dec_ref(v___y_3038_);
lean_dec(v___y_3037_);
lean_dec_ref(v___y_3036_);
return v_res_3045_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_3046_, lean_object* v_m_3047_, lean_object* v_a_3048_){
_start:
{
uint8_t v___x_3049_; 
v___x_3049_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___redArg(v_m_3047_, v_a_3048_);
return v___x_3049_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_3050_, lean_object* v_m_3051_, lean_object* v_a_3052_){
_start:
{
uint8_t v_res_3053_; lean_object* v_r_3054_; 
v_res_3053_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2(v_00_u03b2_3050_, v_m_3051_, v_a_3052_);
lean_dec_ref(v_a_3052_);
lean_dec_ref(v_m_3051_);
v_r_3054_ = lean_box(v_res_3053_);
return v_r_3054_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_3055_, lean_object* v_m_3056_, lean_object* v_a_3057_, lean_object* v_b_3058_){
_start:
{
lean_object* v___x_3059_; 
v___x_3059_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3___redArg(v_m_3056_, v_a_3057_, v_b_3058_);
return v___x_3059_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10(lean_object* v_mvarId_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_){
_start:
{
lean_object* v___x_3071_; 
v___x_3071_ = lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___redArg(v_mvarId_3060_, v___y_3061_, v___y_3067_);
return v___x_3071_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10___boxed(lean_object* v_mvarId_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_, lean_object* v___y_3077_, lean_object* v___y_3078_, lean_object* v___y_3079_, lean_object* v___y_3080_, lean_object* v___y_3081_, lean_object* v___y_3082_){
_start:
{
lean_object* v_res_3083_; 
v_res_3083_ = lp_batteries_Lean_getExprMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__10(v_mvarId_3072_, v___y_3073_, v___y_3074_, v___y_3075_, v___y_3076_, v___y_3077_, v___y_3078_, v___y_3079_, v___y_3080_, v___y_3081_);
lean_dec(v___y_3081_);
lean_dec_ref(v___y_3080_);
lean_dec(v___y_3079_);
lean_dec_ref(v___y_3078_);
lean_dec(v___y_3077_);
lean_dec_ref(v___y_3076_);
lean_dec(v___y_3075_);
lean_dec_ref(v___y_3074_);
lean_dec(v_mvarId_3072_);
return v_res_3083_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11(lean_object* v_mvarId_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_, lean_object* v___y_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_){
_start:
{
lean_object* v___x_3095_; 
v___x_3095_ = lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___redArg(v_mvarId_3084_, v___y_3085_, v___y_3091_);
return v___x_3095_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11___boxed(lean_object* v_mvarId_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_){
_start:
{
lean_object* v_res_3107_; 
v_res_3107_ = lp_batteries_Lean_getDelayedMVarAssignment_x3f___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visitMVar___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__4_spec__11(v_mvarId_3096_, v___y_3097_, v___y_3098_, v___y_3099_, v___y_3100_, v___y_3101_, v___y_3102_, v___y_3103_, v___y_3104_, v___y_3105_);
lean_dec(v___y_3105_);
lean_dec_ref(v___y_3104_);
lean_dec(v___y_3103_);
lean_dec_ref(v___y_3102_);
lean_dec(v___y_3101_);
lean_dec_ref(v___y_3100_);
lean_dec(v___y_3099_);
lean_dec_ref(v___y_3098_);
lean_dec(v_mvarId_3096_);
return v_res_3107_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6(lean_object* v_00_u03b2_3108_, lean_object* v_a_3109_, lean_object* v_x_3110_){
_start:
{
uint8_t v___x_3111_; 
v___x_3111_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___redArg(v_a_3109_, v_x_3110_);
return v___x_3111_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6___boxed(lean_object* v_00_u03b2_3112_, lean_object* v_a_3113_, lean_object* v_x_3114_){
_start:
{
uint8_t v_res_3115_; lean_object* v_r_3116_; 
v_res_3115_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__2_spec__6(v_00_u03b2_3112_, v_a_3113_, v_x_3114_);
lean_dec(v_x_3114_);
lean_dec_ref(v_a_3113_);
v_r_3116_ = lean_box(v_res_3115_);
return v_r_3116_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8(lean_object* v_00_u03b2_3117_, lean_object* v_data_3118_){
_start:
{
lean_object* v___x_3119_; 
v___x_3119_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8___redArg(v_data_3118_);
return v___x_3119_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9(lean_object* v_00_u03b2_3120_, lean_object* v_i_3121_, lean_object* v_source_3122_, lean_object* v_target_3123_){
_start:
{
lean_object* v___x_3124_; 
v___x_3124_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9___redArg(v_i_3121_, v_source_3122_, v_target_3123_);
return v___x_3124_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9_spec__13(lean_object* v_00_u03b2_3125_, lean_object* v_x_3126_, lean_object* v_x_3127_){
_start:
{
lean_object* v___x_3128_; 
v___x_3128_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Lean_Util_OccursCheck_0__Lean_occursCheck_visit___at___00Lean_occursCheck___at___00Batteries_Tactic_evalCase_spec__0_spec__0_spec__3_spec__8_spec__9_spec__13___redArg(v_x_3126_, v_x_3127_);
return v___x_3128_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0(size_t v_sz_3135_, size_t v_i_3136_, lean_object* v_bs_3137_){
_start:
{
uint8_t v___x_3138_; 
v___x_3138_ = lean_usize_dec_lt(v_i_3136_, v_sz_3135_);
if (v___x_3138_ == 0)
{
lean_object* v___x_3139_; 
v___x_3139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3139_, 0, v_bs_3137_);
return v___x_3139_;
}
else
{
lean_object* v_v_3140_; lean_object* v___x_3141_; uint8_t v___x_3142_; 
v_v_3140_ = lean_array_uget(v_bs_3137_, v_i_3136_);
v___x_3141_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattArg___closed__3));
lean_inc(v_v_3140_);
v___x_3142_ = l_Lean_Syntax_isOfKind(v_v_3140_, v___x_3141_);
if (v___x_3142_ == 0)
{
lean_object* v___x_3143_; 
lean_dec(v_v_3140_);
lean_dec_ref(v_bs_3137_);
v___x_3143_ = lean_box(0);
return v___x_3143_;
}
else
{
lean_object* v___x_3144_; lean_object* v___x_3145_; lean_object* v___x_3146_; uint8_t v___x_3147_; 
v___x_3144_ = lean_unsigned_to_nat(0u);
v___x_3145_ = l_Lean_Syntax_getArg(v_v_3140_, v___x_3144_);
v___x_3146_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___closed__1));
lean_inc(v___x_3145_);
v___x_3147_ = l_Lean_Syntax_isOfKind(v___x_3145_, v___x_3146_);
if (v___x_3147_ == 0)
{
lean_object* v___x_3148_; 
lean_dec(v___x_3145_);
lean_dec(v_v_3140_);
lean_dec_ref(v_bs_3137_);
v___x_3148_ = lean_box(0);
return v___x_3148_;
}
else
{
lean_object* v___x_3149_; lean_object* v_bs_x27_3150_; lean_object* v_tags_3151_; lean_object* v___x_3152_; lean_object* v_patts_x3f_3154_; lean_object* v___x_3162_; uint8_t v___x_3163_; 
v___x_3149_ = lean_unsigned_to_nat(1u);
v_bs_x27_3150_ = lean_array_uset(v_bs_3137_, v_i_3136_, v___x_3144_);
v_tags_3151_ = l_Lean_Syntax_getArg(v___x_3145_, v___x_3144_);
v___x_3152_ = l_Lean_Syntax_getArg(v___x_3145_, v___x_3149_);
lean_dec(v___x_3145_);
v___x_3162_ = l_Lean_Syntax_getArg(v_v_3140_, v___x_3149_);
lean_dec(v_v_3140_);
v___x_3163_ = l_Lean_Syntax_isNone(v___x_3162_);
if (v___x_3163_ == 0)
{
lean_object* v___x_3164_; uint8_t v___x_3165_; 
v___x_3164_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_3162_);
v___x_3165_ = l_Lean_Syntax_matchesNull(v___x_3162_, v___x_3164_);
if (v___x_3165_ == 0)
{
lean_object* v___x_3166_; 
lean_dec(v___x_3162_);
lean_dec(v___x_3152_);
lean_dec(v_tags_3151_);
lean_dec_ref(v_bs_x27_3150_);
v___x_3166_ = lean_box(0);
return v___x_3166_;
}
else
{
lean_object* v_patts_x3f_3167_; lean_object* v___x_3168_; 
v_patts_x3f_3167_ = l_Lean_Syntax_getArg(v___x_3162_, v___x_3149_);
lean_dec(v___x_3162_);
v___x_3168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3168_, 0, v_patts_x3f_3167_);
v_patts_x3f_3154_ = v___x_3168_;
goto v___jp_3153_;
}
}
else
{
lean_object* v___x_3169_; 
lean_dec(v___x_3162_);
v___x_3169_ = lean_box(0);
v_patts_x3f_3154_ = v___x_3169_;
goto v___jp_3153_;
}
v___jp_3153_:
{
lean_object* v_hss_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; size_t v___x_3158_; size_t v___x_3159_; lean_object* v___x_3160_; 
v_hss_3155_ = l_Lean_Syntax_getArgs(v___x_3152_);
lean_dec(v___x_3152_);
v___x_3156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3156_, 0, v_hss_3155_);
lean_ctor_set(v___x_3156_, 1, v_patts_x3f_3154_);
v___x_3157_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3157_, 0, v_tags_3151_);
lean_ctor_set(v___x_3157_, 1, v___x_3156_);
v___x_3158_ = ((size_t)1ULL);
v___x_3159_ = lean_usize_add(v_i_3136_, v___x_3158_);
v___x_3160_ = lean_array_uset(v_bs_x27_3150_, v_i_3136_, v___x_3157_);
v_i_3136_ = v___x_3159_;
v_bs_3137_ = v___x_3160_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0___boxed(lean_object* v_sz_3170_, lean_object* v_i_3171_, lean_object* v_bs_3172_){
_start:
{
size_t v_sz_boxed_3173_; size_t v_i_boxed_3174_; lean_object* v_res_3175_; 
v_sz_boxed_3173_ = lean_unbox_usize(v_sz_3170_);
lean_dec(v_sz_3170_);
v_i_boxed_3174_ = lean_unbox_usize(v_i_3171_);
lean_dec(v_i_3171_);
v_res_3175_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0(v_sz_boxed_3173_, v_i_boxed_3174_, v_bs_3172_);
return v_res_3175_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2(size_t v_sz_3176_, size_t v_i_3177_, lean_object* v_bs_3178_){
_start:
{
uint8_t v___x_3179_; 
v___x_3179_ = lean_usize_dec_lt(v_i_3177_, v_sz_3176_);
if (v___x_3179_ == 0)
{
return v_bs_3178_;
}
else
{
lean_object* v_v_3180_; lean_object* v_snd_3181_; lean_object* v_fst_3182_; lean_object* v___x_3183_; lean_object* v_bs_x27_3184_; size_t v___x_3185_; size_t v___x_3186_; lean_object* v___x_3187_; 
v_v_3180_ = lean_array_uget_borrowed(v_bs_3178_, v_i_3177_);
v_snd_3181_ = lean_ctor_get(v_v_3180_, 1);
v_fst_3182_ = lean_ctor_get(v_snd_3181_, 0);
lean_inc(v_fst_3182_);
v___x_3183_ = lean_unsigned_to_nat(0u);
v_bs_x27_3184_ = lean_array_uset(v_bs_3178_, v_i_3177_, v___x_3183_);
v___x_3185_ = ((size_t)1ULL);
v___x_3186_ = lean_usize_add(v_i_3177_, v___x_3185_);
v___x_3187_ = lean_array_uset(v_bs_x27_3184_, v_i_3177_, v_fst_3182_);
v_i_3177_ = v___x_3186_;
v_bs_3178_ = v___x_3187_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2___boxed(lean_object* v_sz_3189_, lean_object* v_i_3190_, lean_object* v_bs_3191_){
_start:
{
size_t v_sz_boxed_3192_; size_t v_i_boxed_3193_; lean_object* v_res_3194_; 
v_sz_boxed_3192_ = lean_unbox_usize(v_sz_3189_);
lean_dec(v_sz_3189_);
v_i_boxed_3193_ = lean_unbox_usize(v_i_3190_);
lean_dec(v_i_3190_);
v_res_3194_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2(v_sz_boxed_3192_, v_i_boxed_3193_, v_bs_3191_);
return v_res_3194_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3(size_t v_sz_3195_, size_t v_i_3196_, lean_object* v_bs_3197_){
_start:
{
uint8_t v___x_3198_; 
v___x_3198_ = lean_usize_dec_lt(v_i_3196_, v_sz_3195_);
if (v___x_3198_ == 0)
{
return v_bs_3197_;
}
else
{
lean_object* v_v_3199_; lean_object* v_fst_3200_; lean_object* v___x_3201_; lean_object* v_bs_x27_3202_; size_t v___x_3203_; size_t v___x_3204_; lean_object* v___x_3205_; 
v_v_3199_ = lean_array_uget_borrowed(v_bs_3197_, v_i_3196_);
v_fst_3200_ = lean_ctor_get(v_v_3199_, 0);
lean_inc(v_fst_3200_);
v___x_3201_ = lean_unsigned_to_nat(0u);
v_bs_x27_3202_ = lean_array_uset(v_bs_3197_, v_i_3196_, v___x_3201_);
v___x_3203_ = ((size_t)1ULL);
v___x_3204_ = lean_usize_add(v_i_3196_, v___x_3203_);
v___x_3205_ = lean_array_uset(v_bs_x27_3202_, v_i_3196_, v_fst_3200_);
v_i_3196_ = v___x_3204_;
v_bs_3197_ = v___x_3205_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3___boxed(lean_object* v_sz_3207_, lean_object* v_i_3208_, lean_object* v_bs_3209_){
_start:
{
size_t v_sz_boxed_3210_; size_t v_i_boxed_3211_; lean_object* v_res_3212_; 
v_sz_boxed_3210_ = lean_unbox_usize(v_sz_3207_);
lean_dec(v_sz_3207_);
v_i_boxed_3211_ = lean_unbox_usize(v_i_3208_);
lean_dec(v_i_3208_);
v_res_3212_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3(v_sz_boxed_3210_, v_i_boxed_3211_, v_bs_3209_);
return v_res_3212_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1(size_t v_sz_3213_, size_t v_i_3214_, lean_object* v_bs_3215_){
_start:
{
uint8_t v___x_3216_; 
v___x_3216_ = lean_usize_dec_lt(v_i_3214_, v_sz_3213_);
if (v___x_3216_ == 0)
{
return v_bs_3215_;
}
else
{
lean_object* v_v_3217_; lean_object* v_snd_3218_; lean_object* v_snd_3219_; lean_object* v___x_3220_; lean_object* v_bs_x27_3221_; size_t v___x_3222_; size_t v___x_3223_; lean_object* v___x_3224_; 
v_v_3217_ = lean_array_uget_borrowed(v_bs_3215_, v_i_3214_);
v_snd_3218_ = lean_ctor_get(v_v_3217_, 1);
v_snd_3219_ = lean_ctor_get(v_snd_3218_, 1);
lean_inc(v_snd_3219_);
v___x_3220_ = lean_unsigned_to_nat(0u);
v_bs_x27_3221_ = lean_array_uset(v_bs_3215_, v_i_3214_, v___x_3220_);
v___x_3222_ = ((size_t)1ULL);
v___x_3223_ = lean_usize_add(v_i_3214_, v___x_3222_);
v___x_3224_ = lean_array_uset(v_bs_x27_3221_, v_i_3214_, v_snd_3219_);
v_i_3214_ = v___x_3223_;
v_bs_3215_ = v___x_3224_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1___boxed(lean_object* v_sz_3226_, lean_object* v_i_3227_, lean_object* v_bs_3228_){
_start:
{
size_t v_sz_boxed_3229_; size_t v_i_boxed_3230_; lean_object* v_res_3231_; 
v_sz_boxed_3229_ = lean_unbox_usize(v_sz_3226_);
lean_dec(v_sz_3226_);
v_i_boxed_3230_ = lean_unbox_usize(v_i_3227_);
lean_dec(v_i_3227_);
v_res_3231_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1(v_sz_boxed_3229_, v_i_boxed_3230_, v_bs_3228_);
return v_res_3231_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1(lean_object* v_x_3232_, lean_object* v_a_3233_, lean_object* v_a_3234_, lean_object* v_a_3235_, lean_object* v_a_3236_, lean_object* v_a_3237_, lean_object* v_a_3238_, lean_object* v_a_3239_, lean_object* v_a_3240_){
_start:
{
lean_object* v___x_3242_; uint8_t v___x_3243_; 
v___x_3242_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt___closed__1));
lean_inc(v_x_3232_);
v___x_3243_ = l_Lean_Syntax_isOfKind(v_x_3232_, v___x_3242_);
if (v___x_3243_ == 0)
{
lean_object* v___x_3244_; 
lean_dec(v_x_3232_);
v___x_3244_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3244_;
}
else
{
lean_object* v___x_3245_; lean_object* v___x_3246_; lean_object* v___y_3248_; lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_3274_; lean_object* v___x_3275_; uint8_t v___x_3276_; 
v___x_3245_ = lean_unsigned_to_nat(0u);
v___x_3246_ = lean_unsigned_to_nat(1u);
v___x_3272_ = l_Lean_Syntax_getArg(v_x_3232_, v___x_3246_);
v___x_3273_ = l_Lean_Syntax_getArgs(v___x_3272_);
lean_dec(v___x_3272_);
v___x_3274_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__18));
v___x_3275_ = lean_array_get_size(v___x_3273_);
v___x_3276_ = lean_nat_dec_lt(v___x_3245_, v___x_3275_);
if (v___x_3276_ == 0)
{
lean_dec_ref(v___x_3273_);
v___y_3248_ = v___x_3274_;
goto v___jp_3247_;
}
else
{
lean_object* v___x_3277_; lean_object* v___x_3278_; uint8_t v___x_3279_; 
v___x_3277_ = lean_box(v___x_3243_);
v___x_3278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3278_, 0, v___x_3277_);
lean_ctor_set(v___x_3278_, 1, v___x_3274_);
v___x_3279_ = lean_nat_dec_le(v___x_3275_, v___x_3275_);
if (v___x_3279_ == 0)
{
if (v___x_3276_ == 0)
{
lean_dec_ref_known(v___x_3278_, 2);
lean_dec_ref(v___x_3273_);
v___y_3248_ = v___x_3274_;
goto v___jp_3247_;
}
else
{
size_t v___x_3280_; size_t v___x_3281_; lean_object* v___x_3282_; lean_object* v_snd_3283_; 
v___x_3280_ = ((size_t)0ULL);
v___x_3281_ = lean_usize_of_nat(v___x_3275_);
v___x_3282_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_3243_, v___x_3273_, v___x_3280_, v___x_3281_, v___x_3278_);
lean_dec_ref(v___x_3273_);
v_snd_3283_ = lean_ctor_get(v___x_3282_, 1);
lean_inc(v_snd_3283_);
lean_dec_ref(v___x_3282_);
v___y_3248_ = v_snd_3283_;
goto v___jp_3247_;
}
}
else
{
size_t v___x_3284_; size_t v___x_3285_; lean_object* v___x_3286_; lean_object* v_snd_3287_; 
v___x_3284_ = ((size_t)0ULL);
v___x_3285_ = lean_usize_of_nat(v___x_3275_);
v___x_3286_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_3243_, v___x_3273_, v___x_3284_, v___x_3285_, v___x_3278_);
lean_dec_ref(v___x_3273_);
v_snd_3287_ = lean_ctor_get(v___x_3286_, 1);
lean_inc(v_snd_3287_);
lean_dec_ref(v___x_3286_);
v___y_3248_ = v_snd_3287_;
goto v___jp_3247_;
}
}
v___jp_3247_:
{
size_t v_sz_3249_; size_t v___x_3250_; lean_object* v___x_3251_; 
v_sz_3249_ = lean_array_size(v___y_3248_);
v___x_3250_ = ((size_t)0ULL);
v___x_3251_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0(v_sz_3249_, v___x_3250_, v___y_3248_);
if (lean_obj_tag(v___x_3251_) == 0)
{
lean_object* v___x_3252_; 
lean_dec(v_x_3232_);
v___x_3252_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3252_;
}
else
{
lean_object* v_val_3253_; size_t v_sz_3254_; lean_object* v___x_3255_; lean_object* v___x_3256_; uint8_t v___x_3257_; 
v_val_3253_ = lean_ctor_get(v___x_3251_, 0);
lean_inc(v_val_3253_);
lean_dec_ref_known(v___x_3251_, 1);
v_sz_3254_ = lean_array_size(v_val_3253_);
v___x_3255_ = lean_unsigned_to_nat(2u);
v___x_3256_ = l_Lean_Syntax_getArg(v_x_3232_, v___x_3255_);
lean_dec(v_x_3232_);
lean_inc(v___x_3256_);
v___x_3257_ = l_Lean_Syntax_matchesNull(v___x_3256_, v___x_3246_);
if (v___x_3257_ == 0)
{
lean_object* v___x_3258_; 
lean_dec(v___x_3256_);
lean_dec(v_val_3253_);
v___x_3258_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3258_;
}
else
{
lean_object* v___x_3259_; lean_object* v___x_3260_; uint8_t v___x_3261_; 
v___x_3259_ = l_Lean_Syntax_getArg(v___x_3256_, v___x_3245_);
lean_dec(v___x_3256_);
v___x_3260_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattBody___closed__1));
lean_inc(v___x_3259_);
v___x_3261_ = l_Lean_Syntax_isOfKind(v___x_3259_, v___x_3260_);
if (v___x_3261_ == 0)
{
lean_object* v___x_3262_; 
lean_dec(v___x_3259_);
lean_dec(v_val_3253_);
v___x_3262_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3262_;
}
else
{
lean_object* v_caseBody_3263_; lean_object* v___x_3264_; uint8_t v___x_3265_; 
v_caseBody_3263_ = l_Lean_Syntax_getArg(v___x_3259_, v___x_3245_);
lean_dec(v___x_3259_);
v___x_3264_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattTac___closed__1));
lean_inc(v_caseBody_3263_);
v___x_3265_ = l_Lean_Syntax_isOfKind(v_caseBody_3263_, v___x_3264_);
if (v___x_3265_ == 0)
{
lean_object* v___x_3266_; 
lean_dec(v_caseBody_3263_);
lean_dec(v_val_3253_);
v___x_3266_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3266_;
}
else
{
lean_object* v_patts_x3f_3267_; lean_object* v_ref_3268_; lean_object* v_hss_3269_; lean_object* v_tags_3270_; lean_object* v___x_3271_; 
lean_inc_n(v_val_3253_, 2);
v_patts_x3f_3267_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1(v_sz_3254_, v___x_3250_, v_val_3253_);
v_ref_3268_ = lean_ctor_get(v_a_3239_, 5);
v_hss_3269_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2(v_sz_3254_, v___x_3250_, v_val_3253_);
v_tags_3270_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3(v_sz_3254_, v___x_3250_, v_val_3253_);
lean_inc(v_ref_3268_);
v___x_3271_ = lp_batteries_Batteries_Tactic_evalCase(v___x_3243_, v_ref_3268_, v_tags_3270_, v_hss_3269_, v_patts_x3f_3267_, v_caseBody_3263_, v_a_3233_, v_a_3234_, v_a_3235_, v_a_3236_, v_a_3237_, v_a_3238_, v_a_3239_, v_a_3240_);
lean_dec_ref(v_tags_3270_);
return v___x_3271_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1___boxed(lean_object* v_x_3288_, lean_object* v_a_3289_, lean_object* v_a_3290_, lean_object* v_a_3291_, lean_object* v_a_3292_, lean_object* v_a_3293_, lean_object* v_a_3294_, lean_object* v_a_3295_, lean_object* v_a_3296_, lean_object* v_a_3297_){
_start:
{
lean_object* v_res_3298_; 
v_res_3298_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1(v_x_3288_, v_a_3289_, v_a_3290_, v_a_3291_, v_a_3292_, v_a_3293_, v_a_3294_, v_a_3295_, v_a_3296_);
lean_dec(v_a_3296_);
lean_dec_ref(v_a_3295_);
lean_dec(v_a_3294_);
lean_dec_ref(v_a_3293_);
lean_dec(v_a_3292_);
lean_dec_ref(v_a_3291_);
lean_dec(v_a_3290_);
lean_dec_ref(v_a_3289_);
return v_res_3298_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt_x27__1(lean_object* v_x_3299_, lean_object* v_a_3300_, lean_object* v_a_3301_, lean_object* v_a_3302_, lean_object* v_a_3303_, lean_object* v_a_3304_, lean_object* v_a_3305_, lean_object* v_a_3306_, lean_object* v_a_3307_){
_start:
{
lean_object* v___y_3310_; lean_object* v___x_3328_; uint8_t v___x_3329_; 
v___x_3328_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePatt_x27___closed__1));
lean_inc(v_x_3299_);
v___x_3329_ = l_Lean_Syntax_isOfKind(v_x_3299_, v___x_3328_);
if (v___x_3329_ == 0)
{
lean_object* v___x_3330_; 
lean_dec(v_x_3299_);
v___x_3330_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3330_;
}
else
{
lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; uint8_t v___x_3337_; 
v___x_3331_ = lean_unsigned_to_nat(1u);
v___x_3332_ = l_Lean_Syntax_getArg(v_x_3299_, v___x_3331_);
v___x_3333_ = l_Lean_Syntax_getArgs(v___x_3332_);
lean_dec(v___x_3332_);
v___x_3334_ = lean_unsigned_to_nat(0u);
v___x_3335_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1___closed__18));
v___x_3336_ = lean_array_get_size(v___x_3333_);
v___x_3337_ = lean_nat_dec_lt(v___x_3334_, v___x_3336_);
if (v___x_3337_ == 0)
{
lean_dec_ref(v___x_3333_);
v___y_3310_ = v___x_3335_;
goto v___jp_3309_;
}
else
{
lean_object* v___x_3338_; lean_object* v___x_3339_; uint8_t v___x_3340_; 
v___x_3338_ = lean_box(v___x_3329_);
v___x_3339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3339_, 0, v___x_3338_);
lean_ctor_set(v___x_3339_, 1, v___x_3335_);
v___x_3340_ = lean_nat_dec_le(v___x_3336_, v___x_3336_);
if (v___x_3340_ == 0)
{
if (v___x_3337_ == 0)
{
lean_dec_ref_known(v___x_3339_, 2);
lean_dec_ref(v___x_3333_);
v___y_3310_ = v___x_3335_;
goto v___jp_3309_;
}
else
{
size_t v___x_3341_; size_t v___x_3342_; lean_object* v___x_3343_; lean_object* v_snd_3344_; 
v___x_3341_ = ((size_t)0ULL);
v___x_3342_ = lean_usize_of_nat(v___x_3336_);
v___x_3343_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_3329_, v___x_3333_, v___x_3341_, v___x_3342_, v___x_3339_);
lean_dec_ref(v___x_3333_);
v_snd_3344_ = lean_ctor_get(v___x_3343_, 1);
lean_inc(v_snd_3344_);
lean_dec_ref(v___x_3343_);
v___y_3310_ = v_snd_3344_;
goto v___jp_3309_;
}
}
else
{
size_t v___x_3345_; size_t v___x_3346_; lean_object* v___x_3347_; lean_object* v_snd_3348_; 
v___x_3345_ = ((size_t)0ULL);
v___x_3346_ = lean_usize_of_nat(v___x_3336_);
v___x_3347_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______macroRules__Batteries__Tactic__casePatt__1_spec__2(v___x_3329_, v___x_3333_, v___x_3345_, v___x_3346_, v___x_3339_);
lean_dec_ref(v___x_3333_);
v_snd_3348_ = lean_ctor_get(v___x_3347_, 1);
lean_inc(v_snd_3348_);
lean_dec_ref(v___x_3347_);
v___y_3310_ = v_snd_3348_;
goto v___jp_3309_;
}
}
}
v___jp_3309_:
{
size_t v_sz_3311_; size_t v___x_3312_; lean_object* v___x_3313_; 
v_sz_3311_ = lean_array_size(v___y_3310_);
v___x_3312_ = ((size_t)0ULL);
v___x_3313_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__0(v_sz_3311_, v___x_3312_, v___y_3310_);
if (lean_obj_tag(v___x_3313_) == 0)
{
lean_object* v___x_3314_; 
lean_dec(v_x_3299_);
v___x_3314_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3314_;
}
else
{
lean_object* v_val_3315_; lean_object* v___x_3316_; lean_object* v_caseBody_3317_; lean_object* v___x_3318_; uint8_t v___x_3319_; 
v_val_3315_ = lean_ctor_get(v___x_3313_, 0);
lean_inc(v_val_3315_);
lean_dec_ref_known(v___x_3313_, 1);
v___x_3316_ = lean_unsigned_to_nat(2u);
v_caseBody_3317_ = l_Lean_Syntax_getArg(v_x_3299_, v___x_3316_);
lean_dec(v_x_3299_);
v___x_3318_ = ((lean_object*)(lp_batteries_Batteries_Tactic_casePattTac___closed__1));
lean_inc(v_caseBody_3317_);
v___x_3319_ = l_Lean_Syntax_isOfKind(v_caseBody_3317_, v___x_3318_);
if (v___x_3319_ == 0)
{
lean_object* v___x_3320_; 
lean_dec(v_caseBody_3317_);
lean_dec(v_val_3315_);
v___x_3320_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic_processCasePattBody_spec__0___redArg();
return v___x_3320_;
}
else
{
size_t v_sz_3321_; lean_object* v_ref_3322_; lean_object* v_patts_x3f_3323_; lean_object* v_hss_3324_; lean_object* v_tags_3325_; uint8_t v___x_3326_; lean_object* v___x_3327_; 
v_sz_3321_ = lean_array_size(v_val_3315_);
v_ref_3322_ = lean_ctor_get(v_a_3306_, 5);
lean_inc_n(v_val_3315_, 2);
v_patts_x3f_3323_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__1(v_sz_3321_, v___x_3312_, v_val_3315_);
v_hss_3324_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__2(v_sz_3321_, v___x_3312_, v_val_3315_);
v_tags_3325_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt__1_spec__3(v_sz_3321_, v___x_3312_, v_val_3315_);
v___x_3326_ = 0;
lean_inc(v_ref_3322_);
v___x_3327_ = lp_batteries_Batteries_Tactic_evalCase(v___x_3326_, v_ref_3322_, v_tags_3325_, v_hss_3324_, v_patts_x3f_3323_, v_caseBody_3317_, v_a_3300_, v_a_3301_, v_a_3302_, v_a_3303_, v_a_3304_, v_a_3305_, v_a_3306_, v_a_3307_);
lean_dec_ref(v_tags_3325_);
return v___x_3327_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt_x27__1___boxed(lean_object* v_x_3349_, lean_object* v_a_3350_, lean_object* v_a_3351_, lean_object* v_a_3352_, lean_object* v_a_3353_, lean_object* v_a_3354_, lean_object* v_a_3355_, lean_object* v_a_3356_, lean_object* v_a_3357_, lean_object* v_a_3358_){
_start:
{
lean_object* v_res_3359_; 
v_res_3359_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Case______elabRules__Batteries__Tactic__casePatt_x27__1(v_x_3349_, v_a_3350_, v_a_3351_, v_a_3352_, v_a_3353_, v_a_3354_, v_a_3355_, v_a_3356_, v_a_3357_);
lean_dec(v_a_3357_);
lean_dec_ref(v_a_3356_);
lean_dec(v_a_3355_);
lean_dec_ref(v_a_3354_);
lean_dec(v_a_3353_);
lean_dec_ref(v_a_3352_);
lean_dec(v_a_3351_);
lean_dec_ref(v_a_3350_);
return v_res_3359_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Case(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_BuiltinTactic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_RenameInaccessibles(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Case(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_BuiltinTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_RenameInaccessibles(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Batteries_Tactic_casePattArg = _init_lp_batteries_Batteries_Tactic_casePattArg();
lean_mark_persistent(lp_batteries_Batteries_Tactic_casePattArg);
lp_batteries_Batteries_Tactic_casePatt = _init_lp_batteries_Batteries_Tactic_casePatt();
lean_mark_persistent(lp_batteries_Batteries_Tactic_casePatt);
lp_batteries_Batteries_Tactic_casePatt_x27 = _init_lp_batteries_Batteries_Tactic_casePatt_x27();
lean_mark_persistent(lp_batteries_Batteries_Tactic_casePatt_x27);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_BuiltinTactic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_RenameInaccessibles(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Case(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_BuiltinTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_RenameInaccessibles(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Case(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Case(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Case(builtin);
}
#ifdef __cplusplus
}
#endif
