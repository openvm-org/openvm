// Lean compiler output
// Module: Batteries.Tactic.Init
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.ElabTerm public meta import Lean.Meta.MatchUtil
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_matchEq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_rcasesPatMed;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalApplyLikeTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
size_t lean_array_size(lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tactic___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tactic___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tactic___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "tactic_"};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tactic___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tactic___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tactic___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(253, 34, 193, 133, 254, 219, 113, 55)}};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tactic___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tactic___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tactic___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tactic___00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tactic__ = (const lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeqBracketed"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(142, 80, 121, 250, 245, 54, 71, 145)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__8_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "exacts"};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 183, 65, 106, 251, 110, 85, 139)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "exacts "};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__9_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__12_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__13_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__13_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__12_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__14_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__15_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__16_value;
static const lean_string_object lp_batteries_Batteries_Tactic_exacts___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__17_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__16_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_exacts___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_exacts___closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__20_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_exacts = (const lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__20_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "done"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(113, 161, 179, 82, 204, 87, 48, 123)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "tacticBy_contra_core"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 81, 148, 161, 164, 160, 122, 158)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "by_contra_core"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__4_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticBy__contra__core = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "guardTarget"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(194, 4, 80, 225, 8, 32, 178, 134)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "guard_target"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "equal"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(155, 80, 62, 47, 176, 56, 79, 244)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "equalR"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(252, 8, 237, 98, 162, 170, 4, 237)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__16_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__20_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__21;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__22 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__22_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__23 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__23_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__22_value)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__24 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__24_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__25 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__25_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__23_value),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__25_value)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__26 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__26_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__27 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__27_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__29 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__29_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "change"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__30 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__30_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__30_value),LEAN_SCALAR_PTR_LITERAL(228, 221, 63, 213, 180, 29, 27, 230)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "arrow"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__32 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__32_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(182, 146, 143, 73, 122, 115, 5, 207)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "→"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__34 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__34_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__35 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__35_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__36;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__35_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__37 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__37_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__37_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__38 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__38_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__37_value)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__39 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__39_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__39_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__40 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__40_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__38_value),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__40_value)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__41 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__41_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__44 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__44_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__44_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__46 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__46_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Decidable.byContradiction"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__47 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__47_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__48;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__49 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__49_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "byContradiction"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__50 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__50_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__49_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(92, 114, 13, 107, 214, 89, 53, 175)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__52 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__52_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__52_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__53 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__53_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__54 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__54_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__54_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__56 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__56_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Classical.byContradiction"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__57 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__57_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__58;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Classical"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__59 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__59_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__59_value),LEAN_SCALAR_PTR_LITERAL(40, 236, 220, 79, 38, 141, 161, 150)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(143, 54, 188, 55, 95, 58, 91, 50)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__61 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__61_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__61_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__62 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__62_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_byContra___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "byContra"};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 151, 1, 107, 2, 236, 248, 130)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_byContra___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "by_contra"};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__3_value;
static const lean_string_object lp_batteries_Batteries_Tactic_byContra___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic_byContra___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__6_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__7_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__8_value;
static const lean_string_object lp_batteries_Batteries_Tactic_byContra___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__9_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__12_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_byContra___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_byContra___closed__13;
static lean_once_cell_t lp_batteries_Batteries_Tactic_byContra___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_byContra___closed__14;
static lean_once_cell_t lp_batteries_Batteries_Tactic_byContra___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_byContra___closed__15;
static const lean_string_object lp_batteries_Batteries_Tactic_byContra___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__17_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_byContra___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__5_value),((lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_byContra___closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_byContra___closed__19_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic_byContra___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_byContra___closed__20;
static lean_once_cell_t lp_batteries_Batteries_Tactic_byContra___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_byContra___closed__21;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_byContra;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rintro"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(170, 254, 242, 235, 94, 162, 254, 146)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rintroPat"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(120, 93, 179, 129, 121, 199, 215, 253)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value_aux_3),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(40, 214, 202, 122, 59, 249, 35, 61)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rcasesPat"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value_aux_3),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 230, 23, 208, 164, 113, 201, 132)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rcasesPatLo"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(133, 222, 245, 138, 122, 92, 170, 214)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__12_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__13_value;
static const lean_array_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__14_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rcasesPatMed"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(253, 13, 65, 195, 228, 27, 47, 149)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value_aux_3),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(186, 152, 172, 228, 11, 240, 156, 168)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "this"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(38, 116, 214, 236, 212, 160, 188, 150)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__19_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__20;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticAbsurd_"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 146, 108, 7, 66, 140, 170, 201)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "absurd "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticAbsurd__ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "absurd"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__1;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(93, 22, 196, 124, 199, 219, 238, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__4_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "tacticSplit_ands"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__0_value),LEAN_SCALAR_PTR_LITERAL(17, 43, 250, 108, 89, 185, 32, 95)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "split_ands"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__4_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticSplit__ands = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "repeat'"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(199, 67, 182, 138, 186, 187, 207, 59)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "And.intro"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(58, 46, 244, 208, 18, 71, 77, 162)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__8_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticFapply_"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticFapply___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(199, 254, 100, 0, 198, 74, 137, 194)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "fapply "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticFapply___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticFapply___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticFapply___00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticFapply___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticFapply___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticFapply__ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticFapply___00__closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticEapply_"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticEapply___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 216, 248, 192, 38, 36, 5, 54)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eapply "};
static const lean_object* lp_batteries_Batteries_Tactic_tacticEapply___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticEapply___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticEapply___00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticEapply___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticEapply___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticEapply__ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticEapply___00__closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_triv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "triv"};
static const lean_object* lp_batteries_Batteries_Tactic_triv___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_triv___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_triv___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_triv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 214, 215, 59, 119, 53, 211, 124)}};
static const lean_object* lp_batteries_Batteries_Tactic_triv___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_triv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_triv___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_triv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_triv___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__3_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_triv = (const lean_object*)&lp_batteries_Batteries_Tactic_triv___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "`triv` has been removed; use `trivial` instead"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Conv_exact___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Conv"};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_exact___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 142, 245, 237, 178, 12, 103, 62)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value_aux_2),((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(88, 165, 153, 227, 27, 68, 59, 235)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_exact___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Conv_exact___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "exact "};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_exact___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_exact___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_exact___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_exact___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_exact___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__5_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Conv_exact = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "nestedTactic"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__0_value),LEAN_SCALAR_PTR_LITERAL(51, 212, 92, 235, 115, 8, 100, 36)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value_aux_3),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 28, 213, 2, 207, 8, 223, 137)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_Conv_equals___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "equals"};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic_tactic___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_exact___closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 142, 245, 237, 178, 12, 103, 62)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value_aux_2),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 156, 188, 185, 29, 64, 171, 138)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Conv_equals___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "equals "};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_Conv_equals___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(13, 106, 54, 236, 164, 218, 24, 154)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_exacts___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__9_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_Conv_equals___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__10_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_Conv_equals___closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__11_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_Conv_equals = (const lean_object*)&lp_batteries_Batteries_Tactic_Conv_equals___closed__11_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__2_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "failed to resolve"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__4;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "\n=\?="};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__5_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__6;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "invalid 'conv' goal"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__7_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__8;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7(void){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = l_Array_mkArray0(lean_box(0));
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1(lean_object* v_x_31_, lean_object* v_a_32_, lean_object* v_a_33_){
_start:
{
lean_object* v___x_34_; uint8_t v___x_35_; 
v___x_34_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tactic___00__closed__3));
v___x_35_ = l_Lean_Syntax_isOfKind(v_x_31_, v___x_34_);
if (v___x_35_ == 0)
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = lean_box(1);
v___x_37_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_37_, 0, v___x_36_);
lean_ctor_set(v___x_37_, 1, v_a_33_);
return v___x_37_;
}
else
{
lean_object* v_ref_38_; uint8_t v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v_ref_38_ = lean_ctor_get(v_a_32_, 5);
v___x_39_ = 0;
v___x_40_ = l_Lean_SourceInfo_fromRef(v_ref_38_, v___x_39_);
v___x_41_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__3));
v___x_42_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__4));
lean_inc_n(v___x_40_, 3);
v___x_43_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_43_, 0, v___x_40_);
lean_ctor_set(v___x_43_, 1, v___x_42_);
v___x_44_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_45_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7);
v___x_46_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_46_, 0, v___x_40_);
lean_ctor_set(v___x_46_, 1, v___x_44_);
lean_ctor_set(v___x_46_, 2, v___x_45_);
v___x_47_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__8));
v___x_48_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_48_, 0, v___x_40_);
lean_ctor_set(v___x_48_, 1, v___x_47_);
v___x_49_ = l_Lean_Syntax_node3(v___x_40_, v___x_41_, v___x_43_, v___x_46_, v___x_48_);
v___x_50_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
lean_ctor_set(v___x_50_, 1, v_a_33_);
return v___x_50_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___boxed(lean_object* v_x_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1(v_x_51_, v_a_52_, v_a_53_);
lean_dec_ref(v_a_52_);
return v_res_54_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_105_ = lean_box(0);
v___x_106_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_107_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v___x_105_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg(){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___closed__0);
v___x_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg___boxed(lean_object* v___y_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0(lean_object* v_00_u03b1_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___boxed(lean_object* v_00_u03b1_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0(v_00_u03b1_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_, v___y_131_, v___y_132_);
lean_dec(v___y_132_);
lean_dec_ref(v___y_131_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1(lean_object* v_as_141_, size_t v_sz_142_, size_t v_i_143_, lean_object* v_b_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
uint8_t v___x_154_; 
v___x_154_ = lean_usize_dec_lt(v_i_143_, v_sz_142_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; 
v___x_155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_155_, 0, v_b_144_);
return v___x_155_;
}
else
{
lean_object* v_ref_156_; lean_object* v_a_157_; uint8_t v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v_ref_156_ = lean_ctor_get(v___y_151_, 5);
v_a_157_ = lean_array_uget_borrowed(v_as_141_, v_i_143_);
v___x_158_ = 0;
v___x_159_ = l_Lean_SourceInfo_fromRef(v_ref_156_, v___x_158_);
v___x_160_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0));
v___x_161_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1));
lean_inc(v___x_159_);
v___x_162_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_159_);
lean_ctor_set(v___x_162_, 1, v___x_160_);
lean_inc(v_a_157_);
v___x_163_ = l_Lean_Syntax_node2(v___x_159_, v___x_161_, v___x_162_, v_a_157_);
v___x_164_ = l_Lean_Elab_Tactic_evalTactic(v___x_163_, v___y_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_, v___y_152_);
if (lean_obj_tag(v___x_164_) == 0)
{
lean_object* v___x_165_; size_t v___x_166_; size_t v___x_167_; 
lean_dec_ref_known(v___x_164_, 1);
v___x_165_ = lean_box(0);
v___x_166_ = ((size_t)1ULL);
v___x_167_ = lean_usize_add(v_i_143_, v___x_166_);
v_i_143_ = v___x_167_;
v_b_144_ = v___x_165_;
goto _start;
}
else
{
return v___x_164_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___boxed(lean_object* v_as_169_, lean_object* v_sz_170_, lean_object* v_i_171_, lean_object* v_b_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
size_t v_sz_boxed_182_; size_t v_i_boxed_183_; lean_object* v_res_184_; 
v_sz_boxed_182_ = lean_unbox_usize(v_sz_170_);
lean_dec(v_sz_170_);
v_i_boxed_183_ = lean_unbox_usize(v_i_171_);
lean_dec(v_i_171_);
v_res_184_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1(v_as_169_, v_sz_boxed_182_, v_i_boxed_183_, v_b_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec(v___y_176_);
lean_dec_ref(v___y_175_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec_ref(v_as_169_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1(lean_object* v_x_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_){
_start:
{
lean_object* v___x_201_; uint8_t v___x_202_; 
v___x_201_ = ((lean_object*)(lp_batteries_Batteries_Tactic_exacts___closed__1));
lean_inc(v_x_191_);
v___x_202_ = l_Lean_Syntax_isOfKind(v_x_191_, v___x_201_);
if (v___x_202_ == 0)
{
lean_object* v___x_203_; 
lean_dec(v_x_191_);
v___x_203_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v___x_203_;
}
else
{
lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v_hs_206_; lean_object* v___x_207_; lean_object* v___x_208_; size_t v_sz_209_; size_t v___x_210_; lean_object* v___x_211_; 
v___x_204_ = lean_unsigned_to_nat(2u);
v___x_205_ = l_Lean_Syntax_getArg(v_x_191_, v___x_204_);
lean_dec(v_x_191_);
v_hs_206_ = l_Lean_Syntax_getArgs(v___x_205_);
lean_dec(v___x_205_);
v___x_207_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_hs_206_);
lean_dec_ref(v_hs_206_);
v___x_208_ = lean_box(0);
v_sz_209_ = lean_array_size(v___x_207_);
v___x_210_ = ((size_t)0ULL);
v___x_211_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1(v___x_207_, v_sz_209_, v___x_210_, v___x_208_, v_a_192_, v_a_193_, v_a_194_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_);
lean_dec_ref(v___x_207_);
if (lean_obj_tag(v___x_211_) == 0)
{
lean_object* v_ref_212_; uint8_t v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
lean_dec_ref_known(v___x_211_, 1);
v_ref_212_ = lean_ctor_get(v_a_198_, 5);
v___x_213_ = 0;
v___x_214_ = l_Lean_SourceInfo_fromRef(v_ref_212_, v___x_213_);
v___x_215_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__0));
v___x_216_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___closed__1));
lean_inc(v___x_214_);
v___x_217_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_214_);
lean_ctor_set(v___x_217_, 1, v___x_215_);
v___x_218_ = l_Lean_Syntax_node1(v___x_214_, v___x_216_, v___x_217_);
v___x_219_ = l_Lean_Elab_Tactic_evalTactic(v___x_218_, v_a_192_, v_a_193_, v_a_194_, v_a_195_, v_a_196_, v_a_197_, v_a_198_, v_a_199_);
return v___x_219_;
}
else
{
return v___x_211_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1___boxed(lean_object* v_x_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_, lean_object* v_a_224_, lean_object* v_a_225_, lean_object* v_a_226_, lean_object* v_a_227_, lean_object* v_a_228_, lean_object* v_a_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1(v_x_220_, v_a_221_, v_a_222_, v_a_223_, v_a_224_, v_a_225_, v_a_226_, v_a_227_, v_a_228_);
lean_dec(v_a_228_);
lean_dec_ref(v_a_227_);
lean_dec(v_a_226_);
lean_dec_ref(v_a_225_);
lean_dec(v_a_224_);
lean_dec_ref(v_a_223_);
lean_dec(v_a_222_);
lean_dec_ref(v_a_221_);
return v_res_230_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__21(void){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_293_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__20));
v___x_294_ = l_String_toRawSubstring_x27(v___x_293_);
return v___x_294_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__36(void){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_329_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__35));
v___x_330_ = l_String_toRawSubstring_x27(v___x_329_);
return v___x_330_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__48(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_358_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__47));
v___x_359_ = l_String_toRawSubstring_x27(v___x_358_);
return v___x_359_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__58(void){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_379_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__57));
v___x_380_ = l_String_toRawSubstring_x27(v___x_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1(lean_object* v_x_391_, lean_object* v_a_392_, lean_object* v_a_393_){
_start:
{
lean_object* v___x_394_; uint8_t v___x_395_; 
v___x_394_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1));
v___x_395_ = l_Lean_Syntax_isOfKind(v_x_391_, v___x_394_);
if (v___x_395_ == 0)
{
lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_396_ = lean_box(1);
v___x_397_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_397_, 0, v___x_396_);
lean_ctor_set(v___x_397_, 1, v_a_393_);
return v___x_397_;
}
else
{
lean_object* v_quotContext_398_; lean_object* v_currMacroScope_399_; lean_object* v_ref_400_; uint8_t v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v_quotContext_398_ = lean_ctor_get(v_a_392_, 1);
v_currMacroScope_399_ = lean_ctor_get(v_a_392_, 2);
v_ref_400_ = lean_ctor_get(v_a_392_, 5);
v___x_401_ = 0;
v___x_402_ = l_Lean_SourceInfo_fromRef(v_ref_400_, v___x_401_);
v___x_403_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__0));
v___x_404_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1));
lean_inc_n(v___x_402_, 46);
v___x_405_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_402_);
lean_ctor_set(v___x_405_, 1, v___x_403_);
v___x_406_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_407_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__3));
v___x_408_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__4));
v___x_409_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_402_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
v___x_410_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6));
v___x_411_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8));
v___x_412_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__10));
v___x_413_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__11));
v___x_414_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_414_, 0, v___x_402_);
lean_ctor_set(v___x_414_, 1, v___x_413_);
v___x_415_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__13));
v___x_416_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__15));
v___x_417_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__16));
v___x_418_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_418_, 0, v___x_402_);
lean_ctor_set(v___x_418_, 1, v___x_417_);
v___x_419_ = l_Lean_Syntax_node1(v___x_402_, v___x_416_, v___x_418_);
v___x_420_ = l_Lean_Syntax_node1(v___x_402_, v___x_415_, v___x_419_);
v___x_421_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19));
v___x_422_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__21, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__21_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__21);
v___x_423_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__22));
lean_inc_n(v_currMacroScope_399_, 4);
lean_inc_n(v_quotContext_398_, 4);
v___x_424_ = l_Lean_addMacroScope(v_quotContext_398_, v___x_423_, v_currMacroScope_399_);
v___x_425_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__26));
v___x_426_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_426_, 0, v___x_402_);
lean_ctor_set(v___x_426_, 1, v___x_422_);
lean_ctor_set(v___x_426_, 2, v___x_424_);
lean_ctor_set(v___x_426_, 3, v___x_425_);
v___x_427_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28));
v___x_428_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tactic___00__closed__4));
v___x_429_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_402_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
lean_inc_ref(v___x_429_);
v___x_430_ = l_Lean_Syntax_node1(v___x_402_, v___x_427_, v___x_429_);
lean_inc_n(v___x_430_, 4);
v___x_431_ = l_Lean_Syntax_node1(v___x_402_, v___x_406_, v___x_430_);
v___x_432_ = l_Lean_Syntax_node2(v___x_402_, v___x_421_, v___x_426_, v___x_431_);
v___x_433_ = l_Lean_Syntax_node3(v___x_402_, v___x_412_, v___x_414_, v___x_420_, v___x_432_);
v___x_434_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__29));
v___x_435_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_435_, 0, v___x_402_);
lean_ctor_set(v___x_435_, 1, v___x_434_);
v___x_436_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__30));
v___x_437_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__31));
v___x_438_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_438_, 0, v___x_402_);
lean_ctor_set(v___x_438_, 1, v___x_436_);
v___x_439_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__33));
v___x_440_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__34));
v___x_441_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_441_, 0, v___x_402_);
lean_ctor_set(v___x_441_, 1, v___x_440_);
v___x_442_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__36, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__36_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__36);
v___x_443_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__37));
v___x_444_ = l_Lean_addMacroScope(v_quotContext_398_, v___x_443_, v_currMacroScope_399_);
v___x_445_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__41));
v___x_446_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_446_, 0, v___x_402_);
lean_ctor_set(v___x_446_, 1, v___x_442_);
lean_ctor_set(v___x_446_, 2, v___x_444_);
lean_ctor_set(v___x_446_, 3, v___x_445_);
v___x_447_ = l_Lean_Syntax_node3(v___x_402_, v___x_439_, v___x_430_, v___x_441_, v___x_446_);
v___x_448_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7);
v___x_449_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_449_, 0, v___x_402_);
lean_ctor_set(v___x_449_, 1, v___x_406_);
lean_ctor_set(v___x_449_, 2, v___x_448_);
v___x_450_ = l_Lean_Syntax_node3(v___x_402_, v___x_437_, v___x_438_, v___x_447_, v___x_449_);
v___x_451_ = l_Lean_Syntax_node3(v___x_402_, v___x_406_, v___x_433_, v___x_435_, v___x_450_);
v___x_452_ = l_Lean_Syntax_node1(v___x_402_, v___x_411_, v___x_451_);
v___x_453_ = l_Lean_Syntax_node1(v___x_402_, v___x_410_, v___x_452_);
lean_inc_ref_n(v___x_409_, 2);
v___x_454_ = l_Lean_Syntax_node2(v___x_402_, v___x_407_, v___x_409_, v___x_453_);
v___x_455_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42));
v___x_456_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43));
v___x_457_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_457_, 0, v___x_402_);
lean_ctor_set(v___x_457_, 1, v___x_455_);
v___x_458_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45));
v___x_459_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__46));
v___x_460_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_460_, 0, v___x_402_);
lean_ctor_set(v___x_460_, 1, v___x_459_);
v___x_461_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__48, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__48_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__48);
v___x_462_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__51));
v___x_463_ = l_Lean_addMacroScope(v_quotContext_398_, v___x_462_, v_currMacroScope_399_);
v___x_464_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__53));
v___x_465_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_465_, 0, v___x_402_);
lean_ctor_set(v___x_465_, 1, v___x_461_);
lean_ctor_set(v___x_465_, 2, v___x_463_);
lean_ctor_set(v___x_465_, 3, v___x_464_);
lean_inc_ref(v___x_460_);
v___x_466_ = l_Lean_Syntax_node2(v___x_402_, v___x_458_, v___x_460_, v___x_465_);
v___x_467_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55));
v___x_468_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__56));
v___x_469_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_469_, 0, v___x_402_);
lean_ctor_set(v___x_469_, 1, v___x_468_);
v___x_470_ = l_Lean_Syntax_node2(v___x_402_, v___x_467_, v___x_469_, v___x_429_);
lean_inc(v___x_470_);
v___x_471_ = l_Lean_Syntax_node3(v___x_402_, v___x_406_, v___x_430_, v___x_430_, v___x_470_);
v___x_472_ = l_Lean_Syntax_node2(v___x_402_, v___x_421_, v___x_466_, v___x_471_);
lean_inc_ref(v___x_457_);
v___x_473_ = l_Lean_Syntax_node2(v___x_402_, v___x_456_, v___x_457_, v___x_472_);
v___x_474_ = l_Lean_Syntax_node1(v___x_402_, v___x_406_, v___x_473_);
v___x_475_ = l_Lean_Syntax_node1(v___x_402_, v___x_411_, v___x_474_);
v___x_476_ = l_Lean_Syntax_node1(v___x_402_, v___x_410_, v___x_475_);
v___x_477_ = l_Lean_Syntax_node2(v___x_402_, v___x_407_, v___x_409_, v___x_476_);
v___x_478_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__58, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__58_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__58);
v___x_479_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__60));
v___x_480_ = l_Lean_addMacroScope(v_quotContext_398_, v___x_479_, v_currMacroScope_399_);
v___x_481_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__62));
v___x_482_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_482_, 0, v___x_402_);
lean_ctor_set(v___x_482_, 1, v___x_478_);
lean_ctor_set(v___x_482_, 2, v___x_480_);
lean_ctor_set(v___x_482_, 3, v___x_481_);
v___x_483_ = l_Lean_Syntax_node2(v___x_402_, v___x_458_, v___x_460_, v___x_482_);
v___x_484_ = l_Lean_Syntax_node2(v___x_402_, v___x_406_, v___x_430_, v___x_470_);
v___x_485_ = l_Lean_Syntax_node2(v___x_402_, v___x_421_, v___x_483_, v___x_484_);
v___x_486_ = l_Lean_Syntax_node2(v___x_402_, v___x_456_, v___x_457_, v___x_485_);
v___x_487_ = l_Lean_Syntax_node1(v___x_402_, v___x_406_, v___x_486_);
v___x_488_ = l_Lean_Syntax_node1(v___x_402_, v___x_411_, v___x_487_);
v___x_489_ = l_Lean_Syntax_node1(v___x_402_, v___x_410_, v___x_488_);
v___x_490_ = l_Lean_Syntax_node2(v___x_402_, v___x_407_, v___x_409_, v___x_489_);
v___x_491_ = l_Lean_Syntax_node3(v___x_402_, v___x_406_, v___x_454_, v___x_477_, v___x_490_);
v___x_492_ = l_Lean_Syntax_node2(v___x_402_, v___x_404_, v___x_405_, v___x_491_);
v___x_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
lean_ctor_set(v___x_493_, 1, v_a_393_);
return v___x_493_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___boxed(lean_object* v_x_494_, lean_object* v_a_495_, lean_object* v_a_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1(v_x_494_, v_a_495_, v_a_496_);
lean_dec_ref(v_a_495_);
return v_res_497_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_byContra___closed__13(void){
_start:
{
lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; 
v___x_524_ = l_Lean_Parser_Tactic_rcasesPatMed;
v___x_525_ = ((lean_object*)(lp_batteries_Batteries_Tactic_byContra___closed__12));
v___x_526_ = ((lean_object*)(lp_batteries_Batteries_Tactic_exacts___closed__3));
v___x_527_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_527_, 0, v___x_526_);
lean_ctor_set(v___x_527_, 1, v___x_525_);
lean_ctor_set(v___x_527_, 2, v___x_524_);
return v___x_527_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_byContra___closed__14(void){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_528_ = lean_obj_once(&lp_batteries_Batteries_Tactic_byContra___closed__13, &lp_batteries_Batteries_Tactic_byContra___closed__13_once, _init_lp_batteries_Batteries_Tactic_byContra___closed__13);
v___x_529_ = ((lean_object*)(lp_batteries_Batteries_Tactic_byContra___closed__5));
v___x_530_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_530_, 0, v___x_529_);
lean_ctor_set(v___x_530_, 1, v___x_528_);
return v___x_530_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_byContra___closed__15(void){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_531_ = lean_obj_once(&lp_batteries_Batteries_Tactic_byContra___closed__14, &lp_batteries_Batteries_Tactic_byContra___closed__14_once, _init_lp_batteries_Batteries_Tactic_byContra___closed__14);
v___x_532_ = ((lean_object*)(lp_batteries_Batteries_Tactic_byContra___closed__3));
v___x_533_ = ((lean_object*)(lp_batteries_Batteries_Tactic_exacts___closed__3));
v___x_534_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
lean_ctor_set(v___x_534_, 1, v___x_532_);
lean_ctor_set(v___x_534_, 2, v___x_531_);
return v___x_534_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_byContra___closed__20(void){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_545_ = ((lean_object*)(lp_batteries_Batteries_Tactic_byContra___closed__19));
v___x_546_ = lean_obj_once(&lp_batteries_Batteries_Tactic_byContra___closed__15, &lp_batteries_Batteries_Tactic_byContra___closed__15_once, _init_lp_batteries_Batteries_Tactic_byContra___closed__15);
v___x_547_ = ((lean_object*)(lp_batteries_Batteries_Tactic_exacts___closed__3));
v___x_548_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_548_, 0, v___x_547_);
lean_ctor_set(v___x_548_, 1, v___x_546_);
lean_ctor_set(v___x_548_, 2, v___x_545_);
return v___x_548_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_byContra___closed__21(void){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_549_ = lean_obj_once(&lp_batteries_Batteries_Tactic_byContra___closed__20, &lp_batteries_Batteries_Tactic_byContra___closed__20_once, _init_lp_batteries_Batteries_Tactic_byContra___closed__20);
v___x_550_ = lean_unsigned_to_nat(1022u);
v___x_551_ = ((lean_object*)(lp_batteries_Batteries_Tactic_byContra___closed__1));
v___x_552_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_552_, 0, v___x_551_);
lean_ctor_set(v___x_552_, 1, v___x_550_);
lean_ctor_set(v___x_552_, 2, v___x_549_);
return v___x_552_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_byContra(void){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lean_obj_once(&lp_batteries_Batteries_Tactic_byContra___closed__21, &lp_batteries_Batteries_Tactic_byContra___closed__21_once, _init_lp_batteries_Batteries_Tactic_byContra___closed__21);
return v___x_553_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__20(void){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_607_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__19));
v___x_608_ = l_Lean_mkIdent(v___x_607_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1(lean_object* v_x_609_, lean_object* v_a_610_, lean_object* v_a_611_){
_start:
{
lean_object* v___y_613_; lean_object* v___y_614_; lean_object* v___y_615_; lean_object* v___y_616_; lean_object* v___y_617_; lean_object* v___y_618_; lean_object* v___y_619_; lean_object* v___y_620_; lean_object* v___y_621_; lean_object* v___y_622_; lean_object* v___y_623_; lean_object* v___y_624_; lean_object* v___y_625_; lean_object* v___y_626_; lean_object* v___y_627_; lean_object* v___y_637_; lean_object* v_ref_638_; lean_object* v_a_639_; lean_object* v_a_640_; lean_object* v___y_675_; lean_object* v_ty_x3f_676_; lean_object* v___y_677_; lean_object* v___y_678_; lean_object* v___x_691_; uint8_t v___x_692_; 
v___x_691_ = ((lean_object*)(lp_batteries_Batteries_Tactic_byContra___closed__1));
lean_inc(v_x_609_);
v___x_692_ = l_Lean_Syntax_isOfKind(v_x_609_, v___x_691_);
if (v___x_692_ == 0)
{
lean_object* v___x_693_; lean_object* v___x_694_; 
lean_dec(v_x_609_);
v___x_693_ = lean_box(1);
v___x_694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_694_, 0, v___x_693_);
lean_ctor_set(v___x_694_, 1, v_a_611_);
return v___x_694_;
}
else
{
lean_object* v___x_695_; lean_object* v_pat_x3f_697_; lean_object* v___y_698_; lean_object* v___y_699_; lean_object* v___x_709_; uint8_t v___x_710_; 
v___x_695_ = lean_unsigned_to_nat(1u);
v___x_709_ = l_Lean_Syntax_getArg(v_x_609_, v___x_695_);
v___x_710_ = l_Lean_Syntax_isNone(v___x_709_);
if (v___x_710_ == 0)
{
uint8_t v___x_711_; 
lean_inc(v___x_709_);
v___x_711_ = l_Lean_Syntax_matchesNull(v___x_709_, v___x_695_);
if (v___x_711_ == 0)
{
lean_object* v___x_712_; lean_object* v___x_713_; 
lean_dec(v___x_709_);
lean_dec(v_x_609_);
v___x_712_ = lean_box(1);
v___x_713_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_713_, 0, v___x_712_);
lean_ctor_set(v___x_713_, 1, v_a_611_);
return v___x_713_;
}
else
{
lean_object* v___x_714_; lean_object* v_pat_x3f_715_; lean_object* v___x_716_; 
v___x_714_ = lean_unsigned_to_nat(0u);
v_pat_x3f_715_ = l_Lean_Syntax_getArg(v___x_709_, v___x_714_);
lean_dec(v___x_709_);
v___x_716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_716_, 0, v_pat_x3f_715_);
v_pat_x3f_697_ = v___x_716_;
v___y_698_ = v_a_610_;
v___y_699_ = v_a_611_;
goto v___jp_696_;
}
}
else
{
lean_object* v___x_717_; 
lean_dec(v___x_709_);
v___x_717_ = lean_box(0);
v_pat_x3f_697_ = v___x_717_;
v___y_698_ = v_a_610_;
v___y_699_ = v_a_611_;
goto v___jp_696_;
}
v___jp_696_:
{
lean_object* v___x_700_; lean_object* v___x_701_; uint8_t v___x_702_; 
v___x_700_ = lean_unsigned_to_nat(2u);
v___x_701_ = l_Lean_Syntax_getArg(v_x_609_, v___x_700_);
lean_dec(v_x_609_);
v___x_702_ = l_Lean_Syntax_isNone(v___x_701_);
if (v___x_702_ == 0)
{
uint8_t v___x_703_; 
lean_inc(v___x_701_);
v___x_703_ = l_Lean_Syntax_matchesNull(v___x_701_, v___x_700_);
if (v___x_703_ == 0)
{
lean_object* v___x_704_; lean_object* v___x_705_; 
lean_dec(v___x_701_);
lean_dec(v_pat_x3f_697_);
v___x_704_ = lean_box(1);
v___x_705_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_704_);
lean_ctor_set(v___x_705_, 1, v___y_699_);
return v___x_705_;
}
else
{
lean_object* v_ty_x3f_706_; lean_object* v___x_707_; 
v_ty_x3f_706_ = l_Lean_Syntax_getArg(v___x_701_, v___x_695_);
lean_dec(v___x_701_);
v___x_707_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_707_, 0, v_ty_x3f_706_);
v___y_675_ = v_pat_x3f_697_;
v_ty_x3f_676_ = v___x_707_;
v___y_677_ = v___y_698_;
v___y_678_ = v___y_699_;
goto v___jp_674_;
}
}
else
{
lean_object* v___x_708_; 
lean_dec(v___x_701_);
v___x_708_ = lean_box(0);
v___y_675_ = v_pat_x3f_697_;
v_ty_x3f_676_ = v___x_708_;
v___y_677_ = v___y_698_;
v___y_678_ = v___y_699_;
goto v___jp_674_;
}
}
}
v___jp_612_:
{
lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; 
lean_inc_ref(v___y_621_);
v___x_628_ = l_Array_append___redArg(v___y_621_, v___y_627_);
lean_dec_ref(v___y_627_);
lean_inc(v___y_618_);
lean_inc_n(v___y_623_, 5);
v___x_629_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_629_, 0, v___y_623_);
lean_ctor_set(v___x_629_, 1, v___y_618_);
lean_ctor_set(v___x_629_, 2, v___x_628_);
lean_inc(v___y_620_);
v___x_630_ = l_Lean_Syntax_node3(v___y_623_, v___y_620_, v___y_626_, v___y_625_, v___x_629_);
v___x_631_ = l_Lean_Syntax_node3(v___y_623_, v___y_618_, v___y_622_, v___y_614_, v___x_630_);
lean_inc(v___y_617_);
v___x_632_ = l_Lean_Syntax_node1(v___y_623_, v___y_617_, v___x_631_);
lean_inc(v___y_615_);
v___x_633_ = l_Lean_Syntax_node1(v___y_623_, v___y_615_, v___x_632_);
lean_inc(v___y_624_);
v___x_634_ = l_Lean_Syntax_node3(v___y_623_, v___y_624_, v___y_619_, v___x_633_, v___y_616_);
v___x_635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_635_, 0, v___x_634_);
lean_ctor_set(v___x_635_, 1, v___y_613_);
return v___x_635_;
}
v___jp_636_:
{
uint8_t v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v___x_641_ = 0;
v___x_642_ = l_Lean_SourceInfo_fromRef(v_ref_638_, v___x_641_);
v___x_643_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__1));
v___x_644_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__2));
lean_inc_n(v___x_642_, 11);
v___x_645_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_645_, 0, v___x_642_);
lean_ctor_set(v___x_645_, 1, v___x_644_);
v___x_646_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6));
v___x_647_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8));
v___x_648_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_649_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__1));
v___x_650_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticBy__contra__core___closed__2));
v___x_651_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_651_, 0, v___x_642_);
lean_ctor_set(v___x_651_, 1, v___x_650_);
v___x_652_ = l_Lean_Syntax_node1(v___x_642_, v___x_649_, v___x_651_);
v___x_653_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__29));
v___x_654_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_654_, 0, v___x_642_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__3));
v___x_656_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__4));
v___x_657_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_642_);
lean_ctor_set(v___x_657_, 1, v___x_655_);
v___x_658_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__7));
v___x_659_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__9));
v___x_660_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__11));
v___x_661_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__7);
v___x_662_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_662_, 0, v___x_642_);
lean_ctor_set(v___x_662_, 1, v___x_648_);
lean_ctor_set(v___x_662_, 2, v___x_661_);
v___x_663_ = l_Lean_Syntax_node2(v___x_642_, v___x_660_, v_a_639_, v___x_662_);
v___x_664_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__12));
v___x_665_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_665_, 0, v___x_642_);
lean_ctor_set(v___x_665_, 1, v___x_664_);
lean_inc_ref(v___x_665_);
lean_inc_ref(v___x_645_);
v___x_666_ = l_Lean_Syntax_node3(v___x_642_, v___x_659_, v___x_645_, v___x_663_, v___x_665_);
v___x_667_ = l_Lean_Syntax_node1(v___x_642_, v___x_658_, v___x_666_);
v___x_668_ = l_Lean_Syntax_node1(v___x_642_, v___x_648_, v___x_667_);
if (lean_obj_tag(v___y_637_) == 1)
{
lean_object* v_val_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v_val_669_ = lean_ctor_get(v___y_637_, 0);
lean_inc(v_val_669_);
lean_dec_ref_known(v___y_637_, 1);
v___x_670_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__13));
lean_inc(v___x_642_);
v___x_671_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_671_, 0, v___x_642_);
lean_ctor_set(v___x_671_, 1, v___x_670_);
v___x_672_ = l_Array_mkArray2___redArg(v___x_671_, v_val_669_);
v___y_613_ = v_a_640_;
v___y_614_ = v___x_654_;
v___y_615_ = v___x_646_;
v___y_616_ = v___x_665_;
v___y_617_ = v___x_647_;
v___y_618_ = v___x_648_;
v___y_619_ = v___x_645_;
v___y_620_ = v___x_656_;
v___y_621_ = v___x_661_;
v___y_622_ = v___x_652_;
v___y_623_ = v___x_642_;
v___y_624_ = v___x_643_;
v___y_625_ = v___x_668_;
v___y_626_ = v___x_657_;
v___y_627_ = v___x_672_;
goto v___jp_612_;
}
else
{
lean_object* v___x_673_; 
lean_dec(v___y_637_);
v___x_673_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__14));
v___y_613_ = v_a_640_;
v___y_614_ = v___x_654_;
v___y_615_ = v___x_646_;
v___y_616_ = v___x_665_;
v___y_617_ = v___x_647_;
v___y_618_ = v___x_648_;
v___y_619_ = v___x_645_;
v___y_620_ = v___x_656_;
v___y_621_ = v___x_661_;
v___y_622_ = v___x_652_;
v___y_623_ = v___x_642_;
v___y_624_ = v___x_643_;
v___y_625_ = v___x_668_;
v___y_626_ = v___x_657_;
v___y_627_ = v___x_673_;
goto v___jp_612_;
}
}
v___jp_674_:
{
if (lean_obj_tag(v___y_675_) == 0)
{
lean_object* v_ref_679_; uint8_t v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; 
v_ref_679_ = lean_ctor_get(v___y_677_, 5);
v___x_680_ = 0;
v___x_681_ = l_Lean_SourceInfo_fromRef(v_ref_679_, v___x_680_);
v___x_682_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__16));
v___x_683_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_684_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__17));
v___x_685_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__20, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__20_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___closed__20);
lean_inc_n(v___x_681_, 2);
v___x_686_ = l_Lean_Syntax_node1(v___x_681_, v___x_684_, v___x_685_);
v___x_687_ = l_Lean_Syntax_node1(v___x_681_, v___x_683_, v___x_686_);
v___x_688_ = l_Lean_Syntax_node1(v___x_681_, v___x_682_, v___x_687_);
v___y_637_ = v_ty_x3f_676_;
v_ref_638_ = v_ref_679_;
v_a_639_ = v___x_688_;
v_a_640_ = v___y_678_;
goto v___jp_636_;
}
else
{
lean_object* v_val_689_; lean_object* v_ref_690_; 
v_val_689_ = lean_ctor_get(v___y_675_, 0);
lean_inc(v_val_689_);
lean_dec_ref_known(v___y_675_, 1);
v_ref_690_ = lean_ctor_get(v___y_677_, 5);
v___y_637_ = v_ty_x3f_676_;
v_ref_638_ = v_ref_690_;
v_a_639_ = v_val_689_;
v_a_640_ = v___y_678_;
goto v___jp_636_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1___boxed(lean_object* v_x_718_, lean_object* v_a_719_, lean_object* v_a_720_){
_start:
{
lean_object* v_res_721_; 
v_res_721_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__byContra__1(v_x_718_, v_a_719_, v_a_720_);
lean_dec_ref(v_a_719_);
return v_res_721_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__1(void){
_start:
{
lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_741_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__0));
v___x_742_ = l_String_toRawSubstring_x27(v___x_741_);
return v___x_742_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1(lean_object* v_x_751_, lean_object* v_a_752_, lean_object* v_a_753_){
_start:
{
lean_object* v___x_754_; uint8_t v___x_755_; 
v___x_754_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticAbsurd___00__closed__1));
lean_inc(v_x_751_);
v___x_755_ = l_Lean_Syntax_isOfKind(v_x_751_, v___x_754_);
if (v___x_755_ == 0)
{
lean_object* v___x_756_; lean_object* v___x_757_; 
lean_dec(v_x_751_);
v___x_756_ = lean_box(1);
v___x_757_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_757_, 0, v___x_756_);
lean_ctor_set(v___x_757_, 1, v_a_753_);
return v___x_757_;
}
else
{
lean_object* v_quotContext_758_; lean_object* v_currMacroScope_759_; lean_object* v_ref_760_; lean_object* v___x_761_; lean_object* v___x_762_; uint8_t v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; 
v_quotContext_758_ = lean_ctor_get(v_a_752_, 1);
v_currMacroScope_759_ = lean_ctor_get(v_a_752_, 2);
v_ref_760_ = lean_ctor_get(v_a_752_, 5);
v___x_761_ = lean_unsigned_to_nat(1u);
v___x_762_ = l_Lean_Syntax_getArg(v_x_751_, v___x_761_);
lean_dec(v_x_751_);
v___x_763_ = 0;
v___x_764_ = l_Lean_SourceInfo_fromRef(v_ref_760_, v___x_763_);
v___x_765_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__0));
v___x_766_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__1));
lean_inc_n(v___x_764_, 25);
v___x_767_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_767_, 0, v___x_764_);
lean_ctor_set(v___x_767_, 1, v___x_765_);
v___x_768_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_769_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__3));
v___x_770_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__4));
v___x_771_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_771_, 0, v___x_764_);
lean_ctor_set(v___x_771_, 1, v___x_770_);
v___x_772_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6));
v___x_773_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8));
v___x_774_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42));
v___x_775_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43));
v___x_776_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_776_, 0, v___x_764_);
lean_ctor_set(v___x_776_, 1, v___x_774_);
v___x_777_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19));
v___x_778_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__45));
v___x_779_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__46));
v___x_780_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_780_, 0, v___x_764_);
lean_ctor_set(v___x_780_, 1, v___x_779_);
v___x_781_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__1, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__1_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__1);
v___x_782_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__2));
lean_inc(v_currMacroScope_759_);
lean_inc(v_quotContext_758_);
v___x_783_ = l_Lean_addMacroScope(v_quotContext_758_, v___x_782_, v_currMacroScope_759_);
v___x_784_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___closed__4));
v___x_785_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_785_, 0, v___x_764_);
lean_ctor_set(v___x_785_, 1, v___x_781_);
lean_ctor_set(v___x_785_, 2, v___x_783_);
lean_ctor_set(v___x_785_, 3, v___x_784_);
v___x_786_ = l_Lean_Syntax_node2(v___x_764_, v___x_778_, v___x_780_, v___x_785_);
v___x_787_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__28));
v___x_788_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tactic___00__closed__4));
v___x_789_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_764_);
lean_ctor_set(v___x_789_, 1, v___x_788_);
lean_inc_ref(v___x_789_);
v___x_790_ = l_Lean_Syntax_node1(v___x_764_, v___x_787_, v___x_789_);
v___x_791_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55));
v___x_792_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__56));
v___x_793_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_793_, 0, v___x_764_);
lean_ctor_set(v___x_793_, 1, v___x_792_);
v___x_794_ = l_Lean_Syntax_node2(v___x_764_, v___x_791_, v___x_793_, v___x_789_);
lean_inc(v___x_762_);
lean_inc(v___x_794_);
lean_inc_n(v___x_790_, 3);
v___x_795_ = l_Lean_Syntax_node4(v___x_764_, v___x_768_, v___x_790_, v___x_790_, v___x_794_, v___x_762_);
lean_inc(v___x_786_);
v___x_796_ = l_Lean_Syntax_node2(v___x_764_, v___x_777_, v___x_786_, v___x_795_);
lean_inc_ref(v___x_776_);
v___x_797_ = l_Lean_Syntax_node2(v___x_764_, v___x_775_, v___x_776_, v___x_796_);
v___x_798_ = l_Lean_Syntax_node1(v___x_764_, v___x_768_, v___x_797_);
v___x_799_ = l_Lean_Syntax_node1(v___x_764_, v___x_773_, v___x_798_);
v___x_800_ = l_Lean_Syntax_node1(v___x_764_, v___x_772_, v___x_799_);
lean_inc_ref(v___x_771_);
v___x_801_ = l_Lean_Syntax_node2(v___x_764_, v___x_769_, v___x_771_, v___x_800_);
v___x_802_ = l_Lean_Syntax_node4(v___x_764_, v___x_768_, v___x_790_, v___x_790_, v___x_762_, v___x_794_);
v___x_803_ = l_Lean_Syntax_node2(v___x_764_, v___x_777_, v___x_786_, v___x_802_);
v___x_804_ = l_Lean_Syntax_node2(v___x_764_, v___x_775_, v___x_776_, v___x_803_);
v___x_805_ = l_Lean_Syntax_node1(v___x_764_, v___x_768_, v___x_804_);
v___x_806_ = l_Lean_Syntax_node1(v___x_764_, v___x_773_, v___x_805_);
v___x_807_ = l_Lean_Syntax_node1(v___x_764_, v___x_772_, v___x_806_);
v___x_808_ = l_Lean_Syntax_node2(v___x_764_, v___x_769_, v___x_771_, v___x_807_);
v___x_809_ = l_Lean_Syntax_node2(v___x_764_, v___x_768_, v___x_801_, v___x_808_);
v___x_810_ = l_Lean_Syntax_node2(v___x_764_, v___x_766_, v___x_767_, v___x_809_);
v___x_811_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_811_, 0, v___x_810_);
lean_ctor_set(v___x_811_, 1, v_a_753_);
return v___x_811_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1___boxed(lean_object* v_x_812_, lean_object* v_a_813_, lean_object* v_a_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticAbsurd____1(v_x_812_, v_a_813_, v_a_814_);
lean_dec_ref(v_a_813_);
return v_res_815_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__3(void){
_start:
{
lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_837_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__2));
v___x_838_ = l_String_toRawSubstring_x27(v___x_837_);
return v___x_838_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1(lean_object* v_x_850_, lean_object* v_a_851_, lean_object* v_a_852_){
_start:
{
lean_object* v___x_853_; uint8_t v___x_854_; 
v___x_853_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticSplit__ands___closed__1));
v___x_854_ = l_Lean_Syntax_isOfKind(v_x_850_, v___x_853_);
if (v___x_854_ == 0)
{
lean_object* v___x_855_; lean_object* v___x_856_; 
v___x_855_ = lean_box(1);
v___x_856_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_856_, 0, v___x_855_);
lean_ctor_set(v___x_856_, 1, v_a_852_);
return v___x_856_;
}
else
{
lean_object* v_quotContext_857_; lean_object* v_currMacroScope_858_; lean_object* v_ref_859_; uint8_t v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; 
v_quotContext_857_ = lean_ctor_get(v_a_851_, 1);
v_currMacroScope_858_ = lean_ctor_get(v_a_851_, 2);
v_ref_859_ = lean_ctor_get(v_a_851_, 5);
v___x_860_ = 0;
v___x_861_ = l_Lean_SourceInfo_fromRef(v_ref_859_, v___x_860_);
v___x_862_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__0));
v___x_863_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__1));
lean_inc_n(v___x_861_, 12);
v___x_864_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_864_, 0, v___x_861_);
lean_ctor_set(v___x_864_, 1, v___x_862_);
v___x_865_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6));
v___x_866_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8));
v___x_867_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_868_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__42));
v___x_869_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__43));
v___x_870_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_870_, 0, v___x_861_);
lean_ctor_set(v___x_870_, 1, v___x_868_);
v___x_871_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__19));
v___x_872_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__3, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__3_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__3);
v___x_873_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__6));
lean_inc(v_currMacroScope_858_);
lean_inc(v_quotContext_857_);
v___x_874_ = l_Lean_addMacroScope(v_quotContext_857_, v___x_873_, v_currMacroScope_858_);
v___x_875_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___closed__8));
v___x_876_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_876_, 0, v___x_861_);
lean_ctor_set(v___x_876_, 1, v___x_872_);
lean_ctor_set(v___x_876_, 2, v___x_874_);
lean_ctor_set(v___x_876_, 3, v___x_875_);
v___x_877_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__55));
v___x_878_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__56));
v___x_879_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_879_, 0, v___x_861_);
lean_ctor_set(v___x_879_, 1, v___x_878_);
v___x_880_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tactic___00__closed__4));
v___x_881_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_881_, 0, v___x_861_);
lean_ctor_set(v___x_881_, 1, v___x_880_);
v___x_882_ = l_Lean_Syntax_node2(v___x_861_, v___x_877_, v___x_879_, v___x_881_);
lean_inc(v___x_882_);
v___x_883_ = l_Lean_Syntax_node2(v___x_861_, v___x_867_, v___x_882_, v___x_882_);
v___x_884_ = l_Lean_Syntax_node2(v___x_861_, v___x_871_, v___x_876_, v___x_883_);
v___x_885_ = l_Lean_Syntax_node2(v___x_861_, v___x_869_, v___x_870_, v___x_884_);
v___x_886_ = l_Lean_Syntax_node1(v___x_861_, v___x_867_, v___x_885_);
v___x_887_ = l_Lean_Syntax_node1(v___x_861_, v___x_866_, v___x_886_);
v___x_888_ = l_Lean_Syntax_node1(v___x_861_, v___x_865_, v___x_887_);
v___x_889_ = l_Lean_Syntax_node2(v___x_861_, v___x_863_, v___x_864_, v___x_888_);
v___x_890_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_890_, 0, v___x_889_);
lean_ctor_set(v___x_890_, 1, v_a_852_);
return v___x_890_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1___boxed(lean_object* v_x_891_, lean_object* v_a_892_, lean_object* v_a_893_){
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticSplit__ands__1(v_x_891_, v_a_892_, v_a_893_);
lean_dec_ref(v_a_892_);
return v_res_894_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___lam__0(uint8_t v___x_913_, lean_object* v_x_914_, lean_object* v_e_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_){
_start:
{
uint8_t v___x_921_; uint8_t v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; 
v___x_921_ = 2;
v___x_922_ = 0;
v___x_923_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_923_, 0, v___x_921_);
lean_ctor_set_uint8(v___x_923_, 1, v___x_913_);
lean_ctor_set_uint8(v___x_923_, 2, v___x_922_);
lean_ctor_set_uint8(v___x_923_, 3, v___x_913_);
v___x_924_ = lean_box(0);
v___x_925_ = l_Lean_MVarId_apply(v_x_914_, v_e_915_, v___x_923_, v___x_924_, v___y_916_, v___y_917_, v___y_918_, v___y_919_);
return v___x_925_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___lam__0___boxed(lean_object* v___x_926_, lean_object* v_x_927_, lean_object* v_e_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
uint8_t v___x_174__boxed_934_; lean_object* v_res_935_; 
v___x_174__boxed_934_ = lean_unbox(v___x_926_);
v_res_935_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___lam__0(v___x_174__boxed_934_, v_x_927_, v_e_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
lean_dec(v___y_930_);
lean_dec_ref(v___y_929_);
return v_res_935_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1(lean_object* v_x_936_, lean_object* v_a_937_, lean_object* v_a_938_, lean_object* v_a_939_, lean_object* v_a_940_, lean_object* v_a_941_, lean_object* v_a_942_, lean_object* v_a_943_, lean_object* v_a_944_){
_start:
{
lean_object* v___x_946_; uint8_t v___x_947_; 
v___x_946_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticFapply___00__closed__1));
lean_inc(v_x_936_);
v___x_947_ = l_Lean_Syntax_isOfKind(v_x_936_, v___x_946_);
if (v___x_947_ == 0)
{
lean_object* v___x_948_; 
lean_dec(v_x_936_);
v___x_948_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v___x_948_;
}
else
{
lean_object* v___x_949_; lean_object* v___f_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_949_ = lean_box(v___x_947_);
v___f_950_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_950_, 0, v___x_949_);
v___x_951_ = lean_unsigned_to_nat(1u);
v___x_952_ = l_Lean_Syntax_getArg(v_x_936_, v___x_951_);
lean_dec(v_x_936_);
v___x_953_ = l_Lean_Elab_Tactic_evalApplyLikeTactic(v___f_950_, v___x_952_, v_a_937_, v_a_938_, v_a_939_, v_a_940_, v_a_941_, v_a_942_, v_a_943_, v_a_944_);
return v___x_953_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1___boxed(lean_object* v_x_954_, lean_object* v_a_955_, lean_object* v_a_956_, lean_object* v_a_957_, lean_object* v_a_958_, lean_object* v_a_959_, lean_object* v_a_960_, lean_object* v_a_961_, lean_object* v_a_962_, lean_object* v_a_963_){
_start:
{
lean_object* v_res_964_; 
v_res_964_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticFapply____1(v_x_954_, v_a_955_, v_a_956_, v_a_957_, v_a_958_, v_a_959_, v_a_960_, v_a_961_, v_a_962_);
lean_dec(v_a_962_);
lean_dec_ref(v_a_961_);
lean_dec(v_a_960_);
lean_dec_ref(v_a_959_);
lean_dec(v_a_958_);
lean_dec_ref(v_a_957_);
lean_dec(v_a_956_);
lean_dec_ref(v_a_955_);
return v_res_964_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___lam__0(uint8_t v___x_983_, lean_object* v_x_984_, lean_object* v_e_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
uint8_t v___x_991_; uint8_t v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; 
v___x_991_ = 1;
v___x_992_ = 0;
v___x_993_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_993_, 0, v___x_991_);
lean_ctor_set_uint8(v___x_993_, 1, v___x_983_);
lean_ctor_set_uint8(v___x_993_, 2, v___x_992_);
lean_ctor_set_uint8(v___x_993_, 3, v___x_983_);
v___x_994_ = lean_box(0);
v___x_995_ = l_Lean_MVarId_apply(v_x_984_, v_e_985_, v___x_993_, v___x_994_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
return v___x_995_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___lam__0___boxed(lean_object* v___x_996_, lean_object* v_x_997_, lean_object* v_e_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_){
_start:
{
uint8_t v___x_174__boxed_1004_; lean_object* v_res_1005_; 
v___x_174__boxed_1004_ = lean_unbox(v___x_996_);
v_res_1005_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___lam__0(v___x_174__boxed_1004_, v_x_997_, v_e_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_);
lean_dec(v___y_1002_);
lean_dec_ref(v___y_1001_);
lean_dec(v___y_1000_);
lean_dec_ref(v___y_999_);
return v_res_1005_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1(lean_object* v_x_1006_, lean_object* v_a_1007_, lean_object* v_a_1008_, lean_object* v_a_1009_, lean_object* v_a_1010_, lean_object* v_a_1011_, lean_object* v_a_1012_, lean_object* v_a_1013_, lean_object* v_a_1014_){
_start:
{
lean_object* v___x_1016_; uint8_t v___x_1017_; 
v___x_1016_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticEapply___00__closed__1));
lean_inc(v_x_1006_);
v___x_1017_ = l_Lean_Syntax_isOfKind(v_x_1006_, v___x_1016_);
if (v___x_1017_ == 0)
{
lean_object* v___x_1018_; 
lean_dec(v_x_1006_);
v___x_1018_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v___x_1018_;
}
else
{
lean_object* v___x_1019_; lean_object* v___f_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; 
v___x_1019_ = lean_box(v___x_1017_);
v___f_1020_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1020_, 0, v___x_1019_);
v___x_1021_ = lean_unsigned_to_nat(1u);
v___x_1022_ = l_Lean_Syntax_getArg(v_x_1006_, v___x_1021_);
lean_dec(v_x_1006_);
v___x_1023_ = l_Lean_Elab_Tactic_evalApplyLikeTactic(v___f_1020_, v___x_1022_, v_a_1007_, v_a_1008_, v_a_1009_, v_a_1010_, v_a_1011_, v_a_1012_, v_a_1013_, v_a_1014_);
return v___x_1023_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1___boxed(lean_object* v_x_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_, lean_object* v_a_1031_, lean_object* v_a_1032_, lean_object* v_a_1033_){
_start:
{
lean_object* v_res_1034_; 
v_res_1034_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__tacticEapply____1(v_x_1024_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_, v_a_1031_, v_a_1032_);
lean_dec(v_a_1032_);
lean_dec_ref(v_a_1031_);
lean_dec(v_a_1030_);
lean_dec_ref(v_a_1029_);
lean_dec(v_a_1028_);
lean_dec_ref(v_a_1027_);
lean_dec(v_a_1026_);
lean_dec_ref(v_a_1025_);
return v_res_1034_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0_spec__0(lean_object* v_msgData_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_){
_start:
{
lean_object* v___x_1054_; lean_object* v_env_1055_; lean_object* v___x_1056_; lean_object* v_mctx_1057_; lean_object* v_lctx_1058_; lean_object* v_options_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1054_ = lean_st_ref_get(v___y_1052_);
v_env_1055_ = lean_ctor_get(v___x_1054_, 0);
lean_inc_ref(v_env_1055_);
lean_dec(v___x_1054_);
v___x_1056_ = lean_st_ref_get(v___y_1050_);
v_mctx_1057_ = lean_ctor_get(v___x_1056_, 0);
lean_inc_ref(v_mctx_1057_);
lean_dec(v___x_1056_);
v_lctx_1058_ = lean_ctor_get(v___y_1049_, 2);
v_options_1059_ = lean_ctor_get(v___y_1051_, 2);
lean_inc_ref(v_options_1059_);
lean_inc_ref(v_lctx_1058_);
v___x_1060_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1060_, 0, v_env_1055_);
lean_ctor_set(v___x_1060_, 1, v_mctx_1057_);
lean_ctor_set(v___x_1060_, 2, v_lctx_1058_);
lean_ctor_set(v___x_1060_, 3, v_options_1059_);
v___x_1061_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1061_, 0, v___x_1060_);
lean_ctor_set(v___x_1061_, 1, v_msgData_1048_);
v___x_1062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1061_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0_spec__0___boxed(lean_object* v_msgData_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
lean_object* v_res_1069_; 
v_res_1069_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0_spec__0(v_msgData_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_);
lean_dec(v___y_1067_);
lean_dec_ref(v___y_1066_);
lean_dec(v___y_1065_);
lean_dec_ref(v___y_1064_);
return v_res_1069_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(lean_object* v_msg_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_){
_start:
{
lean_object* v_ref_1076_; lean_object* v___x_1077_; lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1086_; 
v_ref_1076_ = lean_ctor_get(v___y_1073_, 5);
v___x_1077_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0_spec__0(v_msg_1070_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_);
v_a_1078_ = lean_ctor_get(v___x_1077_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1080_ = v___x_1077_;
v_isShared_1081_ = v_isSharedCheck_1086_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1077_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1086_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1082_; lean_object* v___x_1084_; 
lean_inc(v_ref_1076_);
v___x_1082_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1082_, 0, v_ref_1076_);
lean_ctor_set(v___x_1082_, 1, v_a_1078_);
if (v_isShared_1081_ == 0)
{
lean_ctor_set_tag(v___x_1080_, 1);
lean_ctor_set(v___x_1080_, 0, v___x_1082_);
v___x_1084_ = v___x_1080_;
goto v_reusejp_1083_;
}
else
{
lean_object* v_reuseFailAlloc_1085_; 
v_reuseFailAlloc_1085_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1085_, 0, v___x_1082_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg___boxed(lean_object* v_msg_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v_res_1093_; 
v_res_1093_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(v_msg_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_);
lean_dec(v___y_1091_);
lean_dec_ref(v___y_1090_);
lean_dec(v___y_1089_);
lean_dec_ref(v___y_1088_);
return v_res_1093_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__1(void){
_start:
{
lean_object* v___x_1095_; lean_object* v___x_1096_; 
v___x_1095_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__0));
v___x_1096_ = l_Lean_stringToMessageData(v___x_1095_);
return v___x_1096_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1(lean_object* v_x_1097_, lean_object* v_a_1098_, lean_object* v_a_1099_, lean_object* v_a_1100_, lean_object* v_a_1101_, lean_object* v_a_1102_, lean_object* v_a_1103_, lean_object* v_a_1104_, lean_object* v_a_1105_){
_start:
{
lean_object* v___x_1107_; uint8_t v___x_1108_; 
v___x_1107_ = ((lean_object*)(lp_batteries_Batteries_Tactic_triv___closed__1));
v___x_1108_ = l_Lean_Syntax_isOfKind(v_x_1097_, v___x_1107_);
if (v___x_1108_ == 0)
{
lean_object* v___x_1109_; 
v___x_1109_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v___x_1109_;
}
else
{
lean_object* v___x_1110_; lean_object* v___x_1111_; 
v___x_1110_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__1, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__1_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___closed__1);
v___x_1111_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(v___x_1110_, v_a_1102_, v_a_1103_, v_a_1104_, v_a_1105_);
return v___x_1111_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1___boxed(lean_object* v_x_1112_, lean_object* v_a_1113_, lean_object* v_a_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_, lean_object* v_a_1117_, lean_object* v_a_1118_, lean_object* v_a_1119_, lean_object* v_a_1120_, lean_object* v_a_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1(v_x_1112_, v_a_1113_, v_a_1114_, v_a_1115_, v_a_1116_, v_a_1117_, v_a_1118_, v_a_1119_, v_a_1120_);
lean_dec(v_a_1120_);
lean_dec_ref(v_a_1119_);
lean_dec(v_a_1118_);
lean_dec_ref(v_a_1117_);
lean_dec(v_a_1116_);
lean_dec_ref(v_a_1115_);
lean_dec(v_a_1114_);
lean_dec_ref(v_a_1113_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0(lean_object* v_00_u03b1_1123_, lean_object* v_msg_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_){
_start:
{
lean_object* v___x_1134_; 
v___x_1134_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(v_msg_1124_, v___y_1129_, v___y_1130_, v___y_1131_, v___y_1132_);
return v___x_1134_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___boxed(lean_object* v_00_u03b1_1135_, lean_object* v_msg_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_){
_start:
{
lean_object* v_res_1146_; 
v_res_1146_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0(v_00_u03b1_1135_, v_msg_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_, v___y_1144_);
lean_dec(v___y_1144_);
lean_dec_ref(v___y_1143_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
lean_dec(v___y_1140_);
lean_dec_ref(v___y_1139_);
lean_dec(v___y_1138_);
lean_dec_ref(v___y_1137_);
return v_res_1146_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1(lean_object* v_x_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_){
_start:
{
lean_object* v___x_1178_; lean_object* v___x_1179_; uint8_t v___x_1180_; 
v___x_1178_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__0));
v___x_1179_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Conv_exact___closed__1));
lean_inc(v_x_1175_);
v___x_1180_ = l_Lean_Syntax_isOfKind(v_x_1175_, v___x_1179_);
if (v___x_1180_ == 0)
{
lean_object* v___x_1181_; lean_object* v___x_1182_; 
lean_dec(v_x_1175_);
v___x_1181_ = lean_box(1);
v___x_1182_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1182_, 0, v___x_1181_);
lean_ctor_set(v___x_1182_, 1, v_a_1177_);
return v___x_1182_;
}
else
{
lean_object* v_ref_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; uint8_t v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; 
v_ref_1183_ = lean_ctor_get(v_a_1176_, 5);
v___x_1184_ = lean_unsigned_to_nat(1u);
v___x_1185_ = l_Lean_Syntax_getArg(v_x_1175_, v___x_1184_);
lean_dec(v_x_1175_);
v___x_1186_ = 0;
v___x_1187_ = l_Lean_SourceInfo_fromRef(v_ref_1183_, v___x_1186_);
v___x_1188_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__1));
v___x_1189_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__2));
lean_inc_n(v___x_1187_, 7);
v___x_1190_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1187_);
lean_ctor_set(v___x_1190_, 1, v___x_1189_);
v___x_1191_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__3));
v___x_1192_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1192_, 0, v___x_1187_);
lean_ctor_set(v___x_1192_, 1, v___x_1191_);
v___x_1193_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6));
v___x_1194_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__8));
v___x_1195_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_1196_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__1___closed__1));
v___x_1197_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1187_);
lean_ctor_set(v___x_1197_, 1, v___x_1178_);
v___x_1198_ = l_Lean_Syntax_node2(v___x_1187_, v___x_1196_, v___x_1197_, v___x_1185_);
v___x_1199_ = l_Lean_Syntax_node1(v___x_1187_, v___x_1195_, v___x_1198_);
v___x_1200_ = l_Lean_Syntax_node1(v___x_1187_, v___x_1194_, v___x_1199_);
v___x_1201_ = l_Lean_Syntax_node1(v___x_1187_, v___x_1193_, v___x_1200_);
v___x_1202_ = l_Lean_Syntax_node3(v___x_1187_, v___x_1188_, v___x_1190_, v___x_1192_, v___x_1201_);
v___x_1203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1202_);
lean_ctor_set(v___x_1203_, 1, v_a_1177_);
return v___x_1203_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___boxed(lean_object* v_x_1204_, lean_object* v_a_1205_, lean_object* v_a_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1(v_x_1204_, v_a_1205_, v_a_1206_);
lean_dec_ref(v_a_1205_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___lam__0(lean_object* v_x_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_){
_start:
{
lean_object* v___x_1252_; 
lean_inc(v___y_1246_);
lean_inc_ref(v___y_1245_);
lean_inc(v___y_1244_);
lean_inc_ref(v___y_1243_);
v___x_1252_ = lean_apply_9(v_x_1242_, v___y_1243_, v___y_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_, lean_box(0));
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___lam__0___boxed(lean_object* v_x_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_){
_start:
{
lean_object* v_res_1263_; 
v_res_1263_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___lam__0(v_x_1253_, v___y_1254_, v___y_1255_, v___y_1256_, v___y_1257_, v___y_1258_, v___y_1259_, v___y_1260_, v___y_1261_);
lean_dec(v___y_1257_);
lean_dec_ref(v___y_1256_);
lean_dec(v___y_1255_);
lean_dec_ref(v___y_1254_);
return v_res_1263_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg(lean_object* v_mvarId_1264_, lean_object* v_x_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v___f_1275_; lean_object* v___x_1276_; 
lean_inc(v___y_1269_);
lean_inc_ref(v___y_1268_);
lean_inc(v___y_1267_);
lean_inc_ref(v___y_1266_);
v___f_1275_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1275_, 0, v_x_1265_);
lean_closure_set(v___f_1275_, 1, v___y_1266_);
lean_closure_set(v___f_1275_, 2, v___y_1267_);
lean_closure_set(v___f_1275_, 3, v___y_1268_);
lean_closure_set(v___f_1275_, 4, v___y_1269_);
v___x_1276_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1264_, v___f_1275_, v___y_1270_, v___y_1271_, v___y_1272_, v___y_1273_);
if (lean_obj_tag(v___x_1276_) == 0)
{
return v___x_1276_;
}
else
{
lean_object* v_a_1277_; lean_object* v___x_1279_; uint8_t v_isShared_1280_; uint8_t v_isSharedCheck_1284_; 
v_a_1277_ = lean_ctor_get(v___x_1276_, 0);
v_isSharedCheck_1284_ = !lean_is_exclusive(v___x_1276_);
if (v_isSharedCheck_1284_ == 0)
{
v___x_1279_ = v___x_1276_;
v_isShared_1280_ = v_isSharedCheck_1284_;
goto v_resetjp_1278_;
}
else
{
lean_inc(v_a_1277_);
lean_dec(v___x_1276_);
v___x_1279_ = lean_box(0);
v_isShared_1280_ = v_isSharedCheck_1284_;
goto v_resetjp_1278_;
}
v_resetjp_1278_:
{
lean_object* v___x_1282_; 
if (v_isShared_1280_ == 0)
{
v___x_1282_ = v___x_1279_;
goto v_reusejp_1281_;
}
else
{
lean_object* v_reuseFailAlloc_1283_; 
v_reuseFailAlloc_1283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1283_, 0, v_a_1277_);
v___x_1282_ = v_reuseFailAlloc_1283_;
goto v_reusejp_1281_;
}
v_reusejp_1281_:
{
return v___x_1282_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg___boxed(lean_object* v_mvarId_1285_, lean_object* v_x_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_){
_start:
{
lean_object* v_res_1296_; 
v_res_1296_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg(v_mvarId_1285_, v_x_1286_, v___y_1287_, v___y_1288_, v___y_1289_, v___y_1290_, v___y_1291_, v___y_1292_, v___y_1293_, v___y_1294_);
lean_dec(v___y_1294_);
lean_dec_ref(v___y_1293_);
lean_dec(v___y_1292_);
lean_dec_ref(v___y_1291_);
lean_dec(v___y_1290_);
lean_dec_ref(v___y_1289_);
lean_dec(v___y_1288_);
lean_dec_ref(v___y_1287_);
return v_res_1296_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0(lean_object* v_00_u03b1_1297_, lean_object* v_mvarId_1298_, lean_object* v_x_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_){
_start:
{
lean_object* v___x_1309_; 
v___x_1309_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg(v_mvarId_1298_, v_x_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_);
return v___x_1309_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___boxed(lean_object* v_00_u03b1_1310_, lean_object* v_mvarId_1311_, lean_object* v_x_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
lean_object* v_res_1322_; 
v_res_1322_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0(v_00_u03b1_1310_, v_mvarId_1311_, v_x_1312_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec(v___y_1316_);
lean_dec_ref(v___y_1315_);
lean_dec(v___y_1314_);
lean_dec_ref(v___y_1313_);
return v_res_1322_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__0(lean_object* v___x_1323_, lean_object* v___x_1324_, uint8_t v___x_1325_, lean_object* v___x_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_){
_start:
{
lean_object* v___x_1334_; 
v___x_1334_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_1323_, v___x_1324_, v___x_1325_, v___x_1325_, v___x_1326_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_);
return v___x_1334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__0___boxed(lean_object* v___x_1335_, lean_object* v___x_1336_, lean_object* v___x_1337_, lean_object* v___x_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_){
_start:
{
uint8_t v___x_7242__boxed_1346_; lean_object* v_res_1347_; 
v___x_7242__boxed_1346_ = lean_unbox(v___x_1337_);
v_res_1347_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__0(v___x_1335_, v___x_1336_, v___x_7242__boxed_1346_, v___x_1338_, v___y_1339_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_, v___y_1344_);
lean_dec(v___y_1344_);
lean_dec_ref(v___y_1343_);
lean_dec(v___y_1342_);
lean_dec_ref(v___y_1341_);
lean_dec(v___y_1340_);
lean_dec_ref(v___y_1339_);
return v_res_1347_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__4(void){
_start:
{
lean_object* v___x_1352_; lean_object* v___x_1353_; 
v___x_1352_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__3));
v___x_1353_ = l_Lean_stringToMessageData(v___x_1352_);
return v___x_1353_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__6(void){
_start:
{
lean_object* v___x_1355_; lean_object* v___x_1356_; 
v___x_1355_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__5));
v___x_1356_ = l_Lean_stringToMessageData(v___x_1355_);
return v___x_1356_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__8(void){
_start:
{
lean_object* v___x_1358_; lean_object* v___x_1359_; 
v___x_1358_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__7));
v___x_1359_ = l_Lean_stringToMessageData(v___x_1358_);
return v___x_1359_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1(lean_object* v_a_1360_, lean_object* v___x_1361_, uint8_t v___x_1362_, lean_object* v___x_1363_, lean_object* v___x_1364_, lean_object* v___x_1365_, lean_object* v___x_1366_, lean_object* v___x_1367_, lean_object* v___x_1368_, lean_object* v___y_1369_, lean_object* v___y_1370_, lean_object* v___y_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_){
_start:
{
lean_object* v___y_1379_; lean_object* v___y_1380_; lean_object* v___y_1381_; lean_object* v___y_1382_; lean_object* v___y_1383_; lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v___y_1386_; lean_object* v___x_1412_; 
v___x_1412_ = l_Lean_MVarId_getType(v_a_1360_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
if (lean_obj_tag(v___x_1412_) == 0)
{
lean_object* v_a_1413_; lean_object* v___x_1414_; 
v_a_1413_ = lean_ctor_get(v___x_1412_, 0);
lean_inc(v_a_1413_);
lean_dec_ref_known(v___x_1412_, 1);
v___x_1414_ = l_Lean_Meta_matchEq_x3f(v_a_1413_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
if (lean_obj_tag(v___x_1414_) == 0)
{
lean_object* v_a_1415_; 
v_a_1415_ = lean_ctor_get(v___x_1414_, 0);
lean_inc(v_a_1415_);
lean_dec_ref_known(v___x_1414_, 1);
if (lean_obj_tag(v_a_1415_) == 1)
{
lean_object* v_val_1416_; lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1472_; 
v_val_1416_ = lean_ctor_get(v_a_1415_, 0);
v_isSharedCheck_1472_ = !lean_is_exclusive(v_a_1415_);
if (v_isSharedCheck_1472_ == 0)
{
v___x_1418_ = v_a_1415_;
v_isShared_1419_ = v_isSharedCheck_1472_;
goto v_resetjp_1417_;
}
else
{
lean_inc(v_val_1416_);
lean_dec(v_a_1415_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1472_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
lean_object* v_snd_1420_; lean_object* v_fst_1421_; lean_object* v___x_1423_; uint8_t v_isShared_1424_; uint8_t v_isSharedCheck_1471_; 
v_snd_1420_ = lean_ctor_get(v_val_1416_, 1);
v_fst_1421_ = lean_ctor_get(v_val_1416_, 0);
v_isSharedCheck_1471_ = !lean_is_exclusive(v_val_1416_);
if (v_isSharedCheck_1471_ == 0)
{
v___x_1423_ = v_val_1416_;
v_isShared_1424_ = v_isSharedCheck_1471_;
goto v_resetjp_1422_;
}
else
{
lean_inc(v_snd_1420_);
lean_inc(v_fst_1421_);
lean_dec(v_val_1416_);
v___x_1423_ = lean_box(0);
v_isShared_1424_ = v_isSharedCheck_1471_;
goto v_resetjp_1422_;
}
v_resetjp_1422_:
{
lean_object* v_snd_1425_; lean_object* v___x_1427_; uint8_t v_isShared_1428_; uint8_t v_isSharedCheck_1469_; 
v_snd_1425_ = lean_ctor_get(v_snd_1420_, 1);
v_isSharedCheck_1469_ = !lean_is_exclusive(v_snd_1420_);
if (v_isSharedCheck_1469_ == 0)
{
lean_object* v_unused_1470_; 
v_unused_1470_ = lean_ctor_get(v_snd_1420_, 0);
lean_dec(v_unused_1470_);
v___x_1427_ = v_snd_1420_;
v_isShared_1428_ = v_isSharedCheck_1469_;
goto v_resetjp_1426_;
}
else
{
lean_inc(v_snd_1425_);
lean_dec(v_snd_1420_);
v___x_1427_ = lean_box(0);
v_isShared_1428_ = v_isSharedCheck_1469_;
goto v_resetjp_1426_;
}
v_resetjp_1426_:
{
lean_object* v___x_1430_; 
if (v_isShared_1419_ == 0)
{
lean_ctor_set(v___x_1418_, 0, v_fst_1421_);
v___x_1430_ = v___x_1418_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v_fst_1421_);
v___x_1430_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___f_1433_; uint8_t v___x_1434_; lean_object* v___x_1435_; 
v___x_1431_ = lean_box(0);
v___x_1432_ = lean_box(v___x_1362_);
v___f_1433_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__0___boxed), 11, 4);
lean_closure_set(v___f_1433_, 0, v___x_1361_);
lean_closure_set(v___f_1433_, 1, v___x_1430_);
lean_closure_set(v___f_1433_, 2, v___x_1432_);
lean_closure_set(v___f_1433_, 3, v___x_1431_);
v___x_1434_ = 1;
v___x_1435_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___f_1433_, v___x_1434_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
if (lean_obj_tag(v___x_1435_) == 0)
{
lean_object* v_a_1436_; lean_object* v___x_1437_; 
v_a_1436_ = lean_ctor_get(v___x_1435_, 0);
lean_inc_n(v_a_1436_, 2);
lean_dec_ref_known(v___x_1435_, 1);
lean_inc(v_snd_1425_);
v___x_1437_ = l_Lean_Meta_isExprDefEq(v_snd_1425_, v_a_1436_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
if (lean_obj_tag(v___x_1437_) == 0)
{
lean_object* v_a_1438_; uint8_t v___x_1439_; 
v_a_1438_ = lean_ctor_get(v___x_1437_, 0);
lean_inc(v_a_1438_);
lean_dec_ref_known(v___x_1437_, 1);
v___x_1439_ = lean_unbox(v_a_1438_);
lean_dec(v_a_1438_);
if (v___x_1439_ == 0)
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1443_; 
lean_dec(v___x_1368_);
lean_dec(v___x_1367_);
lean_dec_ref(v___x_1366_);
lean_dec_ref(v___x_1365_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
v___x_1440_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__4, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__4_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__4);
v___x_1441_ = l_Lean_indentExpr(v_snd_1425_);
if (v_isShared_1428_ == 0)
{
lean_ctor_set_tag(v___x_1427_, 7);
lean_ctor_set(v___x_1427_, 1, v___x_1441_);
lean_ctor_set(v___x_1427_, 0, v___x_1440_);
v___x_1443_ = v___x_1427_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1451_; 
v_reuseFailAlloc_1451_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1451_, 0, v___x_1440_);
lean_ctor_set(v_reuseFailAlloc_1451_, 1, v___x_1441_);
v___x_1443_ = v_reuseFailAlloc_1451_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
lean_object* v___x_1444_; lean_object* v___x_1446_; 
v___x_1444_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__6, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__6_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__6);
if (v_isShared_1424_ == 0)
{
lean_ctor_set_tag(v___x_1423_, 7);
lean_ctor_set(v___x_1423_, 1, v___x_1444_);
lean_ctor_set(v___x_1423_, 0, v___x_1443_);
v___x_1446_ = v___x_1423_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v___x_1443_);
lean_ctor_set(v_reuseFailAlloc_1450_, 1, v___x_1444_);
v___x_1446_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; 
v___x_1447_ = l_Lean_indentExpr(v_a_1436_);
v___x_1448_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1448_, 0, v___x_1446_);
lean_ctor_set(v___x_1448_, 1, v___x_1447_);
v___x_1449_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(v___x_1448_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
return v___x_1449_;
}
}
}
else
{
lean_dec(v_a_1436_);
lean_del_object(v___x_1427_);
lean_dec(v_snd_1425_);
lean_del_object(v___x_1423_);
v___y_1379_ = v___y_1369_;
v___y_1380_ = v___y_1370_;
v___y_1381_ = v___y_1371_;
v___y_1382_ = v___y_1372_;
v___y_1383_ = v___y_1373_;
v___y_1384_ = v___y_1374_;
v___y_1385_ = v___y_1375_;
v___y_1386_ = v___y_1376_;
goto v___jp_1378_;
}
}
else
{
lean_object* v_a_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1459_; 
lean_dec(v_a_1436_);
lean_del_object(v___x_1427_);
lean_dec(v_snd_1425_);
lean_del_object(v___x_1423_);
lean_dec(v___x_1368_);
lean_dec(v___x_1367_);
lean_dec_ref(v___x_1366_);
lean_dec_ref(v___x_1365_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
v_a_1452_ = lean_ctor_get(v___x_1437_, 0);
v_isSharedCheck_1459_ = !lean_is_exclusive(v___x_1437_);
if (v_isSharedCheck_1459_ == 0)
{
v___x_1454_ = v___x_1437_;
v_isShared_1455_ = v_isSharedCheck_1459_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_a_1452_);
lean_dec(v___x_1437_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1459_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1457_; 
if (v_isShared_1455_ == 0)
{
v___x_1457_ = v___x_1454_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1458_; 
v_reuseFailAlloc_1458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1458_, 0, v_a_1452_);
v___x_1457_ = v_reuseFailAlloc_1458_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
return v___x_1457_;
}
}
}
}
else
{
lean_object* v_a_1460_; lean_object* v___x_1462_; uint8_t v_isShared_1463_; uint8_t v_isSharedCheck_1467_; 
lean_del_object(v___x_1427_);
lean_dec(v_snd_1425_);
lean_del_object(v___x_1423_);
lean_dec(v___x_1368_);
lean_dec(v___x_1367_);
lean_dec_ref(v___x_1366_);
lean_dec_ref(v___x_1365_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
v_a_1460_ = lean_ctor_get(v___x_1435_, 0);
v_isSharedCheck_1467_ = !lean_is_exclusive(v___x_1435_);
if (v_isSharedCheck_1467_ == 0)
{
v___x_1462_ = v___x_1435_;
v_isShared_1463_ = v_isSharedCheck_1467_;
goto v_resetjp_1461_;
}
else
{
lean_inc(v_a_1460_);
lean_dec(v___x_1435_);
v___x_1462_ = lean_box(0);
v_isShared_1463_ = v_isSharedCheck_1467_;
goto v_resetjp_1461_;
}
v_resetjp_1461_:
{
lean_object* v___x_1465_; 
if (v_isShared_1463_ == 0)
{
v___x_1465_ = v___x_1462_;
goto v_reusejp_1464_;
}
else
{
lean_object* v_reuseFailAlloc_1466_; 
v_reuseFailAlloc_1466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1466_, 0, v_a_1460_);
v___x_1465_ = v_reuseFailAlloc_1466_;
goto v_reusejp_1464_;
}
v_reusejp_1464_:
{
return v___x_1465_;
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
lean_object* v___x_1473_; lean_object* v___x_1474_; 
lean_dec(v_a_1415_);
lean_dec(v___x_1368_);
lean_dec(v___x_1367_);
lean_dec_ref(v___x_1366_);
lean_dec_ref(v___x_1365_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
lean_dec(v___x_1361_);
v___x_1473_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__8, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__8_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__8);
v___x_1474_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__triv__1_spec__0___redArg(v___x_1473_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
return v___x_1474_;
}
}
else
{
lean_object* v_a_1475_; lean_object* v___x_1477_; uint8_t v_isShared_1478_; uint8_t v_isSharedCheck_1482_; 
lean_dec(v___x_1368_);
lean_dec(v___x_1367_);
lean_dec_ref(v___x_1366_);
lean_dec_ref(v___x_1365_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
lean_dec(v___x_1361_);
v_a_1475_ = lean_ctor_get(v___x_1414_, 0);
v_isSharedCheck_1482_ = !lean_is_exclusive(v___x_1414_);
if (v_isSharedCheck_1482_ == 0)
{
v___x_1477_ = v___x_1414_;
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
else
{
lean_inc(v_a_1475_);
lean_dec(v___x_1414_);
v___x_1477_ = lean_box(0);
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
v_resetjp_1476_:
{
lean_object* v___x_1480_; 
if (v_isShared_1478_ == 0)
{
v___x_1480_ = v___x_1477_;
goto v_reusejp_1479_;
}
else
{
lean_object* v_reuseFailAlloc_1481_; 
v_reuseFailAlloc_1481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1481_, 0, v_a_1475_);
v___x_1480_ = v_reuseFailAlloc_1481_;
goto v_reusejp_1479_;
}
v_reusejp_1479_:
{
return v___x_1480_;
}
}
}
}
else
{
lean_object* v_a_1483_; lean_object* v___x_1485_; uint8_t v_isShared_1486_; uint8_t v_isSharedCheck_1490_; 
lean_dec(v___x_1368_);
lean_dec(v___x_1367_);
lean_dec_ref(v___x_1366_);
lean_dec_ref(v___x_1365_);
lean_dec_ref(v___x_1364_);
lean_dec_ref(v___x_1363_);
lean_dec(v___x_1361_);
v_a_1483_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1490_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1490_ == 0)
{
v___x_1485_ = v___x_1412_;
v_isShared_1486_ = v_isSharedCheck_1490_;
goto v_resetjp_1484_;
}
else
{
lean_inc(v_a_1483_);
lean_dec(v___x_1412_);
v___x_1485_ = lean_box(0);
v_isShared_1486_ = v_isSharedCheck_1490_;
goto v_resetjp_1484_;
}
v_resetjp_1484_:
{
lean_object* v___x_1488_; 
if (v_isShared_1486_ == 0)
{
v___x_1488_ = v___x_1485_;
goto v_reusejp_1487_;
}
else
{
lean_object* v_reuseFailAlloc_1489_; 
v_reuseFailAlloc_1489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1489_, 0, v_a_1483_);
v___x_1488_ = v_reuseFailAlloc_1489_;
goto v_reusejp_1487_;
}
v_reusejp_1487_:
{
return v___x_1488_;
}
}
}
v___jp_1378_:
{
lean_object* v_ref_1387_; uint8_t v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; 
v_ref_1387_ = lean_ctor_get(v___y_1385_, 5);
v___x_1388_ = 0;
v___x_1389_ = l_Lean_SourceInfo_fromRef(v_ref_1387_, v___x_1388_);
v___x_1390_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__0));
lean_inc_ref(v___x_1365_);
lean_inc_ref(v___x_1364_);
lean_inc_ref_n(v___x_1363_, 3);
v___x_1391_ = l_Lean_Name_mkStr5(v___x_1363_, v___x_1364_, v___x_1365_, v___x_1366_, v___x_1390_);
v___x_1392_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__2));
lean_inc_n(v___x_1389_, 8);
v___x_1393_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1393_, 0, v___x_1389_);
lean_ctor_set(v___x_1393_, 1, v___x_1392_);
v___x_1394_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__Conv__exact__1___closed__3));
v___x_1395_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1395_, 0, v___x_1389_);
lean_ctor_set(v___x_1395_, 1, v___x_1394_);
v___x_1396_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__7));
v___x_1397_ = l_Lean_Name_mkStr4(v___x_1363_, v___x_1364_, v___x_1365_, v___x_1396_);
v___x_1398_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__6));
v___x_1399_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__0));
v___x_1400_ = l_Lean_Name_mkStr2(v___x_1363_, v___x_1399_);
v___x_1401_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__1));
v___x_1402_ = l_Lean_Name_mkStr2(v___x_1363_, v___x_1401_);
v___x_1403_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___closed__2));
v___x_1404_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1404_, 0, v___x_1389_);
lean_ctor_set(v___x_1404_, 1, v___x_1403_);
v___x_1405_ = l_Lean_Syntax_node1(v___x_1389_, v___x_1402_, v___x_1404_);
v___x_1406_ = l_Lean_Syntax_node2(v___x_1389_, v___x_1400_, v___x_1405_, v___x_1367_);
v___x_1407_ = l_Lean_Syntax_node1(v___x_1389_, v___x_1398_, v___x_1406_);
v___x_1408_ = l_Lean_Syntax_node1(v___x_1389_, v___x_1397_, v___x_1407_);
v___x_1409_ = l_Lean_Syntax_node1(v___x_1389_, v___x_1368_, v___x_1408_);
v___x_1410_ = l_Lean_Syntax_node3(v___x_1389_, v___x_1391_, v___x_1393_, v___x_1395_, v___x_1409_);
v___x_1411_ = l_Lean_Elab_Tactic_evalTactic(v___x_1410_, v___y_1379_, v___y_1380_, v___y_1381_, v___y_1382_, v___y_1383_, v___y_1384_, v___y_1385_, v___y_1386_);
return v___x_1411_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___boxed(lean_object** _args){
lean_object* v_a_1491_ = _args[0];
lean_object* v___x_1492_ = _args[1];
lean_object* v___x_1493_ = _args[2];
lean_object* v___x_1494_ = _args[3];
lean_object* v___x_1495_ = _args[4];
lean_object* v___x_1496_ = _args[5];
lean_object* v___x_1497_ = _args[6];
lean_object* v___x_1498_ = _args[7];
lean_object* v___x_1499_ = _args[8];
lean_object* v___y_1500_ = _args[9];
lean_object* v___y_1501_ = _args[10];
lean_object* v___y_1502_ = _args[11];
lean_object* v___y_1503_ = _args[12];
lean_object* v___y_1504_ = _args[13];
lean_object* v___y_1505_ = _args[14];
lean_object* v___y_1506_ = _args[15];
lean_object* v___y_1507_ = _args[16];
lean_object* v___y_1508_ = _args[17];
_start:
{
uint8_t v___x_7308__boxed_1509_; lean_object* v_res_1510_; 
v___x_7308__boxed_1509_ = lean_unbox(v___x_1493_);
v_res_1510_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1(v_a_1491_, v___x_1492_, v___x_7308__boxed_1509_, v___x_1494_, v___x_1495_, v___x_1496_, v___x_1497_, v___x_1498_, v___x_1499_, v___y_1500_, v___y_1501_, v___y_1502_, v___y_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_);
lean_dec(v___y_1507_);
lean_dec_ref(v___y_1506_);
lean_dec(v___y_1505_);
lean_dec_ref(v___y_1504_);
lean_dec(v___y_1503_);
lean_dec_ref(v___y_1502_);
lean_dec(v___y_1501_);
lean_dec_ref(v___y_1500_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1(lean_object* v_x_1511_, lean_object* v_a_1512_, lean_object* v_a_1513_, lean_object* v_a_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_, lean_object* v_a_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_){
_start:
{
lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; uint8_t v___x_1524_; 
v___x_1521_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tactic___00__closed__1));
v___x_1522_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Conv_exact___closed__0));
v___x_1523_ = ((lean_object*)(lp_batteries_Batteries_Tactic_Conv_equals___closed__1));
lean_inc(v_x_1511_);
v___x_1524_ = l_Lean_Syntax_isOfKind(v_x_1511_, v___x_1523_);
if (v___x_1524_ == 0)
{
lean_object* v___x_1525_; 
lean_dec(v_x_1511_);
v___x_1525_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__exacts__1_spec__0___redArg();
return v___x_1525_;
}
else
{
lean_object* v___x_1526_; 
v___x_1526_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_1513_, v_a_1516_, v_a_1517_, v_a_1518_, v_a_1519_);
if (lean_obj_tag(v___x_1526_) == 0)
{
lean_object* v_a_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; lean_object* v___f_1536_; lean_object* v___x_1537_; 
v_a_1527_ = lean_ctor_get(v___x_1526_, 0);
lean_inc_n(v_a_1527_, 2);
lean_dec_ref_known(v___x_1526_, 1);
v___x_1528_ = lean_unsigned_to_nat(1u);
v___x_1529_ = l_Lean_Syntax_getArg(v_x_1511_, v___x_1528_);
v___x_1530_ = lean_unsigned_to_nat(3u);
v___x_1531_ = l_Lean_Syntax_getArg(v_x_1511_, v___x_1530_);
lean_dec(v_x_1511_);
v___x_1532_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__0));
v___x_1533_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tactic____1___closed__1));
v___x_1534_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______macroRules__Batteries__Tactic__tacticBy__contra__core__1___closed__6));
v___x_1535_ = lean_box(v___x_1524_);
v___f_1536_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___lam__1___boxed), 18, 9);
lean_closure_set(v___f_1536_, 0, v_a_1527_);
lean_closure_set(v___f_1536_, 1, v___x_1529_);
lean_closure_set(v___f_1536_, 2, v___x_1535_);
lean_closure_set(v___f_1536_, 3, v___x_1532_);
lean_closure_set(v___f_1536_, 4, v___x_1533_);
lean_closure_set(v___f_1536_, 5, v___x_1521_);
lean_closure_set(v___f_1536_, 6, v___x_1522_);
lean_closure_set(v___f_1536_, 7, v___x_1531_);
lean_closure_set(v___f_1536_, 8, v___x_1534_);
v___x_1537_ = lp_batteries_Lean_MVarId_withContext___at___00Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1_spec__0___redArg(v_a_1527_, v___f_1536_, v_a_1512_, v_a_1513_, v_a_1514_, v_a_1515_, v_a_1516_, v_a_1517_, v_a_1518_, v_a_1519_);
return v___x_1537_;
}
else
{
lean_object* v_a_1538_; lean_object* v___x_1540_; uint8_t v_isShared_1541_; uint8_t v_isSharedCheck_1545_; 
lean_dec(v_x_1511_);
v_a_1538_ = lean_ctor_get(v___x_1526_, 0);
v_isSharedCheck_1545_ = !lean_is_exclusive(v___x_1526_);
if (v_isSharedCheck_1545_ == 0)
{
v___x_1540_ = v___x_1526_;
v_isShared_1541_ = v_isSharedCheck_1545_;
goto v_resetjp_1539_;
}
else
{
lean_inc(v_a_1538_);
lean_dec(v___x_1526_);
v___x_1540_ = lean_box(0);
v_isShared_1541_ = v_isSharedCheck_1545_;
goto v_resetjp_1539_;
}
v_resetjp_1539_:
{
lean_object* v___x_1543_; 
if (v_isShared_1541_ == 0)
{
v___x_1543_ = v___x_1540_;
goto v_reusejp_1542_;
}
else
{
lean_object* v_reuseFailAlloc_1544_; 
v_reuseFailAlloc_1544_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1544_, 0, v_a_1538_);
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
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1___boxed(lean_object* v_x_1546_, lean_object* v_a_1547_, lean_object* v_a_1548_, lean_object* v_a_1549_, lean_object* v_a_1550_, lean_object* v_a_1551_, lean_object* v_a_1552_, lean_object* v_a_1553_, lean_object* v_a_1554_, lean_object* v_a_1555_){
_start:
{
lean_object* v_res_1556_; 
v_res_1556_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Init______elabRules__Batteries__Tactic__Conv__equals__1(v_x_1546_, v_a_1547_, v_a_1548_, v_a_1549_, v_a_1550_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
lean_dec(v_a_1554_);
lean_dec_ref(v_a_1553_);
lean_dec(v_a_1552_);
lean_dec_ref(v_a_1551_);
lean_dec(v_a_1550_);
lean_dec_ref(v_a_1549_);
lean_dec(v_a_1548_);
lean_dec_ref(v_a_1547_);
return v_res_1556_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Init(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_MatchUtil(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Init(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_MatchUtil(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Batteries_Tactic_byContra = _init_lp_batteries_Batteries_Tactic_byContra();
lean_mark_persistent(lp_batteries_Batteries_Tactic_byContra);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_Lean_Meta_MatchUtil(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_MatchUtil(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Init(builtin);
}
#ifdef __cplusplus
}
#endif
