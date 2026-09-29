// Lean compiler output
// Module: Mathlib.Order.Notation
// Imports: public import Init public meta import Init public import Qq public meta import Mathlib.Lean.PrettyPrinter.Delaborator public import Mathlib.Tactic.Simps public import Mathlib.Tactic.ToDual public meta import Lean.PrettyPrinter.Delaborator.Builtins
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
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_annotateGoToSyntaxDef(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Expr_constLevels_x21(lean_object*);
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toList___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Expr_betaRev(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lp_Qq_Qq_mkFreshExprMVarQ___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLocalInstancesImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u1d9c___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_ᶜ"};
static const lean_object* lp_mathlib_term___u1d9c___closed__0 = (const lean_object*)&lp_mathlib_term___u1d9c___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u1d9c___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d9c___closed__0_value),LEAN_SCALAR_PTR_LITERAL(128, 3, 137, 103, 191, 193, 176, 89)}};
static const lean_object* lp_mathlib_term___u1d9c___closed__1 = (const lean_object*)&lp_mathlib_term___u1d9c___closed__1_value;
static const lean_string_object lp_mathlib_term___u1d9c___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ᶜ"};
static const lean_object* lp_mathlib_term___u1d9c___closed__2 = (const lean_object*)&lp_mathlib_term___u1d9c___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u1d9c___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d9c___closed__2_value)}};
static const lean_object* lp_mathlib_term___u1d9c___closed__3 = (const lean_object*)&lp_mathlib_term___u1d9c___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u1d9c___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u1d9c___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u1d9c___closed__3_value)}};
static const lean_object* lp_mathlib_term___u1d9c___closed__4 = (const lean_object*)&lp_mathlib_term___u1d9c___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u1d9c = (const lean_object*)&lp_mathlib_term___u1d9c___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "compl"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__6;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(67, 51, 0, 102, 49, 143, 132, 80)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Compl"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(86, 104, 100, 165, 159, 188, 212, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(162, 20, 47, 39, 134, 212, 205, 41)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2294___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊔_"};
static const lean_object* lp_mathlib_term___u2294___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2294___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(253, 33, 239, 70, 69, 183, 45, 198)}};
static const lean_object* lp_mathlib_term___u2294___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2294___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2294___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2294___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2294___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2294___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⊔ "};
static const lean_object* lp_mathlib_term___u2294___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2294___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2294___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2294___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2294___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2294___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__7_value),((lean_object*)(((size_t)(69) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2294___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2294___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2294___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2294___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2294___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__1_value),((lean_object*)(((size_t)(68) << 1) | 1)),((lean_object*)(((size_t)(68) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2294___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2294___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2294___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2294__ = (const lean_object*)&lp_mathlib_term___u2294___00__closed__10_value;
static const lean_string_object lp_mathlib_term___u2293___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊓_"};
static const lean_object* lp_mathlib_term___u2293___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2293___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2293___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 218, 188, 48, 229, 20, 44, 30)}};
static const lean_object* lp_mathlib_term___u2293___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2293___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⊓ "};
static const lean_object* lp_mathlib_term___u2293___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2293___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2293___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u2293___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u2293___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__7_value),((lean_object*)(((size_t)(70) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2293___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2293___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2293___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2293___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2293___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u2293___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2293___00__closed__1_value),((lean_object*)(((size_t)(69) << 1) | 1)),((lean_object*)(((size_t)(69) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2293___00__closed__5_value)}};
static const lean_object* lp_mathlib_term___u2293___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2293___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2293__ = (const lean_object*)&lp_mathlib_term___u2293___00__closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Max.max"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Max"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "max"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(169, 95, 226, 81, 206, 208, 89, 76)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(247, 27, 157, 195, 66, 157, 90, 150)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Min.min"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Min"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "min"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 99, 105, 121, 176, 241, 22, 117)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 174, 129, 224, 94, 80, 42, 239)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMax"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__0_value),LEAN_SCALAR_PTR_LITERAL(183, 226, 75, 5, 108, 25, 239, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "toMin"};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 121, 184, 101, 172, 193, 147, 203)}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__0(lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊔"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabSup___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_delabSup___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabSup___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabSup___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabSup___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabSup___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__3_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabSup___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_delabInf___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊓"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabInf___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabInf___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabInf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_delabInf___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabInf___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabInf___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabInf___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_delabInf___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabInf___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabInf___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabInf___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_delabSup___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Meta_delabInf___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabInf___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabInf___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u21e8___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⇨_"};
static const lean_object* lp_mathlib_term___u21e8___00__closed__0 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u21e8___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u21e8___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 90, 166, 103, 252, 89, 56, 18)}};
static const lean_object* lp_mathlib_term___u21e8___00__closed__1 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u21e8___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⇨ "};
static const lean_object* lp_mathlib_term___u21e8___00__closed__2 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u21e8___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u21e8___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u21e8___00__closed__3 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u21e8___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__7_value),((lean_object*)(((size_t)(60) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u21e8___00__closed__4 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u21e8___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__3_value),((lean_object*)&lp_mathlib_term___u21e8___00__closed__3_value),((lean_object*)&lp_mathlib_term___u21e8___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u21e8___00__closed__5 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___u21e8___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u21e8___00__closed__1_value),((lean_object*)(((size_t)(60) << 1) | 1)),((lean_object*)(((size_t)(61) << 1) | 1)),((lean_object*)&lp_mathlib_term___u21e8___00__closed__5_value)}};
static const lean_object* lp_mathlib_term___u21e8___00__closed__6 = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u21e8__ = (const lean_object*)&lp_mathlib_term___u21e8___00__closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "himp"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 93, 163, 74, 45, 235, 252, 9)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HImp"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(92, 198, 231, 201, 58, 189, 79, 50)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 93, 173, 50, 188, 112, 255, 243)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HImp__himp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HImp__himp__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_uffe2___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term￢_"};
static const lean_object* lp_mathlib_term_uffe2___00__closed__0 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_uffe2___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_uffe2___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(134, 250, 70, 114, 249, 232, 48, 35)}};
static const lean_object* lp_mathlib_term_uffe2___00__closed__1 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__1_value;
static const lean_string_object lp_mathlib_term_uffe2___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "￢"};
static const lean_object* lp_mathlib_term_uffe2___00__closed__2 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_uffe2___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_uffe2___00__closed__2_value)}};
static const lean_object* lp_mathlib_term_uffe2___00__closed__3 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term_uffe2___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__7_value),((lean_object*)(((size_t)(72) << 1) | 1))}};
static const lean_object* lp_mathlib_term_uffe2___00__closed__4 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term_uffe2___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2294___00__closed__3_value),((lean_object*)&lp_mathlib_term_uffe2___00__closed__3_value),((lean_object*)&lp_mathlib_term_uffe2___00__closed__4_value)}};
static const lean_object* lp_mathlib_term_uffe2___00__closed__5 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term_uffe2___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_uffe2___00__closed__1_value),((lean_object*)(((size_t)(72) << 1) | 1)),((lean_object*)&lp_mathlib_term_uffe2___00__closed__5_value)}};
static const lean_object* lp_mathlib_term_uffe2___00__closed__6 = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_uffe2__ = (const lean_object*)&lp_mathlib_term_uffe2___00__closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hnot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(206, 73, 113, 22, 73, 144, 163, 231)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "HNot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(200, 90, 78, 181, 255, 197, 234, 147)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 250, 111, 236, 207, 4, 110, 102)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HNot__hnot__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HNot__hnot__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u22a4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = "term⊤"};
static const lean_object* lp_mathlib_term_u22a4___closed__0 = (const lean_object*)&lp_mathlib_term_u22a4___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u22a4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u22a4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 131, 173, 32, 247, 58, 40, 144)}};
static const lean_object* lp_mathlib_term_u22a4___closed__1 = (const lean_object*)&lp_mathlib_term_u22a4___closed__1_value;
static const lean_string_object lp_mathlib_term_u22a4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊤"};
static const lean_object* lp_mathlib_term_u22a4___closed__2 = (const lean_object*)&lp_mathlib_term_u22a4___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u22a4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u22a4___closed__2_value)}};
static const lean_object* lp_mathlib_term_u22a4___closed__3 = (const lean_object*)&lp_mathlib_term_u22a4___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u22a4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u22a4___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u22a4___closed__3_value)}};
static const lean_object* lp_mathlib_term_u22a4___closed__4 = (const lean_object*)&lp_mathlib_term_u22a4___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u22a4 = (const lean_object*)&lp_mathlib_term_u22a4___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Top.top"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Top"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "top"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(17, 209, 230, 57, 51, 197, 162, 233)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(101, 62, 44, 17, 165, 201, 212, 212)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Top__top__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Top__top__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u22a5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = "term⊥"};
static const lean_object* lp_mathlib_term_u22a5___closed__0 = (const lean_object*)&lp_mathlib_term_u22a5___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u22a5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u22a5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(12, 86, 222, 214, 153, 24, 42, 91)}};
static const lean_object* lp_mathlib_term_u22a5___closed__1 = (const lean_object*)&lp_mathlib_term_u22a5___closed__1_value;
static const lean_string_object lp_mathlib_term_u22a5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊥"};
static const lean_object* lp_mathlib_term_u22a5___closed__2 = (const lean_object*)&lp_mathlib_term_u22a5___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u22a5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u22a5___closed__2_value)}};
static const lean_object* lp_mathlib_term_u22a5___closed__3 = (const lean_object*)&lp_mathlib_term_u22a5___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u22a5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u22a5___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u22a5___closed__3_value)}};
static const lean_object* lp_mathlib_term_u22a5___closed__4 = (const lean_object*)&lp_mathlib_term_u22a5___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u22a5 = (const lean_object*)&lp_mathlib_term_u22a5___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Bot.bot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Bot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "bot"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(192, 138, 190, 95, 247, 78, 16, 101)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(98, 132, 46, 181, 27, 87, 250, 96)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Bot__bot__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Bot__bot__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__6(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__5));
v___x_23_ = l_String_toRawSubstring_x27(v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1(lean_object* v_x_39_, lean_object* v_a_40_, lean_object* v_a_41_){
_start:
{
lean_object* v___x_42_; uint8_t v___x_43_; 
v___x_42_ = ((lean_object*)(lp_mathlib_term___u1d9c___closed__1));
lean_inc(v_x_39_);
v___x_43_ = l_Lean_Syntax_isOfKind(v_x_39_, v___x_42_);
if (v___x_43_ == 0)
{
lean_object* v___x_44_; lean_object* v___x_45_; 
lean_dec(v_x_39_);
v___x_44_ = lean_box(1);
v___x_45_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v_a_41_);
return v___x_45_;
}
else
{
lean_object* v_quotContext_46_; lean_object* v_currMacroScope_47_; lean_object* v_ref_48_; lean_object* v___x_49_; lean_object* v___x_50_; uint8_t v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v_quotContext_46_ = lean_ctor_get(v_a_40_, 1);
v_currMacroScope_47_ = lean_ctor_get(v_a_40_, 2);
v_ref_48_ = lean_ctor_get(v_a_40_, 5);
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = l_Lean_Syntax_getArg(v_x_39_, v___x_49_);
lean_dec(v_x_39_);
v___x_51_ = 0;
v___x_52_ = l_Lean_SourceInfo_fromRef(v_ref_48_, v___x_51_);
v___x_53_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
v___x_54_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__6, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__6);
v___x_55_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__7));
lean_inc(v_currMacroScope_47_);
lean_inc(v_quotContext_46_);
v___x_56_ = l_Lean_addMacroScope(v_quotContext_46_, v___x_55_, v_currMacroScope_47_);
v___x_57_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__11));
lean_inc_n(v___x_52_, 2);
v___x_58_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_58_, 0, v___x_52_);
lean_ctor_set(v___x_58_, 1, v___x_54_);
lean_ctor_set(v___x_58_, 2, v___x_56_);
lean_ctor_set(v___x_58_, 3, v___x_57_);
v___x_59_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13));
v___x_60_ = l_Lean_Syntax_node1(v___x_52_, v___x_59_, v___x_50_);
v___x_61_ = l_Lean_Syntax_node2(v___x_52_, v___x_53_, v___x_58_, v___x_60_);
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v_a_41_);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___boxed(lean_object* v_x_63_, lean_object* v_a_64_, lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1(v_x_63_, v_a_64_, v_a_65_);
lean_dec_ref(v_a_64_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1(lean_object* v_x_70_, lean_object* v_a_71_, lean_object* v_a_72_){
_start:
{
lean_object* v___x_73_; uint8_t v___x_74_; 
v___x_73_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
lean_inc(v_x_70_);
v___x_74_ = l_Lean_Syntax_isOfKind(v_x_70_, v___x_73_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; lean_object* v___x_76_; 
lean_dec(v_x_70_);
v___x_75_ = lean_box(0);
v___x_76_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v_a_72_);
return v___x_76_;
}
else
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_77_ = lean_unsigned_to_nat(0u);
v___x_78_ = l_Lean_Syntax_getArg(v_x_70_, v___x_77_);
v___x_79_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1));
lean_inc(v___x_78_);
v___x_80_ = l_Lean_Syntax_isOfKind(v___x_78_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; lean_object* v___x_82_; 
lean_dec(v___x_78_);
lean_dec(v_x_70_);
v___x_81_ = lean_box(0);
v___x_82_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v_a_72_);
return v___x_82_;
}
else
{
lean_object* v___x_83_; lean_object* v___x_84_; uint8_t v___x_85_; 
v___x_83_ = lean_unsigned_to_nat(1u);
v___x_84_ = l_Lean_Syntax_getArg(v_x_70_, v___x_83_);
lean_dec(v_x_70_);
lean_inc(v___x_84_);
v___x_85_ = l_Lean_Syntax_matchesNull(v___x_84_, v___x_83_);
if (v___x_85_ == 0)
{
lean_object* v___x_86_; lean_object* v___x_87_; 
lean_dec(v___x_84_);
lean_dec(v___x_78_);
v___x_86_ = lean_box(0);
v___x_87_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v_a_72_);
return v___x_87_;
}
else
{
lean_object* v___x_88_; lean_object* v_ref_89_; uint8_t v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_88_ = l_Lean_Syntax_getArg(v___x_84_, v___x_77_);
lean_dec(v___x_84_);
v_ref_89_ = l_Lean_replaceRef(v___x_78_, v_a_71_);
lean_dec(v___x_78_);
v___x_90_ = 0;
v___x_91_ = l_Lean_SourceInfo_fromRef(v_ref_89_, v___x_90_);
lean_dec(v_ref_89_);
v___x_92_ = ((lean_object*)(lp_mathlib_term___u1d9c___closed__1));
v___x_93_ = ((lean_object*)(lp_mathlib_term___u1d9c___closed__2));
lean_inc(v___x_91_);
v___x_94_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_91_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = l_Lean_Syntax_node2(v___x_91_, v___x_92_, v___x_88_, v___x_94_);
v___x_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v_a_72_);
return v___x_96_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___boxed(lean_object* v_x_97_, lean_object* v_a_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1(v_x_97_, v_a_98_, v_a_99_);
lean_dec(v_a_98_);
return v_res_100_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__1(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__0));
v___x_145_ = l_String_toRawSubstring_x27(v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1(lean_object* v_x_157_, lean_object* v_a_158_, lean_object* v_a_159_){
_start:
{
lean_object* v___x_160_; uint8_t v___x_161_; 
v___x_160_ = ((lean_object*)(lp_mathlib_term___u2294___00__closed__1));
lean_inc(v_x_157_);
v___x_161_ = l_Lean_Syntax_isOfKind(v_x_157_, v___x_160_);
if (v___x_161_ == 0)
{
lean_object* v___x_162_; lean_object* v___x_163_; 
lean_dec(v_x_157_);
v___x_162_ = lean_box(1);
v___x_163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v_a_159_);
return v___x_163_;
}
else
{
lean_object* v_quotContext_164_; lean_object* v_currMacroScope_165_; lean_object* v_ref_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; uint8_t v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v_quotContext_164_ = lean_ctor_get(v_a_158_, 1);
v_currMacroScope_165_ = lean_ctor_get(v_a_158_, 2);
v_ref_166_ = lean_ctor_get(v_a_158_, 5);
v___x_167_ = lean_unsigned_to_nat(0u);
v___x_168_ = l_Lean_Syntax_getArg(v_x_157_, v___x_167_);
v___x_169_ = lean_unsigned_to_nat(2u);
v___x_170_ = l_Lean_Syntax_getArg(v_x_157_, v___x_169_);
lean_dec(v_x_157_);
v___x_171_ = 0;
v___x_172_ = l_Lean_SourceInfo_fromRef(v_ref_166_, v___x_171_);
v___x_173_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
v___x_174_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__1);
v___x_175_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4));
lean_inc(v_currMacroScope_165_);
lean_inc(v_quotContext_164_);
v___x_176_ = l_Lean_addMacroScope(v_quotContext_164_, v___x_175_, v_currMacroScope_165_);
v___x_177_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__6));
lean_inc_n(v___x_172_, 2);
v___x_178_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_178_, 0, v___x_172_);
lean_ctor_set(v___x_178_, 1, v___x_174_);
lean_ctor_set(v___x_178_, 2, v___x_176_);
lean_ctor_set(v___x_178_, 3, v___x_177_);
v___x_179_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13));
v___x_180_ = l_Lean_Syntax_node2(v___x_172_, v___x_179_, v___x_168_, v___x_170_);
v___x_181_ = l_Lean_Syntax_node2(v___x_172_, v___x_173_, v___x_178_, v___x_180_);
v___x_182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v_a_159_);
return v___x_182_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___boxed(lean_object* v_x_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1(v_x_183_, v_a_184_, v_a_185_);
lean_dec_ref(v_a_184_);
return v_res_186_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__1(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_188_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__0));
v___x_189_ = l_String_toRawSubstring_x27(v___x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1(lean_object* v_x_201_, lean_object* v_a_202_, lean_object* v_a_203_){
_start:
{
lean_object* v___x_204_; uint8_t v___x_205_; 
v___x_204_ = ((lean_object*)(lp_mathlib_term___u2293___00__closed__1));
lean_inc(v_x_201_);
v___x_205_ = l_Lean_Syntax_isOfKind(v_x_201_, v___x_204_);
if (v___x_205_ == 0)
{
lean_object* v___x_206_; lean_object* v___x_207_; 
lean_dec(v_x_201_);
v___x_206_ = lean_box(1);
v___x_207_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_207_, 0, v___x_206_);
lean_ctor_set(v___x_207_, 1, v_a_203_);
return v___x_207_;
}
else
{
lean_object* v_quotContext_208_; lean_object* v_currMacroScope_209_; lean_object* v_ref_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v_quotContext_208_ = lean_ctor_get(v_a_202_, 1);
v_currMacroScope_209_ = lean_ctor_get(v_a_202_, 2);
v_ref_210_ = lean_ctor_get(v_a_202_, 5);
v___x_211_ = lean_unsigned_to_nat(0u);
v___x_212_ = l_Lean_Syntax_getArg(v_x_201_, v___x_211_);
v___x_213_ = lean_unsigned_to_nat(2u);
v___x_214_ = l_Lean_Syntax_getArg(v_x_201_, v___x_213_);
lean_dec(v_x_201_);
v___x_215_ = 0;
v___x_216_ = l_Lean_SourceInfo_fromRef(v_ref_210_, v___x_215_);
v___x_217_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
v___x_218_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__1);
v___x_219_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4));
lean_inc(v_currMacroScope_209_);
lean_inc(v_quotContext_208_);
v___x_220_ = l_Lean_addMacroScope(v_quotContext_208_, v___x_219_, v_currMacroScope_209_);
v___x_221_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__6));
lean_inc_n(v___x_216_, 2);
v___x_222_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_222_, 0, v___x_216_);
lean_ctor_set(v___x_222_, 1, v___x_218_);
lean_ctor_set(v___x_222_, 2, v___x_220_);
lean_ctor_set(v___x_222_, 3, v___x_221_);
v___x_223_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13));
v___x_224_ = l_Lean_Syntax_node2(v___x_216_, v___x_223_, v___x_212_, v___x_214_);
v___x_225_ = l_Lean_Syntax_node2(v___x_216_, v___x_217_, v___x_222_, v___x_224_);
v___x_226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_225_);
lean_ctor_set(v___x_226_, 1, v_a_203_);
return v___x_226_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___boxed(lean_object* v_x_227_, lean_object* v_a_228_, lean_object* v_a_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1(v_x_227_, v_a_228_, v_a_229_);
lean_dec_ref(v_a_228_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr(lean_object* v_u_234_){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_235_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr___closed__1));
v___x_236_ = lean_box(0);
v___x_237_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_237_, 0, v_u_234_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = l_Lean_Expr_const___override(v___x_235_, v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax(lean_object* v_u_243_){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_244_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax___closed__1));
v___x_245_ = lean_box(0);
v___x_246_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_246_, 0, v_u_243_);
lean_ctor_set(v___x_246_, 1, v___x_245_);
v___x_247_ = l_Lean_Expr_const___override(v___x_244_, v___x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin(lean_object* v_u_252_){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_253_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin___closed__1));
v___x_254_ = lean_box(0);
v___x_255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_255_, 0, v_u_252_);
lean_ctor_set(v___x_255_, 1, v___x_254_);
v___x_256_ = l_Lean_Expr_const___override(v___x_253_, v___x_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg(lean_object* v_decls_257_, lean_object* v_x_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = l_Lean_Meta_withLocalInstancesImp___redArg(v_decls_257_, v_x_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_);
if (lean_obj_tag(v___x_264_) == 0)
{
lean_object* v_a_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_272_; 
v_a_265_ = lean_ctor_get(v___x_264_, 0);
v_isSharedCheck_272_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_272_ == 0)
{
v___x_267_ = v___x_264_;
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_a_265_);
lean_dec(v___x_264_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_272_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_270_; 
if (v_isShared_268_ == 0)
{
v___x_270_ = v___x_267_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_a_265_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
else
{
lean_object* v_a_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_280_; 
v_a_273_ = lean_ctor_get(v___x_264_, 0);
v_isSharedCheck_280_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_280_ == 0)
{
v___x_275_ = v___x_264_;
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_a_273_);
lean_dec(v___x_264_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_280_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_278_; 
if (v_isShared_276_ == 0)
{
v___x_278_ = v___x_275_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v_a_273_);
v___x_278_ = v_reuseFailAlloc_279_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
return v___x_278_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg___boxed(lean_object* v_decls_281_, lean_object* v_x_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg(v_decls_281_, v_x_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v_decls_281_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1(lean_object* v_00_u03b1_289_, lean_object* v_decls_290_, lean_object* v_x_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg(v_decls_290_, v_x_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___boxed(lean_object* v_00_u03b1_298_, lean_object* v_decls_299_, lean_object* v_x_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1(v_00_u03b1_298_, v_decls_299_, v_x_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v_decls_299_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg(lean_object* v_k_307_, uint8_t v_allowLevelAssignments_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_308_, v_k_307_, v___y_309_, v___y_310_, v___y_311_, v___y_312_);
if (lean_obj_tag(v___x_314_) == 0)
{
lean_object* v_a_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_322_; 
v_a_315_ = lean_ctor_get(v___x_314_, 0);
v_isSharedCheck_322_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_322_ == 0)
{
v___x_317_ = v___x_314_;
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_a_315_);
lean_dec(v___x_314_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_320_; 
if (v_isShared_318_ == 0)
{
v___x_320_ = v___x_317_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v_a_315_);
v___x_320_ = v_reuseFailAlloc_321_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
return v___x_320_;
}
}
}
else
{
lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
v_a_323_ = lean_ctor_get(v___x_314_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_314_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_314_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_dec(v___x_314_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg___boxed(lean_object* v_k_331_, lean_object* v_allowLevelAssignments_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_338_; lean_object* v_res_339_; 
v_allowLevelAssignments_boxed_338_ = lean_unbox(v_allowLevelAssignments_332_);
v_res_339_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg(v_k_331_, v_allowLevelAssignments_boxed_338_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2(lean_object* v_00_u03b1_340_, lean_object* v_k_341_, uint8_t v_allowLevelAssignments_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg(v_k_341_, v_allowLevelAssignments_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___boxed(lean_object* v_00_u03b1_349_, lean_object* v_k_350_, lean_object* v_allowLevelAssignments_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_357_; lean_object* v_res_358_; 
v_allowLevelAssignments_boxed_357_ = lean_unbox(v_allowLevelAssignments_351_);
v_res_358_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2(v_00_u03b1_349_, v_k_350_, v_allowLevelAssignments_boxed_357_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
lean_dec(v___y_353_);
lean_dec_ref(v___y_352_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__0(lean_object* v___x_359_, uint8_t v___x_360_, lean_object* v___x_361_, lean_object* v___x_362_, lean_object* v_toCls_363_, uint8_t v___x_364_, lean_object* v_inst_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_Qq_Qq_mkFreshExprMVarQ___redArg(v___x_359_, v___x_360_, v___x_361_, v___y_366_, v___y_367_, v___y_368_, v___y_369_);
if (lean_obj_tag(v___x_371_) == 0)
{
lean_object* v_a_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v_keyedConfig_376_; uint8_t v_trackZetaDelta_377_; lean_object* v_zetaDeltaSet_378_; lean_object* v_lctx_379_; lean_object* v_localInstances_380_; lean_object* v_defEqCtx_x3f_381_; lean_object* v_synthPendingDepth_382_; lean_object* v_customCanUnfoldPredicate_x3f_383_; uint8_t v_univApprox_384_; uint8_t v_inTypeClassResolution_385_; uint8_t v_cacheInferType_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_396_; 
v_a_372_ = lean_ctor_get(v___x_371_, 0);
lean_inc(v_a_372_);
lean_dec_ref_known(v___x_371_, 1);
v___x_373_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_373_, 0, v_a_372_);
lean_ctor_set(v___x_373_, 1, v___x_362_);
v___x_374_ = lean_array_mk(v___x_373_);
v___x_375_ = l_Lean_Expr_betaRev(v_toCls_363_, v___x_374_, v___x_364_, v___x_364_);
lean_dec_ref(v___x_374_);
v_keyedConfig_376_ = lean_ctor_get(v___y_366_, 0);
v_trackZetaDelta_377_ = lean_ctor_get_uint8(v___y_366_, sizeof(void*)*7);
v_zetaDeltaSet_378_ = lean_ctor_get(v___y_366_, 1);
v_lctx_379_ = lean_ctor_get(v___y_366_, 2);
v_localInstances_380_ = lean_ctor_get(v___y_366_, 3);
v_defEqCtx_x3f_381_ = lean_ctor_get(v___y_366_, 4);
v_synthPendingDepth_382_ = lean_ctor_get(v___y_366_, 5);
v_customCanUnfoldPredicate_x3f_383_ = lean_ctor_get(v___y_366_, 6);
v_univApprox_384_ = lean_ctor_get_uint8(v___y_366_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_385_ = lean_ctor_get_uint8(v___y_366_, sizeof(void*)*7 + 2);
v_cacheInferType_386_ = lean_ctor_get_uint8(v___y_366_, sizeof(void*)*7 + 3);
v_isSharedCheck_396_ = !lean_is_exclusive(v___y_366_);
if (v_isSharedCheck_396_ == 0)
{
v___x_388_ = v___y_366_;
v_isShared_389_ = v_isSharedCheck_396_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_383_);
lean_inc(v_synthPendingDepth_382_);
lean_inc(v_defEqCtx_x3f_381_);
lean_inc(v_localInstances_380_);
lean_inc(v_lctx_379_);
lean_inc(v_zetaDeltaSet_378_);
lean_inc(v_keyedConfig_376_);
lean_dec(v___y_366_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_396_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
uint8_t v___x_390_; lean_object* v___x_391_; lean_object* v___x_393_; 
v___x_390_ = 5;
v___x_391_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_390_, v_keyedConfig_376_);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 0, v___x_391_);
v___x_393_ = v___x_388_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v___x_391_);
lean_ctor_set(v_reuseFailAlloc_395_, 1, v_zetaDeltaSet_378_);
lean_ctor_set(v_reuseFailAlloc_395_, 2, v_lctx_379_);
lean_ctor_set(v_reuseFailAlloc_395_, 3, v_localInstances_380_);
lean_ctor_set(v_reuseFailAlloc_395_, 4, v_defEqCtx_x3f_381_);
lean_ctor_set(v_reuseFailAlloc_395_, 5, v_synthPendingDepth_382_);
lean_ctor_set(v_reuseFailAlloc_395_, 6, v_customCanUnfoldPredicate_x3f_383_);
lean_ctor_set_uint8(v_reuseFailAlloc_395_, sizeof(void*)*7, v_trackZetaDelta_377_);
lean_ctor_set_uint8(v_reuseFailAlloc_395_, sizeof(void*)*7 + 1, v_univApprox_384_);
lean_ctor_set_uint8(v_reuseFailAlloc_395_, sizeof(void*)*7 + 2, v_inTypeClassResolution_385_);
lean_ctor_set_uint8(v_reuseFailAlloc_395_, sizeof(void*)*7 + 3, v_cacheInferType_386_);
v___x_393_ = v_reuseFailAlloc_395_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
lean_object* v___x_394_; 
v___x_394_ = l_Lean_Meta_isExprDefEq(v_inst_365_, v___x_375_, v___x_393_, v___y_367_, v___y_368_, v___y_369_);
lean_dec_ref(v___x_393_);
return v___x_394_;
}
}
}
else
{
lean_object* v_a_397_; lean_object* v___x_399_; uint8_t v_isShared_400_; uint8_t v_isSharedCheck_404_; 
lean_dec_ref(v___y_366_);
lean_dec_ref(v_inst_365_);
lean_dec_ref(v_toCls_363_);
lean_dec(v___x_362_);
v_a_397_ = lean_ctor_get(v___x_371_, 0);
v_isSharedCheck_404_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_404_ == 0)
{
v___x_399_ = v___x_371_;
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
else
{
lean_inc(v_a_397_);
lean_dec(v___x_371_);
v___x_399_ = lean_box(0);
v_isShared_400_ = v_isSharedCheck_404_;
goto v_resetjp_398_;
}
v_resetjp_398_:
{
lean_object* v___x_402_; 
if (v_isShared_400_ == 0)
{
v___x_402_ = v___x_399_;
goto v_reusejp_401_;
}
else
{
lean_object* v_reuseFailAlloc_403_; 
v_reuseFailAlloc_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_403_, 0, v_a_397_);
v___x_402_ = v_reuseFailAlloc_403_;
goto v_reusejp_401_;
}
v_reusejp_401_:
{
return v___x_402_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__0___boxed(lean_object* v___x_405_, lean_object* v___x_406_, lean_object* v___x_407_, lean_object* v___x_408_, lean_object* v_toCls_409_, lean_object* v___x_410_, lean_object* v_inst_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_){
_start:
{
uint8_t v___x_1960__boxed_417_; uint8_t v___x_1963__boxed_418_; lean_object* v_res_419_; 
v___x_1960__boxed_417_ = lean_unbox(v___x_406_);
v___x_1963__boxed_418_ = lean_unbox(v___x_410_);
v_res_419_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__0(v___x_405_, v___x_1960__boxed_417_, v___x_407_, v___x_408_, v_toCls_409_, v___x_1963__boxed_418_, v_inst_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_);
lean_dec(v___y_415_);
lean_dec_ref(v___y_414_);
lean_dec(v___y_413_);
return v_res_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__0(lean_object* v_a_420_, lean_object* v_a_421_){
_start:
{
if (lean_obj_tag(v_a_420_) == 0)
{
lean_object* v___x_422_; 
v___x_422_ = lean_array_to_list(v_a_421_);
return v___x_422_;
}
else
{
lean_object* v_head_423_; 
v_head_423_ = lean_ctor_get(v_a_420_, 0);
if (lean_obj_tag(v_head_423_) == 0)
{
lean_object* v_tail_424_; 
v_tail_424_ = lean_ctor_get(v_a_420_, 1);
lean_inc(v_tail_424_);
lean_dec_ref_known(v_a_420_, 2);
v_a_420_ = v_tail_424_;
goto _start;
}
else
{
lean_object* v_tail_426_; lean_object* v_val_427_; lean_object* v___x_428_; 
lean_inc_ref(v_head_423_);
v_tail_426_ = lean_ctor_get(v_a_420_, 1);
lean_inc(v_tail_426_);
lean_dec_ref_known(v_a_420_, 2);
v_val_427_ = lean_ctor_get(v_head_423_, 0);
lean_inc(v_val_427_);
lean_dec_ref_known(v_head_423_, 1);
v___x_428_ = lean_array_push(v_a_421_, v_val_427_);
v_a_420_ = v_tail_426_;
v_a_421_ = v___x_428_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1(lean_object* v_u_432_, lean_object* v_00_u03b1_433_, lean_object* v_toCls_434_, lean_object* v_inst_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_){
_start:
{
lean_object* v_lctx_441_; lean_object* v_decls_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; uint8_t v___x_450_; lean_object* v___x_451_; uint8_t v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___f_456_; lean_object* v___x_457_; 
v_lctx_441_ = lean_ctor_get(v___y_436_, 2);
v_decls_442_ = lean_ctor_get(v_lctx_441_, 1);
v___x_443_ = l_Lean_PersistentArray_toList___redArg(v_decls_442_);
v___x_444_ = ((lean_object*)(lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___closed__0));
v___x_445_ = lp_mathlib_List_filterMapTR_go___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__0(v___x_443_, v___x_444_);
v___x_446_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderExpr(v_u_432_);
v___x_447_ = lean_box(0);
v___x_448_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_448_, 0, v_00_u03b1_433_);
lean_ctor_set(v___x_448_, 1, v___x_447_);
lean_inc_ref(v___x_448_);
v___x_449_ = lean_array_mk(v___x_448_);
v___x_450_ = 0;
v___x_451_ = l_Lean_Expr_betaRev(v___x_446_, v___x_449_, v___x_450_, v___x_450_);
lean_dec_ref(v___x_449_);
v___x_452_ = 1;
v___x_453_ = lean_box(0);
v___x_454_ = lean_box(v___x_452_);
v___x_455_ = lean_box(v___x_450_);
v___f_456_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__0___boxed), 12, 7);
lean_closure_set(v___f_456_, 0, v___x_451_);
lean_closure_set(v___f_456_, 1, v___x_454_);
lean_closure_set(v___f_456_, 2, v___x_453_);
lean_closure_set(v___f_456_, 3, v___x_448_);
lean_closure_set(v___f_456_, 4, v_toCls_434_);
lean_closure_set(v___f_456_, 5, v___x_455_);
lean_closure_set(v___f_456_, 6, v_inst_435_);
v___x_457_ = lp_mathlib_Lean_Meta_withLocalInstances___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__1___redArg(v___x_445_, v___f_456_, v___y_436_, v___y_437_, v___y_438_, v___y_439_);
lean_dec(v___x_445_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___boxed(lean_object* v_u_458_, lean_object* v_00_u03b1_459_, lean_object* v_toCls_460_, lean_object* v_inst_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1(v_u_458_, v_00_u03b1_459_, v_toCls_460_, v_inst_461_, v___y_462_, v___y_463_, v___y_464_, v___y_465_);
lean_dec(v___y_465_);
lean_dec_ref(v___y_464_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg(lean_object* v_u_468_, lean_object* v_00_u03b1_469_, lean_object* v_toCls_470_, lean_object* v_inst_471_, lean_object* v_a_472_, lean_object* v_a_473_, lean_object* v_a_474_, lean_object* v_a_475_){
_start:
{
lean_object* v___f_477_; uint8_t v___x_478_; lean_object* v___x_479_; 
v___f_477_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_477_, 0, v_u_468_);
lean_closure_set(v___f_477_, 1, v_00_u03b1_469_);
lean_closure_set(v___f_477_, 2, v_toCls_470_);
lean_closure_set(v___f_477_, 3, v_inst_471_);
v___x_478_ = 0;
v___x_479_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder_spec__2___redArg(v___f_477_, v___x_478_, v_a_472_, v_a_473_, v_a_474_, v_a_475_);
if (lean_obj_tag(v___x_479_) == 0)
{
return v___x_479_;
}
else
{
lean_object* v_a_480_; uint8_t v___y_482_; uint8_t v___x_492_; 
v_a_480_ = lean_ctor_get(v___x_479_, 0);
lean_inc(v_a_480_);
v___x_492_ = l_Lean_Exception_isInterrupt(v_a_480_);
if (v___x_492_ == 0)
{
uint8_t v___x_493_; 
v___x_493_ = l_Lean_Exception_isRuntime(v_a_480_);
v___y_482_ = v___x_493_;
goto v___jp_481_;
}
else
{
lean_dec(v_a_480_);
v___y_482_ = v___x_492_;
goto v___jp_481_;
}
v___jp_481_:
{
if (v___y_482_ == 0)
{
lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_490_; 
v_isSharedCheck_490_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_490_ == 0)
{
lean_object* v_unused_491_; 
v_unused_491_ = lean_ctor_get(v___x_479_, 0);
lean_dec(v_unused_491_);
v___x_484_ = v___x_479_;
v_isShared_485_ = v_isSharedCheck_490_;
goto v_resetjp_483_;
}
else
{
lean_dec(v___x_479_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_490_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; lean_object* v___x_488_; 
v___x_486_ = lean_box(v___x_478_);
if (v_isShared_485_ == 0)
{
lean_ctor_set_tag(v___x_484_, 0);
lean_ctor_set(v___x_484_, 0, v___x_486_);
v___x_488_ = v___x_484_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v___x_486_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
}
else
{
return v___x_479_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg___boxed(lean_object* v_u_494_, lean_object* v_00_u03b1_495_, lean_object* v_toCls_496_, lean_object* v_inst_497_, lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg(v_u_494_, v_00_u03b1_495_, v_toCls_496_, v_inst_497_, v_a_498_, v_a_499_, v_a_500_, v_a_501_);
lean_dec(v_a_501_);
lean_dec_ref(v_a_500_);
lean_dec(v_a_499_);
lean_dec_ref(v_a_498_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder(lean_object* v_u_504_, lean_object* v_00_u03b1_505_, lean_object* v_cls_506_, lean_object* v_toCls_507_, lean_object* v_inst_508_, lean_object* v_a_509_, lean_object* v_a_510_, lean_object* v_a_511_, lean_object* v_a_512_){
_start:
{
lean_object* v___x_514_; 
v___x_514_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg(v_u_504_, v_00_u03b1_505_, v_toCls_507_, v_inst_508_, v_a_509_, v_a_510_, v_a_511_, v_a_512_);
return v___x_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___boxed(lean_object* v_u_515_, lean_object* v_00_u03b1_516_, lean_object* v_cls_517_, lean_object* v_toCls_518_, lean_object* v_inst_519_, lean_object* v_a_520_, lean_object* v_a_521_, lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v_a_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder(v_u_515_, v_00_u03b1_516_, v_cls_517_, v_toCls_518_, v_inst_519_, v_a_520_, v_a_521_, v_a_522_, v_a_523_);
lean_dec(v_a_523_);
lean_dec_ref(v_a_522_);
lean_dec(v_a_521_);
lean_dec_ref(v_a_520_);
lean_dec_ref(v_cls_517_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(lean_object* v___y_526_){
_start:
{
lean_object* v_subExpr_528_; lean_object* v_expr_529_; lean_object* v___x_530_; 
v_subExpr_528_ = lean_ctor_get(v___y_526_, 3);
v_expr_529_ = lean_ctor_get(v_subExpr_528_, 0);
lean_inc_ref(v_expr_529_);
v___x_530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_530_, 0, v_expr_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg___boxed(lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(v___y_531_);
lean_dec_ref(v___y_531_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0(lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_){
_start:
{
lean_object* v___x_541_; 
v___x_541_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(v___y_534_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___boxed(lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_){
_start:
{
lean_object* v_res_549_; 
v_res_549_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0(v___y_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
lean_dec(v___y_545_);
lean_dec_ref(v___y_544_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
return v_res_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg(lean_object* v___y_550_){
_start:
{
lean_object* v_subExpr_552_; lean_object* v_pos_553_; lean_object* v___x_554_; 
v_subExpr_552_ = lean_ctor_get(v___y_550_, 3);
v_pos_553_ = lean_ctor_get(v_subExpr_552_, 1);
lean_inc(v_pos_553_);
v___x_554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_554_, 0, v_pos_553_);
return v___x_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg___boxed(lean_object* v___y_555_, lean_object* v___y_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg(v___y_555_);
lean_dec_ref(v___y_555_);
return v_res_557_;
}
}
static lean_object* _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_558_; lean_object* v_dummy_559_; 
v___x_558_ = lean_box(0);
v_dummy_559_ = l_Lean_Expr_sort___override(v___x_558_);
return v_dummy_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(lean_object* v_argIdx_560_, lean_object* v_x_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_){
_start:
{
lean_object* v___x_569_; lean_object* v_a_570_; lean_object* v___x_571_; lean_object* v_a_572_; lean_object* v_optionsPerPos_573_; lean_object* v_currNamespace_574_; lean_object* v_openDecls_575_; uint8_t v_inPattern_576_; lean_object* v_depth_577_; lean_object* v_lctxInitIndices_578_; lean_object* v_nargs_579_; lean_object* v___x_580_; lean_object* v_dummy_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v_args_585_; lean_object* v___x_586_; lean_object* v_newPos_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_569_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(v___y_562_);
v_a_570_ = lean_ctor_get(v___x_569_, 0);
lean_inc(v_a_570_);
lean_dec_ref(v___x_569_);
v___x_571_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg(v___y_562_);
v_a_572_ = lean_ctor_get(v___x_571_, 0);
lean_inc(v_a_572_);
lean_dec_ref(v___x_571_);
v_optionsPerPos_573_ = lean_ctor_get(v___y_562_, 0);
v_currNamespace_574_ = lean_ctor_get(v___y_562_, 1);
v_openDecls_575_ = lean_ctor_get(v___y_562_, 2);
v_inPattern_576_ = lean_ctor_get_uint8(v___y_562_, sizeof(void*)*6);
v_depth_577_ = lean_ctor_get(v___y_562_, 4);
v_lctxInitIndices_578_ = lean_ctor_get(v___y_562_, 5);
v_nargs_579_ = l_Lean_Expr_getAppNumArgs(v_a_570_);
v___x_580_ = l_Lean_instInhabitedExpr;
v_dummy_581_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___closed__0);
lean_inc(v_nargs_579_);
v___x_582_ = lean_mk_array(v_nargs_579_, v_dummy_581_);
v___x_583_ = lean_unsigned_to_nat(1u);
v___x_584_ = lean_nat_sub(v_nargs_579_, v___x_583_);
lean_dec(v_nargs_579_);
v_args_585_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_570_, v___x_582_, v___x_584_);
v___x_586_ = lean_array_get_size(v_args_585_);
v_newPos_587_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_586_, v_argIdx_560_, v_a_572_);
lean_dec(v_a_572_);
v___x_588_ = lean_array_get(v___x_580_, v_args_585_, v_argIdx_560_);
lean_dec_ref(v_args_585_);
v___x_589_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_588_);
lean_ctor_set(v___x_589_, 1, v_newPos_587_);
lean_inc(v_lctxInitIndices_578_);
lean_inc(v_depth_577_);
lean_inc(v_openDecls_575_);
lean_inc(v_currNamespace_574_);
lean_inc(v_optionsPerPos_573_);
v___x_590_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_590_, 0, v_optionsPerPos_573_);
lean_ctor_set(v___x_590_, 1, v_currNamespace_574_);
lean_ctor_set(v___x_590_, 2, v_openDecls_575_);
lean_ctor_set(v___x_590_, 3, v___x_589_);
lean_ctor_set(v___x_590_, 4, v_depth_577_);
lean_ctor_set(v___x_590_, 5, v_lctxInitIndices_578_);
lean_ctor_set_uint8(v___x_590_, sizeof(void*)*6, v_inPattern_576_);
lean_inc(v___y_567_);
lean_inc_ref(v___y_566_);
lean_inc(v___y_565_);
lean_inc_ref(v___y_564_);
lean_inc(v___y_563_);
v___x_591_ = lean_apply_7(v_x_561_, v___x_590_, v___y_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_, lean_box(0));
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg___boxed(lean_object* v_argIdx_592_, lean_object* v_x_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_){
_start:
{
lean_object* v_res_601_; 
v_res_601_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(v_argIdx_592_, v_x_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_);
lean_dec(v___y_599_);
lean_dec_ref(v___y_598_);
lean_dec(v___y_597_);
lean_dec_ref(v___y_596_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
lean_dec(v_argIdx_592_);
return v_res_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup___lam__0(lean_object* v___x_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_){
_start:
{
lean_object* v___y_613_; lean_object* v___y_614_; lean_object* v___y_615_; lean_object* v___y_616_; lean_object* v___y_617_; lean_object* v___y_618_; lean_object* v___x_634_; lean_object* v_a_635_; lean_object* v___x_636_; uint8_t v___x_637_; 
v___x_634_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(v___y_605_);
v_a_635_ = lean_ctor_get(v___x_634_, 0);
lean_inc(v_a_635_);
lean_dec_ref(v___x_634_);
v___x_636_ = l_Lean_Expr_cleanupAnnotations(v_a_635_);
v___x_637_ = l_Lean_Expr_isApp(v___x_636_);
if (v___x_637_ == 0)
{
lean_object* v___x_638_; 
lean_dec_ref(v___x_636_);
v___x_638_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_638_;
}
else
{
lean_object* v___x_639_; uint8_t v___x_640_; 
v___x_639_ = l_Lean_Expr_appFnCleanup___redArg(v___x_636_);
v___x_640_ = l_Lean_Expr_isApp(v___x_639_);
if (v___x_640_ == 0)
{
lean_object* v___x_641_; 
lean_dec_ref(v___x_639_);
v___x_641_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_641_;
}
else
{
lean_object* v___x_642_; uint8_t v___x_643_; 
v___x_642_ = l_Lean_Expr_appFnCleanup___redArg(v___x_639_);
v___x_643_ = l_Lean_Expr_isApp(v___x_642_);
if (v___x_643_ == 0)
{
lean_object* v___x_644_; 
lean_dec_ref(v___x_642_);
v___x_644_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_644_;
}
else
{
lean_object* v_arg_645_; lean_object* v___x_646_; uint8_t v___x_647_; 
v_arg_645_ = lean_ctor_get(v___x_642_, 1);
lean_inc_ref(v_arg_645_);
v___x_646_ = l_Lean_Expr_appFnCleanup___redArg(v___x_642_);
v___x_647_ = l_Lean_Expr_isApp(v___x_646_);
if (v___x_647_ == 0)
{
lean_object* v___x_648_; 
lean_dec_ref(v___x_646_);
lean_dec_ref(v_arg_645_);
v___x_648_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_648_;
}
else
{
lean_object* v_arg_649_; lean_object* v___x_650_; lean_object* v___x_651_; uint8_t v___x_652_; 
v_arg_649_ = lean_ctor_get(v___x_646_, 1);
lean_inc_ref(v_arg_649_);
v___x_650_ = l_Lean_Expr_appFnCleanup___redArg(v___x_646_);
v___x_651_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2294____1___closed__4));
v___x_652_ = l_Lean_Expr_isConstOf(v___x_650_, v___x_651_);
if (v___x_652_ == 0)
{
lean_object* v___x_653_; 
lean_dec_ref(v___x_650_);
lean_dec_ref(v_arg_649_);
lean_dec_ref(v_arg_645_);
v___x_653_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_653_;
}
else
{
lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v_u_656_; lean_object* v_a_657_; lean_object* v___x_658_; 
v___x_654_ = l_Lean_Expr_constLevels_x21(v___x_650_);
lean_dec_ref(v___x_650_);
v___x_655_ = lean_unsigned_to_nat(0u);
v_u_656_ = l_List_get_x21Internal___redArg(v___x_604_, v___x_654_, v___x_655_);
lean_dec(v___x_654_);
lean_inc(v_u_656_);
v_a_657_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMax(v_u_656_);
v___x_658_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg(v_u_656_, v_arg_649_, v_a_657_, v_arg_645_, v___y_607_, v___y_608_, v___y_609_, v___y_610_);
if (lean_obj_tag(v___x_658_) == 0)
{
lean_object* v_a_659_; uint8_t v___x_660_; 
v_a_659_ = lean_ctor_get(v___x_658_, 0);
lean_inc(v_a_659_);
lean_dec_ref_known(v___x_658_, 1);
v___x_660_ = lean_unbox(v_a_659_);
lean_dec(v_a_659_);
if (v___x_660_ == 0)
{
v___y_613_ = v___y_605_;
v___y_614_ = v___y_606_;
v___y_615_ = v___y_607_;
v___y_616_ = v___y_608_;
v___y_617_ = v___y_609_;
v___y_618_ = v___y_610_;
goto v___jp_612_;
}
else
{
lean_object* v___x_661_; 
v___x_661_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_661_) == 0)
{
lean_dec_ref_known(v___x_661_, 1);
v___y_613_ = v___y_605_;
v___y_614_ = v___y_606_;
v___y_615_ = v___y_607_;
v___y_616_ = v___y_608_;
v___y_617_ = v___y_609_;
v___y_618_ = v___y_610_;
goto v___jp_612_;
}
else
{
lean_object* v_a_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_669_; 
v_a_662_ = lean_ctor_get(v___x_661_, 0);
v_isSharedCheck_669_ = !lean_is_exclusive(v___x_661_);
if (v_isSharedCheck_669_ == 0)
{
v___x_664_ = v___x_661_;
v_isShared_665_ = v_isSharedCheck_669_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_a_662_);
lean_dec(v___x_661_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_669_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
lean_object* v___x_667_; 
if (v_isShared_665_ == 0)
{
v___x_667_ = v___x_664_;
goto v_reusejp_666_;
}
else
{
lean_object* v_reuseFailAlloc_668_; 
v_reuseFailAlloc_668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_668_, 0, v_a_662_);
v___x_667_ = v_reuseFailAlloc_668_;
goto v_reusejp_666_;
}
v_reusejp_666_:
{
return v___x_667_;
}
}
}
}
}
else
{
lean_object* v_a_670_; lean_object* v___x_672_; uint8_t v_isShared_673_; uint8_t v_isSharedCheck_677_; 
v_a_670_ = lean_ctor_get(v___x_658_, 0);
v_isSharedCheck_677_ = !lean_is_exclusive(v___x_658_);
if (v_isSharedCheck_677_ == 0)
{
v___x_672_ = v___x_658_;
v_isShared_673_ = v_isSharedCheck_677_;
goto v_resetjp_671_;
}
else
{
lean_inc(v_a_670_);
lean_dec(v___x_658_);
v___x_672_ = lean_box(0);
v_isShared_673_ = v_isSharedCheck_677_;
goto v_resetjp_671_;
}
v_resetjp_671_:
{
lean_object* v___x_675_; 
if (v_isShared_673_ == 0)
{
v___x_675_ = v___x_672_;
goto v_reusejp_674_;
}
else
{
lean_object* v_reuseFailAlloc_676_; 
v_reuseFailAlloc_676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_676_, 0, v_a_670_);
v___x_675_ = v_reuseFailAlloc_676_;
goto v_reusejp_674_;
}
v_reusejp_674_:
{
return v___x_675_;
}
}
}
}
}
}
}
}
v___jp_612_:
{
lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v___x_619_ = lean_unsigned_to_nat(2u);
v___x_620_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__0));
v___x_621_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(v___x_619_, v___x_620_, v___y_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
if (lean_obj_tag(v___x_621_) == 0)
{
lean_object* v_a_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
v_a_622_ = lean_ctor_get(v___x_621_, 0);
lean_inc(v_a_622_);
lean_dec_ref_known(v___x_621_, 1);
v___x_623_ = lean_unsigned_to_nat(3u);
v___x_624_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(v___x_623_, v___x_620_, v___y_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
if (lean_obj_tag(v___x_624_) == 0)
{
lean_object* v_a_625_; lean_object* v_ref_626_; uint8_t v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v_a_625_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_a_625_);
lean_dec_ref_known(v___x_624_, 1);
v_ref_626_ = lean_ctor_get(v___y_617_, 5);
v___x_627_ = 0;
v___x_628_ = l_Lean_SourceInfo_fromRef(v_ref_626_, v___x_627_);
v___x_629_ = ((lean_object*)(lp_mathlib_term___u2294___00__closed__1));
v___x_630_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__1));
lean_inc(v___x_628_);
v___x_631_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_631_, 0, v___x_628_);
lean_ctor_set(v___x_631_, 1, v___x_630_);
v___x_632_ = l_Lean_Syntax_node3(v___x_628_, v___x_629_, v_a_622_, v___x_631_, v_a_625_);
v___x_633_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_annotateGoToSyntaxDef(v___x_632_, v___y_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
return v___x_633_;
}
else
{
lean_dec(v_a_622_);
return v___x_624_;
}
}
else
{
return v___x_621_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup___lam__0___boxed(lean_object* v___x_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
lean_object* v_res_686_; 
v_res_686_ = lp_mathlib_Mathlib_Meta_delabSup___lam__0(v___x_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___x_678_);
return v_res_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup(lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_){
_start:
{
lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; 
v___x_704_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabSup___closed__1));
v___x_705_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabSup___closed__4));
v___x_706_ = l_Lean_PrettyPrinter_Delaborator_whenNotPPOption(v___x_704_, v___x_705_, v_a_697_, v_a_698_, v_a_699_, v_a_700_, v_a_701_, v_a_702_);
return v___x_706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabSup___boxed(lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_mathlib_Mathlib_Meta_delabSup(v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
lean_dec(v_a_712_);
lean_dec_ref(v_a_711_);
lean_dec(v_a_710_);
lean_dec_ref(v_a_709_);
lean_dec(v_a_708_);
lean_dec_ref(v_a_707_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1(lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___redArg(v___y_715_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1___boxed(lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1_spec__1(v___y_723_, v___y_724_, v___y_725_, v___y_726_, v___y_727_, v___y_728_);
lean_dec(v___y_728_);
lean_dec_ref(v___y_727_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec(v___y_724_);
lean_dec_ref(v___y_723_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1(lean_object* v_00_u03b1_731_, lean_object* v_argIdx_732_, lean_object* v_x_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_){
_start:
{
lean_object* v___x_741_; 
v___x_741_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(v_argIdx_732_, v_x_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_);
return v___x_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___boxed(lean_object* v_00_u03b1_742_, lean_object* v_argIdx_743_, lean_object* v_x_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_){
_start:
{
lean_object* v_res_752_; 
v_res_752_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1(v_00_u03b1_742_, v_argIdx_743_, v_x_744_, v___y_745_, v___y_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_748_);
lean_dec_ref(v___y_747_);
lean_dec(v___y_746_);
lean_dec_ref(v___y_745_);
lean_dec(v_argIdx_743_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf___lam__0(lean_object* v___x_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_){
_start:
{
lean_object* v___y_763_; lean_object* v___y_764_; lean_object* v___y_765_; lean_object* v___y_766_; lean_object* v___y_767_; lean_object* v___y_768_; lean_object* v___x_784_; lean_object* v_a_785_; lean_object* v___x_786_; uint8_t v___x_787_; 
v___x_784_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabSup_spec__0___redArg(v___y_755_);
v_a_785_ = lean_ctor_get(v___x_784_, 0);
lean_inc(v_a_785_);
lean_dec_ref(v___x_784_);
v___x_786_ = l_Lean_Expr_cleanupAnnotations(v_a_785_);
v___x_787_ = l_Lean_Expr_isApp(v___x_786_);
if (v___x_787_ == 0)
{
lean_object* v___x_788_; 
lean_dec_ref(v___x_786_);
v___x_788_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_788_;
}
else
{
lean_object* v___x_789_; uint8_t v___x_790_; 
v___x_789_ = l_Lean_Expr_appFnCleanup___redArg(v___x_786_);
v___x_790_ = l_Lean_Expr_isApp(v___x_789_);
if (v___x_790_ == 0)
{
lean_object* v___x_791_; 
lean_dec_ref(v___x_789_);
v___x_791_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_791_;
}
else
{
lean_object* v___x_792_; uint8_t v___x_793_; 
v___x_792_ = l_Lean_Expr_appFnCleanup___redArg(v___x_789_);
v___x_793_ = l_Lean_Expr_isApp(v___x_792_);
if (v___x_793_ == 0)
{
lean_object* v___x_794_; 
lean_dec_ref(v___x_792_);
v___x_794_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_794_;
}
else
{
lean_object* v_arg_795_; lean_object* v___x_796_; uint8_t v___x_797_; 
v_arg_795_ = lean_ctor_get(v___x_792_, 1);
lean_inc_ref(v_arg_795_);
v___x_796_ = l_Lean_Expr_appFnCleanup___redArg(v___x_792_);
v___x_797_ = l_Lean_Expr_isApp(v___x_796_);
if (v___x_797_ == 0)
{
lean_object* v___x_798_; 
lean_dec_ref(v___x_796_);
lean_dec_ref(v_arg_795_);
v___x_798_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_798_;
}
else
{
lean_object* v_arg_799_; lean_object* v___x_800_; lean_object* v___x_801_; uint8_t v___x_802_; 
v_arg_799_ = lean_ctor_get(v___x_796_, 1);
lean_inc_ref(v_arg_799_);
v___x_800_ = l_Lean_Expr_appFnCleanup___redArg(v___x_796_);
v___x_801_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u2293____1___closed__4));
v___x_802_ = l_Lean_Expr_isConstOf(v___x_800_, v___x_801_);
if (v___x_802_ == 0)
{
lean_object* v___x_803_; 
lean_dec_ref(v___x_800_);
lean_dec_ref(v_arg_799_);
lean_dec_ref(v_arg_795_);
v___x_803_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_803_;
}
else
{
lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v_u_806_; lean_object* v_a_807_; lean_object* v___x_808_; 
v___x_804_ = l_Lean_Expr_constLevels_x21(v___x_800_);
lean_dec_ref(v___x_800_);
v___x_805_ = lean_unsigned_to_nat(0u);
v_u_806_ = l_List_get_x21Internal___redArg(v___x_754_, v___x_804_, v___x_805_);
lean_dec(v___x_804_);
lean_inc(v_u_806_);
v_a_807_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_linearOrderToMin(v_u_806_);
v___x_808_ = lp_mathlib___private_Mathlib_Order_Notation_0__Mathlib_Meta_hasLinearOrder___redArg(v_u_806_, v_arg_799_, v_a_807_, v_arg_795_, v___y_757_, v___y_758_, v___y_759_, v___y_760_);
if (lean_obj_tag(v___x_808_) == 0)
{
lean_object* v_a_809_; uint8_t v___x_810_; 
v_a_809_ = lean_ctor_get(v___x_808_, 0);
lean_inc(v_a_809_);
lean_dec_ref_known(v___x_808_, 1);
v___x_810_ = lean_unbox(v_a_809_);
lean_dec(v_a_809_);
if (v___x_810_ == 0)
{
v___y_763_ = v___y_755_;
v___y_764_ = v___y_756_;
v___y_765_ = v___y_757_;
v___y_766_ = v___y_758_;
v___y_767_ = v___y_759_;
v___y_768_ = v___y_760_;
goto v___jp_762_;
}
else
{
lean_object* v___x_811_; 
v___x_811_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_811_) == 0)
{
lean_dec_ref_known(v___x_811_, 1);
v___y_763_ = v___y_755_;
v___y_764_ = v___y_756_;
v___y_765_ = v___y_757_;
v___y_766_ = v___y_758_;
v___y_767_ = v___y_759_;
v___y_768_ = v___y_760_;
goto v___jp_762_;
}
else
{
lean_object* v_a_812_; lean_object* v___x_814_; uint8_t v_isShared_815_; uint8_t v_isSharedCheck_819_; 
v_a_812_ = lean_ctor_get(v___x_811_, 0);
v_isSharedCheck_819_ = !lean_is_exclusive(v___x_811_);
if (v_isSharedCheck_819_ == 0)
{
v___x_814_ = v___x_811_;
v_isShared_815_ = v_isSharedCheck_819_;
goto v_resetjp_813_;
}
else
{
lean_inc(v_a_812_);
lean_dec(v___x_811_);
v___x_814_ = lean_box(0);
v_isShared_815_ = v_isSharedCheck_819_;
goto v_resetjp_813_;
}
v_resetjp_813_:
{
lean_object* v___x_817_; 
if (v_isShared_815_ == 0)
{
v___x_817_ = v___x_814_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v_a_812_);
v___x_817_ = v_reuseFailAlloc_818_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
return v___x_817_;
}
}
}
}
}
else
{
lean_object* v_a_820_; lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_827_; 
v_a_820_ = lean_ctor_get(v___x_808_, 0);
v_isSharedCheck_827_ = !lean_is_exclusive(v___x_808_);
if (v_isSharedCheck_827_ == 0)
{
v___x_822_ = v___x_808_;
v_isShared_823_ = v_isSharedCheck_827_;
goto v_resetjp_821_;
}
else
{
lean_inc(v_a_820_);
lean_dec(v___x_808_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_827_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v___x_825_; 
if (v_isShared_823_ == 0)
{
v___x_825_ = v___x_822_;
goto v_reusejp_824_;
}
else
{
lean_object* v_reuseFailAlloc_826_; 
v_reuseFailAlloc_826_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_826_, 0, v_a_820_);
v___x_825_ = v_reuseFailAlloc_826_;
goto v_reusejp_824_;
}
v_reusejp_824_:
{
return v___x_825_;
}
}
}
}
}
}
}
}
v___jp_762_:
{
lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; 
v___x_769_ = lean_unsigned_to_nat(2u);
v___x_770_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabSup___lam__0___closed__0));
v___x_771_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(v___x_769_, v___x_770_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_771_) == 0)
{
lean_object* v_a_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v_a_772_ = lean_ctor_get(v___x_771_, 0);
lean_inc(v_a_772_);
lean_dec_ref_known(v___x_771_, 1);
v___x_773_ = lean_unsigned_to_nat(3u);
v___x_774_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabSup_spec__1___redArg(v___x_773_, v___x_770_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_774_) == 0)
{
lean_object* v_a_775_; lean_object* v_ref_776_; uint8_t v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v_a_775_ = lean_ctor_get(v___x_774_, 0);
lean_inc(v_a_775_);
lean_dec_ref_known(v___x_774_, 1);
v_ref_776_ = lean_ctor_get(v___y_767_, 5);
v___x_777_ = 0;
v___x_778_ = l_Lean_SourceInfo_fromRef(v_ref_776_, v___x_777_);
v___x_779_ = ((lean_object*)(lp_mathlib_term___u2293___00__closed__1));
v___x_780_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabInf___lam__0___closed__0));
lean_inc(v___x_778_);
v___x_781_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_781_, 0, v___x_778_);
lean_ctor_set(v___x_781_, 1, v___x_780_);
v___x_782_ = l_Lean_Syntax_node3(v___x_778_, v___x_779_, v_a_772_, v___x_781_, v_a_775_);
v___x_783_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_annotateGoToSyntaxDef(v___x_782_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
return v___x_783_;
}
else
{
lean_dec(v_a_772_);
return v___x_774_;
}
}
else
{
return v___x_771_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf___lam__0___boxed(lean_object* v___x_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_){
_start:
{
lean_object* v_res_836_; 
v_res_836_ = lp_mathlib_Mathlib_Meta_delabInf___lam__0(v___x_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_);
lean_dec(v___y_834_);
lean_dec_ref(v___y_833_);
lean_dec(v___y_832_);
lean_dec_ref(v___y_831_);
lean_dec(v___y_830_);
lean_dec_ref(v___y_829_);
lean_dec(v___x_828_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf(lean_object* v_a_845_, lean_object* v_a_846_, lean_object* v_a_847_, lean_object* v_a_848_, lean_object* v_a_849_, lean_object* v_a_850_){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; 
v___x_852_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabSup___closed__1));
v___x_853_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabInf___closed__2));
v___x_854_ = l_Lean_PrettyPrinter_Delaborator_whenNotPPOption(v___x_852_, v___x_853_, v_a_845_, v_a_846_, v_a_847_, v_a_848_, v_a_849_, v_a_850_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabInf___boxed(lean_object* v_a_855_, lean_object* v_a_856_, lean_object* v_a_857_, lean_object* v_a_858_, lean_object* v_a_859_, lean_object* v_a_860_, lean_object* v_a_861_){
_start:
{
lean_object* v_res_862_; 
v_res_862_ = lp_mathlib_Mathlib_Meta_delabInf(v_a_855_, v_a_856_, v_a_857_, v_a_858_, v_a_859_, v_a_860_);
lean_dec(v_a_860_);
lean_dec_ref(v_a_859_);
lean_dec(v_a_858_);
lean_dec_ref(v_a_857_);
lean_dec(v_a_856_);
lean_dec_ref(v_a_855_);
return v_res_862_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__1(void){
_start:
{
lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_883_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__0));
v___x_884_ = l_String_toRawSubstring_x27(v___x_883_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1(lean_object* v_x_897_, lean_object* v_a_898_, lean_object* v_a_899_){
_start:
{
lean_object* v___x_900_; uint8_t v___x_901_; 
v___x_900_ = ((lean_object*)(lp_mathlib_term___u21e8___00__closed__1));
lean_inc(v_x_897_);
v___x_901_ = l_Lean_Syntax_isOfKind(v_x_897_, v___x_900_);
if (v___x_901_ == 0)
{
lean_object* v___x_902_; lean_object* v___x_903_; 
lean_dec(v_x_897_);
v___x_902_ = lean_box(1);
v___x_903_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_903_, 0, v___x_902_);
lean_ctor_set(v___x_903_, 1, v_a_899_);
return v___x_903_;
}
else
{
lean_object* v_quotContext_904_; lean_object* v_currMacroScope_905_; lean_object* v_ref_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; uint8_t v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v_quotContext_904_ = lean_ctor_get(v_a_898_, 1);
v_currMacroScope_905_ = lean_ctor_get(v_a_898_, 2);
v_ref_906_ = lean_ctor_get(v_a_898_, 5);
v___x_907_ = lean_unsigned_to_nat(0u);
v___x_908_ = l_Lean_Syntax_getArg(v_x_897_, v___x_907_);
v___x_909_ = lean_unsigned_to_nat(2u);
v___x_910_ = l_Lean_Syntax_getArg(v_x_897_, v___x_909_);
lean_dec(v_x_897_);
v___x_911_ = 0;
v___x_912_ = l_Lean_SourceInfo_fromRef(v_ref_906_, v___x_911_);
v___x_913_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
v___x_914_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__1);
v___x_915_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__2));
lean_inc(v_currMacroScope_905_);
lean_inc(v_quotContext_904_);
v___x_916_ = l_Lean_addMacroScope(v_quotContext_904_, v___x_915_, v_currMacroScope_905_);
v___x_917_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___closed__6));
lean_inc_n(v___x_912_, 2);
v___x_918_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_918_, 0, v___x_912_);
lean_ctor_set(v___x_918_, 1, v___x_914_);
lean_ctor_set(v___x_918_, 2, v___x_916_);
lean_ctor_set(v___x_918_, 3, v___x_917_);
v___x_919_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13));
v___x_920_ = l_Lean_Syntax_node2(v___x_912_, v___x_919_, v___x_908_, v___x_910_);
v___x_921_ = l_Lean_Syntax_node2(v___x_912_, v___x_913_, v___x_918_, v___x_920_);
v___x_922_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_922_, 0, v___x_921_);
lean_ctor_set(v___x_922_, 1, v_a_899_);
return v___x_922_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1___boxed(lean_object* v_x_923_, lean_object* v_a_924_, lean_object* v_a_925_){
_start:
{
lean_object* v_res_926_; 
v_res_926_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u21e8____1(v_x_923_, v_a_924_, v_a_925_);
lean_dec_ref(v_a_924_);
return v_res_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HImp__himp__1(lean_object* v_x_927_, lean_object* v_a_928_, lean_object* v_a_929_){
_start:
{
lean_object* v___x_930_; uint8_t v___x_931_; 
v___x_930_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
lean_inc(v_x_927_);
v___x_931_ = l_Lean_Syntax_isOfKind(v_x_927_, v___x_930_);
if (v___x_931_ == 0)
{
lean_object* v___x_932_; lean_object* v___x_933_; 
lean_dec(v_x_927_);
v___x_932_ = lean_box(0);
v___x_933_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_932_);
lean_ctor_set(v___x_933_, 1, v_a_929_);
return v___x_933_;
}
else
{
lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; uint8_t v___x_937_; 
v___x_934_ = lean_unsigned_to_nat(0u);
v___x_935_ = l_Lean_Syntax_getArg(v_x_927_, v___x_934_);
v___x_936_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1));
lean_inc(v___x_935_);
v___x_937_ = l_Lean_Syntax_isOfKind(v___x_935_, v___x_936_);
if (v___x_937_ == 0)
{
lean_object* v___x_938_; lean_object* v___x_939_; 
lean_dec(v___x_935_);
lean_dec(v_x_927_);
v___x_938_ = lean_box(0);
v___x_939_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_939_, 0, v___x_938_);
lean_ctor_set(v___x_939_, 1, v_a_929_);
return v___x_939_;
}
else
{
lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; uint8_t v___x_943_; 
v___x_940_ = lean_unsigned_to_nat(1u);
v___x_941_ = l_Lean_Syntax_getArg(v_x_927_, v___x_940_);
lean_dec(v_x_927_);
v___x_942_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_941_);
v___x_943_ = l_Lean_Syntax_matchesNull(v___x_941_, v___x_942_);
if (v___x_943_ == 0)
{
lean_object* v___x_944_; lean_object* v___x_945_; 
lean_dec(v___x_941_);
lean_dec(v___x_935_);
v___x_944_ = lean_box(0);
v___x_945_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_945_, 0, v___x_944_);
lean_ctor_set(v___x_945_, 1, v_a_929_);
return v___x_945_;
}
else
{
lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v_ref_948_; uint8_t v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; 
v___x_946_ = l_Lean_Syntax_getArg(v___x_941_, v___x_934_);
v___x_947_ = l_Lean_Syntax_getArg(v___x_941_, v___x_940_);
lean_dec(v___x_941_);
v_ref_948_ = l_Lean_replaceRef(v___x_935_, v_a_928_);
lean_dec(v___x_935_);
v___x_949_ = 0;
v___x_950_ = l_Lean_SourceInfo_fromRef(v_ref_948_, v___x_949_);
lean_dec(v_ref_948_);
v___x_951_ = ((lean_object*)(lp_mathlib_term___u21e8___00__closed__1));
v___x_952_ = ((lean_object*)(lp_mathlib_term___u21e8___00__closed__2));
lean_inc(v___x_950_);
v___x_953_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_953_, 0, v___x_950_);
lean_ctor_set(v___x_953_, 1, v___x_952_);
v___x_954_ = l_Lean_Syntax_node3(v___x_950_, v___x_951_, v___x_946_, v___x_953_, v___x_947_);
v___x_955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_955_, 0, v___x_954_);
lean_ctor_set(v___x_955_, 1, v_a_929_);
return v___x_955_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HImp__himp__1___boxed(lean_object* v_x_956_, lean_object* v_a_957_, lean_object* v_a_958_){
_start:
{
lean_object* v_res_959_; 
v_res_959_ = lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HImp__himp__1(v_x_956_, v_a_957_, v_a_958_);
lean_dec(v_a_957_);
return v_res_959_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__1(void){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__0));
v___x_980_ = l_String_toRawSubstring_x27(v___x_979_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1(lean_object* v_x_993_, lean_object* v_a_994_, lean_object* v_a_995_){
_start:
{
lean_object* v___x_996_; uint8_t v___x_997_; 
v___x_996_ = ((lean_object*)(lp_mathlib_term_uffe2___00__closed__1));
lean_inc(v_x_993_);
v___x_997_ = l_Lean_Syntax_isOfKind(v_x_993_, v___x_996_);
if (v___x_997_ == 0)
{
lean_object* v___x_998_; lean_object* v___x_999_; 
lean_dec(v_x_993_);
v___x_998_ = lean_box(1);
v___x_999_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_999_, 0, v___x_998_);
lean_ctor_set(v___x_999_, 1, v_a_995_);
return v___x_999_;
}
else
{
lean_object* v_quotContext_1000_; lean_object* v_currMacroScope_1001_; lean_object* v_ref_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; uint8_t v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; 
v_quotContext_1000_ = lean_ctor_get(v_a_994_, 1);
v_currMacroScope_1001_ = lean_ctor_get(v_a_994_, 2);
v_ref_1002_ = lean_ctor_get(v_a_994_, 5);
v___x_1003_ = lean_unsigned_to_nat(1u);
v___x_1004_ = l_Lean_Syntax_getArg(v_x_993_, v___x_1003_);
lean_dec(v_x_993_);
v___x_1005_ = 0;
v___x_1006_ = l_Lean_SourceInfo_fromRef(v_ref_1002_, v___x_1005_);
v___x_1007_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
v___x_1008_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__1);
v___x_1009_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__2));
lean_inc(v_currMacroScope_1001_);
lean_inc(v_quotContext_1000_);
v___x_1010_ = l_Lean_addMacroScope(v_quotContext_1000_, v___x_1009_, v_currMacroScope_1001_);
v___x_1011_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___closed__6));
lean_inc_n(v___x_1006_, 2);
v___x_1012_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1012_, 0, v___x_1006_);
lean_ctor_set(v___x_1012_, 1, v___x_1008_);
lean_ctor_set(v___x_1012_, 2, v___x_1010_);
lean_ctor_set(v___x_1012_, 3, v___x_1011_);
v___x_1013_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__13));
v___x_1014_ = l_Lean_Syntax_node1(v___x_1006_, v___x_1013_, v___x_1004_);
v___x_1015_ = l_Lean_Syntax_node2(v___x_1006_, v___x_1007_, v___x_1012_, v___x_1014_);
v___x_1016_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1016_, 0, v___x_1015_);
lean_ctor_set(v___x_1016_, 1, v_a_995_);
return v___x_1016_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1___boxed(lean_object* v_x_1017_, lean_object* v_a_1018_, lean_object* v_a_1019_){
_start:
{
lean_object* v_res_1020_; 
v_res_1020_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_uffe2____1(v_x_1017_, v_a_1018_, v_a_1019_);
lean_dec_ref(v_a_1018_);
return v_res_1020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HNot__hnot__1(lean_object* v_x_1021_, lean_object* v_a_1022_, lean_object* v_a_1023_){
_start:
{
lean_object* v___x_1024_; uint8_t v___x_1025_; 
v___x_1024_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term___u1d9c__1___closed__4));
lean_inc(v_x_1021_);
v___x_1025_ = l_Lean_Syntax_isOfKind(v_x_1021_, v___x_1024_);
if (v___x_1025_ == 0)
{
lean_object* v___x_1026_; lean_object* v___x_1027_; 
lean_dec(v_x_1021_);
v___x_1026_ = lean_box(0);
v___x_1027_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1027_, 0, v___x_1026_);
lean_ctor_set(v___x_1027_, 1, v_a_1023_);
return v___x_1027_;
}
else
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; uint8_t v___x_1031_; 
v___x_1028_ = lean_unsigned_to_nat(0u);
v___x_1029_ = l_Lean_Syntax_getArg(v_x_1021_, v___x_1028_);
v___x_1030_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1));
lean_inc(v___x_1029_);
v___x_1031_ = l_Lean_Syntax_isOfKind(v___x_1029_, v___x_1030_);
if (v___x_1031_ == 0)
{
lean_object* v___x_1032_; lean_object* v___x_1033_; 
lean_dec(v___x_1029_);
lean_dec(v_x_1021_);
v___x_1032_ = lean_box(0);
v___x_1033_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1033_, 0, v___x_1032_);
lean_ctor_set(v___x_1033_, 1, v_a_1023_);
return v___x_1033_;
}
else
{
lean_object* v___x_1034_; lean_object* v___x_1035_; uint8_t v___x_1036_; 
v___x_1034_ = lean_unsigned_to_nat(1u);
v___x_1035_ = l_Lean_Syntax_getArg(v_x_1021_, v___x_1034_);
lean_dec(v_x_1021_);
lean_inc(v___x_1035_);
v___x_1036_ = l_Lean_Syntax_matchesNull(v___x_1035_, v___x_1034_);
if (v___x_1036_ == 0)
{
lean_object* v___x_1037_; lean_object* v___x_1038_; 
lean_dec(v___x_1035_);
lean_dec(v___x_1029_);
v___x_1037_ = lean_box(0);
v___x_1038_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1037_);
lean_ctor_set(v___x_1038_, 1, v_a_1023_);
return v___x_1038_;
}
else
{
lean_object* v___x_1039_; lean_object* v_ref_1040_; uint8_t v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; 
v___x_1039_ = l_Lean_Syntax_getArg(v___x_1035_, v___x_1028_);
lean_dec(v___x_1035_);
v_ref_1040_ = l_Lean_replaceRef(v___x_1029_, v_a_1022_);
lean_dec(v___x_1029_);
v___x_1041_ = 0;
v___x_1042_ = l_Lean_SourceInfo_fromRef(v_ref_1040_, v___x_1041_);
lean_dec(v_ref_1040_);
v___x_1043_ = ((lean_object*)(lp_mathlib_term_uffe2___00__closed__1));
v___x_1044_ = ((lean_object*)(lp_mathlib_term_uffe2___00__closed__2));
lean_inc(v___x_1042_);
v___x_1045_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1045_, 0, v___x_1042_);
lean_ctor_set(v___x_1045_, 1, v___x_1044_);
v___x_1046_ = l_Lean_Syntax_node2(v___x_1042_, v___x_1043_, v___x_1045_, v___x_1039_);
v___x_1047_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1047_, 0, v___x_1046_);
lean_ctor_set(v___x_1047_, 1, v_a_1023_);
return v___x_1047_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HNot__hnot__1___boxed(lean_object* v_x_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_mathlib___aux__Mathlib__Order__Notation______unexpand__HNot__hnot__1(v_x_1048_, v_a_1049_, v_a_1050_);
lean_dec(v_a_1049_);
return v_res_1051_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__1(void){
_start:
{
lean_object* v___x_1064_; lean_object* v___x_1065_; 
v___x_1064_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__0));
v___x_1065_ = l_String_toRawSubstring_x27(v___x_1064_);
return v___x_1065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1(lean_object* v_x_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_){
_start:
{
lean_object* v___x_1080_; uint8_t v___x_1081_; 
v___x_1080_ = ((lean_object*)(lp_mathlib_term_u22a4___closed__1));
v___x_1081_ = l_Lean_Syntax_isOfKind(v_x_1077_, v___x_1080_);
if (v___x_1081_ == 0)
{
lean_object* v___x_1082_; lean_object* v___x_1083_; 
v___x_1082_ = lean_box(1);
v___x_1083_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1083_, 0, v___x_1082_);
lean_ctor_set(v___x_1083_, 1, v_a_1079_);
return v___x_1083_;
}
else
{
lean_object* v_quotContext_1084_; lean_object* v_currMacroScope_1085_; lean_object* v_ref_1086_; uint8_t v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; 
v_quotContext_1084_ = lean_ctor_get(v_a_1078_, 1);
v_currMacroScope_1085_ = lean_ctor_get(v_a_1078_, 2);
v_ref_1086_ = lean_ctor_get(v_a_1078_, 5);
v___x_1087_ = 0;
v___x_1088_ = l_Lean_SourceInfo_fromRef(v_ref_1086_, v___x_1087_);
v___x_1089_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__1, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__1);
v___x_1090_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__4));
lean_inc(v_currMacroScope_1085_);
lean_inc(v_quotContext_1084_);
v___x_1091_ = l_Lean_addMacroScope(v_quotContext_1084_, v___x_1090_, v_currMacroScope_1085_);
v___x_1092_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___closed__6));
v___x_1093_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1093_, 0, v___x_1088_);
lean_ctor_set(v___x_1093_, 1, v___x_1089_);
lean_ctor_set(v___x_1093_, 2, v___x_1091_);
lean_ctor_set(v___x_1093_, 3, v___x_1092_);
v___x_1094_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1094_, 0, v___x_1093_);
lean_ctor_set(v___x_1094_, 1, v_a_1079_);
return v___x_1094_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1___boxed(lean_object* v_x_1095_, lean_object* v_a_1096_, lean_object* v_a_1097_){
_start:
{
lean_object* v_res_1098_; 
v_res_1098_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a4__1(v_x_1095_, v_a_1096_, v_a_1097_);
lean_dec_ref(v_a_1096_);
return v_res_1098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Top__top__1(lean_object* v_x_1099_, lean_object* v_a_1100_, lean_object* v_a_1101_){
_start:
{
lean_object* v___x_1102_; uint8_t v___x_1103_; 
v___x_1102_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1));
lean_inc(v_x_1099_);
v___x_1103_ = l_Lean_Syntax_isOfKind(v_x_1099_, v___x_1102_);
if (v___x_1103_ == 0)
{
lean_object* v___x_1104_; lean_object* v___x_1105_; 
lean_dec(v_x_1099_);
v___x_1104_ = lean_box(0);
v___x_1105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1105_, 0, v___x_1104_);
lean_ctor_set(v___x_1105_, 1, v_a_1101_);
return v___x_1105_;
}
else
{
lean_object* v_ref_1106_; uint8_t v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; 
v_ref_1106_ = l_Lean_replaceRef(v_x_1099_, v_a_1100_);
lean_dec(v_x_1099_);
v___x_1107_ = 0;
v___x_1108_ = l_Lean_SourceInfo_fromRef(v_ref_1106_, v___x_1107_);
lean_dec(v_ref_1106_);
v___x_1109_ = ((lean_object*)(lp_mathlib_term_u22a4___closed__1));
v___x_1110_ = ((lean_object*)(lp_mathlib_term_u22a4___closed__2));
lean_inc(v___x_1108_);
v___x_1111_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1111_, 0, v___x_1108_);
lean_ctor_set(v___x_1111_, 1, v___x_1110_);
v___x_1112_ = l_Lean_Syntax_node1(v___x_1108_, v___x_1109_, v___x_1111_);
v___x_1113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1113_, 0, v___x_1112_);
lean_ctor_set(v___x_1113_, 1, v_a_1101_);
return v___x_1113_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Top__top__1___boxed(lean_object* v_x_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_){
_start:
{
lean_object* v_res_1117_; 
v_res_1117_ = lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Top__top__1(v_x_1114_, v_a_1115_, v_a_1116_);
lean_dec(v_a_1115_);
return v_res_1117_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__1(void){
_start:
{
lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1130_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__0));
v___x_1131_ = l_String_toRawSubstring_x27(v___x_1130_);
return v___x_1131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1(lean_object* v_x_1143_, lean_object* v_a_1144_, lean_object* v_a_1145_){
_start:
{
lean_object* v___x_1146_; uint8_t v___x_1147_; 
v___x_1146_ = ((lean_object*)(lp_mathlib_term_u22a5___closed__1));
v___x_1147_ = l_Lean_Syntax_isOfKind(v_x_1143_, v___x_1146_);
if (v___x_1147_ == 0)
{
lean_object* v___x_1148_; lean_object* v___x_1149_; 
v___x_1148_ = lean_box(1);
v___x_1149_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1149_, 0, v___x_1148_);
lean_ctor_set(v___x_1149_, 1, v_a_1145_);
return v___x_1149_;
}
else
{
lean_object* v_quotContext_1150_; lean_object* v_currMacroScope_1151_; lean_object* v_ref_1152_; uint8_t v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; 
v_quotContext_1150_ = lean_ctor_get(v_a_1144_, 1);
v_currMacroScope_1151_ = lean_ctor_get(v_a_1144_, 2);
v_ref_1152_ = lean_ctor_get(v_a_1144_, 5);
v___x_1153_ = 0;
v___x_1154_ = l_Lean_SourceInfo_fromRef(v_ref_1152_, v___x_1153_);
v___x_1155_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__1, &lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__1);
v___x_1156_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__4));
lean_inc(v_currMacroScope_1151_);
lean_inc(v_quotContext_1150_);
v___x_1157_ = l_Lean_addMacroScope(v_quotContext_1150_, v___x_1156_, v_currMacroScope_1151_);
v___x_1158_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___closed__6));
v___x_1159_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1154_);
lean_ctor_set(v___x_1159_, 1, v___x_1155_);
lean_ctor_set(v___x_1159_, 2, v___x_1157_);
lean_ctor_set(v___x_1159_, 3, v___x_1158_);
v___x_1160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1160_, 0, v___x_1159_);
lean_ctor_set(v___x_1160_, 1, v_a_1145_);
return v___x_1160_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1___boxed(lean_object* v_x_1161_, lean_object* v_a_1162_, lean_object* v_a_1163_){
_start:
{
lean_object* v_res_1164_; 
v_res_1164_ = lp_mathlib___aux__Mathlib__Order__Notation______macroRules__term_u22a5__1(v_x_1161_, v_a_1162_, v_a_1163_);
lean_dec_ref(v_a_1162_);
return v_res_1164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Bot__bot__1(lean_object* v_x_1165_, lean_object* v_a_1166_, lean_object* v_a_1167_){
_start:
{
lean_object* v___x_1168_; uint8_t v___x_1169_; 
v___x_1168_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Compl__compl__1___closed__1));
lean_inc(v_x_1165_);
v___x_1169_ = l_Lean_Syntax_isOfKind(v_x_1165_, v___x_1168_);
if (v___x_1169_ == 0)
{
lean_object* v___x_1170_; lean_object* v___x_1171_; 
lean_dec(v_x_1165_);
v___x_1170_ = lean_box(0);
v___x_1171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1171_, 0, v___x_1170_);
lean_ctor_set(v___x_1171_, 1, v_a_1167_);
return v___x_1171_;
}
else
{
lean_object* v_ref_1172_; uint8_t v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; 
v_ref_1172_ = l_Lean_replaceRef(v_x_1165_, v_a_1166_);
lean_dec(v_x_1165_);
v___x_1173_ = 0;
v___x_1174_ = l_Lean_SourceInfo_fromRef(v_ref_1172_, v___x_1173_);
lean_dec(v_ref_1172_);
v___x_1175_ = ((lean_object*)(lp_mathlib_term_u22a5___closed__1));
v___x_1176_ = ((lean_object*)(lp_mathlib_term_u22a5___closed__2));
lean_inc(v___x_1174_);
v___x_1177_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1177_, 0, v___x_1174_);
lean_ctor_set(v___x_1177_, 1, v___x_1176_);
v___x_1178_ = l_Lean_Syntax_node1(v___x_1174_, v___x_1175_, v___x_1177_);
v___x_1179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1179_, 0, v___x_1178_);
lean_ctor_set(v___x_1179_, 1, v_a_1167_);
return v___x_1179_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Bot__bot__1___boxed(lean_object* v_x_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_){
_start:
{
lean_object* v_res_1183_; 
v_res_1183_ = lp_mathlib___aux__Mathlib__Order__Notation______unexpand__Bot__bot__1(v_x_1180_, v_a_1181_, v_a_1182_);
lean_dec(v_a_1181_);
return v_res_1183_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_PrettyPrinter_Delaborator(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_PrettyPrinter_Delaborator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_PrettyPrinter_Delaborator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_PrettyPrinter_Delaborator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Notation(builtin);
}
#ifdef __cplusplus
}
#endif
