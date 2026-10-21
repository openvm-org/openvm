// Lean compiler output
// Module: Mathlib.Order.SetNotation
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Operations public import Mathlib.Util.Notation3
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPFunBinderTypes___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
uint8_t lean_expr_has_loose_bvar(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isLambda(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_getBinders(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchScoped(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u2a06___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term⨆_,_"};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2a06___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 186, 184, 43, 20, 164, 105, 224)}};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_term_u2a06___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2a06___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__3_value;
static const lean_string_object lp_mathlib_term_u2a06___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⨆"};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term_u2a06___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__4_value)}};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__5_value;
static lean_once_cell_t lp_mathlib_term_u2a06___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a06___x2c___00__closed__6;
static const lean_string_object lp_mathlib_term_u2a06___x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__7 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term_u2a06___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__7_value)}};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__8_value;
static lean_once_cell_t lp_mathlib_term_u2a06___x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a06___x2c___00__closed__9;
static const lean_string_object lp_mathlib_term_u2a06___x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__10 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_term_u2a06___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_term_u2a06___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__11_value),((lean_object*)(((size_t)(60) << 1) | 1))}};
static const lean_object* lp_mathlib_term_u2a06___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__12_value;
static lean_once_cell_t lp_mathlib_term_u2a06___x2c___00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a06___x2c___00__closed__13;
static lean_once_cell_t lp_mathlib_term_u2a06___x2c___00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a06___x2c___00__closed__14;
LEAN_EXPORT lean_object* lp_mathlib_term_u2a06___x2c__;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Notation3"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termExpand_binders%(_=>_)_,_"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 176, 22, 214, 10, 13, 147, 22)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(120, 7, 237, 26, 3, 243, 131, 214)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "expand_binders%"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__12_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "iSup"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__15_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__16;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(10, 142, 27, 35, 119, 188, 117, 38)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__17_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__18_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__19_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⨆ "};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extBinders"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(142, 202, 111, 171, 129, 134, 17, 161)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "extBinderCollection"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(144, 58, 22, 199, 215, 82, 42, 232)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__5_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__5_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u2a05___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term⨅_,_"};
static const lean_object* lp_mathlib_term_u2a05___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_term_u2a05___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2a05___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2a05___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(81, 66, 191, 135, 33, 91, 42, 154)}};
static const lean_object* lp_mathlib_term_u2a05___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_term_u2a05___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_term_u2a05___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⨅"};
static const lean_object* lp_mathlib_term_u2a05___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_term_u2a05___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2a05___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a05___x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_term_u2a05___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_term_u2a05___x2c___00__closed__3_value;
static lean_once_cell_t lp_mathlib_term_u2a05___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a05___x2c___00__closed__4;
static lean_once_cell_t lp_mathlib_term_u2a05___x2c___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a05___x2c___00__closed__5;
static lean_once_cell_t lp_mathlib_term_u2a05___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a05___x2c___00__closed__6;
static lean_once_cell_t lp_mathlib_term_u2a05___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u2a05___x2c___00__closed__7;
LEAN_EXPORT lean_object* lp_mathlib_term_u2a05___x2c__;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "iInf"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(180, 16, 117, 176, 174, 45, 54, 18)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⨅ "};
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "extBinderParenthesized"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__0 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 166, 79, 161, 194, 16, 7, 156)}};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__1 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__2 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__3 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__4 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__5 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__6 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__7 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__8 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__9 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__9_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__10_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__10_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__10_value_aux_2),((lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__10 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__10_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_iSup__delab___lam__0___closed__11 = (const lean_object*)&lp_mathlib_iSup__delab___lam__0___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__0(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__1(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_iSup__delab___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_iSup__delab___lam__2___closed__0;
static const lean_closure_object lp_mathlib_iSup__delab___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPFunBinderTypes___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__1 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__1_value;
static const lean_closure_object lp_mathlib_iSup__delab___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__2 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__3 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__3_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__4 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∈_"};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__5 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__5_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(145, 149, 102, 29, 65, 152, 113, 144)}};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__6 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∈_"};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__7 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__7_value;
static const lean_ctor_object lp_mathlib_iSup__delab___lam__2___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_iSup__delab___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__8_value_aux_0),((lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(150, 164, 254, 63, 76, 57, 126, 92)}};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__8 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_iSup__delab___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∈"};
static const lean_object* lp_mathlib_iSup__delab___lam__2___closed__9 = (const lean_object*)&lp_mathlib_iSup__delab___lam__2___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_iSup__delab___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_iSup__delab___lam__2___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1))} };
static const lean_object* lp_mathlib_iSup__delab___closed__0 = (const lean_object*)&lp_mathlib_iSup__delab___closed__0_value;
static const lean_closure_object lp_mathlib_iSup__delab___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib_iSup__delab___closed__0_value)} };
static const lean_object* lp_mathlib_iSup__delab___closed__1 = (const lean_object*)&lp_mathlib_iSup__delab___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__0(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__1(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_iInf__delab___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_iInf__delab___lam__2___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1))} };
static const lean_object* lp_mathlib_iInf__delab___closed__0 = (const lean_object*)&lp_mathlib_iInf__delab___closed__0_value;
static const lean_closure_object lp_mathlib_iInf__delab___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib_iInf__delab___closed__0_value)} };
static const lean_object* lp_mathlib_iInf__delab___closed__1 = (const lean_object*)&lp_mathlib_iInf__delab___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instInfSet(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instSupSet(lean_object*);
static const lean_string_object lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value;
static const lean_string_object lp_mathlib_Set_term_u22c2_u2080___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "term⋂₀_"};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2_u2080___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term_u22c2_u2080___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(192, 106, 119, 4, 67, 136, 52, 99)}};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__2_value;
static const lean_string_object lp_mathlib_Set_term_u22c2_u2080___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "⋂₀ "};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2_u2080___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__3_value)}};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__4 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2_u2080___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__11_value),((lean_object*)(((size_t)(110) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__5 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2_u2080___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__4_value),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__5_value)}};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__6 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2_u2080___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__2_value),((lean_object*)(((size_t)(110) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__6_value)}};
static const lean_object* lp_mathlib_Set_term_u22c2_u2080___00__closed__7 = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_term_u22c2_u2080__ = (const lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__7_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sInter"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__1;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 140, 222, 155, 254, 8, 196, 44)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(252, 41, 190, 32, 200, 62, 232, 223)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sInter__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sInter__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_term_u22c3_u2080___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "term⋃₀_"};
static const lean_object* lp_mathlib_Set_term_u22c3_u2080___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c3_u2080___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term_u22c3_u2080___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(157, 121, 246, 139, 200, 62, 14, 212)}};
static const lean_object* lp_mathlib_Set_term_u22c3_u2080___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__1_value;
static const lean_string_object lp_mathlib_Set_term_u22c3_u2080___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "⋃₀ "};
static const lean_object* lp_mathlib_Set_term_u22c3_u2080___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c3_u2080___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__2_value)}};
static const lean_object* lp_mathlib_Set_term_u22c3_u2080___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c3_u2080___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u2a06___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__3_value),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__5_value)}};
static const lean_object* lp_mathlib_Set_term_u22c3_u2080___00__closed__4 = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c3_u2080___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__1_value),((lean_object*)(((size_t)(110) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__4_value)}};
static const lean_object* lp_mathlib_Set_term_u22c3_u2080___00__closed__5 = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_term_u22c3_u2080__ = (const lean_object*)&lp_mathlib_Set_term_u22c3_u2080___00__closed__5_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sUnion"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__1;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(136, 126, 241, 86, 155, 234, 219, 209)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 253, 223, 125, 22, 232, 248, 60)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sUnion__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sUnion__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_term_u22c3___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term⋃_,_"};
static const lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c3___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term_u22c3___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(218, 66, 216, 118, 216, 3, 1, 105)}};
static const lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Set_term_u22c3___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⋃"};
static const lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c3___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term_u22c3___x2c___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Set_term_u22c3___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__4;
static lean_once_cell_t lp_mathlib_Set_term_u22c3___x2c___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__5;
static lean_once_cell_t lp_mathlib_Set_term_u22c3___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__6;
static lean_once_cell_t lp_mathlib_Set_term_u22c3___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c3___x2c___00__closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Set_term_u22c3___x2c__;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "iUnion"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__1;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 118, 216, 200, 90, 76, 214, 194)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 206, 185, 95, 103, 141, 71, 229)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⋃ "};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_term_u22c2___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term⋂_,_"};
static const lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term_u22c2___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(200, 27, 175, 55, 115, 117, 31, 170)}};
static const lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Set_term_u22c2___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⋂"};
static const lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Set_term_u22c2___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term_u22c2___x2c___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Set_term_u22c2___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__4;
static lean_once_cell_t lp_mathlib_Set_term_u22c2___x2c___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__5;
static lean_once_cell_t lp_mathlib_Set_term_u22c2___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__6;
static lean_once_cell_t lp_mathlib_Set_term_u22c2___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_term_u22c2___x2c___00__closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Set_term_u22c2___x2c__;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "iInter"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__1;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(9, 2, 73, 132, 29, 99, 18, 86)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_u22c2_u2080___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 80, 117, 151, 239, 52, 174, 40)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "⋂ "};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__4_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__0(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__1(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set_iUnion__delab___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_iUnion__delab___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_iUnion__delab___closed__0 = (const lean_object*)&lp_mathlib_Set_iUnion__delab___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__0(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__1(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set_sInter__delab___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Set_sInter__delab___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_sInter__delab___closed__0 = (const lean_object*)&lp_mathlib_Set_sInter__delab___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iSup___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_apply_1(v_inst_1_, lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup(lean_object* v_00_u03b1_3_, lean_object* v_00_u03b9_4_, lean_object* v_inst_5_, lean_object* v_s_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_apply_1(v_inst_5_, lean_box(0));
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup___boxed(lean_object* v_00_u03b1_8_, lean_object* v_00_u03b9_9_, lean_object* v_inst_10_, lean_object* v_s_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_iSup(v_00_u03b1_8_, v_00_u03b9_9_, v_inst_10_, v_s_11_);
lean_dec(v_s_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_apply_1(v_inst_13_, lean_box(0));
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf(lean_object* v_00_u03b1_15_, lean_object* v_00_u03b9_16_, lean_object* v_inst_17_, lean_object* v_s_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_apply_1(v_inst_17_, lean_box(0));
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___boxed(lean_object* v_00_u03b1_20_, lean_object* v_00_u03b9_21_, lean_object* v_inst_22_, lean_object* v_s_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_iInf(v_00_u03b1_20_, v_00_u03b9_21_, v_inst_22_, v_s_23_);
lean_dec(v_s_23_);
return v_res_24_;
}
}
static lean_object* _init_lp_mathlib_term_u2a06___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_34_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_35_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__5));
v___x_36_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_37_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_37_, 0, v___x_36_);
lean_ctor_set(v___x_37_, 1, v___x_35_);
lean_ctor_set(v___x_37_, 2, v___x_34_);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib_term_u2a06___x2c___00__closed__9(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_41_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__8));
v___x_42_ = lean_obj_once(&lp_mathlib_term_u2a06___x2c___00__closed__6, &lp_mathlib_term_u2a06___x2c___00__closed__6_once, _init_lp_mathlib_term_u2a06___x2c___00__closed__6);
v___x_43_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_44_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v___x_42_);
lean_ctor_set(v___x_44_, 2, v___x_41_);
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib_term_u2a06___x2c___00__closed__13(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_51_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__12));
v___x_52_ = lean_obj_once(&lp_mathlib_term_u2a06___x2c___00__closed__9, &lp_mathlib_term_u2a06___x2c___00__closed__9_once, _init_lp_mathlib_term_u2a06___x2c___00__closed__9);
v___x_53_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_54_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v___x_52_);
lean_ctor_set(v___x_54_, 2, v___x_51_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib_term_u2a06___x2c___00__closed__14(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = lean_obj_once(&lp_mathlib_term_u2a06___x2c___00__closed__13, &lp_mathlib_term_u2a06___x2c___00__closed__13_once, _init_lp_mathlib_term_u2a06___x2c___00__closed__13);
v___x_56_ = lean_unsigned_to_nat(1022u);
v___x_57_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
v___x_58_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v___x_56_);
lean_ctor_set(v___x_58_, 2, v___x_55_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_term_u2a06___x2c__(void){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_obj_once(&lp_mathlib_term_u2a06___x2c___00__closed__14, &lp_mathlib_term_u2a06___x2c___00__closed__14_once, _init_lp_mathlib_term_u2a06___x2c___00__closed__14);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__6));
v___x_71_ = l_String_toRawSubstring_x27(v___x_70_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__16(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__15));
v___x_86_ = l_String_toRawSubstring_x27(v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1(lean_object* v_x_100_, lean_object* v_a_101_, lean_object* v_a_102_){
_start:
{
lean_object* v___x_103_; uint8_t v___x_104_; 
v___x_103_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
lean_inc(v_x_100_);
v___x_104_ = l_Lean_Syntax_isOfKind(v_x_100_, v___x_103_);
if (v___x_104_ == 0)
{
lean_object* v___x_105_; lean_object* v___x_106_; 
lean_dec(v_x_100_);
v___x_105_ = lean_box(1);
v___x_106_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_105_);
lean_ctor_set(v___x_106_, 1, v_a_102_);
return v___x_106_;
}
else
{
lean_object* v_quotContext_107_; lean_object* v_currMacroScope_108_; lean_object* v_ref_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v_quotContext_107_ = lean_ctor_get(v_a_101_, 1);
v_currMacroScope_108_ = lean_ctor_get(v_a_101_, 2);
v_ref_109_ = lean_ctor_get(v_a_101_, 5);
v___x_110_ = lean_unsigned_to_nat(1u);
v___x_111_ = l_Lean_Syntax_getArg(v_x_100_, v___x_110_);
v___x_112_ = lean_unsigned_to_nat(3u);
v___x_113_ = l_Lean_Syntax_getArg(v_x_100_, v___x_112_);
lean_dec(v_x_100_);
v___x_114_ = 0;
v___x_115_ = l_Lean_SourceInfo_fromRef(v_ref_109_, v___x_114_);
v___x_116_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3));
v___x_117_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4));
lean_inc_n(v___x_115_, 9);
v___x_118_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_115_);
lean_ctor_set(v___x_118_, 1, v___x_117_);
v___x_119_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_120_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_115_);
lean_ctor_set(v___x_120_, 1, v___x_119_);
v___x_121_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7);
v___x_122_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_108_, 2);
lean_inc_n(v_quotContext_107_, 2);
v___x_123_ = l_Lean_addMacroScope(v_quotContext_107_, v___x_122_, v_currMacroScope_108_);
v___x_124_ = lean_box(0);
v___x_125_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_125_, 0, v___x_115_);
lean_ctor_set(v___x_125_, 1, v___x_121_);
lean_ctor_set(v___x_125_, 2, v___x_123_);
lean_ctor_set(v___x_125_, 3, v___x_124_);
v___x_126_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9));
v___x_127_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_115_);
lean_ctor_set(v___x_127_, 1, v___x_126_);
v___x_128_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
v___x_129_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__16, &lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__16);
v___x_130_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__17));
v___x_131_ = l_Lean_addMacroScope(v_quotContext_107_, v___x_130_, v_currMacroScope_108_);
v___x_132_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__19));
v___x_133_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_133_, 0, v___x_115_);
lean_ctor_set(v___x_133_, 1, v___x_129_);
lean_ctor_set(v___x_133_, 2, v___x_131_);
lean_ctor_set(v___x_133_, 3, v___x_132_);
v___x_134_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
lean_inc_ref(v___x_125_);
v___x_135_ = l_Lean_Syntax_node1(v___x_115_, v___x_134_, v___x_125_);
v___x_136_ = l_Lean_Syntax_node2(v___x_115_, v___x_128_, v___x_133_, v___x_135_);
v___x_137_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_138_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_115_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_140_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_115_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v___x_141_ = lean_unsigned_to_nat(9u);
v___x_142_ = lean_mk_empty_array_with_capacity(v___x_141_);
v___x_143_ = lean_array_push(v___x_142_, v___x_118_);
v___x_144_ = lean_array_push(v___x_143_, v___x_120_);
v___x_145_ = lean_array_push(v___x_144_, v___x_125_);
v___x_146_ = lean_array_push(v___x_145_, v___x_127_);
v___x_147_ = lean_array_push(v___x_146_, v___x_136_);
v___x_148_ = lean_array_push(v___x_147_, v___x_138_);
v___x_149_ = lean_array_push(v___x_148_, v___x_111_);
v___x_150_ = lean_array_push(v___x_149_, v___x_140_);
v___x_151_ = lean_array_push(v___x_150_, v___x_113_);
v___x_152_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_152_, 0, v___x_115_);
lean_ctor_set(v___x_152_, 1, v___x_116_);
lean_ctor_set(v___x_152_, 2, v___x_151_);
v___x_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v_a_102_);
return v___x_153_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___boxed(lean_object* v_x_154_, lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1(v_x_154_, v_a_155_, v_a_156_);
lean_dec_ref(v_a_155_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(lean_object* v___y_158_){
_start:
{
lean_object* v_subExpr_160_; lean_object* v_expr_161_; lean_object* v___x_162_; 
v_subExpr_160_ = lean_ctor_get(v___y_158_, 3);
v_expr_161_ = lean_ctor_get(v_subExpr_160_, 0);
lean_inc_ref(v_expr_161_);
v___x_162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_162_, 0, v_expr_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg___boxed(lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_163_);
lean_dec_ref(v___y_163_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0(lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_166_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___boxed(lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0(v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
return v_res_181_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__0(lean_object* v_x_182_){
_start:
{
lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_183_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__17));
v___x_184_ = l_Lean_Expr_isConstOf(v_x_182_, v___x_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__0___boxed(lean_object* v_x_185_){
_start:
{
uint8_t v_res_186_; lean_object* v_r_187_; 
v_res_186_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__0(v_x_185_);
lean_dec_ref(v_x_185_);
v_r_187_ = lean_box(v_res_186_);
return v_r_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__1(lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_196_, 0, v___y_188_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__1___boxed(lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__1(v___y_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
lean_dec(v___y_203_);
lean_dec_ref(v___y_202_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2(uint8_t v___x_207_, lean_object* v___x_208_, lean_object* v_a_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_ref_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v_ref_217_ = lean_ctor_get(v___y_214_, 5);
v___x_218_ = l_Lean_SourceInfo_fromRef(v_ref_217_, v___x_207_);
v___x_219_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
v___x_220_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_218_, 2);
v___x_221_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_221_, 0, v___x_218_);
lean_ctor_set(v___x_221_, 1, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__7));
v___x_223_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_218_);
lean_ctor_set(v___x_223_, 1, v___x_222_);
v___x_224_ = l_Lean_Syntax_node4(v___x_218_, v___x_219_, v___x_221_, v___x_208_, v___x_223_, v_a_209_);
v___x_225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___boxed(lean_object* v___x_226_, lean_object* v___x_227_, lean_object* v_a_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_){
_start:
{
uint8_t v___x_6998__boxed_236_; lean_object* v_res_237_; 
v___x_6998__boxed_236_ = lean_unbox(v___x_226_);
v_res_237_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2(v___x_6998__boxed_236_, v___x_227_, v_a_228_, v___y_229_, v___y_230_, v___y_231_, v___y_232_, v___y_233_, v___y_234_);
lean_dec(v___y_234_);
lean_dec_ref(v___y_233_);
lean_dec(v___y_232_);
lean_dec_ref(v___y_231_);
lean_dec(v___y_230_);
lean_dec_ref(v___y_229_);
return v_res_237_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6(void){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = l_Array_mkArray0(lean_box(0));
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3(lean_object* v___f_256_, lean_object* v___f_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v___x_265_; lean_object* v_a_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_313_; 
v___x_265_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_258_);
v_a_266_ = lean_ctor_get(v___x_265_, 0);
v_isSharedCheck_313_ = !lean_is_exclusive(v___x_265_);
if (v_isSharedCheck_313_ == 0)
{
v___x_268_ = v___x_265_;
v_isShared_269_ = v_isSharedCheck_313_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_a_266_);
lean_dec(v___x_265_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_313_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_270_; lean_object* v___y_272_; lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_270_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1));
v___x_302_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_303_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_270_, v___x_302_, v___y_258_, v___y_260_);
if (lean_obj_tag(v___x_303_) == 0)
{
lean_object* v_a_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v_a_304_ = lean_ctor_get(v___x_303_, 0);
lean_inc(v_a_304_);
lean_dec_ref_known(v___x_303_, 1);
v___x_305_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_305_, 0, v___f_256_);
lean_inc_ref_n(v___f_257_, 2);
v___x_306_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_306_, 0, v___x_305_);
lean_closure_set(v___x_306_, 1, v___f_257_);
v___x_307_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_307_, 0, v___x_306_);
lean_closure_set(v___x_307_, 1, v___f_257_);
v___x_308_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
v___x_309_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_309_, 0, v___x_307_);
lean_closure_set(v___x_309_, 1, v___f_257_);
v___x_310_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9));
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_311_, 0, v___x_309_);
lean_closure_set(v___x_311_, 1, v___x_310_);
v___x_312_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_270_, v___x_308_, v___x_311_, v_a_304_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
v___y_272_ = v___x_312_;
goto v___jp_271_;
}
else
{
lean_dec_ref(v___f_257_);
lean_dec_ref(v___f_256_);
v___y_272_ = v___x_303_;
goto v___jp_271_;
}
v___jp_271_:
{
if (lean_obj_tag(v___y_272_) == 0)
{
lean_object* v_a_273_; lean_object* v_ref_274_; lean_object* v___x_276_; 
v_a_273_ = lean_ctor_get(v___y_272_, 0);
lean_inc(v_a_273_);
lean_dec_ref_known(v___y_272_, 1);
v_ref_274_ = lean_ctor_get(v___y_262_, 5);
if (v_isShared_269_ == 0)
{
lean_ctor_set_tag(v___x_268_, 1);
v___x_276_ = v___x_268_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v_a_266_);
v___x_276_ = v_reuseFailAlloc_293_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_273_, v___x_270_, v___x_276_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
if (lean_obj_tag(v___x_277_) == 0)
{
lean_object* v_a_278_; uint8_t v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___f_291_; lean_object* v___x_292_; 
v_a_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc(v_a_278_);
lean_dec_ref_known(v___x_277_, 1);
v___x_279_ = 0;
v___x_280_ = l_Lean_SourceInfo_fromRef(v_ref_274_, v___x_279_);
v___x_281_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_282_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_283_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_284_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_273_);
lean_dec(v_a_273_);
v___x_285_ = l_Array_append___redArg(v___x_283_, v___x_284_);
lean_dec_ref(v___x_284_);
lean_inc_n(v___x_280_, 2);
v___x_286_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_286_, 0, v___x_280_);
lean_ctor_set(v___x_286_, 1, v___x_282_);
lean_ctor_set(v___x_286_, 2, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_288_ = l_Lean_Syntax_node1(v___x_280_, v___x_287_, v___x_286_);
v___x_289_ = l_Lean_Syntax_node1(v___x_280_, v___x_281_, v___x_288_);
v___x_290_ = lean_box(v___x_279_);
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_291_, 0, v___x_290_);
lean_closure_set(v___f_291_, 1, v___x_289_);
lean_closure_set(v___f_291_, 2, v_a_278_);
v___x_292_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_291_, v___y_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
return v___x_292_;
}
else
{
lean_dec(v_a_273_);
return v___x_277_;
}
}
}
else
{
lean_object* v_a_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_301_; 
lean_del_object(v___x_268_);
lean_dec(v_a_266_);
v_a_294_ = lean_ctor_get(v___y_272_, 0);
v_isSharedCheck_301_ = !lean_is_exclusive(v___y_272_);
if (v_isSharedCheck_301_ == 0)
{
v___x_296_ = v___y_272_;
v_isShared_297_ = v_isSharedCheck_301_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_a_294_);
lean_dec(v___y_272_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_301_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_299_; 
if (v_isShared_297_ == 0)
{
v___x_299_ = v___x_296_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_300_; 
v_reuseFailAlloc_300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_300_, 0, v_a_294_);
v___x_299_ = v_reuseFailAlloc_300_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
return v___x_299_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___boxed(lean_object* v___f_314_, lean_object* v___f_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3(v___f_314_, v___f_315_, v___y_316_, v___y_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v___y_317_);
lean_dec_ref(v___y_316_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1(lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_344_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_345_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__6));
v___x_346_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_344_, v___x_345_, v_a_337_, v_a_338_, v_a_339_, v_a_340_, v_a_341_, v_a_342_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___boxed(lean_object* v_a_347_, lean_object* v_a_348_, lean_object* v_a_349_, lean_object* v_a_350_, lean_object* v_a_351_, lean_object* v_a_352_, lean_object* v_a_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1(v_a_347_, v_a_348_, v_a_349_, v_a_350_, v_a_351_, v_a_352_);
lean_dec(v_a_352_);
lean_dec_ref(v_a_351_);
lean_dec(v_a_350_);
lean_dec_ref(v_a_349_);
lean_dec(v_a_348_);
lean_dec_ref(v_a_347_);
return v_res_354_;
}
}
static lean_object* _init_lp_mathlib_term_u2a05___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_361_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_362_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__3));
v___x_363_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_364_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_362_);
lean_ctor_set(v___x_364_, 2, v___x_361_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_term_u2a05___x2c___00__closed__5(void){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_365_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__8));
v___x_366_ = lean_obj_once(&lp_mathlib_term_u2a05___x2c___00__closed__4, &lp_mathlib_term_u2a05___x2c___00__closed__4_once, _init_lp_mathlib_term_u2a05___x2c___00__closed__4);
v___x_367_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_368_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_368_, 0, v___x_367_);
lean_ctor_set(v___x_368_, 1, v___x_366_);
lean_ctor_set(v___x_368_, 2, v___x_365_);
return v___x_368_;
}
}
static lean_object* _init_lp_mathlib_term_u2a05___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_369_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__12));
v___x_370_ = lean_obj_once(&lp_mathlib_term_u2a05___x2c___00__closed__5, &lp_mathlib_term_u2a05___x2c___00__closed__5_once, _init_lp_mathlib_term_u2a05___x2c___00__closed__5);
v___x_371_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_372_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_372_, 0, v___x_371_);
lean_ctor_set(v___x_372_, 1, v___x_370_);
lean_ctor_set(v___x_372_, 2, v___x_369_);
return v___x_372_;
}
}
static lean_object* _init_lp_mathlib_term_u2a05___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_373_ = lean_obj_once(&lp_mathlib_term_u2a05___x2c___00__closed__6, &lp_mathlib_term_u2a05___x2c___00__closed__6_once, _init_lp_mathlib_term_u2a05___x2c___00__closed__6);
v___x_374_ = lean_unsigned_to_nat(1022u);
v___x_375_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
v___x_376_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
lean_ctor_set(v___x_376_, 1, v___x_374_);
lean_ctor_set(v___x_376_, 2, v___x_373_);
return v___x_376_;
}
}
static lean_object* _init_lp_mathlib_term_u2a05___x2c__(void){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lean_obj_once(&lp_mathlib_term_u2a05___x2c___00__closed__7, &lp_mathlib_term_u2a05___x2c___00__closed__7_once, _init_lp_mathlib_term_u2a05___x2c___00__closed__7);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__1(void){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_379_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__0));
v___x_380_ = l_String_toRawSubstring_x27(v___x_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1(lean_object* v_x_389_, lean_object* v_a_390_, lean_object* v_a_391_){
_start:
{
lean_object* v___x_392_; uint8_t v___x_393_; 
v___x_392_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
lean_inc(v_x_389_);
v___x_393_ = l_Lean_Syntax_isOfKind(v_x_389_, v___x_392_);
if (v___x_393_ == 0)
{
lean_object* v___x_394_; lean_object* v___x_395_; 
lean_dec(v_x_389_);
v___x_394_ = lean_box(1);
v___x_395_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
lean_ctor_set(v___x_395_, 1, v_a_391_);
return v___x_395_;
}
else
{
lean_object* v_quotContext_396_; lean_object* v_currMacroScope_397_; lean_object* v_ref_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
v_quotContext_396_ = lean_ctor_get(v_a_390_, 1);
v_currMacroScope_397_ = lean_ctor_get(v_a_390_, 2);
v_ref_398_ = lean_ctor_get(v_a_390_, 5);
v___x_399_ = lean_unsigned_to_nat(1u);
v___x_400_ = l_Lean_Syntax_getArg(v_x_389_, v___x_399_);
v___x_401_ = lean_unsigned_to_nat(3u);
v___x_402_ = l_Lean_Syntax_getArg(v_x_389_, v___x_401_);
lean_dec(v_x_389_);
v___x_403_ = 0;
v___x_404_ = l_Lean_SourceInfo_fromRef(v_ref_398_, v___x_403_);
v___x_405_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3));
v___x_406_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4));
lean_inc_n(v___x_404_, 9);
v___x_407_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_407_, 0, v___x_404_);
lean_ctor_set(v___x_407_, 1, v___x_406_);
v___x_408_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_409_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_404_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
v___x_410_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7);
v___x_411_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_397_, 2);
lean_inc_n(v_quotContext_396_, 2);
v___x_412_ = l_Lean_addMacroScope(v_quotContext_396_, v___x_411_, v_currMacroScope_397_);
v___x_413_ = lean_box(0);
v___x_414_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_414_, 0, v___x_404_);
lean_ctor_set(v___x_414_, 1, v___x_410_);
lean_ctor_set(v___x_414_, 2, v___x_412_);
lean_ctor_set(v___x_414_, 3, v___x_413_);
v___x_415_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9));
v___x_416_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_416_, 0, v___x_404_);
lean_ctor_set(v___x_416_, 1, v___x_415_);
v___x_417_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
v___x_418_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__1, &lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__1);
v___x_419_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__2));
v___x_420_ = l_Lean_addMacroScope(v_quotContext_396_, v___x_419_, v_currMacroScope_397_);
v___x_421_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__4));
v___x_422_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_422_, 0, v___x_404_);
lean_ctor_set(v___x_422_, 1, v___x_418_);
lean_ctor_set(v___x_422_, 2, v___x_420_);
lean_ctor_set(v___x_422_, 3, v___x_421_);
v___x_423_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
lean_inc_ref(v___x_414_);
v___x_424_ = l_Lean_Syntax_node1(v___x_404_, v___x_423_, v___x_414_);
v___x_425_ = l_Lean_Syntax_node2(v___x_404_, v___x_417_, v___x_422_, v___x_424_);
v___x_426_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_427_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_427_, 0, v___x_404_);
lean_ctor_set(v___x_427_, 1, v___x_426_);
v___x_428_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_429_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_404_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
v___x_430_ = lean_unsigned_to_nat(9u);
v___x_431_ = lean_mk_empty_array_with_capacity(v___x_430_);
v___x_432_ = lean_array_push(v___x_431_, v___x_407_);
v___x_433_ = lean_array_push(v___x_432_, v___x_409_);
v___x_434_ = lean_array_push(v___x_433_, v___x_414_);
v___x_435_ = lean_array_push(v___x_434_, v___x_416_);
v___x_436_ = lean_array_push(v___x_435_, v___x_425_);
v___x_437_ = lean_array_push(v___x_436_, v___x_427_);
v___x_438_ = lean_array_push(v___x_437_, v___x_400_);
v___x_439_ = lean_array_push(v___x_438_, v___x_429_);
v___x_440_ = lean_array_push(v___x_439_, v___x_402_);
v___x_441_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_441_, 0, v___x_404_);
lean_ctor_set(v___x_441_, 1, v___x_405_);
lean_ctor_set(v___x_441_, 2, v___x_440_);
v___x_442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
lean_ctor_set(v___x_442_, 1, v_a_391_);
return v___x_442_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___boxed(lean_object* v_x_443_, lean_object* v_a_444_, lean_object* v_a_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1(v_x_443_, v_a_444_, v_a_445_);
lean_dec_ref(v_a_444_);
return v_res_446_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__0(lean_object* v_x_447_){
_start:
{
lean_object* v___x_448_; uint8_t v___x_449_; 
v___x_448_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a05___x2c____1___closed__2));
v___x_449_ = l_Lean_Expr_isConstOf(v_x_447_, v___x_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__0___boxed(lean_object* v_x_450_){
_start:
{
uint8_t v_res_451_; lean_object* v_r_452_; 
v_res_451_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__0(v_x_450_);
lean_dec_ref(v_x_450_);
v_r_452_ = lean_box(v_res_451_);
return v_r_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2(uint8_t v___x_454_, lean_object* v___x_455_, lean_object* v_a_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_){
_start:
{
lean_object* v_ref_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v_ref_464_ = lean_ctor_get(v___y_461_, 5);
v___x_465_ = l_Lean_SourceInfo_fromRef(v_ref_464_, v___x_454_);
v___x_466_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
v___x_467_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_465_, 2);
v___x_468_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_468_, 0, v___x_465_);
lean_ctor_set(v___x_468_, 1, v___x_467_);
v___x_469_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__7));
v___x_470_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_470_, 0, v___x_465_);
lean_ctor_set(v___x_470_, 1, v___x_469_);
v___x_471_ = l_Lean_Syntax_node4(v___x_465_, v___x_466_, v___x_468_, v___x_455_, v___x_470_, v_a_456_);
v___x_472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___boxed(lean_object* v___x_473_, lean_object* v___x_474_, lean_object* v_a_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_){
_start:
{
uint8_t v___x_6595__boxed_483_; lean_object* v_res_484_; 
v___x_6595__boxed_483_ = lean_unbox(v___x_473_);
v_res_484_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2(v___x_6595__boxed_483_, v___x_474_, v_a_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
lean_dec(v___y_481_);
lean_dec_ref(v___y_480_);
lean_dec(v___y_479_);
lean_dec_ref(v___y_478_);
lean_dec(v___y_477_);
lean_dec_ref(v___y_476_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__1(lean_object* v___f_485_, lean_object* v___f_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v___x_494_; lean_object* v_a_495_; lean_object* v___x_497_; uint8_t v_isShared_498_; uint8_t v_isSharedCheck_542_; 
v___x_494_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_487_);
v_a_495_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_542_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_542_ == 0)
{
v___x_497_ = v___x_494_;
v_isShared_498_ = v_isSharedCheck_542_;
goto v_resetjp_496_;
}
else
{
lean_inc(v_a_495_);
lean_dec(v___x_494_);
v___x_497_ = lean_box(0);
v_isShared_498_ = v_isSharedCheck_542_;
goto v_resetjp_496_;
}
v_resetjp_496_:
{
lean_object* v___x_499_; lean_object* v___y_501_; lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_499_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1));
v___x_531_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_532_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_499_, v___x_531_, v___y_487_, v___y_489_);
if (lean_obj_tag(v___x_532_) == 0)
{
lean_object* v_a_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
v_a_533_ = lean_ctor_get(v___x_532_, 0);
lean_inc(v_a_533_);
lean_dec_ref_known(v___x_532_, 1);
v___x_534_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_534_, 0, v___f_485_);
lean_inc_ref_n(v___f_486_, 2);
v___x_535_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_535_, 0, v___x_534_);
lean_closure_set(v___x_535_, 1, v___f_486_);
v___x_536_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_536_, 0, v___x_535_);
lean_closure_set(v___x_536_, 1, v___f_486_);
v___x_537_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
v___x_538_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_538_, 0, v___x_536_);
lean_closure_set(v___x_538_, 1, v___f_486_);
v___x_539_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9));
v___x_540_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_540_, 0, v___x_538_);
lean_closure_set(v___x_540_, 1, v___x_539_);
v___x_541_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_499_, v___x_537_, v___x_540_, v_a_533_, v___y_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_, v___y_492_);
v___y_501_ = v___x_541_;
goto v___jp_500_;
}
else
{
lean_dec_ref(v___f_486_);
lean_dec_ref(v___f_485_);
v___y_501_ = v___x_532_;
goto v___jp_500_;
}
v___jp_500_:
{
if (lean_obj_tag(v___y_501_) == 0)
{
lean_object* v_a_502_; lean_object* v_ref_503_; lean_object* v___x_505_; 
v_a_502_ = lean_ctor_get(v___y_501_, 0);
lean_inc(v_a_502_);
lean_dec_ref_known(v___y_501_, 1);
v_ref_503_ = lean_ctor_get(v___y_491_, 5);
if (v_isShared_498_ == 0)
{
lean_ctor_set_tag(v___x_497_, 1);
v___x_505_ = v___x_497_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_a_495_);
v___x_505_ = v_reuseFailAlloc_522_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
lean_object* v___x_506_; 
v___x_506_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_502_, v___x_499_, v___x_505_, v___y_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_, v___y_492_);
if (lean_obj_tag(v___x_506_) == 0)
{
lean_object* v_a_507_; uint8_t v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___f_520_; lean_object* v___x_521_; 
v_a_507_ = lean_ctor_get(v___x_506_, 0);
lean_inc(v_a_507_);
lean_dec_ref_known(v___x_506_, 1);
v___x_508_ = 0;
v___x_509_ = l_Lean_SourceInfo_fromRef(v_ref_503_, v___x_508_);
v___x_510_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_511_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_512_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_513_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_502_);
lean_dec(v_a_502_);
v___x_514_ = l_Array_append___redArg(v___x_512_, v___x_513_);
lean_dec_ref(v___x_513_);
lean_inc_n(v___x_509_, 2);
v___x_515_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_515_, 0, v___x_509_);
lean_ctor_set(v___x_515_, 1, v___x_511_);
lean_ctor_set(v___x_515_, 2, v___x_514_);
v___x_516_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_517_ = l_Lean_Syntax_node1(v___x_509_, v___x_516_, v___x_515_);
v___x_518_ = l_Lean_Syntax_node1(v___x_509_, v___x_510_, v___x_517_);
v___x_519_ = lean_box(v___x_508_);
v___f_520_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_520_, 0, v___x_519_);
lean_closure_set(v___f_520_, 1, v___x_518_);
lean_closure_set(v___f_520_, 2, v_a_507_);
v___x_521_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_520_, v___y_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_, v___y_492_);
return v___x_521_;
}
else
{
lean_dec(v_a_502_);
return v___x_506_;
}
}
}
else
{
lean_object* v_a_523_; lean_object* v___x_525_; uint8_t v_isShared_526_; uint8_t v_isSharedCheck_530_; 
lean_del_object(v___x_497_);
lean_dec(v_a_495_);
v_a_523_ = lean_ctor_get(v___y_501_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v___y_501_);
if (v_isSharedCheck_530_ == 0)
{
v___x_525_ = v___y_501_;
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
else
{
lean_inc(v_a_523_);
lean_dec(v___y_501_);
v___x_525_ = lean_box(0);
v_isShared_526_ = v_isSharedCheck_530_;
goto v_resetjp_524_;
}
v_resetjp_524_:
{
lean_object* v___x_528_; 
if (v_isShared_526_ == 0)
{
v___x_528_ = v___x_525_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v_a_523_);
v___x_528_ = v_reuseFailAlloc_529_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
return v___x_528_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__1___boxed(lean_object* v___f_543_, lean_object* v___f_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___lam__1(v___f_543_, v___f_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_, v___y_550_);
lean_dec(v___y_550_);
lean_dec_ref(v___y_549_);
lean_dec(v___y_548_);
lean_dec_ref(v___y_547_);
lean_dec(v___y_546_);
lean_dec_ref(v___y_545_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1(lean_object* v_a_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_, lean_object* v_a_568_){
_start:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_570_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_571_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___closed__3));
v___x_572_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_570_, v___x_571_, v_a_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_, v_a_568_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1___boxed(lean_object* v_a_573_, lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a05___x2c____1(v_a_573_, v_a_574_, v_a_575_, v_a_576_, v_a_577_, v_a_578_);
lean_dec(v_a_578_);
lean_dec_ref(v_a_577_);
lean_dec(v_a_576_);
lean_dec_ref(v_a_575_);
lean_dec(v_a_574_);
lean_dec_ref(v_a_573_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__0(lean_object* v_a_606_, uint8_t v_a_607_, uint8_t v_a_608_, uint8_t v___x_609_, lean_object* v_x_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_){
_start:
{
lean_object* v___x_618_; 
v___x_618_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_611_, v___y_612_, v___y_613_, v___y_614_, v___y_615_, v___y_616_);
if (lean_obj_tag(v___x_618_) == 0)
{
lean_object* v_a_619_; lean_object* v___x_621_; uint8_t v_isShared_622_; uint8_t v_isSharedCheck_712_; 
v_a_619_ = lean_ctor_get(v___x_618_, 0);
v_isSharedCheck_712_ = !lean_is_exclusive(v___x_618_);
if (v_isSharedCheck_712_ == 0)
{
v___x_621_ = v___x_618_;
v_isShared_622_ = v_isSharedCheck_712_;
goto v_resetjp_620_;
}
else
{
lean_inc(v_a_619_);
lean_dec(v___x_618_);
v___x_621_ = lean_box(0);
v_isShared_622_ = v_isSharedCheck_712_;
goto v_resetjp_620_;
}
v_resetjp_620_:
{
uint8_t v___y_624_; uint8_t v___y_658_; 
if (v_a_607_ == 0)
{
v___y_658_ = v_a_607_;
goto v___jp_657_;
}
else
{
if (v___x_609_ == 0)
{
lean_object* v_ref_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
lean_del_object(v___x_621_);
lean_dec(v_x_610_);
v_ref_677_ = lean_ctor_get(v___y_615_, 5);
v___x_678_ = l_Lean_SourceInfo_fromRef(v_ref_677_, v___x_609_);
v___x_679_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
v___x_680_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__4));
lean_inc_n(v___x_678_, 15);
v___x_681_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_678_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
v___x_682_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_683_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_684_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_685_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_686_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_687_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_687_, 0, v___x_678_);
lean_ctor_set(v___x_687_, 1, v___x_686_);
v___x_688_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_689_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_690_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_691_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__11));
v___x_692_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_692_, 0, v___x_678_);
lean_ctor_set(v___x_692_, 1, v___x_691_);
v___x_693_ = l_Lean_Syntax_node1(v___x_678_, v___x_690_, v___x_692_);
v___x_694_ = l_Lean_Syntax_node1(v___x_678_, v___x_689_, v___x_693_);
v___x_695_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_696_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_697_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_697_, 0, v___x_678_);
lean_ctor_set(v___x_697_, 1, v___x_696_);
v___x_698_ = l_Lean_Syntax_node2(v___x_678_, v___x_695_, v___x_697_, v_a_606_);
v___x_699_ = l_Lean_Syntax_node1(v___x_678_, v___x_684_, v___x_698_);
v___x_700_ = l_Lean_Syntax_node2(v___x_678_, v___x_688_, v___x_694_, v___x_699_);
v___x_701_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_702_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_702_, 0, v___x_678_);
lean_ctor_set(v___x_702_, 1, v___x_701_);
v___x_703_ = l_Lean_Syntax_node3(v___x_678_, v___x_685_, v___x_687_, v___x_700_, v___x_702_);
v___x_704_ = l_Lean_Syntax_node1(v___x_678_, v___x_684_, v___x_703_);
v___x_705_ = l_Lean_Syntax_node1(v___x_678_, v___x_683_, v___x_704_);
v___x_706_ = l_Lean_Syntax_node1(v___x_678_, v___x_682_, v___x_705_);
v___x_707_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_708_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_708_, 0, v___x_678_);
lean_ctor_set(v___x_708_, 1, v___x_707_);
v___x_709_ = l_Lean_Syntax_node4(v___x_678_, v___x_679_, v___x_681_, v___x_706_, v___x_708_, v_a_619_);
v___x_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
return v___x_710_;
}
else
{
uint8_t v___x_711_; 
v___x_711_ = 0;
v___y_658_ = v___x_711_;
goto v___jp_657_;
}
}
v___jp_623_:
{
lean_object* v_ref_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_655_; 
v_ref_625_ = lean_ctor_get(v___y_615_, 5);
v___x_626_ = l_Lean_SourceInfo_fromRef(v_ref_625_, v___y_624_);
v___x_627_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
v___x_628_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__4));
lean_inc_n(v___x_626_, 13);
v___x_629_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_629_, 0, v___x_626_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
v___x_630_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_631_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_632_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_633_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_634_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_635_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_635_, 0, v___x_626_);
lean_ctor_set(v___x_635_, 1, v___x_634_);
v___x_636_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_637_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_638_ = l_Lean_Syntax_node1(v___x_626_, v___x_637_, v_x_610_);
v___x_639_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_640_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_641_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_641_, 0, v___x_626_);
lean_ctor_set(v___x_641_, 1, v___x_640_);
v___x_642_ = l_Lean_Syntax_node2(v___x_626_, v___x_639_, v___x_641_, v_a_606_);
v___x_643_ = l_Lean_Syntax_node1(v___x_626_, v___x_632_, v___x_642_);
v___x_644_ = l_Lean_Syntax_node2(v___x_626_, v___x_636_, v___x_638_, v___x_643_);
v___x_645_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_646_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_646_, 0, v___x_626_);
lean_ctor_set(v___x_646_, 1, v___x_645_);
v___x_647_ = l_Lean_Syntax_node3(v___x_626_, v___x_633_, v___x_635_, v___x_644_, v___x_646_);
v___x_648_ = l_Lean_Syntax_node1(v___x_626_, v___x_632_, v___x_647_);
v___x_649_ = l_Lean_Syntax_node1(v___x_626_, v___x_631_, v___x_648_);
v___x_650_ = l_Lean_Syntax_node1(v___x_626_, v___x_630_, v___x_649_);
v___x_651_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_652_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_652_, 0, v___x_626_);
lean_ctor_set(v___x_652_, 1, v___x_651_);
v___x_653_ = l_Lean_Syntax_node4(v___x_626_, v___x_627_, v___x_629_, v___x_650_, v___x_652_, v_a_619_);
if (v_isShared_622_ == 0)
{
lean_ctor_set(v___x_621_, 0, v___x_653_);
v___x_655_ = v___x_621_;
goto v_reusejp_654_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v___x_653_);
v___x_655_ = v_reuseFailAlloc_656_;
goto v_reusejp_654_;
}
v_reusejp_654_:
{
return v___x_655_;
}
}
v___jp_657_:
{
if (v_a_607_ == 0)
{
if (v_a_608_ == 0)
{
lean_object* v_ref_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
lean_del_object(v___x_621_);
lean_dec(v_a_606_);
v_ref_659_ = lean_ctor_get(v___y_615_, 5);
v___x_660_ = l_Lean_SourceInfo_fromRef(v_ref_659_, v_a_608_);
v___x_661_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
v___x_662_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__4));
lean_inc_n(v___x_660_, 6);
v___x_663_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_663_, 0, v___x_660_);
lean_ctor_set(v___x_663_, 1, v___x_662_);
v___x_664_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_665_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_666_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_667_ = l_Lean_Syntax_node1(v___x_660_, v___x_666_, v_x_610_);
v___x_668_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_669_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_670_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_670_, 0, v___x_660_);
lean_ctor_set(v___x_670_, 1, v___x_668_);
lean_ctor_set(v___x_670_, 2, v___x_669_);
v___x_671_ = l_Lean_Syntax_node2(v___x_660_, v___x_665_, v___x_667_, v___x_670_);
v___x_672_ = l_Lean_Syntax_node1(v___x_660_, v___x_664_, v___x_671_);
v___x_673_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_674_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_674_, 0, v___x_660_);
lean_ctor_set(v___x_674_, 1, v___x_673_);
v___x_675_ = l_Lean_Syntax_node4(v___x_660_, v___x_661_, v___x_663_, v___x_672_, v___x_674_, v_a_619_);
v___x_676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_676_, 0, v___x_675_);
return v___x_676_;
}
else
{
v___y_624_ = v___y_658_;
goto v___jp_623_;
}
}
else
{
v___y_624_ = v___y_658_;
goto v___jp_623_;
}
}
}
}
else
{
lean_dec(v_x_610_);
lean_dec(v_a_606_);
return v___x_618_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__0___boxed(lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_, lean_object* v___x_716_, lean_object* v_x_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_){
_start:
{
uint8_t v_a_69410__boxed_725_; uint8_t v_a_69411__boxed_726_; uint8_t v___x_69412__boxed_727_; lean_object* v_res_728_; 
v_a_69410__boxed_725_ = lean_unbox(v_a_714_);
v_a_69411__boxed_726_ = lean_unbox(v_a_715_);
v___x_69412__boxed_727_ = lean_unbox(v___x_716_);
v_res_728_ = lp_mathlib_iSup__delab___lam__0(v_a_713_, v_a_69410__boxed_725_, v_a_69411__boxed_726_, v___x_69412__boxed_727_, v_x_717_, v___y_718_, v___y_719_, v___y_720_, v___y_721_, v___y_722_, v___y_723_);
lean_dec(v___y_723_);
lean_dec_ref(v___y_722_);
lean_dec(v___y_721_);
lean_dec_ref(v___y_720_);
lean_dec(v___y_719_);
lean_dec_ref(v___y_718_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg(lean_object* v_child_729_, lean_object* v_childIdx_730_, lean_object* v_x_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_){
_start:
{
lean_object* v_subExpr_739_; lean_object* v_optionsPerPos_740_; lean_object* v_currNamespace_741_; lean_object* v_openDecls_742_; uint8_t v_inPattern_743_; lean_object* v_depth_744_; lean_object* v_lctxInitIndices_745_; lean_object* v_pos_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; 
v_subExpr_739_ = lean_ctor_get(v___y_732_, 3);
v_optionsPerPos_740_ = lean_ctor_get(v___y_732_, 0);
v_currNamespace_741_ = lean_ctor_get(v___y_732_, 1);
v_openDecls_742_ = lean_ctor_get(v___y_732_, 2);
v_inPattern_743_ = lean_ctor_get_uint8(v___y_732_, sizeof(void*)*6);
v_depth_744_ = lean_ctor_get(v___y_732_, 4);
v_lctxInitIndices_745_ = lean_ctor_get(v___y_732_, 5);
v_pos_746_ = lean_ctor_get(v_subExpr_739_, 1);
v___x_747_ = l_Lean_SubExpr_Pos_push(v_pos_746_, v_childIdx_730_);
v___x_748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_748_, 0, v_child_729_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
lean_inc(v_lctxInitIndices_745_);
lean_inc(v_depth_744_);
lean_inc(v_openDecls_742_);
lean_inc(v_currNamespace_741_);
lean_inc(v_optionsPerPos_740_);
v___x_749_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_749_, 0, v_optionsPerPos_740_);
lean_ctor_set(v___x_749_, 1, v_currNamespace_741_);
lean_ctor_set(v___x_749_, 2, v_openDecls_742_);
lean_ctor_set(v___x_749_, 3, v___x_748_);
lean_ctor_set(v___x_749_, 4, v_depth_744_);
lean_ctor_set(v___x_749_, 5, v_lctxInitIndices_745_);
lean_ctor_set_uint8(v___x_749_, sizeof(void*)*6, v_inPattern_743_);
lean_inc(v___y_737_);
lean_inc_ref(v___y_736_);
lean_inc(v___y_735_);
lean_inc_ref(v___y_734_);
lean_inc(v___y_733_);
v___x_750_ = lean_apply_7(v_x_731_, v___x_749_, v___y_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_, lean_box(0));
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg___boxed(lean_object* v_child_751_, lean_object* v_childIdx_752_, lean_object* v_x_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg(v_child_751_, v_childIdx_752_, v_x_753_, v___y_754_, v___y_755_, v___y_756_, v___y_757_, v___y_758_, v___y_759_);
lean_dec(v___y_759_);
lean_dec_ref(v___y_758_);
lean_dec(v___y_757_);
lean_dec_ref(v___y_756_);
lean_dec(v___y_755_);
lean_dec_ref(v___y_754_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(lean_object* v_x_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
lean_object* v___x_770_; lean_object* v_a_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_770_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_763_);
v_a_771_ = lean_ctor_get(v___x_770_, 0);
lean_inc(v_a_771_);
lean_dec_ref(v___x_770_);
v___x_772_ = l_Lean_Expr_bindingDomain_x21(v_a_771_);
lean_dec(v_a_771_);
v___x_773_ = lean_unsigned_to_nat(0u);
v___x_774_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg(v___x_772_, v___x_773_, v_x_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg___boxed(lean_object* v_x_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_){
_start:
{
lean_object* v_res_783_; 
v_res_783_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(v_x_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_);
lean_dec(v___y_781_);
lean_dec_ref(v___y_780_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
lean_dec(v___y_777_);
lean_dec_ref(v___y_776_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__1(lean_object* v___x_784_, uint8_t v_a_785_, uint8_t v_a_786_, uint8_t v___x_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v___x_795_; 
v___x_795_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(v___x_784_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_);
if (lean_obj_tag(v___x_795_) == 0)
{
lean_object* v_a_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___f_800_; uint8_t v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; 
v_a_796_ = lean_ctor_get(v___x_795_, 0);
lean_inc(v_a_796_);
lean_dec_ref_known(v___x_795_, 1);
v___x_797_ = lean_box(v_a_785_);
v___x_798_ = lean_box(v_a_786_);
v___x_799_ = lean_box(v___x_787_);
v___f_800_ = lean_alloc_closure((void*)(lp_mathlib_iSup__delab___lam__0___boxed), 12, 4);
lean_closure_set(v___f_800_, 0, v_a_796_);
lean_closure_set(v___f_800_, 1, v___x_797_);
lean_closure_set(v___f_800_, 2, v___x_798_);
lean_closure_set(v___f_800_, 3, v___x_799_);
v___x_801_ = 0;
v___x_802_ = l_Lean_NameSet_empty;
v___x_803_ = l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(v___f_800_, v___x_801_, v___x_802_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_, v___y_793_);
return v___x_803_;
}
else
{
return v___x_795_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__1___boxed(lean_object* v___x_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v___x_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
uint8_t v_a_69714__boxed_815_; uint8_t v_a_69715__boxed_816_; uint8_t v___x_69716__boxed_817_; lean_object* v_res_818_; 
v_a_69714__boxed_815_ = lean_unbox(v_a_805_);
v_a_69715__boxed_816_ = lean_unbox(v_a_806_);
v___x_69716__boxed_817_ = lean_unbox(v___x_807_);
v_res_818_ = lp_mathlib_iSup__delab___lam__1(v___x_804_, v_a_69714__boxed_815_, v_a_69715__boxed_816_, v___x_69716__boxed_817_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_, v___y_813_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
lean_dec(v___y_811_);
lean_dec_ref(v___y_810_);
lean_dec(v___y_809_);
lean_dec_ref(v___y_808_);
return v_res_818_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(lean_object* v_x_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v___x_827_; lean_object* v_a_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; 
v___x_827_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_820_);
v_a_828_ = lean_ctor_get(v___x_827_, 0);
lean_inc(v_a_828_);
lean_dec_ref(v___x_827_);
v___x_829_ = l_Lean_Expr_appArg_x21(v_a_828_);
lean_dec(v_a_828_);
v___x_830_ = lean_unsigned_to_nat(1u);
v___x_831_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg(v___x_829_, v___x_830_, v_x_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_, v___y_825_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg___boxed(lean_object* v_x_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(v_x_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_, v___y_837_, v___y_838_);
lean_dec(v___y_838_);
lean_dec_ref(v___y_837_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
lean_dec(v___y_834_);
lean_dec_ref(v___y_833_);
return v_res_840_;
}
}
static lean_object* _init_lp_mathlib_iSup__delab___lam__2___closed__0(void){
_start:
{
lean_object* v___x_841_; lean_object* v_dummy_842_; 
v___x_841_ = lean_box(0);
v_dummy_842_ = l_Lean_Expr_sort___override(v___x_841_);
return v_dummy_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__2(lean_object* v___x_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_){
_start:
{
lean_object* v___x_864_; lean_object* v_a_865_; lean_object* v_dummy_866_; lean_object* v_nargs_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; uint8_t v___x_873_; 
v___x_864_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_857_);
v_a_865_ = lean_ctor_get(v___x_864_, 0);
lean_inc(v_a_865_);
lean_dec_ref(v___x_864_);
v_dummy_866_ = lean_obj_once(&lp_mathlib_iSup__delab___lam__2___closed__0, &lp_mathlib_iSup__delab___lam__2___closed__0_once, _init_lp_mathlib_iSup__delab___lam__2___closed__0);
v_nargs_867_ = l_Lean_Expr_getAppNumArgs(v_a_865_);
lean_inc(v_nargs_867_);
v___x_868_ = lean_mk_array(v_nargs_867_, v_dummy_866_);
v___x_869_ = lean_unsigned_to_nat(1u);
v___x_870_ = lean_nat_sub(v_nargs_867_, v___x_869_);
lean_dec(v_nargs_867_);
v___x_871_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_865_, v___x_868_, v___x_870_);
v___x_872_ = lean_array_get_size(v___x_871_);
v___x_873_ = lean_nat_dec_eq(v___x_872_, v___x_856_);
if (v___x_873_ == 0)
{
lean_object* v___x_874_; 
lean_dec_ref(v___x_871_);
v___x_874_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_874_;
}
else
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v___y_879_; lean_object* v___y_880_; lean_object* v___y_881_; lean_object* v___y_882_; lean_object* v___y_883_; lean_object* v___y_884_; uint8_t v___x_1062_; 
v___x_875_ = lean_array_fget(v___x_871_, v___x_869_);
v___x_876_ = lean_unsigned_to_nat(3u);
v___x_877_ = lean_array_fget(v___x_871_, v___x_876_);
lean_dec_ref(v___x_871_);
v___x_1062_ = l_Lean_Expr_isLambda(v___x_877_);
if (v___x_1062_ == 0)
{
lean_object* v___x_1063_; 
v___x_1063_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1063_) == 0)
{
lean_dec_ref_known(v___x_1063_, 1);
v___y_879_ = v___y_857_;
v___y_880_ = v___y_858_;
v___y_881_ = v___y_859_;
v___y_882_ = v___y_860_;
v___y_883_ = v___y_861_;
v___y_884_ = v___y_862_;
goto v___jp_878_;
}
else
{
lean_object* v_a_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1071_; 
lean_dec(v___x_877_);
lean_dec(v___x_875_);
v_a_1064_ = lean_ctor_get(v___x_1063_, 0);
v_isSharedCheck_1071_ = !lean_is_exclusive(v___x_1063_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_1066_ = v___x_1063_;
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_a_1064_);
lean_dec(v___x_1063_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1069_; 
if (v_isShared_1067_ == 0)
{
v___x_1069_ = v___x_1066_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v_a_1064_);
v___x_1069_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
return v___x_1069_;
}
}
}
}
else
{
v___y_879_ = v___y_857_;
v___y_880_ = v___y_858_;
v___y_881_ = v___y_859_;
v___y_882_ = v___y_860_;
v___y_883_ = v___y_861_;
v___y_884_ = v___y_862_;
goto v___jp_878_;
}
v___jp_878_:
{
lean_object* v___x_885_; 
v___x_885_ = l_Lean_Meta_isProp(v___x_875_, v___y_881_, v___y_882_, v___y_883_, v___y_884_);
if (lean_obj_tag(v___x_885_) == 0)
{
lean_object* v_a_886_; lean_object* v___x_887_; lean_object* v___x_888_; 
v_a_886_ = lean_ctor_get(v___x_885_, 0);
lean_inc(v_a_886_);
lean_dec_ref_known(v___x_885_, 1);
v___x_887_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__1));
v___x_888_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_887_, v___y_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_);
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v_a_889_; lean_object* v___x_890_; lean_object* v___x_891_; uint8_t v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___f_895_; lean_object* v___x_896_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc(v_a_889_);
lean_dec_ref_known(v___x_888_, 1);
v___x_890_ = l_Lean_Expr_bindingBody_x21(v___x_877_);
lean_dec(v___x_877_);
v___x_891_ = lean_unsigned_to_nat(0u);
v___x_892_ = lean_expr_has_loose_bvar(v___x_890_, v___x_891_);
lean_dec_ref(v___x_890_);
v___x_893_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__2));
v___x_894_ = lean_box(v___x_892_);
v___f_895_ = lean_alloc_closure((void*)(lp_mathlib_iSup__delab___lam__1___boxed), 11, 4);
lean_closure_set(v___f_895_, 0, v___x_893_);
lean_closure_set(v___f_895_, 1, v_a_886_);
lean_closure_set(v___f_895_, 2, v_a_889_);
lean_closure_set(v___f_895_, 3, v___x_894_);
v___x_896_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(v___f_895_, v___y_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_);
if (lean_obj_tag(v___x_896_) == 0)
{
lean_object* v_a_897_; lean_object* v___x_898_; uint8_t v___x_899_; 
v_a_897_ = lean_ctor_get(v___x_896_, 0);
lean_inc_n(v_a_897_, 2);
v___x_898_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__1));
v___x_899_ = l_Lean_Syntax_isOfKind(v_a_897_, v___x_898_);
if (v___x_899_ == 0)
{
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_900_; lean_object* v___x_901_; uint8_t v___x_902_; 
v___x_900_ = l_Lean_Syntax_getArg(v_a_897_, v___x_869_);
v___x_901_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
lean_inc(v___x_900_);
v___x_902_ = l_Lean_Syntax_isOfKind(v___x_900_, v___x_901_);
if (v___x_902_ == 0)
{
lean_dec(v___x_900_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_903_; lean_object* v___x_904_; uint8_t v___x_905_; 
v___x_903_ = l_Lean_Syntax_getArg(v___x_900_, v___x_891_);
lean_dec(v___x_900_);
v___x_904_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
lean_inc(v___x_903_);
v___x_905_ = l_Lean_Syntax_isOfKind(v___x_903_, v___x_904_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; uint8_t v___x_907_; 
v___x_906_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_903_);
v___x_907_ = l_Lean_Syntax_isOfKind(v___x_903_, v___x_906_);
if (v___x_907_ == 0)
{
lean_dec(v___x_903_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_908_; uint8_t v___x_909_; 
v___x_908_ = l_Lean_Syntax_getArg(v___x_903_, v___x_891_);
lean_dec(v___x_903_);
lean_inc(v___x_908_);
v___x_909_ = l_Lean_Syntax_matchesNull(v___x_908_, v___x_869_);
if (v___x_909_ == 0)
{
lean_dec(v___x_908_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_910_; lean_object* v___x_911_; uint8_t v___x_912_; 
v___x_910_ = l_Lean_Syntax_getArg(v___x_908_, v___x_891_);
lean_dec(v___x_908_);
v___x_911_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_910_);
v___x_912_ = l_Lean_Syntax_isOfKind(v___x_910_, v___x_911_);
if (v___x_912_ == 0)
{
lean_dec(v___x_910_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_913_; uint8_t v___x_914_; 
v___x_913_ = l_Lean_Syntax_getArg(v___x_910_, v___x_869_);
lean_dec(v___x_910_);
lean_inc(v___x_913_);
v___x_914_ = l_Lean_Syntax_isOfKind(v___x_913_, v___x_904_);
if (v___x_914_ == 0)
{
lean_dec(v___x_913_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_915_; lean_object* v___x_916_; uint8_t v___x_917_; 
v___x_915_ = l_Lean_Syntax_getArg(v___x_913_, v___x_891_);
v___x_916_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_915_);
v___x_917_ = l_Lean_Syntax_isOfKind(v___x_915_, v___x_916_);
if (v___x_917_ == 0)
{
lean_dec(v___x_915_);
lean_dec(v___x_913_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_918_; lean_object* v___x_919_; uint8_t v___x_920_; 
v___x_918_ = l_Lean_Syntax_getArg(v___x_915_, v___x_891_);
lean_dec(v___x_915_);
v___x_919_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_918_);
v___x_920_ = l_Lean_Syntax_isOfKind(v___x_918_, v___x_919_);
if (v___x_920_ == 0)
{
lean_dec(v___x_918_);
lean_dec(v___x_913_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_921_; uint8_t v___x_922_; 
v___x_921_ = l_Lean_Syntax_getArg(v___x_913_, v___x_869_);
lean_dec(v___x_913_);
lean_inc(v___x_921_);
v___x_922_ = l_Lean_Syntax_matchesNull(v___x_921_, v___x_869_);
if (v___x_922_ == 0)
{
lean_dec(v___x_921_);
lean_dec(v___x_918_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_923_; lean_object* v___x_924_; uint8_t v___x_925_; 
v___x_923_ = l_Lean_Syntax_getArg(v___x_921_, v___x_891_);
lean_dec(v___x_921_);
v___x_924_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_925_ = l_Lean_Syntax_isOfKind(v___x_923_, v___x_924_);
if (v___x_925_ == 0)
{
lean_dec(v___x_918_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_926_; uint8_t v___x_927_; 
v___x_926_ = l_Lean_Syntax_getArg(v_a_897_, v___x_876_);
lean_dec(v_a_897_);
lean_inc(v___x_926_);
v___x_927_ = l_Lean_Syntax_isOfKind(v___x_926_, v___x_898_);
if (v___x_927_ == 0)
{
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_928_; uint8_t v___x_929_; 
v___x_928_ = l_Lean_Syntax_getArg(v___x_926_, v___x_869_);
lean_inc(v___x_928_);
v___x_929_ = l_Lean_Syntax_isOfKind(v___x_928_, v___x_901_);
if (v___x_929_ == 0)
{
lean_dec(v___x_928_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_930_; uint8_t v___x_931_; 
v___x_930_ = l_Lean_Syntax_getArg(v___x_928_, v___x_891_);
lean_dec(v___x_928_);
lean_inc(v___x_930_);
v___x_931_ = l_Lean_Syntax_isOfKind(v___x_930_, v___x_906_);
if (v___x_931_ == 0)
{
lean_dec(v___x_930_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_932_; uint8_t v___x_933_; 
v___x_932_ = l_Lean_Syntax_getArg(v___x_930_, v___x_891_);
lean_dec(v___x_930_);
lean_inc(v___x_932_);
v___x_933_ = l_Lean_Syntax_matchesNull(v___x_932_, v___x_869_);
if (v___x_933_ == 0)
{
lean_dec(v___x_932_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_934_; uint8_t v___x_935_; 
v___x_934_ = l_Lean_Syntax_getArg(v___x_932_, v___x_891_);
lean_dec(v___x_932_);
lean_inc(v___x_934_);
v___x_935_ = l_Lean_Syntax_isOfKind(v___x_934_, v___x_911_);
if (v___x_935_ == 0)
{
lean_dec(v___x_934_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_936_; uint8_t v___x_937_; 
v___x_936_ = l_Lean_Syntax_getArg(v___x_934_, v___x_869_);
lean_dec(v___x_934_);
lean_inc(v___x_936_);
v___x_937_ = l_Lean_Syntax_isOfKind(v___x_936_, v___x_904_);
if (v___x_937_ == 0)
{
lean_dec(v___x_936_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_938_; uint8_t v___x_939_; 
v___x_938_ = l_Lean_Syntax_getArg(v___x_936_, v___x_891_);
lean_inc(v___x_938_);
v___x_939_ = l_Lean_Syntax_isOfKind(v___x_938_, v___x_916_);
if (v___x_939_ == 0)
{
lean_dec(v___x_938_);
lean_dec(v___x_936_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_940_; lean_object* v___x_941_; uint8_t v___x_942_; 
v___x_940_ = l_Lean_Syntax_getArg(v___x_938_, v___x_891_);
lean_dec(v___x_938_);
v___x_941_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_942_ = l_Lean_Syntax_isOfKind(v___x_940_, v___x_941_);
if (v___x_942_ == 0)
{
lean_dec(v___x_936_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_943_; uint8_t v___x_944_; 
v___x_943_ = l_Lean_Syntax_getArg(v___x_936_, v___x_869_);
lean_dec(v___x_936_);
lean_inc(v___x_943_);
v___x_944_ = l_Lean_Syntax_matchesNull(v___x_943_, v___x_869_);
if (v___x_944_ == 0)
{
lean_dec(v___x_943_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_945_; uint8_t v___x_946_; 
v___x_945_ = l_Lean_Syntax_getArg(v___x_943_, v___x_891_);
lean_dec(v___x_943_);
lean_inc(v___x_945_);
v___x_946_ = l_Lean_Syntax_isOfKind(v___x_945_, v___x_924_);
if (v___x_946_ == 0)
{
lean_dec(v___x_945_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_947_; lean_object* v___x_948_; uint8_t v___x_949_; 
v___x_947_ = l_Lean_Syntax_getArg(v___x_945_, v___x_869_);
lean_dec(v___x_945_);
v___x_948_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_947_);
v___x_949_ = l_Lean_Syntax_isOfKind(v___x_947_, v___x_948_);
if (v___x_949_ == 0)
{
lean_dec(v___x_947_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_950_; uint8_t v___x_951_; 
v___x_950_ = l_Lean_Syntax_getArg(v___x_947_, v___x_891_);
lean_inc(v___x_950_);
v___x_951_ = l_Lean_Syntax_isOfKind(v___x_950_, v___x_919_);
if (v___x_951_ == 0)
{
lean_dec(v___x_950_);
lean_dec(v___x_947_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
uint8_t v___x_952_; 
v___x_952_ = l_Lean_Syntax_structEq(v___x_918_, v___x_950_);
lean_dec(v___x_950_);
if (v___x_952_ == 0)
{
lean_dec(v___x_947_);
lean_dec(v___x_926_);
lean_dec(v___x_918_);
return v___x_896_;
}
else
{
lean_object* v___x_954_; uint8_t v_isShared_955_; uint8_t v_isSharedCheck_978_; 
v_isSharedCheck_978_ = !lean_is_exclusive(v___x_896_);
if (v_isSharedCheck_978_ == 0)
{
lean_object* v_unused_979_; 
v_unused_979_ = lean_ctor_get(v___x_896_, 0);
lean_dec(v_unused_979_);
v___x_954_ = v___x_896_;
v_isShared_955_ = v_isSharedCheck_978_;
goto v_resetjp_953_;
}
else
{
lean_dec(v___x_896_);
v___x_954_ = lean_box(0);
v_isShared_955_ = v_isSharedCheck_978_;
goto v_resetjp_953_;
}
v_resetjp_953_:
{
lean_object* v_ref_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_976_; 
v_ref_956_ = lean_ctor_get(v___y_883_, 5);
v___x_957_ = lean_unsigned_to_nat(2u);
v___x_958_ = l_Lean_Syntax_getArg(v___x_947_, v___x_957_);
lean_dec(v___x_947_);
v___x_959_ = l_Lean_Syntax_getArg(v___x_926_, v___x_876_);
lean_dec(v___x_926_);
v___x_960_ = l_Lean_SourceInfo_fromRef(v_ref_956_, v___x_905_);
v___x_961_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__4));
lean_inc_n(v___x_960_, 8);
v___x_962_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_962_, 0, v___x_960_);
lean_ctor_set(v___x_962_, 1, v___x_961_);
v___x_963_ = l_Lean_Syntax_node1(v___x_960_, v___x_916_, v___x_918_);
v___x_964_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_965_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_966_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_967_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_967_, 0, v___x_960_);
lean_ctor_set(v___x_967_, 1, v___x_966_);
v___x_968_ = l_Lean_Syntax_node2(v___x_960_, v___x_965_, v___x_967_, v___x_958_);
v___x_969_ = l_Lean_Syntax_node1(v___x_960_, v___x_964_, v___x_968_);
v___x_970_ = l_Lean_Syntax_node2(v___x_960_, v___x_904_, v___x_963_, v___x_969_);
v___x_971_ = l_Lean_Syntax_node1(v___x_960_, v___x_901_, v___x_970_);
v___x_972_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_973_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_973_, 0, v___x_960_);
lean_ctor_set(v___x_973_, 1, v___x_972_);
v___x_974_ = l_Lean_Syntax_node4(v___x_960_, v___x_898_, v___x_962_, v___x_971_, v___x_973_, v___x_959_);
if (v_isShared_955_ == 0)
{
lean_ctor_set(v___x_954_, 0, v___x_974_);
v___x_976_ = v___x_954_;
goto v_reusejp_975_;
}
else
{
lean_object* v_reuseFailAlloc_977_; 
v_reuseFailAlloc_977_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_977_, 0, v___x_974_);
v___x_976_ = v_reuseFailAlloc_977_;
goto v_reusejp_975_;
}
v_reusejp_975_:
{
return v___x_976_;
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
}
}
}
}
}
}
}
else
{
lean_object* v___x_980_; lean_object* v___x_981_; uint8_t v___x_982_; 
v___x_980_ = l_Lean_Syntax_getArg(v___x_903_, v___x_891_);
v___x_981_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_980_);
v___x_982_ = l_Lean_Syntax_isOfKind(v___x_980_, v___x_981_);
if (v___x_982_ == 0)
{
lean_dec(v___x_980_);
lean_dec(v___x_903_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_983_; lean_object* v___x_984_; uint8_t v___x_985_; 
v___x_983_ = l_Lean_Syntax_getArg(v___x_980_, v___x_891_);
lean_dec(v___x_980_);
v___x_984_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_983_);
v___x_985_ = l_Lean_Syntax_isOfKind(v___x_983_, v___x_984_);
if (v___x_985_ == 0)
{
lean_dec(v___x_983_);
lean_dec(v___x_903_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_986_; uint8_t v___x_987_; 
v___x_986_ = l_Lean_Syntax_getArg(v___x_903_, v___x_869_);
lean_dec(v___x_903_);
v___x_987_ = l_Lean_Syntax_matchesNull(v___x_986_, v___x_891_);
if (v___x_987_ == 0)
{
lean_dec(v___x_983_);
lean_dec(v_a_897_);
return v___x_896_;
}
else
{
lean_object* v___x_988_; uint8_t v___x_989_; 
v___x_988_ = l_Lean_Syntax_getArg(v_a_897_, v___x_876_);
lean_dec(v_a_897_);
lean_inc(v___x_988_);
v___x_989_ = l_Lean_Syntax_isOfKind(v___x_988_, v___x_898_);
if (v___x_989_ == 0)
{
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_990_; uint8_t v___x_991_; 
v___x_990_ = l_Lean_Syntax_getArg(v___x_988_, v___x_869_);
lean_inc(v___x_990_);
v___x_991_ = l_Lean_Syntax_isOfKind(v___x_990_, v___x_901_);
if (v___x_991_ == 0)
{
lean_dec(v___x_990_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_992_; lean_object* v___x_993_; uint8_t v___x_994_; 
v___x_992_ = l_Lean_Syntax_getArg(v___x_990_, v___x_891_);
lean_dec(v___x_990_);
v___x_993_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_992_);
v___x_994_ = l_Lean_Syntax_isOfKind(v___x_992_, v___x_993_);
if (v___x_994_ == 0)
{
lean_dec(v___x_992_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_995_; uint8_t v___x_996_; 
v___x_995_ = l_Lean_Syntax_getArg(v___x_992_, v___x_891_);
lean_dec(v___x_992_);
lean_inc(v___x_995_);
v___x_996_ = l_Lean_Syntax_matchesNull(v___x_995_, v___x_869_);
if (v___x_996_ == 0)
{
lean_dec(v___x_995_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_997_; lean_object* v___x_998_; uint8_t v___x_999_; 
v___x_997_ = l_Lean_Syntax_getArg(v___x_995_, v___x_891_);
lean_dec(v___x_995_);
v___x_998_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_997_);
v___x_999_ = l_Lean_Syntax_isOfKind(v___x_997_, v___x_998_);
if (v___x_999_ == 0)
{
lean_dec(v___x_997_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1000_; uint8_t v___x_1001_; 
v___x_1000_ = l_Lean_Syntax_getArg(v___x_997_, v___x_869_);
lean_dec(v___x_997_);
lean_inc(v___x_1000_);
v___x_1001_ = l_Lean_Syntax_isOfKind(v___x_1000_, v___x_904_);
if (v___x_1001_ == 0)
{
lean_dec(v___x_1000_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1002_; uint8_t v___x_1003_; 
v___x_1002_ = l_Lean_Syntax_getArg(v___x_1000_, v___x_891_);
lean_inc(v___x_1002_);
v___x_1003_ = l_Lean_Syntax_isOfKind(v___x_1002_, v___x_981_);
if (v___x_1003_ == 0)
{
lean_dec(v___x_1002_);
lean_dec(v___x_1000_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1004_; lean_object* v___x_1005_; uint8_t v___x_1006_; 
v___x_1004_ = l_Lean_Syntax_getArg(v___x_1002_, v___x_891_);
lean_dec(v___x_1002_);
v___x_1005_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_1006_ = l_Lean_Syntax_isOfKind(v___x_1004_, v___x_1005_);
if (v___x_1006_ == 0)
{
lean_dec(v___x_1000_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1007_; uint8_t v___x_1008_; 
v___x_1007_ = l_Lean_Syntax_getArg(v___x_1000_, v___x_869_);
lean_dec(v___x_1000_);
lean_inc(v___x_1007_);
v___x_1008_ = l_Lean_Syntax_matchesNull(v___x_1007_, v___x_869_);
if (v___x_1008_ == 0)
{
lean_dec(v___x_1007_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1009_; lean_object* v___x_1010_; uint8_t v___x_1011_; 
v___x_1009_ = l_Lean_Syntax_getArg(v___x_1007_, v___x_891_);
lean_dec(v___x_1007_);
v___x_1010_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
lean_inc(v___x_1009_);
v___x_1011_ = l_Lean_Syntax_isOfKind(v___x_1009_, v___x_1010_);
if (v___x_1011_ == 0)
{
lean_dec(v___x_1009_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1012_; lean_object* v___x_1013_; uint8_t v___x_1014_; 
v___x_1012_ = l_Lean_Syntax_getArg(v___x_1009_, v___x_869_);
lean_dec(v___x_1009_);
v___x_1013_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_1012_);
v___x_1014_ = l_Lean_Syntax_isOfKind(v___x_1012_, v___x_1013_);
if (v___x_1014_ == 0)
{
lean_dec(v___x_1012_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1015_; uint8_t v___x_1016_; 
v___x_1015_ = l_Lean_Syntax_getArg(v___x_1012_, v___x_891_);
lean_inc(v___x_1015_);
v___x_1016_ = l_Lean_Syntax_isOfKind(v___x_1015_, v___x_984_);
if (v___x_1016_ == 0)
{
lean_dec(v___x_1015_);
lean_dec(v___x_1012_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
uint8_t v___x_1017_; 
v___x_1017_ = l_Lean_Syntax_structEq(v___x_983_, v___x_1015_);
lean_dec(v___x_1015_);
if (v___x_1017_ == 0)
{
lean_dec(v___x_1012_);
lean_dec(v___x_988_);
lean_dec(v___x_983_);
return v___x_896_;
}
else
{
lean_object* v___x_1019_; uint8_t v_isShared_1020_; uint8_t v_isSharedCheck_1044_; 
v_isSharedCheck_1044_ = !lean_is_exclusive(v___x_896_);
if (v_isSharedCheck_1044_ == 0)
{
lean_object* v_unused_1045_; 
v_unused_1045_ = lean_ctor_get(v___x_896_, 0);
lean_dec(v_unused_1045_);
v___x_1019_ = v___x_896_;
v_isShared_1020_ = v_isSharedCheck_1044_;
goto v_resetjp_1018_;
}
else
{
lean_dec(v___x_896_);
v___x_1019_ = lean_box(0);
v_isShared_1020_ = v_isSharedCheck_1044_;
goto v_resetjp_1018_;
}
v_resetjp_1018_:
{
lean_object* v_ref_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; uint8_t v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1042_; 
v_ref_1021_ = lean_ctor_get(v___y_883_, 5);
v___x_1022_ = lean_unsigned_to_nat(2u);
v___x_1023_ = l_Lean_Syntax_getArg(v___x_1012_, v___x_1022_);
lean_dec(v___x_1012_);
v___x_1024_ = l_Lean_Syntax_getArg(v___x_988_, v___x_876_);
lean_dec(v___x_988_);
v___x_1025_ = 0;
v___x_1026_ = l_Lean_SourceInfo_fromRef(v_ref_1021_, v___x_1025_);
v___x_1027_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__4));
lean_inc_n(v___x_1026_, 8);
v___x_1028_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1026_);
lean_ctor_set(v___x_1028_, 1, v___x_1027_);
v___x_1029_ = l_Lean_Syntax_node1(v___x_1026_, v___x_981_, v___x_983_);
v___x_1030_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1031_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_1032_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_1033_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1033_, 0, v___x_1026_);
lean_ctor_set(v___x_1033_, 1, v___x_1032_);
v___x_1034_ = l_Lean_Syntax_node2(v___x_1026_, v___x_1031_, v___x_1033_, v___x_1023_);
v___x_1035_ = l_Lean_Syntax_node1(v___x_1026_, v___x_1030_, v___x_1034_);
v___x_1036_ = l_Lean_Syntax_node2(v___x_1026_, v___x_904_, v___x_1029_, v___x_1035_);
v___x_1037_ = l_Lean_Syntax_node1(v___x_1026_, v___x_901_, v___x_1036_);
v___x_1038_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1039_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1039_, 0, v___x_1026_);
lean_ctor_set(v___x_1039_, 1, v___x_1038_);
v___x_1040_ = l_Lean_Syntax_node4(v___x_1026_, v___x_898_, v___x_1028_, v___x_1037_, v___x_1039_, v___x_1024_);
if (v_isShared_1020_ == 0)
{
lean_ctor_set(v___x_1019_, 0, v___x_1040_);
v___x_1042_ = v___x_1019_;
goto v_reusejp_1041_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v___x_1040_);
v___x_1042_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1041_;
}
v_reusejp_1041_:
{
return v___x_1042_;
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
}
}
}
}
}
else
{
return v___x_896_;
}
}
else
{
lean_object* v_a_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1053_; 
lean_dec(v_a_886_);
lean_dec(v___x_877_);
v_a_1046_ = lean_ctor_get(v___x_888_, 0);
v_isSharedCheck_1053_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_1053_ == 0)
{
v___x_1048_ = v___x_888_;
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_a_1046_);
lean_dec(v___x_888_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1051_; 
if (v_isShared_1049_ == 0)
{
v___x_1051_ = v___x_1048_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v_a_1046_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
}
else
{
lean_object* v_a_1054_; lean_object* v___x_1056_; uint8_t v_isShared_1057_; uint8_t v_isSharedCheck_1061_; 
lean_dec(v___x_877_);
v_a_1054_ = lean_ctor_get(v___x_885_, 0);
v_isSharedCheck_1061_ = !lean_is_exclusive(v___x_885_);
if (v_isSharedCheck_1061_ == 0)
{
v___x_1056_ = v___x_885_;
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
else
{
lean_inc(v_a_1054_);
lean_dec(v___x_885_);
v___x_1056_ = lean_box(0);
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
v_resetjp_1055_:
{
lean_object* v___x_1059_; 
if (v_isShared_1057_ == 0)
{
v___x_1059_ = v___x_1056_;
goto v_reusejp_1058_;
}
else
{
lean_object* v_reuseFailAlloc_1060_; 
v_reuseFailAlloc_1060_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1060_, 0, v_a_1054_);
v___x_1059_ = v_reuseFailAlloc_1060_;
goto v_reusejp_1058_;
}
v_reusejp_1058_:
{
return v___x_1059_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___lam__2___boxed(lean_object* v___x_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_){
_start:
{
lean_object* v_res_1080_; 
v_res_1080_ = lp_mathlib_iSup__delab___lam__2(v___x_1072_, v___y_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_);
lean_dec(v___y_1078_);
lean_dec_ref(v___y_1077_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
lean_dec(v___y_1074_);
lean_dec_ref(v___y_1073_);
lean_dec(v___x_1072_);
return v_res_1080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab(lean_object* v_a_1086_, lean_object* v_a_1087_, lean_object* v_a_1088_, lean_object* v_a_1089_, lean_object* v_a_1090_, lean_object* v_a_1091_){
_start:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
v___x_1093_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_1094_ = ((lean_object*)(lp_mathlib_iSup__delab___closed__1));
v___x_1095_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_1093_, v___x_1094_, v_a_1086_, v_a_1087_, v_a_1088_, v_a_1089_, v_a_1090_, v_a_1091_);
return v___x_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iSup__delab___boxed(lean_object* v_a_1096_, lean_object* v_a_1097_, lean_object* v_a_1098_, lean_object* v_a_1099_, lean_object* v_a_1100_, lean_object* v_a_1101_, lean_object* v_a_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_mathlib_iSup__delab(v_a_1096_, v_a_1097_, v_a_1098_, v_a_1099_, v_a_1100_, v_a_1101_);
lean_dec(v_a_1101_);
lean_dec_ref(v_a_1100_);
lean_dec(v_a_1099_);
lean_dec_ref(v_a_1098_);
lean_dec(v_a_1097_);
lean_dec_ref(v_a_1096_);
return v_res_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0(lean_object* v_00_u03b1_1104_, lean_object* v_child_1105_, lean_object* v_childIdx_1106_, lean_object* v_x_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
lean_object* v___x_1115_; 
v___x_1115_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___redArg(v_child_1105_, v_childIdx_1106_, v_x_1107_, v___y_1108_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_, v___y_1113_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1116_, lean_object* v_child_1117_, lean_object* v_childIdx_1118_, lean_object* v_x_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_){
_start:
{
lean_object* v_res_1127_; 
v_res_1127_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0_spec__0(v_00_u03b1_1116_, v_child_1117_, v_childIdx_1118_, v_x_1119_, v___y_1120_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_, v___y_1125_);
lean_dec(v___y_1125_);
lean_dec_ref(v___y_1124_);
lean_dec(v___y_1123_);
lean_dec_ref(v___y_1122_);
lean_dec(v___y_1121_);
lean_dec_ref(v___y_1120_);
return v_res_1127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0(lean_object* v_00_u03b1_1128_, lean_object* v_x_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_, lean_object* v___y_1132_, lean_object* v___y_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_){
_start:
{
lean_object* v___x_1137_; 
v___x_1137_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(v_x_1129_, v___y_1130_, v___y_1131_, v___y_1132_, v___y_1133_, v___y_1134_, v___y_1135_);
return v___x_1137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___boxed(lean_object* v_00_u03b1_1138_, lean_object* v_x_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v_res_1147_; 
v_res_1147_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0(v_00_u03b1_1138_, v_x_1139_, v___y_1140_, v___y_1141_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_);
lean_dec(v___y_1145_);
lean_dec_ref(v___y_1144_);
lean_dec(v___y_1143_);
lean_dec_ref(v___y_1142_);
lean_dec(v___y_1141_);
lean_dec_ref(v___y_1140_);
return v_res_1147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1(lean_object* v_00_u03b1_1148_, lean_object* v_x_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_){
_start:
{
lean_object* v___x_1157_; 
v___x_1157_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(v_x_1149_, v___y_1150_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_);
return v___x_1157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___boxed(lean_object* v_00_u03b1_1158_, lean_object* v_x_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1(v_00_u03b1_1158_, v_x_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
lean_dec(v___y_1161_);
lean_dec_ref(v___y_1160_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__0(lean_object* v_a_1168_, uint8_t v_a_1169_, uint8_t v_a_1170_, uint8_t v___x_1171_, lean_object* v_x_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_){
_start:
{
lean_object* v___x_1180_; 
v___x_1180_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_);
if (lean_obj_tag(v___x_1180_) == 0)
{
lean_object* v_a_1181_; lean_object* v___x_1183_; uint8_t v_isShared_1184_; uint8_t v_isSharedCheck_1274_; 
v_a_1181_ = lean_ctor_get(v___x_1180_, 0);
v_isSharedCheck_1274_ = !lean_is_exclusive(v___x_1180_);
if (v_isSharedCheck_1274_ == 0)
{
v___x_1183_ = v___x_1180_;
v_isShared_1184_ = v_isSharedCheck_1274_;
goto v_resetjp_1182_;
}
else
{
lean_inc(v_a_1181_);
lean_dec(v___x_1180_);
v___x_1183_ = lean_box(0);
v_isShared_1184_ = v_isSharedCheck_1274_;
goto v_resetjp_1182_;
}
v_resetjp_1182_:
{
uint8_t v___y_1186_; uint8_t v___y_1220_; 
if (v_a_1169_ == 0)
{
v___y_1220_ = v_a_1169_;
goto v___jp_1219_;
}
else
{
if (v___x_1171_ == 0)
{
lean_object* v_ref_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; 
lean_del_object(v___x_1183_);
lean_dec(v_x_1172_);
v_ref_1239_ = lean_ctor_get(v___y_1177_, 5);
v___x_1240_ = l_Lean_SourceInfo_fromRef(v_ref_1239_, v___x_1171_);
v___x_1241_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
v___x_1242_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__2));
lean_inc_n(v___x_1240_, 15);
v___x_1243_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1243_, 0, v___x_1240_);
lean_ctor_set(v___x_1243_, 1, v___x_1242_);
v___x_1244_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_1245_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_1246_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1247_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_1248_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_1249_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1240_);
lean_ctor_set(v___x_1249_, 1, v___x_1248_);
v___x_1250_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_1251_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_1252_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_1253_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__11));
v___x_1254_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1254_, 0, v___x_1240_);
lean_ctor_set(v___x_1254_, 1, v___x_1253_);
v___x_1255_ = l_Lean_Syntax_node1(v___x_1240_, v___x_1252_, v___x_1254_);
v___x_1256_ = l_Lean_Syntax_node1(v___x_1240_, v___x_1251_, v___x_1255_);
v___x_1257_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_1258_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_1259_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1259_, 0, v___x_1240_);
lean_ctor_set(v___x_1259_, 1, v___x_1258_);
v___x_1260_ = l_Lean_Syntax_node2(v___x_1240_, v___x_1257_, v___x_1259_, v_a_1168_);
v___x_1261_ = l_Lean_Syntax_node1(v___x_1240_, v___x_1246_, v___x_1260_);
v___x_1262_ = l_Lean_Syntax_node2(v___x_1240_, v___x_1250_, v___x_1256_, v___x_1261_);
v___x_1263_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_1264_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1240_);
lean_ctor_set(v___x_1264_, 1, v___x_1263_);
v___x_1265_ = l_Lean_Syntax_node3(v___x_1240_, v___x_1247_, v___x_1249_, v___x_1262_, v___x_1264_);
v___x_1266_ = l_Lean_Syntax_node1(v___x_1240_, v___x_1246_, v___x_1265_);
v___x_1267_ = l_Lean_Syntax_node1(v___x_1240_, v___x_1245_, v___x_1266_);
v___x_1268_ = l_Lean_Syntax_node1(v___x_1240_, v___x_1244_, v___x_1267_);
v___x_1269_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1270_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1240_);
lean_ctor_set(v___x_1270_, 1, v___x_1269_);
v___x_1271_ = l_Lean_Syntax_node4(v___x_1240_, v___x_1241_, v___x_1243_, v___x_1268_, v___x_1270_, v_a_1181_);
v___x_1272_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1271_);
return v___x_1272_;
}
else
{
uint8_t v___x_1273_; 
v___x_1273_ = 0;
v___y_1220_ = v___x_1273_;
goto v___jp_1219_;
}
}
v___jp_1185_:
{
lean_object* v_ref_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1217_; 
v_ref_1187_ = lean_ctor_get(v___y_1177_, 5);
v___x_1188_ = l_Lean_SourceInfo_fromRef(v_ref_1187_, v___y_1186_);
v___x_1189_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
v___x_1190_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__2));
lean_inc_n(v___x_1188_, 13);
v___x_1191_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1188_);
lean_ctor_set(v___x_1191_, 1, v___x_1190_);
v___x_1192_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_1193_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_1194_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1195_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_1196_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_1197_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1188_);
lean_ctor_set(v___x_1197_, 1, v___x_1196_);
v___x_1198_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_1199_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_1200_ = l_Lean_Syntax_node1(v___x_1188_, v___x_1199_, v_x_1172_);
v___x_1201_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_1202_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_1203_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1188_);
lean_ctor_set(v___x_1203_, 1, v___x_1202_);
v___x_1204_ = l_Lean_Syntax_node2(v___x_1188_, v___x_1201_, v___x_1203_, v_a_1168_);
v___x_1205_ = l_Lean_Syntax_node1(v___x_1188_, v___x_1194_, v___x_1204_);
v___x_1206_ = l_Lean_Syntax_node2(v___x_1188_, v___x_1198_, v___x_1200_, v___x_1205_);
v___x_1207_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_1208_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1208_, 0, v___x_1188_);
lean_ctor_set(v___x_1208_, 1, v___x_1207_);
v___x_1209_ = l_Lean_Syntax_node3(v___x_1188_, v___x_1195_, v___x_1197_, v___x_1206_, v___x_1208_);
v___x_1210_ = l_Lean_Syntax_node1(v___x_1188_, v___x_1194_, v___x_1209_);
v___x_1211_ = l_Lean_Syntax_node1(v___x_1188_, v___x_1193_, v___x_1210_);
v___x_1212_ = l_Lean_Syntax_node1(v___x_1188_, v___x_1192_, v___x_1211_);
v___x_1213_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1214_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1214_, 0, v___x_1188_);
lean_ctor_set(v___x_1214_, 1, v___x_1213_);
v___x_1215_ = l_Lean_Syntax_node4(v___x_1188_, v___x_1189_, v___x_1191_, v___x_1212_, v___x_1214_, v_a_1181_);
if (v_isShared_1184_ == 0)
{
lean_ctor_set(v___x_1183_, 0, v___x_1215_);
v___x_1217_ = v___x_1183_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v___x_1215_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
}
}
v___jp_1219_:
{
if (v_a_1169_ == 0)
{
if (v_a_1170_ == 0)
{
lean_object* v_ref_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; 
lean_del_object(v___x_1183_);
lean_dec(v_a_1168_);
v_ref_1221_ = lean_ctor_get(v___y_1177_, 5);
v___x_1222_ = l_Lean_SourceInfo_fromRef(v_ref_1221_, v_a_1170_);
v___x_1223_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
v___x_1224_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__2));
lean_inc_n(v___x_1222_, 6);
v___x_1225_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1225_, 0, v___x_1222_);
lean_ctor_set(v___x_1225_, 1, v___x_1224_);
v___x_1226_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_1227_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_1228_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_1229_ = l_Lean_Syntax_node1(v___x_1222_, v___x_1228_, v_x_1172_);
v___x_1230_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1231_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_1232_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1232_, 0, v___x_1222_);
lean_ctor_set(v___x_1232_, 1, v___x_1230_);
lean_ctor_set(v___x_1232_, 2, v___x_1231_);
v___x_1233_ = l_Lean_Syntax_node2(v___x_1222_, v___x_1227_, v___x_1229_, v___x_1232_);
v___x_1234_ = l_Lean_Syntax_node1(v___x_1222_, v___x_1226_, v___x_1233_);
v___x_1235_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1236_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1236_, 0, v___x_1222_);
lean_ctor_set(v___x_1236_, 1, v___x_1235_);
v___x_1237_ = l_Lean_Syntax_node4(v___x_1222_, v___x_1223_, v___x_1225_, v___x_1234_, v___x_1236_, v_a_1181_);
v___x_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1238_, 0, v___x_1237_);
return v___x_1238_;
}
else
{
v___y_1186_ = v___y_1220_;
goto v___jp_1185_;
}
}
else
{
v___y_1186_ = v___y_1220_;
goto v___jp_1185_;
}
}
}
}
else
{
lean_dec(v_x_1172_);
lean_dec(v_a_1168_);
return v___x_1180_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__0___boxed(lean_object* v_a_1275_, lean_object* v_a_1276_, lean_object* v_a_1277_, lean_object* v___x_1278_, lean_object* v_x_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_){
_start:
{
uint8_t v_a_68897__boxed_1287_; uint8_t v_a_68898__boxed_1288_; uint8_t v___x_68899__boxed_1289_; lean_object* v_res_1290_; 
v_a_68897__boxed_1287_ = lean_unbox(v_a_1276_);
v_a_68898__boxed_1288_ = lean_unbox(v_a_1277_);
v___x_68899__boxed_1289_ = lean_unbox(v___x_1278_);
v_res_1290_ = lp_mathlib_iInf__delab___lam__0(v_a_1275_, v_a_68897__boxed_1287_, v_a_68898__boxed_1288_, v___x_68899__boxed_1289_, v_x_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v___y_1281_);
lean_dec_ref(v___y_1280_);
return v_res_1290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__1(lean_object* v___x_1291_, uint8_t v_a_1292_, uint8_t v_a_1293_, uint8_t v___x_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v___x_1302_; 
v___x_1302_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(v___x_1291_, v___y_1295_, v___y_1296_, v___y_1297_, v___y_1298_, v___y_1299_, v___y_1300_);
if (lean_obj_tag(v___x_1302_) == 0)
{
lean_object* v_a_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___f_1307_; uint8_t v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; 
v_a_1303_ = lean_ctor_get(v___x_1302_, 0);
lean_inc(v_a_1303_);
lean_dec_ref_known(v___x_1302_, 1);
v___x_1304_ = lean_box(v_a_1292_);
v___x_1305_ = lean_box(v_a_1293_);
v___x_1306_ = lean_box(v___x_1294_);
v___f_1307_ = lean_alloc_closure((void*)(lp_mathlib_iInf__delab___lam__0___boxed), 12, 4);
lean_closure_set(v___f_1307_, 0, v_a_1303_);
lean_closure_set(v___f_1307_, 1, v___x_1304_);
lean_closure_set(v___f_1307_, 2, v___x_1305_);
lean_closure_set(v___f_1307_, 3, v___x_1306_);
v___x_1308_ = 0;
v___x_1309_ = l_Lean_NameSet_empty;
v___x_1310_ = l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(v___f_1307_, v___x_1308_, v___x_1309_, v___y_1295_, v___y_1296_, v___y_1297_, v___y_1298_, v___y_1299_, v___y_1300_);
return v___x_1310_;
}
else
{
return v___x_1302_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__1___boxed(lean_object* v___x_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_, lean_object* v___x_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
uint8_t v_a_69148__boxed_1322_; uint8_t v_a_69149__boxed_1323_; uint8_t v___x_69150__boxed_1324_; lean_object* v_res_1325_; 
v_a_69148__boxed_1322_ = lean_unbox(v_a_1312_);
v_a_69149__boxed_1323_ = lean_unbox(v_a_1313_);
v___x_69150__boxed_1324_ = lean_unbox(v___x_1314_);
v_res_1325_ = lp_mathlib_iInf__delab___lam__1(v___x_1311_, v_a_69148__boxed_1322_, v_a_69149__boxed_1323_, v___x_69150__boxed_1324_, v___y_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
lean_dec(v___y_1316_);
lean_dec_ref(v___y_1315_);
return v_res_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__2(lean_object* v___x_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_){
_start:
{
lean_object* v___x_1334_; lean_object* v_a_1335_; lean_object* v_dummy_1336_; lean_object* v_nargs_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; uint8_t v___x_1343_; 
v___x_1334_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_1327_);
v_a_1335_ = lean_ctor_get(v___x_1334_, 0);
lean_inc(v_a_1335_);
lean_dec_ref(v___x_1334_);
v_dummy_1336_ = lean_obj_once(&lp_mathlib_iSup__delab___lam__2___closed__0, &lp_mathlib_iSup__delab___lam__2___closed__0_once, _init_lp_mathlib_iSup__delab___lam__2___closed__0);
v_nargs_1337_ = l_Lean_Expr_getAppNumArgs(v_a_1335_);
lean_inc(v_nargs_1337_);
v___x_1338_ = lean_mk_array(v_nargs_1337_, v_dummy_1336_);
v___x_1339_ = lean_unsigned_to_nat(1u);
v___x_1340_ = lean_nat_sub(v_nargs_1337_, v___x_1339_);
lean_dec(v_nargs_1337_);
v___x_1341_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_1335_, v___x_1338_, v___x_1340_);
v___x_1342_ = lean_array_get_size(v___x_1341_);
v___x_1343_ = lean_nat_dec_eq(v___x_1342_, v___x_1326_);
if (v___x_1343_ == 0)
{
lean_object* v___x_1344_; 
lean_dec_ref(v___x_1341_);
v___x_1344_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_1344_;
}
else
{
lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v___y_1349_; lean_object* v___y_1350_; lean_object* v___y_1351_; lean_object* v___y_1352_; lean_object* v___y_1353_; lean_object* v___y_1354_; uint8_t v___x_1532_; 
v___x_1345_ = lean_array_fget(v___x_1341_, v___x_1339_);
v___x_1346_ = lean_unsigned_to_nat(3u);
v___x_1347_ = lean_array_fget(v___x_1341_, v___x_1346_);
lean_dec_ref(v___x_1341_);
v___x_1532_ = l_Lean_Expr_isLambda(v___x_1347_);
if (v___x_1532_ == 0)
{
lean_object* v___x_1533_; 
v___x_1533_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1533_) == 0)
{
lean_dec_ref_known(v___x_1533_, 1);
v___y_1349_ = v___y_1327_;
v___y_1350_ = v___y_1328_;
v___y_1351_ = v___y_1329_;
v___y_1352_ = v___y_1330_;
v___y_1353_ = v___y_1331_;
v___y_1354_ = v___y_1332_;
goto v___jp_1348_;
}
else
{
lean_object* v_a_1534_; lean_object* v___x_1536_; uint8_t v_isShared_1537_; uint8_t v_isSharedCheck_1541_; 
lean_dec(v___x_1347_);
lean_dec(v___x_1345_);
v_a_1534_ = lean_ctor_get(v___x_1533_, 0);
v_isSharedCheck_1541_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1541_ == 0)
{
v___x_1536_ = v___x_1533_;
v_isShared_1537_ = v_isSharedCheck_1541_;
goto v_resetjp_1535_;
}
else
{
lean_inc(v_a_1534_);
lean_dec(v___x_1533_);
v___x_1536_ = lean_box(0);
v_isShared_1537_ = v_isSharedCheck_1541_;
goto v_resetjp_1535_;
}
v_resetjp_1535_:
{
lean_object* v___x_1539_; 
if (v_isShared_1537_ == 0)
{
v___x_1539_ = v___x_1536_;
goto v_reusejp_1538_;
}
else
{
lean_object* v_reuseFailAlloc_1540_; 
v_reuseFailAlloc_1540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1540_, 0, v_a_1534_);
v___x_1539_ = v_reuseFailAlloc_1540_;
goto v_reusejp_1538_;
}
v_reusejp_1538_:
{
return v___x_1539_;
}
}
}
}
else
{
v___y_1349_ = v___y_1327_;
v___y_1350_ = v___y_1328_;
v___y_1351_ = v___y_1329_;
v___y_1352_ = v___y_1330_;
v___y_1353_ = v___y_1331_;
v___y_1354_ = v___y_1332_;
goto v___jp_1348_;
}
v___jp_1348_:
{
lean_object* v___x_1355_; 
v___x_1355_ = l_Lean_Meta_isProp(v___x_1345_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1355_) == 0)
{
lean_object* v_a_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; 
v_a_1356_ = lean_ctor_get(v___x_1355_, 0);
lean_inc(v_a_1356_);
lean_dec_ref_known(v___x_1355_, 1);
v___x_1357_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__1));
v___x_1358_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_1357_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1358_) == 0)
{
lean_object* v_a_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; uint8_t v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___f_1365_; lean_object* v___x_1366_; 
v_a_1359_ = lean_ctor_get(v___x_1358_, 0);
lean_inc(v_a_1359_);
lean_dec_ref_known(v___x_1358_, 1);
v___x_1360_ = l_Lean_Expr_bindingBody_x21(v___x_1347_);
lean_dec(v___x_1347_);
v___x_1361_ = lean_unsigned_to_nat(0u);
v___x_1362_ = lean_expr_has_loose_bvar(v___x_1360_, v___x_1361_);
lean_dec_ref(v___x_1360_);
v___x_1363_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__2));
v___x_1364_ = lean_box(v___x_1362_);
v___f_1365_ = lean_alloc_closure((void*)(lp_mathlib_iInf__delab___lam__1___boxed), 11, 4);
lean_closure_set(v___f_1365_, 0, v___x_1363_);
lean_closure_set(v___f_1365_, 1, v_a_1356_);
lean_closure_set(v___f_1365_, 2, v_a_1359_);
lean_closure_set(v___f_1365_, 3, v___x_1364_);
v___x_1366_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(v___f_1365_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_, v___y_1354_);
if (lean_obj_tag(v___x_1366_) == 0)
{
lean_object* v_a_1367_; lean_object* v___x_1368_; uint8_t v___x_1369_; 
v_a_1367_ = lean_ctor_get(v___x_1366_, 0);
lean_inc_n(v_a_1367_, 2);
v___x_1368_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__1));
v___x_1369_ = l_Lean_Syntax_isOfKind(v_a_1367_, v___x_1368_);
if (v___x_1369_ == 0)
{
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1370_; lean_object* v___x_1371_; uint8_t v___x_1372_; 
v___x_1370_ = l_Lean_Syntax_getArg(v_a_1367_, v___x_1339_);
v___x_1371_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
lean_inc(v___x_1370_);
v___x_1372_ = l_Lean_Syntax_isOfKind(v___x_1370_, v___x_1371_);
if (v___x_1372_ == 0)
{
lean_dec(v___x_1370_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1373_; lean_object* v___x_1374_; uint8_t v___x_1375_; 
v___x_1373_ = l_Lean_Syntax_getArg(v___x_1370_, v___x_1361_);
lean_dec(v___x_1370_);
v___x_1374_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
lean_inc(v___x_1373_);
v___x_1375_ = l_Lean_Syntax_isOfKind(v___x_1373_, v___x_1374_);
if (v___x_1375_ == 0)
{
lean_object* v___x_1376_; uint8_t v___x_1377_; 
v___x_1376_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_1373_);
v___x_1377_ = l_Lean_Syntax_isOfKind(v___x_1373_, v___x_1376_);
if (v___x_1377_ == 0)
{
lean_dec(v___x_1373_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1378_; uint8_t v___x_1379_; 
v___x_1378_ = l_Lean_Syntax_getArg(v___x_1373_, v___x_1361_);
lean_dec(v___x_1373_);
lean_inc(v___x_1378_);
v___x_1379_ = l_Lean_Syntax_matchesNull(v___x_1378_, v___x_1339_);
if (v___x_1379_ == 0)
{
lean_dec(v___x_1378_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1380_; lean_object* v___x_1381_; uint8_t v___x_1382_; 
v___x_1380_ = l_Lean_Syntax_getArg(v___x_1378_, v___x_1361_);
lean_dec(v___x_1378_);
v___x_1381_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_1380_);
v___x_1382_ = l_Lean_Syntax_isOfKind(v___x_1380_, v___x_1381_);
if (v___x_1382_ == 0)
{
lean_dec(v___x_1380_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1383_; uint8_t v___x_1384_; 
v___x_1383_ = l_Lean_Syntax_getArg(v___x_1380_, v___x_1339_);
lean_dec(v___x_1380_);
lean_inc(v___x_1383_);
v___x_1384_ = l_Lean_Syntax_isOfKind(v___x_1383_, v___x_1374_);
if (v___x_1384_ == 0)
{
lean_dec(v___x_1383_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1385_; lean_object* v___x_1386_; uint8_t v___x_1387_; 
v___x_1385_ = l_Lean_Syntax_getArg(v___x_1383_, v___x_1361_);
v___x_1386_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_1385_);
v___x_1387_ = l_Lean_Syntax_isOfKind(v___x_1385_, v___x_1386_);
if (v___x_1387_ == 0)
{
lean_dec(v___x_1385_);
lean_dec(v___x_1383_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1388_; lean_object* v___x_1389_; uint8_t v___x_1390_; 
v___x_1388_ = l_Lean_Syntax_getArg(v___x_1385_, v___x_1361_);
lean_dec(v___x_1385_);
v___x_1389_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_1388_);
v___x_1390_ = l_Lean_Syntax_isOfKind(v___x_1388_, v___x_1389_);
if (v___x_1390_ == 0)
{
lean_dec(v___x_1388_);
lean_dec(v___x_1383_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1391_; uint8_t v___x_1392_; 
v___x_1391_ = l_Lean_Syntax_getArg(v___x_1383_, v___x_1339_);
lean_dec(v___x_1383_);
lean_inc(v___x_1391_);
v___x_1392_ = l_Lean_Syntax_matchesNull(v___x_1391_, v___x_1339_);
if (v___x_1392_ == 0)
{
lean_dec(v___x_1391_);
lean_dec(v___x_1388_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1393_; lean_object* v___x_1394_; uint8_t v___x_1395_; 
v___x_1393_ = l_Lean_Syntax_getArg(v___x_1391_, v___x_1361_);
lean_dec(v___x_1391_);
v___x_1394_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_1395_ = l_Lean_Syntax_isOfKind(v___x_1393_, v___x_1394_);
if (v___x_1395_ == 0)
{
lean_dec(v___x_1388_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1396_; uint8_t v___x_1397_; 
v___x_1396_ = l_Lean_Syntax_getArg(v_a_1367_, v___x_1346_);
lean_dec(v_a_1367_);
lean_inc(v___x_1396_);
v___x_1397_ = l_Lean_Syntax_isOfKind(v___x_1396_, v___x_1368_);
if (v___x_1397_ == 0)
{
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1398_; uint8_t v___x_1399_; 
v___x_1398_ = l_Lean_Syntax_getArg(v___x_1396_, v___x_1339_);
lean_inc(v___x_1398_);
v___x_1399_ = l_Lean_Syntax_isOfKind(v___x_1398_, v___x_1371_);
if (v___x_1399_ == 0)
{
lean_dec(v___x_1398_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1400_; uint8_t v___x_1401_; 
v___x_1400_ = l_Lean_Syntax_getArg(v___x_1398_, v___x_1361_);
lean_dec(v___x_1398_);
lean_inc(v___x_1400_);
v___x_1401_ = l_Lean_Syntax_isOfKind(v___x_1400_, v___x_1376_);
if (v___x_1401_ == 0)
{
lean_dec(v___x_1400_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1402_; uint8_t v___x_1403_; 
v___x_1402_ = l_Lean_Syntax_getArg(v___x_1400_, v___x_1361_);
lean_dec(v___x_1400_);
lean_inc(v___x_1402_);
v___x_1403_ = l_Lean_Syntax_matchesNull(v___x_1402_, v___x_1339_);
if (v___x_1403_ == 0)
{
lean_dec(v___x_1402_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1404_; uint8_t v___x_1405_; 
v___x_1404_ = l_Lean_Syntax_getArg(v___x_1402_, v___x_1361_);
lean_dec(v___x_1402_);
lean_inc(v___x_1404_);
v___x_1405_ = l_Lean_Syntax_isOfKind(v___x_1404_, v___x_1381_);
if (v___x_1405_ == 0)
{
lean_dec(v___x_1404_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1406_; uint8_t v___x_1407_; 
v___x_1406_ = l_Lean_Syntax_getArg(v___x_1404_, v___x_1339_);
lean_dec(v___x_1404_);
lean_inc(v___x_1406_);
v___x_1407_ = l_Lean_Syntax_isOfKind(v___x_1406_, v___x_1374_);
if (v___x_1407_ == 0)
{
lean_dec(v___x_1406_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1408_; uint8_t v___x_1409_; 
v___x_1408_ = l_Lean_Syntax_getArg(v___x_1406_, v___x_1361_);
lean_inc(v___x_1408_);
v___x_1409_ = l_Lean_Syntax_isOfKind(v___x_1408_, v___x_1386_);
if (v___x_1409_ == 0)
{
lean_dec(v___x_1408_);
lean_dec(v___x_1406_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1410_; lean_object* v___x_1411_; uint8_t v___x_1412_; 
v___x_1410_ = l_Lean_Syntax_getArg(v___x_1408_, v___x_1361_);
lean_dec(v___x_1408_);
v___x_1411_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_1412_ = l_Lean_Syntax_isOfKind(v___x_1410_, v___x_1411_);
if (v___x_1412_ == 0)
{
lean_dec(v___x_1406_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1413_; uint8_t v___x_1414_; 
v___x_1413_ = l_Lean_Syntax_getArg(v___x_1406_, v___x_1339_);
lean_dec(v___x_1406_);
lean_inc(v___x_1413_);
v___x_1414_ = l_Lean_Syntax_matchesNull(v___x_1413_, v___x_1339_);
if (v___x_1414_ == 0)
{
lean_dec(v___x_1413_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1415_; uint8_t v___x_1416_; 
v___x_1415_ = l_Lean_Syntax_getArg(v___x_1413_, v___x_1361_);
lean_dec(v___x_1413_);
lean_inc(v___x_1415_);
v___x_1416_ = l_Lean_Syntax_isOfKind(v___x_1415_, v___x_1394_);
if (v___x_1416_ == 0)
{
lean_dec(v___x_1415_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1417_; lean_object* v___x_1418_; uint8_t v___x_1419_; 
v___x_1417_ = l_Lean_Syntax_getArg(v___x_1415_, v___x_1339_);
lean_dec(v___x_1415_);
v___x_1418_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_1417_);
v___x_1419_ = l_Lean_Syntax_isOfKind(v___x_1417_, v___x_1418_);
if (v___x_1419_ == 0)
{
lean_dec(v___x_1417_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1420_; uint8_t v___x_1421_; 
v___x_1420_ = l_Lean_Syntax_getArg(v___x_1417_, v___x_1361_);
lean_inc(v___x_1420_);
v___x_1421_ = l_Lean_Syntax_isOfKind(v___x_1420_, v___x_1389_);
if (v___x_1421_ == 0)
{
lean_dec(v___x_1420_);
lean_dec(v___x_1417_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
uint8_t v___x_1422_; 
v___x_1422_ = l_Lean_Syntax_structEq(v___x_1388_, v___x_1420_);
lean_dec(v___x_1420_);
if (v___x_1422_ == 0)
{
lean_dec(v___x_1417_);
lean_dec(v___x_1396_);
lean_dec(v___x_1388_);
return v___x_1366_;
}
else
{
lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1448_; 
v_isSharedCheck_1448_ = !lean_is_exclusive(v___x_1366_);
if (v_isSharedCheck_1448_ == 0)
{
lean_object* v_unused_1449_; 
v_unused_1449_ = lean_ctor_get(v___x_1366_, 0);
lean_dec(v_unused_1449_);
v___x_1424_ = v___x_1366_;
v_isShared_1425_ = v_isSharedCheck_1448_;
goto v_resetjp_1423_;
}
else
{
lean_dec(v___x_1366_);
v___x_1424_ = lean_box(0);
v_isShared_1425_ = v_isSharedCheck_1448_;
goto v_resetjp_1423_;
}
v_resetjp_1423_:
{
lean_object* v_ref_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1446_; 
v_ref_1426_ = lean_ctor_get(v___y_1353_, 5);
v___x_1427_ = lean_unsigned_to_nat(2u);
v___x_1428_ = l_Lean_Syntax_getArg(v___x_1417_, v___x_1427_);
lean_dec(v___x_1417_);
v___x_1429_ = l_Lean_Syntax_getArg(v___x_1396_, v___x_1346_);
lean_dec(v___x_1396_);
v___x_1430_ = l_Lean_SourceInfo_fromRef(v_ref_1426_, v___x_1375_);
v___x_1431_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__2));
lean_inc_n(v___x_1430_, 8);
v___x_1432_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1432_, 0, v___x_1430_);
lean_ctor_set(v___x_1432_, 1, v___x_1431_);
v___x_1433_ = l_Lean_Syntax_node1(v___x_1430_, v___x_1386_, v___x_1388_);
v___x_1434_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1435_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_1436_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_1437_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1437_, 0, v___x_1430_);
lean_ctor_set(v___x_1437_, 1, v___x_1436_);
v___x_1438_ = l_Lean_Syntax_node2(v___x_1430_, v___x_1435_, v___x_1437_, v___x_1428_);
v___x_1439_ = l_Lean_Syntax_node1(v___x_1430_, v___x_1434_, v___x_1438_);
v___x_1440_ = l_Lean_Syntax_node2(v___x_1430_, v___x_1374_, v___x_1433_, v___x_1439_);
v___x_1441_ = l_Lean_Syntax_node1(v___x_1430_, v___x_1371_, v___x_1440_);
v___x_1442_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1443_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1443_, 0, v___x_1430_);
lean_ctor_set(v___x_1443_, 1, v___x_1442_);
v___x_1444_ = l_Lean_Syntax_node4(v___x_1430_, v___x_1368_, v___x_1432_, v___x_1441_, v___x_1443_, v___x_1429_);
if (v_isShared_1425_ == 0)
{
lean_ctor_set(v___x_1424_, 0, v___x_1444_);
v___x_1446_ = v___x_1424_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1447_; 
v_reuseFailAlloc_1447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1447_, 0, v___x_1444_);
v___x_1446_ = v_reuseFailAlloc_1447_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
return v___x_1446_;
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
}
}
}
}
}
}
}
else
{
lean_object* v___x_1450_; lean_object* v___x_1451_; uint8_t v___x_1452_; 
v___x_1450_ = l_Lean_Syntax_getArg(v___x_1373_, v___x_1361_);
v___x_1451_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_1450_);
v___x_1452_ = l_Lean_Syntax_isOfKind(v___x_1450_, v___x_1451_);
if (v___x_1452_ == 0)
{
lean_dec(v___x_1450_);
lean_dec(v___x_1373_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1453_; lean_object* v___x_1454_; uint8_t v___x_1455_; 
v___x_1453_ = l_Lean_Syntax_getArg(v___x_1450_, v___x_1361_);
lean_dec(v___x_1450_);
v___x_1454_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_1453_);
v___x_1455_ = l_Lean_Syntax_isOfKind(v___x_1453_, v___x_1454_);
if (v___x_1455_ == 0)
{
lean_dec(v___x_1453_);
lean_dec(v___x_1373_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1456_; uint8_t v___x_1457_; 
v___x_1456_ = l_Lean_Syntax_getArg(v___x_1373_, v___x_1339_);
lean_dec(v___x_1373_);
v___x_1457_ = l_Lean_Syntax_matchesNull(v___x_1456_, v___x_1361_);
if (v___x_1457_ == 0)
{
lean_dec(v___x_1453_);
lean_dec(v_a_1367_);
return v___x_1366_;
}
else
{
lean_object* v___x_1458_; uint8_t v___x_1459_; 
v___x_1458_ = l_Lean_Syntax_getArg(v_a_1367_, v___x_1346_);
lean_dec(v_a_1367_);
lean_inc(v___x_1458_);
v___x_1459_ = l_Lean_Syntax_isOfKind(v___x_1458_, v___x_1368_);
if (v___x_1459_ == 0)
{
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1460_; uint8_t v___x_1461_; 
v___x_1460_ = l_Lean_Syntax_getArg(v___x_1458_, v___x_1339_);
lean_inc(v___x_1460_);
v___x_1461_ = l_Lean_Syntax_isOfKind(v___x_1460_, v___x_1371_);
if (v___x_1461_ == 0)
{
lean_dec(v___x_1460_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1462_; lean_object* v___x_1463_; uint8_t v___x_1464_; 
v___x_1462_ = l_Lean_Syntax_getArg(v___x_1460_, v___x_1361_);
lean_dec(v___x_1460_);
v___x_1463_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_1462_);
v___x_1464_ = l_Lean_Syntax_isOfKind(v___x_1462_, v___x_1463_);
if (v___x_1464_ == 0)
{
lean_dec(v___x_1462_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1465_; uint8_t v___x_1466_; 
v___x_1465_ = l_Lean_Syntax_getArg(v___x_1462_, v___x_1361_);
lean_dec(v___x_1462_);
lean_inc(v___x_1465_);
v___x_1466_ = l_Lean_Syntax_matchesNull(v___x_1465_, v___x_1339_);
if (v___x_1466_ == 0)
{
lean_dec(v___x_1465_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1467_; lean_object* v___x_1468_; uint8_t v___x_1469_; 
v___x_1467_ = l_Lean_Syntax_getArg(v___x_1465_, v___x_1361_);
lean_dec(v___x_1465_);
v___x_1468_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_1467_);
v___x_1469_ = l_Lean_Syntax_isOfKind(v___x_1467_, v___x_1468_);
if (v___x_1469_ == 0)
{
lean_dec(v___x_1467_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1470_; uint8_t v___x_1471_; 
v___x_1470_ = l_Lean_Syntax_getArg(v___x_1467_, v___x_1339_);
lean_dec(v___x_1467_);
lean_inc(v___x_1470_);
v___x_1471_ = l_Lean_Syntax_isOfKind(v___x_1470_, v___x_1374_);
if (v___x_1471_ == 0)
{
lean_dec(v___x_1470_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1472_; uint8_t v___x_1473_; 
v___x_1472_ = l_Lean_Syntax_getArg(v___x_1470_, v___x_1361_);
lean_inc(v___x_1472_);
v___x_1473_ = l_Lean_Syntax_isOfKind(v___x_1472_, v___x_1451_);
if (v___x_1473_ == 0)
{
lean_dec(v___x_1472_);
lean_dec(v___x_1470_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1474_; lean_object* v___x_1475_; uint8_t v___x_1476_; 
v___x_1474_ = l_Lean_Syntax_getArg(v___x_1472_, v___x_1361_);
lean_dec(v___x_1472_);
v___x_1475_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_1476_ = l_Lean_Syntax_isOfKind(v___x_1474_, v___x_1475_);
if (v___x_1476_ == 0)
{
lean_dec(v___x_1470_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1477_; uint8_t v___x_1478_; 
v___x_1477_ = l_Lean_Syntax_getArg(v___x_1470_, v___x_1339_);
lean_dec(v___x_1470_);
lean_inc(v___x_1477_);
v___x_1478_ = l_Lean_Syntax_matchesNull(v___x_1477_, v___x_1339_);
if (v___x_1478_ == 0)
{
lean_dec(v___x_1477_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1479_; lean_object* v___x_1480_; uint8_t v___x_1481_; 
v___x_1479_ = l_Lean_Syntax_getArg(v___x_1477_, v___x_1361_);
lean_dec(v___x_1477_);
v___x_1480_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
lean_inc(v___x_1479_);
v___x_1481_ = l_Lean_Syntax_isOfKind(v___x_1479_, v___x_1480_);
if (v___x_1481_ == 0)
{
lean_dec(v___x_1479_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1482_; lean_object* v___x_1483_; uint8_t v___x_1484_; 
v___x_1482_ = l_Lean_Syntax_getArg(v___x_1479_, v___x_1339_);
lean_dec(v___x_1479_);
v___x_1483_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_1482_);
v___x_1484_ = l_Lean_Syntax_isOfKind(v___x_1482_, v___x_1483_);
if (v___x_1484_ == 0)
{
lean_dec(v___x_1482_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1485_; uint8_t v___x_1486_; 
v___x_1485_ = l_Lean_Syntax_getArg(v___x_1482_, v___x_1361_);
lean_inc(v___x_1485_);
v___x_1486_ = l_Lean_Syntax_isOfKind(v___x_1485_, v___x_1454_);
if (v___x_1486_ == 0)
{
lean_dec(v___x_1485_);
lean_dec(v___x_1482_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
uint8_t v___x_1487_; 
v___x_1487_ = l_Lean_Syntax_structEq(v___x_1453_, v___x_1485_);
lean_dec(v___x_1485_);
if (v___x_1487_ == 0)
{
lean_dec(v___x_1482_);
lean_dec(v___x_1458_);
lean_dec(v___x_1453_);
return v___x_1366_;
}
else
{
lean_object* v___x_1489_; uint8_t v_isShared_1490_; uint8_t v_isSharedCheck_1514_; 
v_isSharedCheck_1514_ = !lean_is_exclusive(v___x_1366_);
if (v_isSharedCheck_1514_ == 0)
{
lean_object* v_unused_1515_; 
v_unused_1515_ = lean_ctor_get(v___x_1366_, 0);
lean_dec(v_unused_1515_);
v___x_1489_ = v___x_1366_;
v_isShared_1490_ = v_isSharedCheck_1514_;
goto v_resetjp_1488_;
}
else
{
lean_dec(v___x_1366_);
v___x_1489_ = lean_box(0);
v_isShared_1490_ = v_isSharedCheck_1514_;
goto v_resetjp_1488_;
}
v_resetjp_1488_:
{
lean_object* v_ref_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; uint8_t v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1512_; 
v_ref_1491_ = lean_ctor_get(v___y_1353_, 5);
v___x_1492_ = lean_unsigned_to_nat(2u);
v___x_1493_ = l_Lean_Syntax_getArg(v___x_1482_, v___x_1492_);
lean_dec(v___x_1482_);
v___x_1494_ = l_Lean_Syntax_getArg(v___x_1458_, v___x_1346_);
lean_dec(v___x_1458_);
v___x_1495_ = 0;
v___x_1496_ = l_Lean_SourceInfo_fromRef(v_ref_1491_, v___x_1495_);
v___x_1497_ = ((lean_object*)(lp_mathlib_term_u2a05___x2c___00__closed__2));
lean_inc_n(v___x_1496_, 8);
v___x_1498_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1496_);
lean_ctor_set(v___x_1498_, 1, v___x_1497_);
v___x_1499_ = l_Lean_Syntax_node1(v___x_1496_, v___x_1451_, v___x_1453_);
v___x_1500_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1501_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_1502_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_1503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1503_, 0, v___x_1496_);
lean_ctor_set(v___x_1503_, 1, v___x_1502_);
v___x_1504_ = l_Lean_Syntax_node2(v___x_1496_, v___x_1501_, v___x_1503_, v___x_1493_);
v___x_1505_ = l_Lean_Syntax_node1(v___x_1496_, v___x_1500_, v___x_1504_);
v___x_1506_ = l_Lean_Syntax_node2(v___x_1496_, v___x_1374_, v___x_1499_, v___x_1505_);
v___x_1507_ = l_Lean_Syntax_node1(v___x_1496_, v___x_1371_, v___x_1506_);
v___x_1508_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1509_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1509_, 0, v___x_1496_);
lean_ctor_set(v___x_1509_, 1, v___x_1508_);
v___x_1510_ = l_Lean_Syntax_node4(v___x_1496_, v___x_1368_, v___x_1498_, v___x_1507_, v___x_1509_, v___x_1494_);
if (v_isShared_1490_ == 0)
{
lean_ctor_set(v___x_1489_, 0, v___x_1510_);
v___x_1512_ = v___x_1489_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1513_; 
v_reuseFailAlloc_1513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1513_, 0, v___x_1510_);
v___x_1512_ = v_reuseFailAlloc_1513_;
goto v_reusejp_1511_;
}
v_reusejp_1511_:
{
return v___x_1512_;
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
}
}
}
}
}
else
{
return v___x_1366_;
}
}
else
{
lean_object* v_a_1516_; lean_object* v___x_1518_; uint8_t v_isShared_1519_; uint8_t v_isSharedCheck_1523_; 
lean_dec(v_a_1356_);
lean_dec(v___x_1347_);
v_a_1516_ = lean_ctor_get(v___x_1358_, 0);
v_isSharedCheck_1523_ = !lean_is_exclusive(v___x_1358_);
if (v_isSharedCheck_1523_ == 0)
{
v___x_1518_ = v___x_1358_;
v_isShared_1519_ = v_isSharedCheck_1523_;
goto v_resetjp_1517_;
}
else
{
lean_inc(v_a_1516_);
lean_dec(v___x_1358_);
v___x_1518_ = lean_box(0);
v_isShared_1519_ = v_isSharedCheck_1523_;
goto v_resetjp_1517_;
}
v_resetjp_1517_:
{
lean_object* v___x_1521_; 
if (v_isShared_1519_ == 0)
{
v___x_1521_ = v___x_1518_;
goto v_reusejp_1520_;
}
else
{
lean_object* v_reuseFailAlloc_1522_; 
v_reuseFailAlloc_1522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1522_, 0, v_a_1516_);
v___x_1521_ = v_reuseFailAlloc_1522_;
goto v_reusejp_1520_;
}
v_reusejp_1520_:
{
return v___x_1521_;
}
}
}
}
else
{
lean_object* v_a_1524_; lean_object* v___x_1526_; uint8_t v_isShared_1527_; uint8_t v_isSharedCheck_1531_; 
lean_dec(v___x_1347_);
v_a_1524_ = lean_ctor_get(v___x_1355_, 0);
v_isSharedCheck_1531_ = !lean_is_exclusive(v___x_1355_);
if (v_isSharedCheck_1531_ == 0)
{
v___x_1526_ = v___x_1355_;
v_isShared_1527_ = v_isSharedCheck_1531_;
goto v_resetjp_1525_;
}
else
{
lean_inc(v_a_1524_);
lean_dec(v___x_1355_);
v___x_1526_ = lean_box(0);
v_isShared_1527_ = v_isSharedCheck_1531_;
goto v_resetjp_1525_;
}
v_resetjp_1525_:
{
lean_object* v___x_1529_; 
if (v_isShared_1527_ == 0)
{
v___x_1529_ = v___x_1526_;
goto v_reusejp_1528_;
}
else
{
lean_object* v_reuseFailAlloc_1530_; 
v_reuseFailAlloc_1530_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1530_, 0, v_a_1524_);
v___x_1529_ = v_reuseFailAlloc_1530_;
goto v_reusejp_1528_;
}
v_reusejp_1528_:
{
return v___x_1529_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___lam__2___boxed(lean_object* v___x_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_){
_start:
{
lean_object* v_res_1550_; 
v_res_1550_ = lp_mathlib_iInf__delab___lam__2(v___x_1542_, v___y_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_);
lean_dec(v___y_1548_);
lean_dec_ref(v___y_1547_);
lean_dec(v___y_1546_);
lean_dec_ref(v___y_1545_);
lean_dec(v___y_1544_);
lean_dec_ref(v___y_1543_);
lean_dec(v___x_1542_);
return v_res_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab(lean_object* v_a_1556_, lean_object* v_a_1557_, lean_object* v_a_1558_, lean_object* v_a_1559_, lean_object* v_a_1560_, lean_object* v_a_1561_){
_start:
{
lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___x_1563_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_1564_ = ((lean_object*)(lp_mathlib_iInf__delab___closed__1));
v___x_1565_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_1563_, v___x_1564_, v_a_1556_, v_a_1557_, v_a_1558_, v_a_1559_, v_a_1560_, v_a_1561_);
return v___x_1565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf__delab___boxed(lean_object* v_a_1566_, lean_object* v_a_1567_, lean_object* v_a_1568_, lean_object* v_a_1569_, lean_object* v_a_1570_, lean_object* v_a_1571_, lean_object* v_a_1572_){
_start:
{
lean_object* v_res_1573_; 
v_res_1573_ = lp_mathlib_iInf__delab(v_a_1566_, v_a_1567_, v_a_1568_, v_a_1569_, v_a_1570_, v_a_1571_);
lean_dec(v_a_1571_);
lean_dec_ref(v_a_1570_);
lean_dec(v_a_1569_);
lean_dec_ref(v_a_1568_);
lean_dec(v_a_1567_);
lean_dec_ref(v_a_1566_);
return v_res_1573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instInfSet(lean_object* v_00_u03b1_1574_){
_start:
{
lean_object* v___x_1575_; 
v___x_1575_ = lean_box(0);
return v___x_1575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instSupSet(lean_object* v_00_u03b1_1576_){
_start:
{
lean_object* v___x_1577_; 
v___x_1577_ = lean_box(0);
return v___x_1577_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__1(void){
_start:
{
lean_object* v___x_1599_; lean_object* v___x_1600_; 
v___x_1599_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__0));
v___x_1600_ = l_String_toRawSubstring_x27(v___x_1599_);
return v___x_1600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1(lean_object* v_x_1612_, lean_object* v_a_1613_, lean_object* v_a_1614_){
_start:
{
lean_object* v___x_1615_; uint8_t v___x_1616_; 
v___x_1615_ = ((lean_object*)(lp_mathlib_Set_term_u22c2_u2080___00__closed__2));
lean_inc(v_x_1612_);
v___x_1616_ = l_Lean_Syntax_isOfKind(v_x_1612_, v___x_1615_);
if (v___x_1616_ == 0)
{
lean_object* v___x_1617_; lean_object* v___x_1618_; 
lean_dec(v_x_1612_);
v___x_1617_ = lean_box(1);
v___x_1618_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1618_, 0, v___x_1617_);
lean_ctor_set(v___x_1618_, 1, v_a_1614_);
return v___x_1618_;
}
else
{
lean_object* v_quotContext_1619_; lean_object* v_currMacroScope_1620_; lean_object* v_ref_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; uint8_t v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; 
v_quotContext_1619_ = lean_ctor_get(v_a_1613_, 1);
v_currMacroScope_1620_ = lean_ctor_get(v_a_1613_, 2);
v_ref_1621_ = lean_ctor_get(v_a_1613_, 5);
v___x_1622_ = lean_unsigned_to_nat(1u);
v___x_1623_ = l_Lean_Syntax_getArg(v_x_1612_, v___x_1622_);
lean_dec(v_x_1612_);
v___x_1624_ = 0;
v___x_1625_ = l_Lean_SourceInfo_fromRef(v_ref_1621_, v___x_1624_);
v___x_1626_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
v___x_1627_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__1, &lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__1_once, _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__1);
v___x_1628_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__2));
lean_inc(v_currMacroScope_1620_);
lean_inc(v_quotContext_1619_);
v___x_1629_ = l_Lean_addMacroScope(v_quotContext_1619_, v___x_1628_, v_currMacroScope_1620_);
v___x_1630_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___closed__5));
lean_inc_n(v___x_1625_, 2);
v___x_1631_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1631_, 0, v___x_1625_);
lean_ctor_set(v___x_1631_, 1, v___x_1627_);
lean_ctor_set(v___x_1631_, 2, v___x_1629_);
lean_ctor_set(v___x_1631_, 3, v___x_1630_);
v___x_1632_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1633_ = l_Lean_Syntax_node1(v___x_1625_, v___x_1632_, v___x_1623_);
v___x_1634_ = l_Lean_Syntax_node2(v___x_1625_, v___x_1626_, v___x_1631_, v___x_1633_);
v___x_1635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1635_, 0, v___x_1634_);
lean_ctor_set(v___x_1635_, 1, v_a_1614_);
return v___x_1635_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1___boxed(lean_object* v_x_1636_, lean_object* v_a_1637_, lean_object* v_a_1638_){
_start:
{
lean_object* v_res_1639_; 
v_res_1639_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2_u2080____1(v_x_1636_, v_a_1637_, v_a_1638_);
lean_dec_ref(v_a_1637_);
return v_res_1639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sInter__1(lean_object* v_x_1640_, lean_object* v_a_1641_, lean_object* v_a_1642_){
_start:
{
lean_object* v___x_1643_; uint8_t v___x_1644_; 
v___x_1643_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
lean_inc(v_x_1640_);
v___x_1644_ = l_Lean_Syntax_isOfKind(v_x_1640_, v___x_1643_);
if (v___x_1644_ == 0)
{
lean_object* v___x_1645_; lean_object* v___x_1646_; 
lean_dec(v_x_1640_);
v___x_1645_ = lean_box(0);
v___x_1646_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1646_, 0, v___x_1645_);
lean_ctor_set(v___x_1646_, 1, v_a_1642_);
return v___x_1646_;
}
else
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; uint8_t v___x_1650_; 
v___x_1647_ = lean_unsigned_to_nat(0u);
v___x_1648_ = l_Lean_Syntax_getArg(v_x_1640_, v___x_1647_);
v___x_1649_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_1648_);
v___x_1650_ = l_Lean_Syntax_isOfKind(v___x_1648_, v___x_1649_);
if (v___x_1650_ == 0)
{
lean_object* v___x_1651_; lean_object* v___x_1652_; 
lean_dec(v___x_1648_);
lean_dec(v_x_1640_);
v___x_1651_ = lean_box(0);
v___x_1652_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1652_, 0, v___x_1651_);
lean_ctor_set(v___x_1652_, 1, v_a_1642_);
return v___x_1652_;
}
else
{
lean_object* v___x_1653_; lean_object* v___x_1654_; uint8_t v___x_1655_; 
v___x_1653_ = lean_unsigned_to_nat(1u);
v___x_1654_ = l_Lean_Syntax_getArg(v_x_1640_, v___x_1653_);
lean_dec(v_x_1640_);
lean_inc(v___x_1654_);
v___x_1655_ = l_Lean_Syntax_matchesNull(v___x_1654_, v___x_1653_);
if (v___x_1655_ == 0)
{
lean_object* v___x_1656_; lean_object* v___x_1657_; 
lean_dec(v___x_1654_);
lean_dec(v___x_1648_);
v___x_1656_ = lean_box(0);
v___x_1657_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1657_, 0, v___x_1656_);
lean_ctor_set(v___x_1657_, 1, v_a_1642_);
return v___x_1657_;
}
else
{
lean_object* v___x_1658_; lean_object* v_ref_1659_; uint8_t v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; 
v___x_1658_ = l_Lean_Syntax_getArg(v___x_1654_, v___x_1647_);
lean_dec(v___x_1654_);
v_ref_1659_ = l_Lean_replaceRef(v___x_1648_, v_a_1641_);
lean_dec(v___x_1648_);
v___x_1660_ = 0;
v___x_1661_ = l_Lean_SourceInfo_fromRef(v_ref_1659_, v___x_1660_);
lean_dec(v_ref_1659_);
v___x_1662_ = ((lean_object*)(lp_mathlib_Set_term_u22c2_u2080___00__closed__2));
v___x_1663_ = ((lean_object*)(lp_mathlib_Set_term_u22c2_u2080___00__closed__3));
lean_inc(v___x_1661_);
v___x_1664_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1664_, 0, v___x_1661_);
lean_ctor_set(v___x_1664_, 1, v___x_1663_);
v___x_1665_ = l_Lean_Syntax_node2(v___x_1661_, v___x_1662_, v___x_1664_, v___x_1658_);
v___x_1666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1666_, 0, v___x_1665_);
lean_ctor_set(v___x_1666_, 1, v_a_1642_);
return v___x_1666_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sInter__1___boxed(lean_object* v_x_1667_, lean_object* v_a_1668_, lean_object* v_a_1669_){
_start:
{
lean_object* v_res_1670_; 
v_res_1670_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sInter__1(v_x_1667_, v_a_1668_, v_a_1669_);
lean_dec(v_a_1668_);
return v_res_1670_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__1(void){
_start:
{
lean_object* v___x_1688_; lean_object* v___x_1689_; 
v___x_1688_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__0));
v___x_1689_ = l_String_toRawSubstring_x27(v___x_1688_);
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1(lean_object* v_x_1701_, lean_object* v_a_1702_, lean_object* v_a_1703_){
_start:
{
lean_object* v___x_1704_; uint8_t v___x_1705_; 
v___x_1704_ = ((lean_object*)(lp_mathlib_Set_term_u22c3_u2080___00__closed__1));
lean_inc(v_x_1701_);
v___x_1705_ = l_Lean_Syntax_isOfKind(v_x_1701_, v___x_1704_);
if (v___x_1705_ == 0)
{
lean_object* v___x_1706_; lean_object* v___x_1707_; 
lean_dec(v_x_1701_);
v___x_1706_ = lean_box(1);
v___x_1707_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1707_, 0, v___x_1706_);
lean_ctor_set(v___x_1707_, 1, v_a_1703_);
return v___x_1707_;
}
else
{
lean_object* v_quotContext_1708_; lean_object* v_currMacroScope_1709_; lean_object* v_ref_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; uint8_t v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; 
v_quotContext_1708_ = lean_ctor_get(v_a_1702_, 1);
v_currMacroScope_1709_ = lean_ctor_get(v_a_1702_, 2);
v_ref_1710_ = lean_ctor_get(v_a_1702_, 5);
v___x_1711_ = lean_unsigned_to_nat(1u);
v___x_1712_ = l_Lean_Syntax_getArg(v_x_1701_, v___x_1711_);
lean_dec(v_x_1701_);
v___x_1713_ = 0;
v___x_1714_ = l_Lean_SourceInfo_fromRef(v_ref_1710_, v___x_1713_);
v___x_1715_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
v___x_1716_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__1, &lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__1_once, _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__1);
v___x_1717_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__2));
lean_inc(v_currMacroScope_1709_);
lean_inc(v_quotContext_1708_);
v___x_1718_ = l_Lean_addMacroScope(v_quotContext_1708_, v___x_1717_, v_currMacroScope_1709_);
v___x_1719_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___closed__5));
lean_inc_n(v___x_1714_, 2);
v___x_1720_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1720_, 0, v___x_1714_);
lean_ctor_set(v___x_1720_, 1, v___x_1716_);
lean_ctor_set(v___x_1720_, 2, v___x_1718_);
lean_ctor_set(v___x_1720_, 3, v___x_1719_);
v___x_1721_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1722_ = l_Lean_Syntax_node1(v___x_1714_, v___x_1721_, v___x_1712_);
v___x_1723_ = l_Lean_Syntax_node2(v___x_1714_, v___x_1715_, v___x_1720_, v___x_1722_);
v___x_1724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1724_, 0, v___x_1723_);
lean_ctor_set(v___x_1724_, 1, v_a_1703_);
return v___x_1724_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1___boxed(lean_object* v_x_1725_, lean_object* v_a_1726_, lean_object* v_a_1727_){
_start:
{
lean_object* v_res_1728_; 
v_res_1728_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3_u2080____1(v_x_1725_, v_a_1726_, v_a_1727_);
lean_dec_ref(v_a_1726_);
return v_res_1728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sUnion__1(lean_object* v_x_1729_, lean_object* v_a_1730_, lean_object* v_a_1731_){
_start:
{
lean_object* v___x_1732_; uint8_t v___x_1733_; 
v___x_1732_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
lean_inc(v_x_1729_);
v___x_1733_ = l_Lean_Syntax_isOfKind(v_x_1729_, v___x_1732_);
if (v___x_1733_ == 0)
{
lean_object* v___x_1734_; lean_object* v___x_1735_; 
lean_dec(v_x_1729_);
v___x_1734_ = lean_box(0);
v___x_1735_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1735_, 0, v___x_1734_);
lean_ctor_set(v___x_1735_, 1, v_a_1731_);
return v___x_1735_;
}
else
{
lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; uint8_t v___x_1739_; 
v___x_1736_ = lean_unsigned_to_nat(0u);
v___x_1737_ = l_Lean_Syntax_getArg(v_x_1729_, v___x_1736_);
v___x_1738_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_1737_);
v___x_1739_ = l_Lean_Syntax_isOfKind(v___x_1737_, v___x_1738_);
if (v___x_1739_ == 0)
{
lean_object* v___x_1740_; lean_object* v___x_1741_; 
lean_dec(v___x_1737_);
lean_dec(v_x_1729_);
v___x_1740_ = lean_box(0);
v___x_1741_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1741_, 0, v___x_1740_);
lean_ctor_set(v___x_1741_, 1, v_a_1731_);
return v___x_1741_;
}
else
{
lean_object* v___x_1742_; lean_object* v___x_1743_; uint8_t v___x_1744_; 
v___x_1742_ = lean_unsigned_to_nat(1u);
v___x_1743_ = l_Lean_Syntax_getArg(v_x_1729_, v___x_1742_);
lean_dec(v_x_1729_);
lean_inc(v___x_1743_);
v___x_1744_ = l_Lean_Syntax_matchesNull(v___x_1743_, v___x_1742_);
if (v___x_1744_ == 0)
{
lean_object* v___x_1745_; lean_object* v___x_1746_; 
lean_dec(v___x_1743_);
lean_dec(v___x_1737_);
v___x_1745_ = lean_box(0);
v___x_1746_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1746_, 0, v___x_1745_);
lean_ctor_set(v___x_1746_, 1, v_a_1731_);
return v___x_1746_;
}
else
{
lean_object* v___x_1747_; lean_object* v_ref_1748_; uint8_t v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; 
v___x_1747_ = l_Lean_Syntax_getArg(v___x_1743_, v___x_1736_);
lean_dec(v___x_1743_);
v_ref_1748_ = l_Lean_replaceRef(v___x_1737_, v_a_1730_);
lean_dec(v___x_1737_);
v___x_1749_ = 0;
v___x_1750_ = l_Lean_SourceInfo_fromRef(v_ref_1748_, v___x_1749_);
lean_dec(v_ref_1748_);
v___x_1751_ = ((lean_object*)(lp_mathlib_Set_term_u22c3_u2080___00__closed__1));
v___x_1752_ = ((lean_object*)(lp_mathlib_Set_term_u22c3_u2080___00__closed__2));
lean_inc(v___x_1750_);
v___x_1753_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1753_, 0, v___x_1750_);
lean_ctor_set(v___x_1753_, 1, v___x_1752_);
v___x_1754_ = l_Lean_Syntax_node2(v___x_1750_, v___x_1751_, v___x_1753_, v___x_1747_);
v___x_1755_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1755_, 0, v___x_1754_);
lean_ctor_set(v___x_1755_, 1, v_a_1731_);
return v___x_1755_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sUnion__1___boxed(lean_object* v_x_1756_, lean_object* v_a_1757_, lean_object* v_a_1758_){
_start:
{
lean_object* v_res_1759_; 
v_res_1759_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______unexpand__Set__sUnion__1(v_x_1756_, v_a_1757_, v_a_1758_);
lean_dec(v_a_1757_);
return v_res_1759_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; 
v___x_1767_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_1768_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__3));
v___x_1769_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_1770_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1770_, 0, v___x_1769_);
lean_ctor_set(v___x_1770_, 1, v___x_1768_);
lean_ctor_set(v___x_1770_, 2, v___x_1767_);
return v___x_1770_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__5(void){
_start:
{
lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; 
v___x_1771_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__8));
v___x_1772_ = lean_obj_once(&lp_mathlib_Set_term_u22c3___x2c___00__closed__4, &lp_mathlib_Set_term_u22c3___x2c___00__closed__4_once, _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__4);
v___x_1773_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_1774_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1774_, 0, v___x_1773_);
lean_ctor_set(v___x_1774_, 1, v___x_1772_);
lean_ctor_set(v___x_1774_, 2, v___x_1771_);
return v___x_1774_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; 
v___x_1775_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__12));
v___x_1776_ = lean_obj_once(&lp_mathlib_Set_term_u22c3___x2c___00__closed__5, &lp_mathlib_Set_term_u22c3___x2c___00__closed__5_once, _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__5);
v___x_1777_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_1778_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1778_, 0, v___x_1777_);
lean_ctor_set(v___x_1778_, 1, v___x_1776_);
lean_ctor_set(v___x_1778_, 2, v___x_1775_);
return v___x_1778_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; 
v___x_1779_ = lean_obj_once(&lp_mathlib_Set_term_u22c3___x2c___00__closed__6, &lp_mathlib_Set_term_u22c3___x2c___00__closed__6_once, _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__6);
v___x_1780_ = lean_unsigned_to_nat(1022u);
v___x_1781_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
v___x_1782_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1782_, 0, v___x_1781_);
lean_ctor_set(v___x_1782_, 1, v___x_1780_);
lean_ctor_set(v___x_1782_, 2, v___x_1779_);
return v___x_1782_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c3___x2c__(void){
_start:
{
lean_object* v___x_1783_; 
v___x_1783_ = lean_obj_once(&lp_mathlib_Set_term_u22c3___x2c___00__closed__7, &lp_mathlib_Set_term_u22c3___x2c___00__closed__7_once, _init_lp_mathlib_Set_term_u22c3___x2c___00__closed__7);
return v___x_1783_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__1(void){
_start:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; 
v___x_1785_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__0));
v___x_1786_ = l_String_toRawSubstring_x27(v___x_1785_);
return v___x_1786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1(lean_object* v_x_1798_, lean_object* v_a_1799_, lean_object* v_a_1800_){
_start:
{
lean_object* v___x_1801_; uint8_t v___x_1802_; 
v___x_1801_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
lean_inc(v_x_1798_);
v___x_1802_ = l_Lean_Syntax_isOfKind(v_x_1798_, v___x_1801_);
if (v___x_1802_ == 0)
{
lean_object* v___x_1803_; lean_object* v___x_1804_; 
lean_dec(v_x_1798_);
v___x_1803_ = lean_box(1);
v___x_1804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1804_, 0, v___x_1803_);
lean_ctor_set(v___x_1804_, 1, v_a_1800_);
return v___x_1804_;
}
else
{
lean_object* v_quotContext_1805_; lean_object* v_currMacroScope_1806_; lean_object* v_ref_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; uint8_t v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; 
v_quotContext_1805_ = lean_ctor_get(v_a_1799_, 1);
v_currMacroScope_1806_ = lean_ctor_get(v_a_1799_, 2);
v_ref_1807_ = lean_ctor_get(v_a_1799_, 5);
v___x_1808_ = lean_unsigned_to_nat(1u);
v___x_1809_ = l_Lean_Syntax_getArg(v_x_1798_, v___x_1808_);
v___x_1810_ = lean_unsigned_to_nat(3u);
v___x_1811_ = l_Lean_Syntax_getArg(v_x_1798_, v___x_1810_);
lean_dec(v_x_1798_);
v___x_1812_ = 0;
v___x_1813_ = l_Lean_SourceInfo_fromRef(v_ref_1807_, v___x_1812_);
v___x_1814_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3));
v___x_1815_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4));
lean_inc_n(v___x_1813_, 9);
v___x_1816_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1816_, 0, v___x_1813_);
lean_ctor_set(v___x_1816_, 1, v___x_1815_);
v___x_1817_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_1818_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1818_, 0, v___x_1813_);
lean_ctor_set(v___x_1818_, 1, v___x_1817_);
v___x_1819_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7);
v___x_1820_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_1806_, 2);
lean_inc_n(v_quotContext_1805_, 2);
v___x_1821_ = l_Lean_addMacroScope(v_quotContext_1805_, v___x_1820_, v_currMacroScope_1806_);
v___x_1822_ = lean_box(0);
v___x_1823_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1823_, 0, v___x_1813_);
lean_ctor_set(v___x_1823_, 1, v___x_1819_);
lean_ctor_set(v___x_1823_, 2, v___x_1821_);
lean_ctor_set(v___x_1823_, 3, v___x_1822_);
v___x_1824_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9));
v___x_1825_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1825_, 0, v___x_1813_);
lean_ctor_set(v___x_1825_, 1, v___x_1824_);
v___x_1826_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
v___x_1827_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__1, &lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__1_once, _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__1);
v___x_1828_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__2));
v___x_1829_ = l_Lean_addMacroScope(v_quotContext_1805_, v___x_1828_, v_currMacroScope_1806_);
v___x_1830_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__5));
v___x_1831_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1831_, 0, v___x_1813_);
lean_ctor_set(v___x_1831_, 1, v___x_1827_);
lean_ctor_set(v___x_1831_, 2, v___x_1829_);
lean_ctor_set(v___x_1831_, 3, v___x_1830_);
v___x_1832_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
lean_inc_ref(v___x_1823_);
v___x_1833_ = l_Lean_Syntax_node1(v___x_1813_, v___x_1832_, v___x_1823_);
v___x_1834_ = l_Lean_Syntax_node2(v___x_1813_, v___x_1826_, v___x_1831_, v___x_1833_);
v___x_1835_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_1836_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1813_);
lean_ctor_set(v___x_1836_, 1, v___x_1835_);
v___x_1837_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_1838_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1838_, 0, v___x_1813_);
lean_ctor_set(v___x_1838_, 1, v___x_1837_);
v___x_1839_ = lean_unsigned_to_nat(9u);
v___x_1840_ = lean_mk_empty_array_with_capacity(v___x_1839_);
v___x_1841_ = lean_array_push(v___x_1840_, v___x_1816_);
v___x_1842_ = lean_array_push(v___x_1841_, v___x_1818_);
v___x_1843_ = lean_array_push(v___x_1842_, v___x_1823_);
v___x_1844_ = lean_array_push(v___x_1843_, v___x_1825_);
v___x_1845_ = lean_array_push(v___x_1844_, v___x_1834_);
v___x_1846_ = lean_array_push(v___x_1845_, v___x_1836_);
v___x_1847_ = lean_array_push(v___x_1846_, v___x_1809_);
v___x_1848_ = lean_array_push(v___x_1847_, v___x_1838_);
v___x_1849_ = lean_array_push(v___x_1848_, v___x_1811_);
v___x_1850_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1850_, 0, v___x_1813_);
lean_ctor_set(v___x_1850_, 1, v___x_1814_);
lean_ctor_set(v___x_1850_, 2, v___x_1849_);
v___x_1851_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1851_, 0, v___x_1850_);
lean_ctor_set(v___x_1851_, 1, v_a_1800_);
return v___x_1851_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___boxed(lean_object* v_x_1852_, lean_object* v_a_1853_, lean_object* v_a_1854_){
_start:
{
lean_object* v_res_1855_; 
v_res_1855_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1(v_x_1852_, v_a_1853_, v_a_1854_);
lean_dec_ref(v_a_1853_);
return v_res_1855_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__0(lean_object* v_x_1856_){
_start:
{
lean_object* v___x_1857_; uint8_t v___x_1858_; 
v___x_1857_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c3___x2c____1___closed__3));
v___x_1858_ = l_Lean_Expr_isConstOf(v_x_1856_, v___x_1857_);
return v___x_1858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__0___boxed(lean_object* v_x_1859_){
_start:
{
uint8_t v_res_1860_; lean_object* v_r_1861_; 
v_res_1860_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__0(v_x_1859_);
lean_dec_ref(v_x_1859_);
v_r_1861_ = lean_box(v_res_1860_);
return v_r_1861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2(uint8_t v___x_1863_, lean_object* v___x_1864_, lean_object* v_a_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_){
_start:
{
lean_object* v_ref_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; 
v_ref_1873_ = lean_ctor_get(v___y_1870_, 5);
v___x_1874_ = l_Lean_SourceInfo_fromRef(v_ref_1873_, v___x_1863_);
v___x_1875_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
v___x_1876_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_1874_, 2);
v___x_1877_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1877_, 0, v___x_1874_);
lean_ctor_set(v___x_1877_, 1, v___x_1876_);
v___x_1878_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__7));
v___x_1879_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1879_, 0, v___x_1874_);
lean_ctor_set(v___x_1879_, 1, v___x_1878_);
v___x_1880_ = l_Lean_Syntax_node4(v___x_1874_, v___x_1875_, v___x_1877_, v___x_1864_, v___x_1879_, v_a_1865_);
v___x_1881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1881_, 0, v___x_1880_);
return v___x_1881_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___boxed(lean_object* v___x_1882_, lean_object* v___x_1883_, lean_object* v_a_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_){
_start:
{
uint8_t v___x_6645__boxed_1892_; lean_object* v_res_1893_; 
v___x_6645__boxed_1892_ = lean_unbox(v___x_1882_);
v_res_1893_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2(v___x_6645__boxed_1892_, v___x_1883_, v_a_1884_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_, v___y_1889_, v___y_1890_);
lean_dec(v___y_1890_);
lean_dec_ref(v___y_1889_);
lean_dec(v___y_1888_);
lean_dec_ref(v___y_1887_);
lean_dec(v___y_1886_);
lean_dec_ref(v___y_1885_);
return v_res_1893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__1(lean_object* v___f_1894_, lean_object* v___f_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_){
_start:
{
lean_object* v___x_1903_; lean_object* v_a_1904_; lean_object* v___x_1906_; uint8_t v_isShared_1907_; uint8_t v_isSharedCheck_1950_; 
v___x_1903_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_1896_);
v_a_1904_ = lean_ctor_get(v___x_1903_, 0);
v_isSharedCheck_1950_ = !lean_is_exclusive(v___x_1903_);
if (v_isSharedCheck_1950_ == 0)
{
v___x_1906_ = v___x_1903_;
v_isShared_1907_ = v_isSharedCheck_1950_;
goto v_resetjp_1905_;
}
else
{
lean_inc(v_a_1904_);
lean_dec(v___x_1903_);
v___x_1906_ = lean_box(0);
v_isShared_1907_ = v_isSharedCheck_1950_;
goto v_resetjp_1905_;
}
v_resetjp_1905_:
{
lean_object* v___x_1908_; lean_object* v___y_1910_; lean_object* v___x_1940_; lean_object* v___x_1941_; 
v___x_1908_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1));
v___x_1940_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_1941_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_1908_, v___x_1940_, v___y_1896_, v___y_1898_);
if (lean_obj_tag(v___x_1941_) == 0)
{
lean_object* v_a_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; 
v_a_1942_ = lean_ctor_get(v___x_1941_, 0);
lean_inc(v_a_1942_);
lean_dec_ref_known(v___x_1941_, 1);
v___x_1943_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_1943_, 0, v___f_1894_);
lean_inc_ref(v___f_1895_);
v___x_1944_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1944_, 0, v___x_1943_);
lean_closure_set(v___x_1944_, 1, v___f_1895_);
v___x_1945_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
v___x_1946_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1946_, 0, v___x_1944_);
lean_closure_set(v___x_1946_, 1, v___f_1895_);
v___x_1947_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9));
v___x_1948_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_1948_, 0, v___x_1946_);
lean_closure_set(v___x_1948_, 1, v___x_1947_);
v___x_1949_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_1908_, v___x_1945_, v___x_1948_, v_a_1942_, v___y_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_);
v___y_1910_ = v___x_1949_;
goto v___jp_1909_;
}
else
{
lean_dec_ref(v___f_1895_);
lean_dec_ref(v___f_1894_);
v___y_1910_ = v___x_1941_;
goto v___jp_1909_;
}
v___jp_1909_:
{
if (lean_obj_tag(v___y_1910_) == 0)
{
lean_object* v_a_1911_; lean_object* v_ref_1912_; lean_object* v___x_1914_; 
v_a_1911_ = lean_ctor_get(v___y_1910_, 0);
lean_inc(v_a_1911_);
lean_dec_ref_known(v___y_1910_, 1);
v_ref_1912_ = lean_ctor_get(v___y_1900_, 5);
if (v_isShared_1907_ == 0)
{
lean_ctor_set_tag(v___x_1906_, 1);
v___x_1914_ = v___x_1906_;
goto v_reusejp_1913_;
}
else
{
lean_object* v_reuseFailAlloc_1931_; 
v_reuseFailAlloc_1931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1931_, 0, v_a_1904_);
v___x_1914_ = v_reuseFailAlloc_1931_;
goto v_reusejp_1913_;
}
v_reusejp_1913_:
{
lean_object* v___x_1915_; 
v___x_1915_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_1911_, v___x_1908_, v___x_1914_, v___y_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_);
if (lean_obj_tag(v___x_1915_) == 0)
{
lean_object* v_a_1916_; uint8_t v___x_1917_; lean_object* v___x_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___f_1929_; lean_object* v___x_1930_; 
v_a_1916_ = lean_ctor_get(v___x_1915_, 0);
lean_inc(v_a_1916_);
lean_dec_ref_known(v___x_1915_, 1);
v___x_1917_ = 0;
v___x_1918_ = l_Lean_SourceInfo_fromRef(v_ref_1912_, v___x_1917_);
v___x_1919_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_1920_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_1921_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_1922_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_1911_);
lean_dec(v_a_1911_);
v___x_1923_ = l_Array_append___redArg(v___x_1921_, v___x_1922_);
lean_dec_ref(v___x_1922_);
lean_inc_n(v___x_1918_, 2);
v___x_1924_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1924_, 0, v___x_1918_);
lean_ctor_set(v___x_1924_, 1, v___x_1920_);
lean_ctor_set(v___x_1924_, 2, v___x_1923_);
v___x_1925_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_1926_ = l_Lean_Syntax_node1(v___x_1918_, v___x_1925_, v___x_1924_);
v___x_1927_ = l_Lean_Syntax_node1(v___x_1918_, v___x_1919_, v___x_1926_);
v___x_1928_ = lean_box(v___x_1917_);
v___f_1929_ = lean_alloc_closure((void*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_1929_, 0, v___x_1928_);
lean_closure_set(v___f_1929_, 1, v___x_1927_);
lean_closure_set(v___f_1929_, 2, v_a_1916_);
v___x_1930_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_1929_, v___y_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_);
return v___x_1930_;
}
else
{
lean_dec(v_a_1911_);
return v___x_1915_;
}
}
}
else
{
lean_object* v_a_1932_; lean_object* v___x_1934_; uint8_t v_isShared_1935_; uint8_t v_isSharedCheck_1939_; 
lean_del_object(v___x_1906_);
lean_dec(v_a_1904_);
v_a_1932_ = lean_ctor_get(v___y_1910_, 0);
v_isSharedCheck_1939_ = !lean_is_exclusive(v___y_1910_);
if (v_isSharedCheck_1939_ == 0)
{
v___x_1934_ = v___y_1910_;
v_isShared_1935_ = v_isSharedCheck_1939_;
goto v_resetjp_1933_;
}
else
{
lean_inc(v_a_1932_);
lean_dec(v___y_1910_);
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
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__1___boxed(lean_object* v___f_1951_, lean_object* v___f_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_){
_start:
{
lean_object* v_res_1960_; 
v_res_1960_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___lam__1(v___f_1951_, v___f_1952_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
lean_dec(v___y_1956_);
lean_dec_ref(v___y_1955_);
lean_dec(v___y_1954_);
lean_dec_ref(v___y_1953_);
return v_res_1960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1(lean_object* v_a_1971_, lean_object* v_a_1972_, lean_object* v_a_1973_, lean_object* v_a_1974_, lean_object* v_a_1975_, lean_object* v_a_1976_){
_start:
{
lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; 
v___x_1978_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_1979_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___closed__3));
v___x_1980_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_1978_, v___x_1979_, v_a_1971_, v_a_1972_, v_a_1973_, v_a_1974_, v_a_1975_, v_a_1976_);
return v___x_1980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1___boxed(lean_object* v_a_1981_, lean_object* v_a_1982_, lean_object* v_a_1983_, lean_object* v_a_1984_, lean_object* v_a_1985_, lean_object* v_a_1986_, lean_object* v_a_1987_){
_start:
{
lean_object* v_res_1988_; 
v_res_1988_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c3___x2c____1(v_a_1981_, v_a_1982_, v_a_1983_, v_a_1984_, v_a_1985_, v_a_1986_);
lean_dec(v_a_1986_);
lean_dec_ref(v_a_1985_);
lean_dec(v_a_1984_);
lean_dec_ref(v_a_1983_);
lean_dec(v_a_1982_);
lean_dec_ref(v_a_1981_);
return v_res_1988_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; 
v___x_1996_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_1997_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__3));
v___x_1998_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_1999_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1999_, 0, v___x_1998_);
lean_ctor_set(v___x_1999_, 1, v___x_1997_);
lean_ctor_set(v___x_1999_, 2, v___x_1996_);
return v___x_1999_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__5(void){
_start:
{
lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; 
v___x_2000_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__8));
v___x_2001_ = lean_obj_once(&lp_mathlib_Set_term_u22c2___x2c___00__closed__4, &lp_mathlib_Set_term_u22c2___x2c___00__closed__4_once, _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__4);
v___x_2002_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_2003_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2003_, 0, v___x_2002_);
lean_ctor_set(v___x_2003_, 1, v___x_2001_);
lean_ctor_set(v___x_2003_, 2, v___x_2000_);
return v___x_2003_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; 
v___x_2004_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__12));
v___x_2005_ = lean_obj_once(&lp_mathlib_Set_term_u22c2___x2c___00__closed__5, &lp_mathlib_Set_term_u22c2___x2c___00__closed__5_once, _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__5);
v___x_2006_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__3));
v___x_2007_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2007_, 0, v___x_2006_);
lean_ctor_set(v___x_2007_, 1, v___x_2005_);
lean_ctor_set(v___x_2007_, 2, v___x_2004_);
return v___x_2007_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; 
v___x_2008_ = lean_obj_once(&lp_mathlib_Set_term_u22c2___x2c___00__closed__6, &lp_mathlib_Set_term_u22c2___x2c___00__closed__6_once, _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__6);
v___x_2009_ = lean_unsigned_to_nat(1022u);
v___x_2010_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
v___x_2011_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_2011_, 0, v___x_2010_);
lean_ctor_set(v___x_2011_, 1, v___x_2009_);
lean_ctor_set(v___x_2011_, 2, v___x_2008_);
return v___x_2011_;
}
}
static lean_object* _init_lp_mathlib_Set_term_u22c2___x2c__(void){
_start:
{
lean_object* v___x_2012_; 
v___x_2012_ = lean_obj_once(&lp_mathlib_Set_term_u22c2___x2c___00__closed__7, &lp_mathlib_Set_term_u22c2___x2c___00__closed__7_once, _init_lp_mathlib_Set_term_u22c2___x2c___00__closed__7);
return v___x_2012_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__1(void){
_start:
{
lean_object* v___x_2014_; lean_object* v___x_2015_; 
v___x_2014_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__0));
v___x_2015_ = l_String_toRawSubstring_x27(v___x_2014_);
return v___x_2015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1(lean_object* v_x_2027_, lean_object* v_a_2028_, lean_object* v_a_2029_){
_start:
{
lean_object* v___x_2030_; uint8_t v___x_2031_; 
v___x_2030_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
lean_inc(v_x_2027_);
v___x_2031_ = l_Lean_Syntax_isOfKind(v_x_2027_, v___x_2030_);
if (v___x_2031_ == 0)
{
lean_object* v___x_2032_; lean_object* v___x_2033_; 
lean_dec(v_x_2027_);
v___x_2032_ = lean_box(1);
v___x_2033_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2033_, 0, v___x_2032_);
lean_ctor_set(v___x_2033_, 1, v_a_2029_);
return v___x_2033_;
}
else
{
lean_object* v_quotContext_2034_; lean_object* v_currMacroScope_2035_; lean_object* v_ref_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; uint8_t v___x_2041_; lean_object* v___x_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; lean_object* v___x_2072_; lean_object* v___x_2073_; lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; lean_object* v___x_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; 
v_quotContext_2034_ = lean_ctor_get(v_a_2028_, 1);
v_currMacroScope_2035_ = lean_ctor_get(v_a_2028_, 2);
v_ref_2036_ = lean_ctor_get(v_a_2028_, 5);
v___x_2037_ = lean_unsigned_to_nat(1u);
v___x_2038_ = l_Lean_Syntax_getArg(v_x_2027_, v___x_2037_);
v___x_2039_ = lean_unsigned_to_nat(3u);
v___x_2040_ = l_Lean_Syntax_getArg(v_x_2027_, v___x_2039_);
lean_dec(v_x_2027_);
v___x_2041_ = 0;
v___x_2042_ = l_Lean_SourceInfo_fromRef(v_ref_2036_, v___x_2041_);
v___x_2043_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__3));
v___x_2044_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__4));
lean_inc_n(v___x_2042_, 9);
v___x_2045_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2045_, 0, v___x_2042_);
lean_ctor_set(v___x_2045_, 1, v___x_2044_);
v___x_2046_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_2047_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2047_, 0, v___x_2042_);
lean_ctor_set(v___x_2047_, 1, v___x_2046_);
v___x_2048_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__7);
v___x_2049_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_2035_, 2);
lean_inc_n(v_quotContext_2034_, 2);
v___x_2050_ = l_Lean_addMacroScope(v_quotContext_2034_, v___x_2049_, v_currMacroScope_2035_);
v___x_2051_ = lean_box(0);
v___x_2052_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2052_, 0, v___x_2042_);
lean_ctor_set(v___x_2052_, 1, v___x_2048_);
lean_ctor_set(v___x_2052_, 2, v___x_2050_);
lean_ctor_set(v___x_2052_, 3, v___x_2051_);
v___x_2053_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__9));
v___x_2054_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2054_, 0, v___x_2042_);
lean_ctor_set(v___x_2054_, 1, v___x_2053_);
v___x_2055_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__14));
v___x_2056_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__1, &lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__1_once, _init_lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__1);
v___x_2057_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__2));
v___x_2058_ = l_Lean_addMacroScope(v_quotContext_2034_, v___x_2057_, v_currMacroScope_2035_);
v___x_2059_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__5));
v___x_2060_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2060_, 0, v___x_2042_);
lean_ctor_set(v___x_2060_, 1, v___x_2056_);
lean_ctor_set(v___x_2060_, 2, v___x_2058_);
lean_ctor_set(v___x_2060_, 3, v___x_2059_);
v___x_2061_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
lean_inc_ref(v___x_2052_);
v___x_2062_ = l_Lean_Syntax_node1(v___x_2042_, v___x_2061_, v___x_2052_);
v___x_2063_ = l_Lean_Syntax_node2(v___x_2042_, v___x_2055_, v___x_2060_, v___x_2062_);
v___x_2064_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_2065_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2065_, 0, v___x_2042_);
lean_ctor_set(v___x_2065_, 1, v___x_2064_);
v___x_2066_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2067_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2067_, 0, v___x_2042_);
lean_ctor_set(v___x_2067_, 1, v___x_2066_);
v___x_2068_ = lean_unsigned_to_nat(9u);
v___x_2069_ = lean_mk_empty_array_with_capacity(v___x_2068_);
v___x_2070_ = lean_array_push(v___x_2069_, v___x_2045_);
v___x_2071_ = lean_array_push(v___x_2070_, v___x_2047_);
v___x_2072_ = lean_array_push(v___x_2071_, v___x_2052_);
v___x_2073_ = lean_array_push(v___x_2072_, v___x_2054_);
v___x_2074_ = lean_array_push(v___x_2073_, v___x_2063_);
v___x_2075_ = lean_array_push(v___x_2074_, v___x_2065_);
v___x_2076_ = lean_array_push(v___x_2075_, v___x_2038_);
v___x_2077_ = lean_array_push(v___x_2076_, v___x_2067_);
v___x_2078_ = lean_array_push(v___x_2077_, v___x_2040_);
v___x_2079_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2079_, 0, v___x_2042_);
lean_ctor_set(v___x_2079_, 1, v___x_2043_);
lean_ctor_set(v___x_2079_, 2, v___x_2078_);
v___x_2080_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2080_, 0, v___x_2079_);
lean_ctor_set(v___x_2080_, 1, v_a_2029_);
return v___x_2080_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___boxed(lean_object* v_x_2081_, lean_object* v_a_2082_, lean_object* v_a_2083_){
_start:
{
lean_object* v_res_2084_; 
v_res_2084_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1(v_x_2081_, v_a_2082_, v_a_2083_);
lean_dec_ref(v_a_2082_);
return v_res_2084_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__0(lean_object* v_x_2085_){
_start:
{
lean_object* v___x_2086_; uint8_t v___x_2087_; 
v___x_2086_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______macroRules__Set__term_u22c2___x2c____1___closed__3));
v___x_2087_ = l_Lean_Expr_isConstOf(v_x_2085_, v___x_2086_);
return v___x_2087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__0___boxed(lean_object* v_x_2088_){
_start:
{
uint8_t v_res_2089_; lean_object* v_r_2090_; 
v_res_2089_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__0(v_x_2088_);
lean_dec_ref(v_x_2088_);
v_r_2090_ = lean_box(v_res_2089_);
return v_r_2090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2(uint8_t v___x_2092_, lean_object* v___x_2093_, lean_object* v_a_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_, lean_object* v___y_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_){
_start:
{
lean_object* v_ref_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; 
v_ref_2102_ = lean_ctor_get(v___y_2099_, 5);
v___x_2103_ = l_Lean_SourceInfo_fromRef(v_ref_2102_, v___x_2092_);
v___x_2104_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
v___x_2105_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_2103_, 2);
v___x_2106_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2106_, 0, v___x_2103_);
lean_ctor_set(v___x_2106_, 1, v___x_2105_);
v___x_2107_ = ((lean_object*)(lp_mathlib_term_u2a06___x2c___00__closed__7));
v___x_2108_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2108_, 0, v___x_2103_);
lean_ctor_set(v___x_2108_, 1, v___x_2107_);
v___x_2109_ = l_Lean_Syntax_node4(v___x_2103_, v___x_2104_, v___x_2106_, v___x_2093_, v___x_2108_, v_a_2094_);
v___x_2110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2110_, 0, v___x_2109_);
return v___x_2110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___boxed(lean_object* v___x_2111_, lean_object* v___x_2112_, lean_object* v_a_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_){
_start:
{
uint8_t v___x_6645__boxed_2121_; lean_object* v_res_2122_; 
v___x_6645__boxed_2121_ = lean_unbox(v___x_2111_);
v_res_2122_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2(v___x_6645__boxed_2121_, v___x_2112_, v_a_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_);
lean_dec(v___y_2119_);
lean_dec_ref(v___y_2118_);
lean_dec(v___y_2117_);
lean_dec_ref(v___y_2116_);
lean_dec(v___y_2115_);
lean_dec_ref(v___y_2114_);
return v_res_2122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__1(lean_object* v___f_2123_, lean_object* v___f_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_){
_start:
{
lean_object* v___x_2132_; lean_object* v_a_2133_; lean_object* v___x_2135_; uint8_t v_isShared_2136_; uint8_t v_isSharedCheck_2179_; 
v___x_2132_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_2125_);
v_a_2133_ = lean_ctor_get(v___x_2132_, 0);
v_isSharedCheck_2179_ = !lean_is_exclusive(v___x_2132_);
if (v_isSharedCheck_2179_ == 0)
{
v___x_2135_ = v___x_2132_;
v_isShared_2136_ = v_isSharedCheck_2179_;
goto v_resetjp_2134_;
}
else
{
lean_inc(v_a_2133_);
lean_dec(v___x_2132_);
v___x_2135_ = lean_box(0);
v_isShared_2136_ = v_isSharedCheck_2179_;
goto v_resetjp_2134_;
}
v_resetjp_2134_:
{
lean_object* v___x_2137_; lean_object* v___y_2139_; lean_object* v___x_2169_; lean_object* v___x_2170_; 
v___x_2137_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__1));
v___x_2169_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_2170_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_2137_, v___x_2169_, v___y_2125_, v___y_2127_);
if (lean_obj_tag(v___x_2170_) == 0)
{
lean_object* v_a_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; 
v_a_2171_ = lean_ctor_get(v___x_2170_, 0);
lean_inc(v_a_2171_);
lean_dec_ref_known(v___x_2170_, 1);
v___x_2172_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_2172_, 0, v___f_2123_);
lean_inc_ref(v___f_2124_);
v___x_2173_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_2173_, 0, v___x_2172_);
lean_closure_set(v___x_2173_, 1, v___f_2124_);
v___x_2174_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__8));
v___x_2175_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_2175_, 0, v___x_2173_);
lean_closure_set(v___x_2175_, 1, v___f_2124_);
v___x_2176_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__9));
v___x_2177_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_2177_, 0, v___x_2175_);
lean_closure_set(v___x_2177_, 1, v___x_2176_);
v___x_2178_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_2137_, v___x_2174_, v___x_2177_, v_a_2171_, v___y_2125_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_);
v___y_2139_ = v___x_2178_;
goto v___jp_2138_;
}
else
{
lean_dec_ref(v___f_2124_);
lean_dec_ref(v___f_2123_);
v___y_2139_ = v___x_2170_;
goto v___jp_2138_;
}
v___jp_2138_:
{
if (lean_obj_tag(v___y_2139_) == 0)
{
lean_object* v_a_2140_; lean_object* v_ref_2141_; lean_object* v___x_2143_; 
v_a_2140_ = lean_ctor_get(v___y_2139_, 0);
lean_inc(v_a_2140_);
lean_dec_ref_known(v___y_2139_, 1);
v_ref_2141_ = lean_ctor_get(v___y_2129_, 5);
if (v_isShared_2136_ == 0)
{
lean_ctor_set_tag(v___x_2135_, 1);
v___x_2143_ = v___x_2135_;
goto v_reusejp_2142_;
}
else
{
lean_object* v_reuseFailAlloc_2160_; 
v_reuseFailAlloc_2160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2160_, 0, v_a_2133_);
v___x_2143_ = v_reuseFailAlloc_2160_;
goto v_reusejp_2142_;
}
v_reusejp_2142_:
{
lean_object* v___x_2144_; 
v___x_2144_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_2140_, v___x_2137_, v___x_2143_, v___y_2125_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_);
if (lean_obj_tag(v___x_2144_) == 0)
{
lean_object* v_a_2145_; uint8_t v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; lean_object* v___x_2157_; lean_object* v___f_2158_; lean_object* v___x_2159_; 
v_a_2145_ = lean_ctor_get(v___x_2144_, 0);
lean_inc(v_a_2145_);
lean_dec_ref_known(v___x_2144_, 1);
v___x_2146_ = 0;
v___x_2147_ = l_Lean_SourceInfo_fromRef(v_ref_2141_, v___x_2146_);
v___x_2148_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2149_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2150_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_2151_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_2140_);
lean_dec(v_a_2140_);
v___x_2152_ = l_Array_append___redArg(v___x_2150_, v___x_2151_);
lean_dec_ref(v___x_2151_);
lean_inc_n(v___x_2147_, 2);
v___x_2153_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2153_, 0, v___x_2147_);
lean_ctor_set(v___x_2153_, 1, v___x_2149_);
lean_ctor_set(v___x_2153_, 2, v___x_2152_);
v___x_2154_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_2155_ = l_Lean_Syntax_node1(v___x_2147_, v___x_2154_, v___x_2153_);
v___x_2156_ = l_Lean_Syntax_node1(v___x_2147_, v___x_2148_, v___x_2155_);
v___x_2157_ = lean_box(v___x_2146_);
v___f_2158_ = lean_alloc_closure((void*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_2158_, 0, v___x_2157_);
lean_closure_set(v___f_2158_, 1, v___x_2156_);
lean_closure_set(v___f_2158_, 2, v_a_2145_);
v___x_2159_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_2158_, v___y_2125_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_);
return v___x_2159_;
}
else
{
lean_dec(v_a_2140_);
return v___x_2144_;
}
}
}
else
{
lean_object* v_a_2161_; lean_object* v___x_2163_; uint8_t v_isShared_2164_; uint8_t v_isSharedCheck_2168_; 
lean_del_object(v___x_2135_);
lean_dec(v_a_2133_);
v_a_2161_ = lean_ctor_get(v___y_2139_, 0);
v_isSharedCheck_2168_ = !lean_is_exclusive(v___y_2139_);
if (v_isSharedCheck_2168_ == 0)
{
v___x_2163_ = v___y_2139_;
v_isShared_2164_ = v_isSharedCheck_2168_;
goto v_resetjp_2162_;
}
else
{
lean_inc(v_a_2161_);
lean_dec(v___y_2139_);
v___x_2163_ = lean_box(0);
v_isShared_2164_ = v_isSharedCheck_2168_;
goto v_resetjp_2162_;
}
v_resetjp_2162_:
{
lean_object* v___x_2166_; 
if (v_isShared_2164_ == 0)
{
v___x_2166_ = v___x_2163_;
goto v_reusejp_2165_;
}
else
{
lean_object* v_reuseFailAlloc_2167_; 
v_reuseFailAlloc_2167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2167_, 0, v_a_2161_);
v___x_2166_ = v_reuseFailAlloc_2167_;
goto v_reusejp_2165_;
}
v_reusejp_2165_:
{
return v___x_2166_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__1___boxed(lean_object* v___f_2180_, lean_object* v___f_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_){
_start:
{
lean_object* v_res_2189_; 
v_res_2189_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___lam__1(v___f_2180_, v___f_2181_, v___y_2182_, v___y_2183_, v___y_2184_, v___y_2185_, v___y_2186_, v___y_2187_);
lean_dec(v___y_2187_);
lean_dec_ref(v___y_2186_);
lean_dec(v___y_2185_);
lean_dec_ref(v___y_2184_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
return v_res_2189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1(lean_object* v_a_2200_, lean_object* v_a_2201_, lean_object* v_a_2202_, lean_object* v_a_2203_, lean_object* v_a_2204_, lean_object* v_a_2205_){
_start:
{
lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; 
v___x_2207_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_2208_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___closed__3));
v___x_2209_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_2207_, v___x_2208_, v_a_2200_, v_a_2201_, v_a_2202_, v_a_2203_, v_a_2204_, v_a_2205_);
return v___x_2209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1___boxed(lean_object* v_a_2210_, lean_object* v_a_2211_, lean_object* v_a_2212_, lean_object* v_a_2213_, lean_object* v_a_2214_, lean_object* v_a_2215_, lean_object* v_a_2216_){
_start:
{
lean_object* v_res_2217_; 
v_res_2217_ = lp_mathlib_Set___aux__Mathlib__Order__SetNotation______delab__app__Set__term_u22c2___x2c____1(v_a_2210_, v_a_2211_, v_a_2212_, v_a_2213_, v_a_2214_, v_a_2215_);
lean_dec(v_a_2215_);
lean_dec_ref(v_a_2214_);
lean_dec(v_a_2213_);
lean_dec_ref(v_a_2212_);
lean_dec(v_a_2211_);
lean_dec_ref(v_a_2210_);
return v_res_2217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__0(lean_object* v_a_2218_, uint8_t v_a_2219_, uint8_t v_a_2220_, uint8_t v___x_2221_, lean_object* v_x_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_){
_start:
{
lean_object* v___x_2230_; 
v___x_2230_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_2223_, v___y_2224_, v___y_2225_, v___y_2226_, v___y_2227_, v___y_2228_);
if (lean_obj_tag(v___x_2230_) == 0)
{
lean_object* v_a_2231_; lean_object* v___x_2233_; uint8_t v_isShared_2234_; uint8_t v_isSharedCheck_2324_; 
v_a_2231_ = lean_ctor_get(v___x_2230_, 0);
v_isSharedCheck_2324_ = !lean_is_exclusive(v___x_2230_);
if (v_isSharedCheck_2324_ == 0)
{
v___x_2233_ = v___x_2230_;
v_isShared_2234_ = v_isSharedCheck_2324_;
goto v_resetjp_2232_;
}
else
{
lean_inc(v_a_2231_);
lean_dec(v___x_2230_);
v___x_2233_ = lean_box(0);
v_isShared_2234_ = v_isSharedCheck_2324_;
goto v_resetjp_2232_;
}
v_resetjp_2232_:
{
uint8_t v___y_2236_; uint8_t v___y_2270_; 
if (v_a_2219_ == 0)
{
v___y_2270_ = v_a_2219_;
goto v___jp_2269_;
}
else
{
if (v___x_2221_ == 0)
{
lean_object* v_ref_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; lean_object* v___x_2321_; lean_object* v___x_2322_; 
lean_del_object(v___x_2233_);
lean_dec(v_x_2222_);
v_ref_2289_ = lean_ctor_get(v___y_2227_, 5);
v___x_2290_ = l_Lean_SourceInfo_fromRef(v_ref_2289_, v___x_2221_);
v___x_2291_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
v___x_2292_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__2));
lean_inc_n(v___x_2290_, 15);
v___x_2293_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2293_, 0, v___x_2290_);
lean_ctor_set(v___x_2293_, 1, v___x_2292_);
v___x_2294_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2295_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_2296_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2297_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_2298_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_2299_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2299_, 0, v___x_2290_);
lean_ctor_set(v___x_2299_, 1, v___x_2298_);
v___x_2300_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_2301_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_2302_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_2303_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__11));
v___x_2304_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2304_, 0, v___x_2290_);
lean_ctor_set(v___x_2304_, 1, v___x_2303_);
v___x_2305_ = l_Lean_Syntax_node1(v___x_2290_, v___x_2302_, v___x_2304_);
v___x_2306_ = l_Lean_Syntax_node1(v___x_2290_, v___x_2301_, v___x_2305_);
v___x_2307_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_2308_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_2309_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2309_, 0, v___x_2290_);
lean_ctor_set(v___x_2309_, 1, v___x_2308_);
v___x_2310_ = l_Lean_Syntax_node2(v___x_2290_, v___x_2307_, v___x_2309_, v_a_2218_);
v___x_2311_ = l_Lean_Syntax_node1(v___x_2290_, v___x_2296_, v___x_2310_);
v___x_2312_ = l_Lean_Syntax_node2(v___x_2290_, v___x_2300_, v___x_2306_, v___x_2311_);
v___x_2313_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_2314_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2314_, 0, v___x_2290_);
lean_ctor_set(v___x_2314_, 1, v___x_2313_);
v___x_2315_ = l_Lean_Syntax_node3(v___x_2290_, v___x_2297_, v___x_2299_, v___x_2312_, v___x_2314_);
v___x_2316_ = l_Lean_Syntax_node1(v___x_2290_, v___x_2296_, v___x_2315_);
v___x_2317_ = l_Lean_Syntax_node1(v___x_2290_, v___x_2295_, v___x_2316_);
v___x_2318_ = l_Lean_Syntax_node1(v___x_2290_, v___x_2294_, v___x_2317_);
v___x_2319_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2320_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2320_, 0, v___x_2290_);
lean_ctor_set(v___x_2320_, 1, v___x_2319_);
v___x_2321_ = l_Lean_Syntax_node4(v___x_2290_, v___x_2291_, v___x_2293_, v___x_2318_, v___x_2320_, v_a_2231_);
v___x_2322_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2322_, 0, v___x_2321_);
return v___x_2322_;
}
else
{
uint8_t v___x_2323_; 
v___x_2323_ = 0;
v___y_2270_ = v___x_2323_;
goto v___jp_2269_;
}
}
v___jp_2235_:
{
lean_object* v_ref_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2267_; 
v_ref_2237_ = lean_ctor_get(v___y_2227_, 5);
v___x_2238_ = l_Lean_SourceInfo_fromRef(v_ref_2237_, v___y_2236_);
v___x_2239_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
v___x_2240_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__2));
lean_inc_n(v___x_2238_, 13);
v___x_2241_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2241_, 0, v___x_2238_);
lean_ctor_set(v___x_2241_, 1, v___x_2240_);
v___x_2242_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2243_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_2244_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2245_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_2246_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_2247_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2247_, 0, v___x_2238_);
lean_ctor_set(v___x_2247_, 1, v___x_2246_);
v___x_2248_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_2249_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_2250_ = l_Lean_Syntax_node1(v___x_2238_, v___x_2249_, v_x_2222_);
v___x_2251_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_2252_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_2253_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2253_, 0, v___x_2238_);
lean_ctor_set(v___x_2253_, 1, v___x_2252_);
v___x_2254_ = l_Lean_Syntax_node2(v___x_2238_, v___x_2251_, v___x_2253_, v_a_2218_);
v___x_2255_ = l_Lean_Syntax_node1(v___x_2238_, v___x_2244_, v___x_2254_);
v___x_2256_ = l_Lean_Syntax_node2(v___x_2238_, v___x_2248_, v___x_2250_, v___x_2255_);
v___x_2257_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_2258_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2258_, 0, v___x_2238_);
lean_ctor_set(v___x_2258_, 1, v___x_2257_);
v___x_2259_ = l_Lean_Syntax_node3(v___x_2238_, v___x_2245_, v___x_2247_, v___x_2256_, v___x_2258_);
v___x_2260_ = l_Lean_Syntax_node1(v___x_2238_, v___x_2244_, v___x_2259_);
v___x_2261_ = l_Lean_Syntax_node1(v___x_2238_, v___x_2243_, v___x_2260_);
v___x_2262_ = l_Lean_Syntax_node1(v___x_2238_, v___x_2242_, v___x_2261_);
v___x_2263_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2264_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2264_, 0, v___x_2238_);
lean_ctor_set(v___x_2264_, 1, v___x_2263_);
v___x_2265_ = l_Lean_Syntax_node4(v___x_2238_, v___x_2239_, v___x_2241_, v___x_2262_, v___x_2264_, v_a_2231_);
if (v_isShared_2234_ == 0)
{
lean_ctor_set(v___x_2233_, 0, v___x_2265_);
v___x_2267_ = v___x_2233_;
goto v_reusejp_2266_;
}
else
{
lean_object* v_reuseFailAlloc_2268_; 
v_reuseFailAlloc_2268_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2268_, 0, v___x_2265_);
v___x_2267_ = v_reuseFailAlloc_2268_;
goto v_reusejp_2266_;
}
v_reusejp_2266_:
{
return v___x_2267_;
}
}
v___jp_2269_:
{
if (v_a_2219_ == 0)
{
if (v_a_2220_ == 0)
{
lean_object* v_ref_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; 
lean_del_object(v___x_2233_);
lean_dec(v_a_2218_);
v_ref_2271_ = lean_ctor_get(v___y_2227_, 5);
v___x_2272_ = l_Lean_SourceInfo_fromRef(v_ref_2271_, v_a_2220_);
v___x_2273_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
v___x_2274_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__2));
lean_inc_n(v___x_2272_, 6);
v___x_2275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2272_);
lean_ctor_set(v___x_2275_, 1, v___x_2274_);
v___x_2276_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2277_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_2278_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_2279_ = l_Lean_Syntax_node1(v___x_2272_, v___x_2278_, v_x_2222_);
v___x_2280_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2281_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_2282_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2282_, 0, v___x_2272_);
lean_ctor_set(v___x_2282_, 1, v___x_2280_);
lean_ctor_set(v___x_2282_, 2, v___x_2281_);
v___x_2283_ = l_Lean_Syntax_node2(v___x_2272_, v___x_2277_, v___x_2279_, v___x_2282_);
v___x_2284_ = l_Lean_Syntax_node1(v___x_2272_, v___x_2276_, v___x_2283_);
v___x_2285_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2286_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2286_, 0, v___x_2272_);
lean_ctor_set(v___x_2286_, 1, v___x_2285_);
v___x_2287_ = l_Lean_Syntax_node4(v___x_2272_, v___x_2273_, v___x_2275_, v___x_2284_, v___x_2286_, v_a_2231_);
v___x_2288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2288_, 0, v___x_2287_);
return v___x_2288_;
}
else
{
v___y_2236_ = v___y_2270_;
goto v___jp_2235_;
}
}
else
{
v___y_2236_ = v___y_2270_;
goto v___jp_2235_;
}
}
}
}
else
{
lean_dec(v_x_2222_);
lean_dec(v_a_2218_);
return v___x_2230_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__0___boxed(lean_object* v_a_2325_, lean_object* v_a_2326_, lean_object* v_a_2327_, lean_object* v___x_2328_, lean_object* v_x_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_){
_start:
{
uint8_t v_a_69105__boxed_2337_; uint8_t v_a_69106__boxed_2338_; uint8_t v___x_69107__boxed_2339_; lean_object* v_res_2340_; 
v_a_69105__boxed_2337_ = lean_unbox(v_a_2326_);
v_a_69106__boxed_2338_ = lean_unbox(v_a_2327_);
v___x_69107__boxed_2339_ = lean_unbox(v___x_2328_);
v_res_2340_ = lp_mathlib_Set_iUnion__delab___lam__0(v_a_2325_, v_a_69105__boxed_2337_, v_a_69106__boxed_2338_, v___x_69107__boxed_2339_, v_x_2329_, v___y_2330_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, v___y_2335_);
lean_dec(v___y_2335_);
lean_dec_ref(v___y_2334_);
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec(v___y_2331_);
lean_dec_ref(v___y_2330_);
return v_res_2340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__1(lean_object* v___x_2341_, uint8_t v_a_2342_, uint8_t v_a_2343_, uint8_t v___x_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_, lean_object* v___y_2347_, lean_object* v___y_2348_, lean_object* v___y_2349_, lean_object* v___y_2350_){
_start:
{
lean_object* v___x_2352_; 
v___x_2352_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(v___x_2341_, v___y_2345_, v___y_2346_, v___y_2347_, v___y_2348_, v___y_2349_, v___y_2350_);
if (lean_obj_tag(v___x_2352_) == 0)
{
lean_object* v_a_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___f_2357_; uint8_t v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; 
v_a_2353_ = lean_ctor_get(v___x_2352_, 0);
lean_inc(v_a_2353_);
lean_dec_ref_known(v___x_2352_, 1);
v___x_2354_ = lean_box(v_a_2342_);
v___x_2355_ = lean_box(v_a_2343_);
v___x_2356_ = lean_box(v___x_2344_);
v___f_2357_ = lean_alloc_closure((void*)(lp_mathlib_Set_iUnion__delab___lam__0___boxed), 12, 4);
lean_closure_set(v___f_2357_, 0, v_a_2353_);
lean_closure_set(v___f_2357_, 1, v___x_2354_);
lean_closure_set(v___f_2357_, 2, v___x_2355_);
lean_closure_set(v___f_2357_, 3, v___x_2356_);
v___x_2358_ = 0;
v___x_2359_ = l_Lean_NameSet_empty;
v___x_2360_ = l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(v___f_2357_, v___x_2358_, v___x_2359_, v___y_2345_, v___y_2346_, v___y_2347_, v___y_2348_, v___y_2349_, v___y_2350_);
return v___x_2360_;
}
else
{
return v___x_2352_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__1___boxed(lean_object* v___x_2361_, lean_object* v_a_2362_, lean_object* v_a_2363_, lean_object* v___x_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_, lean_object* v___y_2370_, lean_object* v___y_2371_){
_start:
{
uint8_t v_a_69359__boxed_2372_; uint8_t v_a_69360__boxed_2373_; uint8_t v___x_69361__boxed_2374_; lean_object* v_res_2375_; 
v_a_69359__boxed_2372_ = lean_unbox(v_a_2362_);
v_a_69360__boxed_2373_ = lean_unbox(v_a_2363_);
v___x_69361__boxed_2374_ = lean_unbox(v___x_2364_);
v_res_2375_ = lp_mathlib_Set_iUnion__delab___lam__1(v___x_2361_, v_a_69359__boxed_2372_, v_a_69360__boxed_2373_, v___x_69361__boxed_2374_, v___y_2365_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_, v___y_2370_);
lean_dec(v___y_2370_);
lean_dec_ref(v___y_2369_);
lean_dec(v___y_2368_);
lean_dec_ref(v___y_2367_);
lean_dec(v___y_2366_);
lean_dec_ref(v___y_2365_);
return v_res_2375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__2(lean_object* v___y_2376_, lean_object* v___y_2377_, lean_object* v___y_2378_, lean_object* v___y_2379_, lean_object* v___y_2380_, lean_object* v___y_2381_){
_start:
{
lean_object* v___x_2383_; lean_object* v_a_2384_; lean_object* v_dummy_2385_; lean_object* v_nargs_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; uint8_t v___x_2393_; 
v___x_2383_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_2376_);
v_a_2384_ = lean_ctor_get(v___x_2383_, 0);
lean_inc(v_a_2384_);
lean_dec_ref(v___x_2383_);
v_dummy_2385_ = lean_obj_once(&lp_mathlib_iSup__delab___lam__2___closed__0, &lp_mathlib_iSup__delab___lam__2___closed__0_once, _init_lp_mathlib_iSup__delab___lam__2___closed__0);
v_nargs_2386_ = l_Lean_Expr_getAppNumArgs(v_a_2384_);
lean_inc(v_nargs_2386_);
v___x_2387_ = lean_mk_array(v_nargs_2386_, v_dummy_2385_);
v___x_2388_ = lean_unsigned_to_nat(1u);
v___x_2389_ = lean_nat_sub(v_nargs_2386_, v___x_2388_);
lean_dec(v_nargs_2386_);
v___x_2390_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_2384_, v___x_2387_, v___x_2389_);
v___x_2391_ = lean_array_get_size(v___x_2390_);
v___x_2392_ = lean_unsigned_to_nat(3u);
v___x_2393_ = lean_nat_dec_eq(v___x_2391_, v___x_2392_);
if (v___x_2393_ == 0)
{
lean_object* v___x_2394_; 
lean_dec_ref(v___x_2390_);
v___x_2394_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2394_;
}
else
{
lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v___y_2399_; lean_object* v___y_2400_; lean_object* v___y_2401_; lean_object* v___y_2402_; lean_object* v___y_2403_; lean_object* v___y_2404_; uint8_t v___x_2580_; 
v___x_2395_ = lean_array_fget(v___x_2390_, v___x_2388_);
v___x_2396_ = lean_unsigned_to_nat(2u);
v___x_2397_ = lean_array_fget(v___x_2390_, v___x_2396_);
lean_dec_ref(v___x_2390_);
v___x_2580_ = l_Lean_Expr_isLambda(v___x_2397_);
if (v___x_2580_ == 0)
{
lean_object* v___x_2581_; 
v___x_2581_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_2581_) == 0)
{
lean_dec_ref_known(v___x_2581_, 1);
v___y_2399_ = v___y_2376_;
v___y_2400_ = v___y_2377_;
v___y_2401_ = v___y_2378_;
v___y_2402_ = v___y_2379_;
v___y_2403_ = v___y_2380_;
v___y_2404_ = v___y_2381_;
goto v___jp_2398_;
}
else
{
lean_object* v_a_2582_; lean_object* v___x_2584_; uint8_t v_isShared_2585_; uint8_t v_isSharedCheck_2589_; 
lean_dec(v___x_2397_);
lean_dec(v___x_2395_);
v_a_2582_ = lean_ctor_get(v___x_2581_, 0);
v_isSharedCheck_2589_ = !lean_is_exclusive(v___x_2581_);
if (v_isSharedCheck_2589_ == 0)
{
v___x_2584_ = v___x_2581_;
v_isShared_2585_ = v_isSharedCheck_2589_;
goto v_resetjp_2583_;
}
else
{
lean_inc(v_a_2582_);
lean_dec(v___x_2581_);
v___x_2584_ = lean_box(0);
v_isShared_2585_ = v_isSharedCheck_2589_;
goto v_resetjp_2583_;
}
v_resetjp_2583_:
{
lean_object* v___x_2587_; 
if (v_isShared_2585_ == 0)
{
v___x_2587_ = v___x_2584_;
goto v_reusejp_2586_;
}
else
{
lean_object* v_reuseFailAlloc_2588_; 
v_reuseFailAlloc_2588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2588_, 0, v_a_2582_);
v___x_2587_ = v_reuseFailAlloc_2588_;
goto v_reusejp_2586_;
}
v_reusejp_2586_:
{
return v___x_2587_;
}
}
}
}
else
{
v___y_2399_ = v___y_2376_;
v___y_2400_ = v___y_2377_;
v___y_2401_ = v___y_2378_;
v___y_2402_ = v___y_2379_;
v___y_2403_ = v___y_2380_;
v___y_2404_ = v___y_2381_;
goto v___jp_2398_;
}
v___jp_2398_:
{
lean_object* v___x_2405_; 
v___x_2405_ = l_Lean_Meta_isProp(v___x_2395_, v___y_2401_, v___y_2402_, v___y_2403_, v___y_2404_);
if (lean_obj_tag(v___x_2405_) == 0)
{
lean_object* v_a_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; 
v_a_2406_ = lean_ctor_get(v___x_2405_, 0);
lean_inc(v_a_2406_);
lean_dec_ref_known(v___x_2405_, 1);
v___x_2407_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__1));
v___x_2408_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_2407_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_, v___y_2403_, v___y_2404_);
if (lean_obj_tag(v___x_2408_) == 0)
{
lean_object* v_a_2409_; lean_object* v___x_2410_; lean_object* v___x_2411_; uint8_t v___x_2412_; lean_object* v___x_2413_; lean_object* v___x_2414_; lean_object* v___f_2415_; lean_object* v___x_2416_; 
v_a_2409_ = lean_ctor_get(v___x_2408_, 0);
lean_inc(v_a_2409_);
lean_dec_ref_known(v___x_2408_, 1);
v___x_2410_ = l_Lean_Expr_bindingBody_x21(v___x_2397_);
lean_dec(v___x_2397_);
v___x_2411_ = lean_unsigned_to_nat(0u);
v___x_2412_ = lean_expr_has_loose_bvar(v___x_2410_, v___x_2411_);
lean_dec_ref(v___x_2410_);
v___x_2413_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__2));
v___x_2414_ = lean_box(v___x_2412_);
v___f_2415_ = lean_alloc_closure((void*)(lp_mathlib_Set_iUnion__delab___lam__1___boxed), 11, 4);
lean_closure_set(v___f_2415_, 0, v___x_2413_);
lean_closure_set(v___f_2415_, 1, v_a_2406_);
lean_closure_set(v___f_2415_, 2, v_a_2409_);
lean_closure_set(v___f_2415_, 3, v___x_2414_);
v___x_2416_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(v___f_2415_, v___y_2399_, v___y_2400_, v___y_2401_, v___y_2402_, v___y_2403_, v___y_2404_);
if (lean_obj_tag(v___x_2416_) == 0)
{
lean_object* v_a_2417_; lean_object* v___x_2418_; uint8_t v___x_2419_; 
v_a_2417_ = lean_ctor_get(v___x_2416_, 0);
lean_inc_n(v_a_2417_, 2);
v___x_2418_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__1));
v___x_2419_ = l_Lean_Syntax_isOfKind(v_a_2417_, v___x_2418_);
if (v___x_2419_ == 0)
{
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2420_; lean_object* v___x_2421_; uint8_t v___x_2422_; 
v___x_2420_ = l_Lean_Syntax_getArg(v_a_2417_, v___x_2388_);
v___x_2421_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
lean_inc(v___x_2420_);
v___x_2422_ = l_Lean_Syntax_isOfKind(v___x_2420_, v___x_2421_);
if (v___x_2422_ == 0)
{
lean_dec(v___x_2420_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2423_; lean_object* v___x_2424_; uint8_t v___x_2425_; 
v___x_2423_ = l_Lean_Syntax_getArg(v___x_2420_, v___x_2411_);
lean_dec(v___x_2420_);
v___x_2424_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
lean_inc(v___x_2423_);
v___x_2425_ = l_Lean_Syntax_isOfKind(v___x_2423_, v___x_2424_);
if (v___x_2425_ == 0)
{
lean_object* v___x_2426_; uint8_t v___x_2427_; 
v___x_2426_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_2423_);
v___x_2427_ = l_Lean_Syntax_isOfKind(v___x_2423_, v___x_2426_);
if (v___x_2427_ == 0)
{
lean_dec(v___x_2423_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2428_; uint8_t v___x_2429_; 
v___x_2428_ = l_Lean_Syntax_getArg(v___x_2423_, v___x_2411_);
lean_dec(v___x_2423_);
lean_inc(v___x_2428_);
v___x_2429_ = l_Lean_Syntax_matchesNull(v___x_2428_, v___x_2388_);
if (v___x_2429_ == 0)
{
lean_dec(v___x_2428_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2430_; lean_object* v___x_2431_; uint8_t v___x_2432_; 
v___x_2430_ = l_Lean_Syntax_getArg(v___x_2428_, v___x_2411_);
lean_dec(v___x_2428_);
v___x_2431_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_2430_);
v___x_2432_ = l_Lean_Syntax_isOfKind(v___x_2430_, v___x_2431_);
if (v___x_2432_ == 0)
{
lean_dec(v___x_2430_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2433_; uint8_t v___x_2434_; 
v___x_2433_ = l_Lean_Syntax_getArg(v___x_2430_, v___x_2388_);
lean_dec(v___x_2430_);
lean_inc(v___x_2433_);
v___x_2434_ = l_Lean_Syntax_isOfKind(v___x_2433_, v___x_2424_);
if (v___x_2434_ == 0)
{
lean_dec(v___x_2433_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2435_; lean_object* v___x_2436_; uint8_t v___x_2437_; 
v___x_2435_ = l_Lean_Syntax_getArg(v___x_2433_, v___x_2411_);
v___x_2436_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_2435_);
v___x_2437_ = l_Lean_Syntax_isOfKind(v___x_2435_, v___x_2436_);
if (v___x_2437_ == 0)
{
lean_dec(v___x_2435_);
lean_dec(v___x_2433_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2438_; lean_object* v___x_2439_; uint8_t v___x_2440_; 
v___x_2438_ = l_Lean_Syntax_getArg(v___x_2435_, v___x_2411_);
lean_dec(v___x_2435_);
v___x_2439_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_2438_);
v___x_2440_ = l_Lean_Syntax_isOfKind(v___x_2438_, v___x_2439_);
if (v___x_2440_ == 0)
{
lean_dec(v___x_2438_);
lean_dec(v___x_2433_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2441_; uint8_t v___x_2442_; 
v___x_2441_ = l_Lean_Syntax_getArg(v___x_2433_, v___x_2388_);
lean_dec(v___x_2433_);
lean_inc(v___x_2441_);
v___x_2442_ = l_Lean_Syntax_matchesNull(v___x_2441_, v___x_2388_);
if (v___x_2442_ == 0)
{
lean_dec(v___x_2441_);
lean_dec(v___x_2438_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2443_; lean_object* v___x_2444_; uint8_t v___x_2445_; 
v___x_2443_ = l_Lean_Syntax_getArg(v___x_2441_, v___x_2411_);
lean_dec(v___x_2441_);
v___x_2444_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_2445_ = l_Lean_Syntax_isOfKind(v___x_2443_, v___x_2444_);
if (v___x_2445_ == 0)
{
lean_dec(v___x_2438_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2446_; uint8_t v___x_2447_; 
v___x_2446_ = l_Lean_Syntax_getArg(v_a_2417_, v___x_2392_);
lean_dec(v_a_2417_);
lean_inc(v___x_2446_);
v___x_2447_ = l_Lean_Syntax_isOfKind(v___x_2446_, v___x_2418_);
if (v___x_2447_ == 0)
{
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2448_; uint8_t v___x_2449_; 
v___x_2448_ = l_Lean_Syntax_getArg(v___x_2446_, v___x_2388_);
lean_inc(v___x_2448_);
v___x_2449_ = l_Lean_Syntax_isOfKind(v___x_2448_, v___x_2421_);
if (v___x_2449_ == 0)
{
lean_dec(v___x_2448_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2450_; uint8_t v___x_2451_; 
v___x_2450_ = l_Lean_Syntax_getArg(v___x_2448_, v___x_2411_);
lean_dec(v___x_2448_);
lean_inc(v___x_2450_);
v___x_2451_ = l_Lean_Syntax_isOfKind(v___x_2450_, v___x_2426_);
if (v___x_2451_ == 0)
{
lean_dec(v___x_2450_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2452_; uint8_t v___x_2453_; 
v___x_2452_ = l_Lean_Syntax_getArg(v___x_2450_, v___x_2411_);
lean_dec(v___x_2450_);
lean_inc(v___x_2452_);
v___x_2453_ = l_Lean_Syntax_matchesNull(v___x_2452_, v___x_2388_);
if (v___x_2453_ == 0)
{
lean_dec(v___x_2452_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2454_; uint8_t v___x_2455_; 
v___x_2454_ = l_Lean_Syntax_getArg(v___x_2452_, v___x_2411_);
lean_dec(v___x_2452_);
lean_inc(v___x_2454_);
v___x_2455_ = l_Lean_Syntax_isOfKind(v___x_2454_, v___x_2431_);
if (v___x_2455_ == 0)
{
lean_dec(v___x_2454_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2456_; uint8_t v___x_2457_; 
v___x_2456_ = l_Lean_Syntax_getArg(v___x_2454_, v___x_2388_);
lean_dec(v___x_2454_);
lean_inc(v___x_2456_);
v___x_2457_ = l_Lean_Syntax_isOfKind(v___x_2456_, v___x_2424_);
if (v___x_2457_ == 0)
{
lean_dec(v___x_2456_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2458_; uint8_t v___x_2459_; 
v___x_2458_ = l_Lean_Syntax_getArg(v___x_2456_, v___x_2411_);
lean_inc(v___x_2458_);
v___x_2459_ = l_Lean_Syntax_isOfKind(v___x_2458_, v___x_2436_);
if (v___x_2459_ == 0)
{
lean_dec(v___x_2458_);
lean_dec(v___x_2456_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2460_; lean_object* v___x_2461_; uint8_t v___x_2462_; 
v___x_2460_ = l_Lean_Syntax_getArg(v___x_2458_, v___x_2411_);
lean_dec(v___x_2458_);
v___x_2461_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_2462_ = l_Lean_Syntax_isOfKind(v___x_2460_, v___x_2461_);
if (v___x_2462_ == 0)
{
lean_dec(v___x_2456_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2463_; uint8_t v___x_2464_; 
v___x_2463_ = l_Lean_Syntax_getArg(v___x_2456_, v___x_2388_);
lean_dec(v___x_2456_);
lean_inc(v___x_2463_);
v___x_2464_ = l_Lean_Syntax_matchesNull(v___x_2463_, v___x_2388_);
if (v___x_2464_ == 0)
{
lean_dec(v___x_2463_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2465_; uint8_t v___x_2466_; 
v___x_2465_ = l_Lean_Syntax_getArg(v___x_2463_, v___x_2411_);
lean_dec(v___x_2463_);
lean_inc(v___x_2465_);
v___x_2466_ = l_Lean_Syntax_isOfKind(v___x_2465_, v___x_2444_);
if (v___x_2466_ == 0)
{
lean_dec(v___x_2465_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2467_; lean_object* v___x_2468_; uint8_t v___x_2469_; 
v___x_2467_ = l_Lean_Syntax_getArg(v___x_2465_, v___x_2388_);
lean_dec(v___x_2465_);
v___x_2468_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_2467_);
v___x_2469_ = l_Lean_Syntax_isOfKind(v___x_2467_, v___x_2468_);
if (v___x_2469_ == 0)
{
lean_dec(v___x_2467_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2470_; uint8_t v___x_2471_; 
v___x_2470_ = l_Lean_Syntax_getArg(v___x_2467_, v___x_2411_);
lean_inc(v___x_2470_);
v___x_2471_ = l_Lean_Syntax_isOfKind(v___x_2470_, v___x_2439_);
if (v___x_2471_ == 0)
{
lean_dec(v___x_2470_);
lean_dec(v___x_2467_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
uint8_t v___x_2472_; 
v___x_2472_ = l_Lean_Syntax_structEq(v___x_2438_, v___x_2470_);
lean_dec(v___x_2470_);
if (v___x_2472_ == 0)
{
lean_dec(v___x_2467_);
lean_dec(v___x_2446_);
lean_dec(v___x_2438_);
return v___x_2416_;
}
else
{
lean_object* v___x_2474_; uint8_t v_isShared_2475_; uint8_t v_isSharedCheck_2497_; 
v_isSharedCheck_2497_ = !lean_is_exclusive(v___x_2416_);
if (v_isSharedCheck_2497_ == 0)
{
lean_object* v_unused_2498_; 
v_unused_2498_ = lean_ctor_get(v___x_2416_, 0);
lean_dec(v_unused_2498_);
v___x_2474_ = v___x_2416_;
v_isShared_2475_ = v_isSharedCheck_2497_;
goto v_resetjp_2473_;
}
else
{
lean_dec(v___x_2416_);
v___x_2474_ = lean_box(0);
v_isShared_2475_ = v_isSharedCheck_2497_;
goto v_resetjp_2473_;
}
v_resetjp_2473_:
{
lean_object* v_ref_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2495_; 
v_ref_2476_ = lean_ctor_get(v___y_2403_, 5);
v___x_2477_ = l_Lean_Syntax_getArg(v___x_2467_, v___x_2396_);
lean_dec(v___x_2467_);
v___x_2478_ = l_Lean_Syntax_getArg(v___x_2446_, v___x_2392_);
lean_dec(v___x_2446_);
v___x_2479_ = l_Lean_SourceInfo_fromRef(v_ref_2476_, v___x_2425_);
v___x_2480_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__2));
lean_inc_n(v___x_2479_, 8);
v___x_2481_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2481_, 0, v___x_2479_);
lean_ctor_set(v___x_2481_, 1, v___x_2480_);
v___x_2482_ = l_Lean_Syntax_node1(v___x_2479_, v___x_2436_, v___x_2438_);
v___x_2483_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2484_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_2485_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_2486_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2486_, 0, v___x_2479_);
lean_ctor_set(v___x_2486_, 1, v___x_2485_);
v___x_2487_ = l_Lean_Syntax_node2(v___x_2479_, v___x_2484_, v___x_2486_, v___x_2477_);
v___x_2488_ = l_Lean_Syntax_node1(v___x_2479_, v___x_2483_, v___x_2487_);
v___x_2489_ = l_Lean_Syntax_node2(v___x_2479_, v___x_2424_, v___x_2482_, v___x_2488_);
v___x_2490_ = l_Lean_Syntax_node1(v___x_2479_, v___x_2421_, v___x_2489_);
v___x_2491_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2492_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2492_, 0, v___x_2479_);
lean_ctor_set(v___x_2492_, 1, v___x_2491_);
v___x_2493_ = l_Lean_Syntax_node4(v___x_2479_, v___x_2418_, v___x_2481_, v___x_2490_, v___x_2492_, v___x_2478_);
if (v_isShared_2475_ == 0)
{
lean_ctor_set(v___x_2474_, 0, v___x_2493_);
v___x_2495_ = v___x_2474_;
goto v_reusejp_2494_;
}
else
{
lean_object* v_reuseFailAlloc_2496_; 
v_reuseFailAlloc_2496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2496_, 0, v___x_2493_);
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
}
}
}
else
{
lean_object* v___x_2499_; lean_object* v___x_2500_; uint8_t v___x_2501_; 
v___x_2499_ = l_Lean_Syntax_getArg(v___x_2423_, v___x_2411_);
v___x_2500_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_2499_);
v___x_2501_ = l_Lean_Syntax_isOfKind(v___x_2499_, v___x_2500_);
if (v___x_2501_ == 0)
{
lean_dec(v___x_2499_);
lean_dec(v___x_2423_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2502_; lean_object* v___x_2503_; uint8_t v___x_2504_; 
v___x_2502_ = l_Lean_Syntax_getArg(v___x_2499_, v___x_2411_);
lean_dec(v___x_2499_);
v___x_2503_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_2502_);
v___x_2504_ = l_Lean_Syntax_isOfKind(v___x_2502_, v___x_2503_);
if (v___x_2504_ == 0)
{
lean_dec(v___x_2502_);
lean_dec(v___x_2423_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2505_; uint8_t v___x_2506_; 
v___x_2505_ = l_Lean_Syntax_getArg(v___x_2423_, v___x_2388_);
lean_dec(v___x_2423_);
v___x_2506_ = l_Lean_Syntax_matchesNull(v___x_2505_, v___x_2411_);
if (v___x_2506_ == 0)
{
lean_dec(v___x_2502_);
lean_dec(v_a_2417_);
return v___x_2416_;
}
else
{
lean_object* v___x_2507_; uint8_t v___x_2508_; 
v___x_2507_ = l_Lean_Syntax_getArg(v_a_2417_, v___x_2392_);
lean_dec(v_a_2417_);
lean_inc(v___x_2507_);
v___x_2508_ = l_Lean_Syntax_isOfKind(v___x_2507_, v___x_2418_);
if (v___x_2508_ == 0)
{
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2509_; uint8_t v___x_2510_; 
v___x_2509_ = l_Lean_Syntax_getArg(v___x_2507_, v___x_2388_);
lean_inc(v___x_2509_);
v___x_2510_ = l_Lean_Syntax_isOfKind(v___x_2509_, v___x_2421_);
if (v___x_2510_ == 0)
{
lean_dec(v___x_2509_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2511_; lean_object* v___x_2512_; uint8_t v___x_2513_; 
v___x_2511_ = l_Lean_Syntax_getArg(v___x_2509_, v___x_2411_);
lean_dec(v___x_2509_);
v___x_2512_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_2511_);
v___x_2513_ = l_Lean_Syntax_isOfKind(v___x_2511_, v___x_2512_);
if (v___x_2513_ == 0)
{
lean_dec(v___x_2511_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2514_; uint8_t v___x_2515_; 
v___x_2514_ = l_Lean_Syntax_getArg(v___x_2511_, v___x_2411_);
lean_dec(v___x_2511_);
lean_inc(v___x_2514_);
v___x_2515_ = l_Lean_Syntax_matchesNull(v___x_2514_, v___x_2388_);
if (v___x_2515_ == 0)
{
lean_dec(v___x_2514_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2516_; lean_object* v___x_2517_; uint8_t v___x_2518_; 
v___x_2516_ = l_Lean_Syntax_getArg(v___x_2514_, v___x_2411_);
lean_dec(v___x_2514_);
v___x_2517_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_2516_);
v___x_2518_ = l_Lean_Syntax_isOfKind(v___x_2516_, v___x_2517_);
if (v___x_2518_ == 0)
{
lean_dec(v___x_2516_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2519_; uint8_t v___x_2520_; 
v___x_2519_ = l_Lean_Syntax_getArg(v___x_2516_, v___x_2388_);
lean_dec(v___x_2516_);
lean_inc(v___x_2519_);
v___x_2520_ = l_Lean_Syntax_isOfKind(v___x_2519_, v___x_2424_);
if (v___x_2520_ == 0)
{
lean_dec(v___x_2519_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2521_; uint8_t v___x_2522_; 
v___x_2521_ = l_Lean_Syntax_getArg(v___x_2519_, v___x_2411_);
lean_inc(v___x_2521_);
v___x_2522_ = l_Lean_Syntax_isOfKind(v___x_2521_, v___x_2500_);
if (v___x_2522_ == 0)
{
lean_dec(v___x_2521_);
lean_dec(v___x_2519_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2523_; lean_object* v___x_2524_; uint8_t v___x_2525_; 
v___x_2523_ = l_Lean_Syntax_getArg(v___x_2521_, v___x_2411_);
lean_dec(v___x_2521_);
v___x_2524_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_2525_ = l_Lean_Syntax_isOfKind(v___x_2523_, v___x_2524_);
if (v___x_2525_ == 0)
{
lean_dec(v___x_2519_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2526_; uint8_t v___x_2527_; 
v___x_2526_ = l_Lean_Syntax_getArg(v___x_2519_, v___x_2388_);
lean_dec(v___x_2519_);
lean_inc(v___x_2526_);
v___x_2527_ = l_Lean_Syntax_matchesNull(v___x_2526_, v___x_2388_);
if (v___x_2527_ == 0)
{
lean_dec(v___x_2526_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2528_; lean_object* v___x_2529_; uint8_t v___x_2530_; 
v___x_2528_ = l_Lean_Syntax_getArg(v___x_2526_, v___x_2411_);
lean_dec(v___x_2526_);
v___x_2529_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
lean_inc(v___x_2528_);
v___x_2530_ = l_Lean_Syntax_isOfKind(v___x_2528_, v___x_2529_);
if (v___x_2530_ == 0)
{
lean_dec(v___x_2528_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2531_; lean_object* v___x_2532_; uint8_t v___x_2533_; 
v___x_2531_ = l_Lean_Syntax_getArg(v___x_2528_, v___x_2388_);
lean_dec(v___x_2528_);
v___x_2532_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_2531_);
v___x_2533_ = l_Lean_Syntax_isOfKind(v___x_2531_, v___x_2532_);
if (v___x_2533_ == 0)
{
lean_dec(v___x_2531_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2534_; uint8_t v___x_2535_; 
v___x_2534_ = l_Lean_Syntax_getArg(v___x_2531_, v___x_2411_);
lean_inc(v___x_2534_);
v___x_2535_ = l_Lean_Syntax_isOfKind(v___x_2534_, v___x_2503_);
if (v___x_2535_ == 0)
{
lean_dec(v___x_2534_);
lean_dec(v___x_2531_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
uint8_t v___x_2536_; 
v___x_2536_ = l_Lean_Syntax_structEq(v___x_2502_, v___x_2534_);
lean_dec(v___x_2534_);
if (v___x_2536_ == 0)
{
lean_dec(v___x_2531_);
lean_dec(v___x_2507_);
lean_dec(v___x_2502_);
return v___x_2416_;
}
else
{
lean_object* v___x_2538_; uint8_t v_isShared_2539_; uint8_t v_isSharedCheck_2562_; 
v_isSharedCheck_2562_ = !lean_is_exclusive(v___x_2416_);
if (v_isSharedCheck_2562_ == 0)
{
lean_object* v_unused_2563_; 
v_unused_2563_ = lean_ctor_get(v___x_2416_, 0);
lean_dec(v_unused_2563_);
v___x_2538_ = v___x_2416_;
v_isShared_2539_ = v_isSharedCheck_2562_;
goto v_resetjp_2537_;
}
else
{
lean_dec(v___x_2416_);
v___x_2538_ = lean_box(0);
v_isShared_2539_ = v_isSharedCheck_2562_;
goto v_resetjp_2537_;
}
v_resetjp_2537_:
{
lean_object* v_ref_2540_; lean_object* v___x_2541_; lean_object* v___x_2542_; uint8_t v___x_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2547_; lean_object* v___x_2548_; lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v___x_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2560_; 
v_ref_2540_ = lean_ctor_get(v___y_2403_, 5);
v___x_2541_ = l_Lean_Syntax_getArg(v___x_2531_, v___x_2396_);
lean_dec(v___x_2531_);
v___x_2542_ = l_Lean_Syntax_getArg(v___x_2507_, v___x_2392_);
lean_dec(v___x_2507_);
v___x_2543_ = 0;
v___x_2544_ = l_Lean_SourceInfo_fromRef(v_ref_2540_, v___x_2543_);
v___x_2545_ = ((lean_object*)(lp_mathlib_Set_term_u22c3___x2c___00__closed__2));
lean_inc_n(v___x_2544_, 8);
v___x_2546_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2546_, 0, v___x_2544_);
lean_ctor_set(v___x_2546_, 1, v___x_2545_);
v___x_2547_ = l_Lean_Syntax_node1(v___x_2544_, v___x_2500_, v___x_2502_);
v___x_2548_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2549_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_2550_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_2551_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2551_, 0, v___x_2544_);
lean_ctor_set(v___x_2551_, 1, v___x_2550_);
v___x_2552_ = l_Lean_Syntax_node2(v___x_2544_, v___x_2549_, v___x_2551_, v___x_2541_);
v___x_2553_ = l_Lean_Syntax_node1(v___x_2544_, v___x_2548_, v___x_2552_);
v___x_2554_ = l_Lean_Syntax_node2(v___x_2544_, v___x_2424_, v___x_2547_, v___x_2553_);
v___x_2555_ = l_Lean_Syntax_node1(v___x_2544_, v___x_2421_, v___x_2554_);
v___x_2556_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2557_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2557_, 0, v___x_2544_);
lean_ctor_set(v___x_2557_, 1, v___x_2556_);
v___x_2558_ = l_Lean_Syntax_node4(v___x_2544_, v___x_2418_, v___x_2546_, v___x_2555_, v___x_2557_, v___x_2542_);
if (v_isShared_2539_ == 0)
{
lean_ctor_set(v___x_2538_, 0, v___x_2558_);
v___x_2560_ = v___x_2538_;
goto v_reusejp_2559_;
}
else
{
lean_object* v_reuseFailAlloc_2561_; 
v_reuseFailAlloc_2561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2561_, 0, v___x_2558_);
v___x_2560_ = v_reuseFailAlloc_2561_;
goto v_reusejp_2559_;
}
v_reusejp_2559_:
{
return v___x_2560_;
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
}
}
}
}
}
else
{
return v___x_2416_;
}
}
else
{
lean_object* v_a_2564_; lean_object* v___x_2566_; uint8_t v_isShared_2567_; uint8_t v_isSharedCheck_2571_; 
lean_dec(v_a_2406_);
lean_dec(v___x_2397_);
v_a_2564_ = lean_ctor_get(v___x_2408_, 0);
v_isSharedCheck_2571_ = !lean_is_exclusive(v___x_2408_);
if (v_isSharedCheck_2571_ == 0)
{
v___x_2566_ = v___x_2408_;
v_isShared_2567_ = v_isSharedCheck_2571_;
goto v_resetjp_2565_;
}
else
{
lean_inc(v_a_2564_);
lean_dec(v___x_2408_);
v___x_2566_ = lean_box(0);
v_isShared_2567_ = v_isSharedCheck_2571_;
goto v_resetjp_2565_;
}
v_resetjp_2565_:
{
lean_object* v___x_2569_; 
if (v_isShared_2567_ == 0)
{
v___x_2569_ = v___x_2566_;
goto v_reusejp_2568_;
}
else
{
lean_object* v_reuseFailAlloc_2570_; 
v_reuseFailAlloc_2570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2570_, 0, v_a_2564_);
v___x_2569_ = v_reuseFailAlloc_2570_;
goto v_reusejp_2568_;
}
v_reusejp_2568_:
{
return v___x_2569_;
}
}
}
}
else
{
lean_object* v_a_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2579_; 
lean_dec(v___x_2397_);
v_a_2572_ = lean_ctor_get(v___x_2405_, 0);
v_isSharedCheck_2579_ = !lean_is_exclusive(v___x_2405_);
if (v_isSharedCheck_2579_ == 0)
{
v___x_2574_ = v___x_2405_;
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
else
{
lean_inc(v_a_2572_);
lean_dec(v___x_2405_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2579_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v___x_2577_; 
if (v_isShared_2575_ == 0)
{
v___x_2577_ = v___x_2574_;
goto v_reusejp_2576_;
}
else
{
lean_object* v_reuseFailAlloc_2578_; 
v_reuseFailAlloc_2578_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2578_, 0, v_a_2572_);
v___x_2577_ = v_reuseFailAlloc_2578_;
goto v_reusejp_2576_;
}
v_reusejp_2576_:
{
return v___x_2577_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___lam__2___boxed(lean_object* v___y_2590_, lean_object* v___y_2591_, lean_object* v___y_2592_, lean_object* v___y_2593_, lean_object* v___y_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_){
_start:
{
lean_object* v_res_2597_; 
v_res_2597_ = lp_mathlib_Set_iUnion__delab___lam__2(v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_, v___y_2594_, v___y_2595_);
lean_dec(v___y_2595_);
lean_dec_ref(v___y_2594_);
lean_dec(v___y_2593_);
lean_dec_ref(v___y_2592_);
lean_dec(v___y_2591_);
lean_dec_ref(v___y_2590_);
return v_res_2597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab(lean_object* v_a_2599_, lean_object* v_a_2600_, lean_object* v_a_2601_, lean_object* v_a_2602_, lean_object* v_a_2603_, lean_object* v_a_2604_){
_start:
{
lean_object* v___f_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; 
v___f_2606_ = ((lean_object*)(lp_mathlib_Set_iUnion__delab___closed__0));
v___x_2607_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_2608_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_2607_, v___f_2606_, v_a_2599_, v_a_2600_, v_a_2601_, v_a_2602_, v_a_2603_, v_a_2604_);
return v___x_2608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_iUnion__delab___boxed(lean_object* v_a_2609_, lean_object* v_a_2610_, lean_object* v_a_2611_, lean_object* v_a_2612_, lean_object* v_a_2613_, lean_object* v_a_2614_, lean_object* v_a_2615_){
_start:
{
lean_object* v_res_2616_; 
v_res_2616_ = lp_mathlib_Set_iUnion__delab(v_a_2609_, v_a_2610_, v_a_2611_, v_a_2612_, v_a_2613_, v_a_2614_);
lean_dec(v_a_2614_);
lean_dec_ref(v_a_2613_);
lean_dec(v_a_2612_);
lean_dec_ref(v_a_2611_);
lean_dec(v_a_2610_);
lean_dec_ref(v_a_2609_);
return v_res_2616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__0(lean_object* v_a_2617_, uint8_t v_a_2618_, uint8_t v_a_2619_, uint8_t v___x_2620_, lean_object* v_x_2621_, lean_object* v___y_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_, lean_object* v___y_2625_, lean_object* v___y_2626_, lean_object* v___y_2627_){
_start:
{
lean_object* v___x_2629_; 
v___x_2629_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_2622_, v___y_2623_, v___y_2624_, v___y_2625_, v___y_2626_, v___y_2627_);
if (lean_obj_tag(v___x_2629_) == 0)
{
lean_object* v_a_2630_; lean_object* v___x_2632_; uint8_t v_isShared_2633_; uint8_t v_isSharedCheck_2723_; 
v_a_2630_ = lean_ctor_get(v___x_2629_, 0);
v_isSharedCheck_2723_ = !lean_is_exclusive(v___x_2629_);
if (v_isSharedCheck_2723_ == 0)
{
v___x_2632_ = v___x_2629_;
v_isShared_2633_ = v_isSharedCheck_2723_;
goto v_resetjp_2631_;
}
else
{
lean_inc(v_a_2630_);
lean_dec(v___x_2629_);
v___x_2632_ = lean_box(0);
v_isShared_2633_ = v_isSharedCheck_2723_;
goto v_resetjp_2631_;
}
v_resetjp_2631_:
{
uint8_t v___y_2635_; uint8_t v___y_2669_; 
if (v_a_2618_ == 0)
{
v___y_2669_ = v_a_2618_;
goto v___jp_2668_;
}
else
{
if (v___x_2620_ == 0)
{
lean_object* v_ref_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; lean_object* v___x_2718_; lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; 
lean_del_object(v___x_2632_);
lean_dec(v_x_2621_);
v_ref_2688_ = lean_ctor_get(v___y_2626_, 5);
v___x_2689_ = l_Lean_SourceInfo_fromRef(v_ref_2688_, v___x_2620_);
v___x_2690_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
v___x_2691_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__2));
lean_inc_n(v___x_2689_, 15);
v___x_2692_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2692_, 0, v___x_2689_);
lean_ctor_set(v___x_2692_, 1, v___x_2691_);
v___x_2693_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2694_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_2695_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2696_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_2697_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_2698_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2698_, 0, v___x_2689_);
lean_ctor_set(v___x_2698_, 1, v___x_2697_);
v___x_2699_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_2700_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_2701_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_2702_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__11));
v___x_2703_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2703_, 0, v___x_2689_);
lean_ctor_set(v___x_2703_, 1, v___x_2702_);
v___x_2704_ = l_Lean_Syntax_node1(v___x_2689_, v___x_2701_, v___x_2703_);
v___x_2705_ = l_Lean_Syntax_node1(v___x_2689_, v___x_2700_, v___x_2704_);
v___x_2706_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_2707_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_2708_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2708_, 0, v___x_2689_);
lean_ctor_set(v___x_2708_, 1, v___x_2707_);
v___x_2709_ = l_Lean_Syntax_node2(v___x_2689_, v___x_2706_, v___x_2708_, v_a_2617_);
v___x_2710_ = l_Lean_Syntax_node1(v___x_2689_, v___x_2695_, v___x_2709_);
v___x_2711_ = l_Lean_Syntax_node2(v___x_2689_, v___x_2699_, v___x_2705_, v___x_2710_);
v___x_2712_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_2713_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2713_, 0, v___x_2689_);
lean_ctor_set(v___x_2713_, 1, v___x_2712_);
v___x_2714_ = l_Lean_Syntax_node3(v___x_2689_, v___x_2696_, v___x_2698_, v___x_2711_, v___x_2713_);
v___x_2715_ = l_Lean_Syntax_node1(v___x_2689_, v___x_2695_, v___x_2714_);
v___x_2716_ = l_Lean_Syntax_node1(v___x_2689_, v___x_2694_, v___x_2715_);
v___x_2717_ = l_Lean_Syntax_node1(v___x_2689_, v___x_2693_, v___x_2716_);
v___x_2718_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2719_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2719_, 0, v___x_2689_);
lean_ctor_set(v___x_2719_, 1, v___x_2718_);
v___x_2720_ = l_Lean_Syntax_node4(v___x_2689_, v___x_2690_, v___x_2692_, v___x_2717_, v___x_2719_, v_a_2630_);
v___x_2721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2721_, 0, v___x_2720_);
return v___x_2721_;
}
else
{
uint8_t v___x_2722_; 
v___x_2722_ = 0;
v___y_2669_ = v___x_2722_;
goto v___jp_2668_;
}
}
v___jp_2634_:
{
lean_object* v_ref_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; lean_object* v___x_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2644_; lean_object* v___x_2645_; lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; lean_object* v___x_2655_; lean_object* v___x_2656_; lean_object* v___x_2657_; lean_object* v___x_2658_; lean_object* v___x_2659_; lean_object* v___x_2660_; lean_object* v___x_2661_; lean_object* v___x_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2666_; 
v_ref_2636_ = lean_ctor_get(v___y_2626_, 5);
v___x_2637_ = l_Lean_SourceInfo_fromRef(v_ref_2636_, v___y_2635_);
v___x_2638_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
v___x_2639_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__2));
lean_inc_n(v___x_2637_, 13);
v___x_2640_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2640_, 0, v___x_2637_);
lean_ctor_set(v___x_2640_, 1, v___x_2639_);
v___x_2641_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2642_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
v___x_2643_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2644_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
v___x_2645_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__5));
v___x_2646_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2646_, 0, v___x_2637_);
lean_ctor_set(v___x_2646_, 1, v___x_2645_);
v___x_2647_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_2648_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_2649_ = l_Lean_Syntax_node1(v___x_2637_, v___x_2648_, v_x_2621_);
v___x_2650_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_2651_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__8));
v___x_2652_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2652_, 0, v___x_2637_);
lean_ctor_set(v___x_2652_, 1, v___x_2651_);
v___x_2653_ = l_Lean_Syntax_node2(v___x_2637_, v___x_2650_, v___x_2652_, v_a_2617_);
v___x_2654_ = l_Lean_Syntax_node1(v___x_2637_, v___x_2643_, v___x_2653_);
v___x_2655_ = l_Lean_Syntax_node2(v___x_2637_, v___x_2647_, v___x_2649_, v___x_2654_);
v___x_2656_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__22));
v___x_2657_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2657_, 0, v___x_2637_);
lean_ctor_set(v___x_2657_, 1, v___x_2656_);
v___x_2658_ = l_Lean_Syntax_node3(v___x_2637_, v___x_2644_, v___x_2646_, v___x_2655_, v___x_2657_);
v___x_2659_ = l_Lean_Syntax_node1(v___x_2637_, v___x_2643_, v___x_2658_);
v___x_2660_ = l_Lean_Syntax_node1(v___x_2637_, v___x_2642_, v___x_2659_);
v___x_2661_ = l_Lean_Syntax_node1(v___x_2637_, v___x_2641_, v___x_2660_);
v___x_2662_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2663_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2663_, 0, v___x_2637_);
lean_ctor_set(v___x_2663_, 1, v___x_2662_);
v___x_2664_ = l_Lean_Syntax_node4(v___x_2637_, v___x_2638_, v___x_2640_, v___x_2661_, v___x_2663_, v_a_2630_);
if (v_isShared_2633_ == 0)
{
lean_ctor_set(v___x_2632_, 0, v___x_2664_);
v___x_2666_ = v___x_2632_;
goto v_reusejp_2665_;
}
else
{
lean_object* v_reuseFailAlloc_2667_; 
v_reuseFailAlloc_2667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2667_, 0, v___x_2664_);
v___x_2666_ = v_reuseFailAlloc_2667_;
goto v_reusejp_2665_;
}
v_reusejp_2665_:
{
return v___x_2666_;
}
}
v___jp_2668_:
{
if (v_a_2618_ == 0)
{
if (v_a_2619_ == 0)
{
lean_object* v_ref_2670_; lean_object* v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; lean_object* v___x_2679_; lean_object* v___x_2680_; lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; 
lean_del_object(v___x_2632_);
lean_dec(v_a_2617_);
v_ref_2670_ = lean_ctor_get(v___y_2626_, 5);
v___x_2671_ = l_Lean_SourceInfo_fromRef(v_ref_2670_, v_a_2619_);
v___x_2672_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
v___x_2673_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__2));
lean_inc_n(v___x_2671_, 6);
v___x_2674_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2674_, 0, v___x_2671_);
lean_ctor_set(v___x_2674_, 1, v___x_2673_);
v___x_2675_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
v___x_2676_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
v___x_2677_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
v___x_2678_ = l_Lean_Syntax_node1(v___x_2671_, v___x_2677_, v_x_2621_);
v___x_2679_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2680_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__6);
v___x_2681_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2681_, 0, v___x_2671_);
lean_ctor_set(v___x_2681_, 1, v___x_2679_);
lean_ctor_set(v___x_2681_, 2, v___x_2680_);
v___x_2682_ = l_Lean_Syntax_node2(v___x_2671_, v___x_2676_, v___x_2678_, v___x_2681_);
v___x_2683_ = l_Lean_Syntax_node1(v___x_2671_, v___x_2675_, v___x_2682_);
v___x_2684_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2685_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2685_, 0, v___x_2671_);
lean_ctor_set(v___x_2685_, 1, v___x_2684_);
v___x_2686_ = l_Lean_Syntax_node4(v___x_2671_, v___x_2672_, v___x_2674_, v___x_2683_, v___x_2685_, v_a_2630_);
v___x_2687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2687_, 0, v___x_2686_);
return v___x_2687_;
}
else
{
v___y_2635_ = v___y_2669_;
goto v___jp_2634_;
}
}
else
{
v___y_2635_ = v___y_2669_;
goto v___jp_2634_;
}
}
}
}
else
{
lean_dec(v_x_2621_);
lean_dec(v_a_2617_);
return v___x_2629_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__0___boxed(lean_object* v_a_2724_, lean_object* v_a_2725_, lean_object* v_a_2726_, lean_object* v___x_2727_, lean_object* v_x_2728_, lean_object* v___y_2729_, lean_object* v___y_2730_, lean_object* v___y_2731_, lean_object* v___y_2732_, lean_object* v___y_2733_, lean_object* v___y_2734_, lean_object* v___y_2735_){
_start:
{
uint8_t v_a_69105__boxed_2736_; uint8_t v_a_69106__boxed_2737_; uint8_t v___x_69107__boxed_2738_; lean_object* v_res_2739_; 
v_a_69105__boxed_2736_ = lean_unbox(v_a_2725_);
v_a_69106__boxed_2737_ = lean_unbox(v_a_2726_);
v___x_69107__boxed_2738_ = lean_unbox(v___x_2727_);
v_res_2739_ = lp_mathlib_Set_sInter__delab___lam__0(v_a_2724_, v_a_69105__boxed_2736_, v_a_69106__boxed_2737_, v___x_69107__boxed_2738_, v_x_2728_, v___y_2729_, v___y_2730_, v___y_2731_, v___y_2732_, v___y_2733_, v___y_2734_);
lean_dec(v___y_2734_);
lean_dec_ref(v___y_2733_);
lean_dec(v___y_2732_);
lean_dec_ref(v___y_2731_);
lean_dec(v___y_2730_);
lean_dec_ref(v___y_2729_);
return v_res_2739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__1(lean_object* v___x_2740_, uint8_t v_a_2741_, uint8_t v_a_2742_, uint8_t v___x_2743_, lean_object* v___y_2744_, lean_object* v___y_2745_, lean_object* v___y_2746_, lean_object* v___y_2747_, lean_object* v___y_2748_, lean_object* v___y_2749_){
_start:
{
lean_object* v___x_2751_; 
v___x_2751_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00iSup__delab_spec__0___redArg(v___x_2740_, v___y_2744_, v___y_2745_, v___y_2746_, v___y_2747_, v___y_2748_, v___y_2749_);
if (lean_obj_tag(v___x_2751_) == 0)
{
lean_object* v_a_2752_; lean_object* v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___f_2756_; uint8_t v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; 
v_a_2752_ = lean_ctor_get(v___x_2751_, 0);
lean_inc(v_a_2752_);
lean_dec_ref_known(v___x_2751_, 1);
v___x_2753_ = lean_box(v_a_2741_);
v___x_2754_ = lean_box(v_a_2742_);
v___x_2755_ = lean_box(v___x_2743_);
v___f_2756_ = lean_alloc_closure((void*)(lp_mathlib_Set_sInter__delab___lam__0___boxed), 12, 4);
lean_closure_set(v___f_2756_, 0, v_a_2752_);
lean_closure_set(v___f_2756_, 1, v___x_2753_);
lean_closure_set(v___f_2756_, 2, v___x_2754_);
lean_closure_set(v___f_2756_, 3, v___x_2755_);
v___x_2757_ = 0;
v___x_2758_ = l_Lean_NameSet_empty;
v___x_2759_ = l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(v___f_2756_, v___x_2757_, v___x_2758_, v___y_2744_, v___y_2745_, v___y_2746_, v___y_2747_, v___y_2748_, v___y_2749_);
return v___x_2759_;
}
else
{
return v___x_2751_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__1___boxed(lean_object* v___x_2760_, lean_object* v_a_2761_, lean_object* v_a_2762_, lean_object* v___x_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_, lean_object* v___y_2768_, lean_object* v___y_2769_, lean_object* v___y_2770_){
_start:
{
uint8_t v_a_69359__boxed_2771_; uint8_t v_a_69360__boxed_2772_; uint8_t v___x_69361__boxed_2773_; lean_object* v_res_2774_; 
v_a_69359__boxed_2771_ = lean_unbox(v_a_2761_);
v_a_69360__boxed_2772_ = lean_unbox(v_a_2762_);
v___x_69361__boxed_2773_ = lean_unbox(v___x_2763_);
v_res_2774_ = lp_mathlib_Set_sInter__delab___lam__1(v___x_2760_, v_a_69359__boxed_2771_, v_a_69360__boxed_2772_, v___x_69361__boxed_2773_, v___y_2764_, v___y_2765_, v___y_2766_, v___y_2767_, v___y_2768_, v___y_2769_);
lean_dec(v___y_2769_);
lean_dec_ref(v___y_2768_);
lean_dec(v___y_2767_);
lean_dec_ref(v___y_2766_);
lean_dec(v___y_2765_);
lean_dec_ref(v___y_2764_);
return v_res_2774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__2(lean_object* v___y_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_){
_start:
{
lean_object* v___x_2782_; lean_object* v_a_2783_; lean_object* v_dummy_2784_; lean_object* v_nargs_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; uint8_t v___x_2792_; 
v___x_2782_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1_spec__0___redArg(v___y_2775_);
v_a_2783_ = lean_ctor_get(v___x_2782_, 0);
lean_inc(v_a_2783_);
lean_dec_ref(v___x_2782_);
v_dummy_2784_ = lean_obj_once(&lp_mathlib_iSup__delab___lam__2___closed__0, &lp_mathlib_iSup__delab___lam__2___closed__0_once, _init_lp_mathlib_iSup__delab___lam__2___closed__0);
v_nargs_2785_ = l_Lean_Expr_getAppNumArgs(v_a_2783_);
lean_inc(v_nargs_2785_);
v___x_2786_ = lean_mk_array(v_nargs_2785_, v_dummy_2784_);
v___x_2787_ = lean_unsigned_to_nat(1u);
v___x_2788_ = lean_nat_sub(v_nargs_2785_, v___x_2787_);
lean_dec(v_nargs_2785_);
v___x_2789_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_2783_, v___x_2786_, v___x_2788_);
v___x_2790_ = lean_array_get_size(v___x_2789_);
v___x_2791_ = lean_unsigned_to_nat(3u);
v___x_2792_ = lean_nat_dec_eq(v___x_2790_, v___x_2791_);
if (v___x_2792_ == 0)
{
lean_object* v___x_2793_; 
lean_dec_ref(v___x_2789_);
v___x_2793_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2793_;
}
else
{
lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2796_; lean_object* v___y_2798_; lean_object* v___y_2799_; lean_object* v___y_2800_; lean_object* v___y_2801_; lean_object* v___y_2802_; lean_object* v___y_2803_; uint8_t v___x_2979_; 
v___x_2794_ = lean_array_fget(v___x_2789_, v___x_2787_);
v___x_2795_ = lean_unsigned_to_nat(2u);
v___x_2796_ = lean_array_fget(v___x_2789_, v___x_2795_);
lean_dec_ref(v___x_2789_);
v___x_2979_ = l_Lean_Expr_isLambda(v___x_2796_);
if (v___x_2979_ == 0)
{
lean_object* v___x_2980_; 
v___x_2980_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_2980_) == 0)
{
lean_dec_ref_known(v___x_2980_, 1);
v___y_2798_ = v___y_2775_;
v___y_2799_ = v___y_2776_;
v___y_2800_ = v___y_2777_;
v___y_2801_ = v___y_2778_;
v___y_2802_ = v___y_2779_;
v___y_2803_ = v___y_2780_;
goto v___jp_2797_;
}
else
{
lean_object* v_a_2981_; lean_object* v___x_2983_; uint8_t v_isShared_2984_; uint8_t v_isSharedCheck_2988_; 
lean_dec(v___x_2796_);
lean_dec(v___x_2794_);
v_a_2981_ = lean_ctor_get(v___x_2980_, 0);
v_isSharedCheck_2988_ = !lean_is_exclusive(v___x_2980_);
if (v_isSharedCheck_2988_ == 0)
{
v___x_2983_ = v___x_2980_;
v_isShared_2984_ = v_isSharedCheck_2988_;
goto v_resetjp_2982_;
}
else
{
lean_inc(v_a_2981_);
lean_dec(v___x_2980_);
v___x_2983_ = lean_box(0);
v_isShared_2984_ = v_isSharedCheck_2988_;
goto v_resetjp_2982_;
}
v_resetjp_2982_:
{
lean_object* v___x_2986_; 
if (v_isShared_2984_ == 0)
{
v___x_2986_ = v___x_2983_;
goto v_reusejp_2985_;
}
else
{
lean_object* v_reuseFailAlloc_2987_; 
v_reuseFailAlloc_2987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2987_, 0, v_a_2981_);
v___x_2986_ = v_reuseFailAlloc_2987_;
goto v_reusejp_2985_;
}
v_reusejp_2985_:
{
return v___x_2986_;
}
}
}
}
else
{
v___y_2798_ = v___y_2775_;
v___y_2799_ = v___y_2776_;
v___y_2800_ = v___y_2777_;
v___y_2801_ = v___y_2778_;
v___y_2802_ = v___y_2779_;
v___y_2803_ = v___y_2780_;
goto v___jp_2797_;
}
v___jp_2797_:
{
lean_object* v___x_2804_; 
v___x_2804_ = l_Lean_Meta_isProp(v___x_2794_, v___y_2800_, v___y_2801_, v___y_2802_, v___y_2803_);
if (lean_obj_tag(v___x_2804_) == 0)
{
lean_object* v_a_2805_; lean_object* v___x_2806_; lean_object* v___x_2807_; 
v_a_2805_ = lean_ctor_get(v___x_2804_, 0);
lean_inc(v_a_2805_);
lean_dec_ref_known(v___x_2804_, 1);
v___x_2806_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__1));
v___x_2807_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_2806_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_, v___y_2802_, v___y_2803_);
if (lean_obj_tag(v___x_2807_) == 0)
{
lean_object* v_a_2808_; lean_object* v___x_2809_; lean_object* v___x_2810_; uint8_t v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___f_2814_; lean_object* v___x_2815_; 
v_a_2808_ = lean_ctor_get(v___x_2807_, 0);
lean_inc(v_a_2808_);
lean_dec_ref_known(v___x_2807_, 1);
v___x_2809_ = l_Lean_Expr_bindingBody_x21(v___x_2796_);
lean_dec(v___x_2796_);
v___x_2810_ = lean_unsigned_to_nat(0u);
v___x_2811_ = lean_expr_has_loose_bvar(v___x_2809_, v___x_2810_);
lean_dec_ref(v___x_2809_);
v___x_2812_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__2));
v___x_2813_ = lean_box(v___x_2811_);
v___f_2814_ = lean_alloc_closure((void*)(lp_mathlib_Set_sInter__delab___lam__1___boxed), 11, 4);
lean_closure_set(v___f_2814_, 0, v___x_2812_);
lean_closure_set(v___f_2814_, 1, v_a_2805_);
lean_closure_set(v___f_2814_, 2, v_a_2808_);
lean_closure_set(v___f_2814_, 3, v___x_2813_);
v___x_2815_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00iSup__delab_spec__1___redArg(v___f_2814_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_, v___y_2802_, v___y_2803_);
if (lean_obj_tag(v___x_2815_) == 0)
{
lean_object* v_a_2816_; lean_object* v___x_2817_; uint8_t v___x_2818_; 
v_a_2816_ = lean_ctor_get(v___x_2815_, 0);
lean_inc_n(v_a_2816_, 2);
v___x_2817_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__1));
v___x_2818_ = l_Lean_Syntax_isOfKind(v_a_2816_, v___x_2817_);
if (v___x_2818_ == 0)
{
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2819_; lean_object* v___x_2820_; uint8_t v___x_2821_; 
v___x_2819_ = l_Lean_Syntax_getArg(v_a_2816_, v___x_2787_);
v___x_2820_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__5));
lean_inc(v___x_2819_);
v___x_2821_ = l_Lean_Syntax_isOfKind(v___x_2819_, v___x_2820_);
if (v___x_2821_ == 0)
{
lean_dec(v___x_2819_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2822_; lean_object* v___x_2823_; uint8_t v___x_2824_; 
v___x_2822_ = l_Lean_Syntax_getArg(v___x_2819_, v___x_2810_);
lean_dec(v___x_2819_);
v___x_2823_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__3));
lean_inc(v___x_2822_);
v___x_2824_ = l_Lean_Syntax_isOfKind(v___x_2822_, v___x_2823_);
if (v___x_2824_ == 0)
{
lean_object* v___x_2825_; uint8_t v___x_2826_; 
v___x_2825_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_2822_);
v___x_2826_ = l_Lean_Syntax_isOfKind(v___x_2822_, v___x_2825_);
if (v___x_2826_ == 0)
{
lean_dec(v___x_2822_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2827_; uint8_t v___x_2828_; 
v___x_2827_ = l_Lean_Syntax_getArg(v___x_2822_, v___x_2810_);
lean_dec(v___x_2822_);
lean_inc(v___x_2827_);
v___x_2828_ = l_Lean_Syntax_matchesNull(v___x_2827_, v___x_2787_);
if (v___x_2828_ == 0)
{
lean_dec(v___x_2827_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2829_; lean_object* v___x_2830_; uint8_t v___x_2831_; 
v___x_2829_ = l_Lean_Syntax_getArg(v___x_2827_, v___x_2810_);
lean_dec(v___x_2827_);
v___x_2830_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_2829_);
v___x_2831_ = l_Lean_Syntax_isOfKind(v___x_2829_, v___x_2830_);
if (v___x_2831_ == 0)
{
lean_dec(v___x_2829_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2832_; uint8_t v___x_2833_; 
v___x_2832_ = l_Lean_Syntax_getArg(v___x_2829_, v___x_2787_);
lean_dec(v___x_2829_);
lean_inc(v___x_2832_);
v___x_2833_ = l_Lean_Syntax_isOfKind(v___x_2832_, v___x_2823_);
if (v___x_2833_ == 0)
{
lean_dec(v___x_2832_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2834_; lean_object* v___x_2835_; uint8_t v___x_2836_; 
v___x_2834_ = l_Lean_Syntax_getArg(v___x_2832_, v___x_2810_);
v___x_2835_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_2834_);
v___x_2836_ = l_Lean_Syntax_isOfKind(v___x_2834_, v___x_2835_);
if (v___x_2836_ == 0)
{
lean_dec(v___x_2834_);
lean_dec(v___x_2832_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2837_; lean_object* v___x_2838_; uint8_t v___x_2839_; 
v___x_2837_ = l_Lean_Syntax_getArg(v___x_2834_, v___x_2810_);
lean_dec(v___x_2834_);
v___x_2838_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_2837_);
v___x_2839_ = l_Lean_Syntax_isOfKind(v___x_2837_, v___x_2838_);
if (v___x_2839_ == 0)
{
lean_dec(v___x_2837_);
lean_dec(v___x_2832_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2840_; uint8_t v___x_2841_; 
v___x_2840_ = l_Lean_Syntax_getArg(v___x_2832_, v___x_2787_);
lean_dec(v___x_2832_);
lean_inc(v___x_2840_);
v___x_2841_ = l_Lean_Syntax_matchesNull(v___x_2840_, v___x_2787_);
if (v___x_2841_ == 0)
{
lean_dec(v___x_2840_);
lean_dec(v___x_2837_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2842_; lean_object* v___x_2843_; uint8_t v___x_2844_; 
v___x_2842_ = l_Lean_Syntax_getArg(v___x_2840_, v___x_2810_);
lean_dec(v___x_2840_);
v___x_2843_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
v___x_2844_ = l_Lean_Syntax_isOfKind(v___x_2842_, v___x_2843_);
if (v___x_2844_ == 0)
{
lean_dec(v___x_2837_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2845_; uint8_t v___x_2846_; 
v___x_2845_ = l_Lean_Syntax_getArg(v_a_2816_, v___x_2791_);
lean_dec(v_a_2816_);
lean_inc(v___x_2845_);
v___x_2846_ = l_Lean_Syntax_isOfKind(v___x_2845_, v___x_2817_);
if (v___x_2846_ == 0)
{
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2847_; uint8_t v___x_2848_; 
v___x_2847_ = l_Lean_Syntax_getArg(v___x_2845_, v___x_2787_);
lean_inc(v___x_2847_);
v___x_2848_ = l_Lean_Syntax_isOfKind(v___x_2847_, v___x_2820_);
if (v___x_2848_ == 0)
{
lean_dec(v___x_2847_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2849_; uint8_t v___x_2850_; 
v___x_2849_ = l_Lean_Syntax_getArg(v___x_2847_, v___x_2810_);
lean_dec(v___x_2847_);
lean_inc(v___x_2849_);
v___x_2850_ = l_Lean_Syntax_isOfKind(v___x_2849_, v___x_2825_);
if (v___x_2850_ == 0)
{
lean_dec(v___x_2849_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2851_; uint8_t v___x_2852_; 
v___x_2851_ = l_Lean_Syntax_getArg(v___x_2849_, v___x_2810_);
lean_dec(v___x_2849_);
lean_inc(v___x_2851_);
v___x_2852_ = l_Lean_Syntax_matchesNull(v___x_2851_, v___x_2787_);
if (v___x_2852_ == 0)
{
lean_dec(v___x_2851_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2853_; uint8_t v___x_2854_; 
v___x_2853_ = l_Lean_Syntax_getArg(v___x_2851_, v___x_2810_);
lean_dec(v___x_2851_);
lean_inc(v___x_2853_);
v___x_2854_ = l_Lean_Syntax_isOfKind(v___x_2853_, v___x_2830_);
if (v___x_2854_ == 0)
{
lean_dec(v___x_2853_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2855_; uint8_t v___x_2856_; 
v___x_2855_ = l_Lean_Syntax_getArg(v___x_2853_, v___x_2787_);
lean_dec(v___x_2853_);
lean_inc(v___x_2855_);
v___x_2856_ = l_Lean_Syntax_isOfKind(v___x_2855_, v___x_2823_);
if (v___x_2856_ == 0)
{
lean_dec(v___x_2855_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2857_; uint8_t v___x_2858_; 
v___x_2857_ = l_Lean_Syntax_getArg(v___x_2855_, v___x_2810_);
lean_inc(v___x_2857_);
v___x_2858_ = l_Lean_Syntax_isOfKind(v___x_2857_, v___x_2835_);
if (v___x_2858_ == 0)
{
lean_dec(v___x_2857_);
lean_dec(v___x_2855_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2859_; lean_object* v___x_2860_; uint8_t v___x_2861_; 
v___x_2859_ = l_Lean_Syntax_getArg(v___x_2857_, v___x_2810_);
lean_dec(v___x_2857_);
v___x_2860_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_2861_ = l_Lean_Syntax_isOfKind(v___x_2859_, v___x_2860_);
if (v___x_2861_ == 0)
{
lean_dec(v___x_2855_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2862_; uint8_t v___x_2863_; 
v___x_2862_ = l_Lean_Syntax_getArg(v___x_2855_, v___x_2787_);
lean_dec(v___x_2855_);
lean_inc(v___x_2862_);
v___x_2863_ = l_Lean_Syntax_matchesNull(v___x_2862_, v___x_2787_);
if (v___x_2863_ == 0)
{
lean_dec(v___x_2862_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2864_; uint8_t v___x_2865_; 
v___x_2864_ = l_Lean_Syntax_getArg(v___x_2862_, v___x_2810_);
lean_dec(v___x_2862_);
lean_inc(v___x_2864_);
v___x_2865_ = l_Lean_Syntax_isOfKind(v___x_2864_, v___x_2843_);
if (v___x_2865_ == 0)
{
lean_dec(v___x_2864_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2866_; lean_object* v___x_2867_; uint8_t v___x_2868_; 
v___x_2866_ = l_Lean_Syntax_getArg(v___x_2864_, v___x_2787_);
lean_dec(v___x_2864_);
v___x_2867_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_2866_);
v___x_2868_ = l_Lean_Syntax_isOfKind(v___x_2866_, v___x_2867_);
if (v___x_2868_ == 0)
{
lean_dec(v___x_2866_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2869_; uint8_t v___x_2870_; 
v___x_2869_ = l_Lean_Syntax_getArg(v___x_2866_, v___x_2810_);
lean_inc(v___x_2869_);
v___x_2870_ = l_Lean_Syntax_isOfKind(v___x_2869_, v___x_2838_);
if (v___x_2870_ == 0)
{
lean_dec(v___x_2869_);
lean_dec(v___x_2866_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
uint8_t v___x_2871_; 
v___x_2871_ = l_Lean_Syntax_structEq(v___x_2837_, v___x_2869_);
lean_dec(v___x_2869_);
if (v___x_2871_ == 0)
{
lean_dec(v___x_2866_);
lean_dec(v___x_2845_);
lean_dec(v___x_2837_);
return v___x_2815_;
}
else
{
lean_object* v___x_2873_; uint8_t v_isShared_2874_; uint8_t v_isSharedCheck_2896_; 
v_isSharedCheck_2896_ = !lean_is_exclusive(v___x_2815_);
if (v_isSharedCheck_2896_ == 0)
{
lean_object* v_unused_2897_; 
v_unused_2897_ = lean_ctor_get(v___x_2815_, 0);
lean_dec(v_unused_2897_);
v___x_2873_ = v___x_2815_;
v_isShared_2874_ = v_isSharedCheck_2896_;
goto v_resetjp_2872_;
}
else
{
lean_dec(v___x_2815_);
v___x_2873_ = lean_box(0);
v_isShared_2874_ = v_isSharedCheck_2896_;
goto v_resetjp_2872_;
}
v_resetjp_2872_:
{
lean_object* v_ref_2875_; lean_object* v___x_2876_; lean_object* v___x_2877_; lean_object* v___x_2878_; lean_object* v___x_2879_; lean_object* v___x_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v___x_2884_; lean_object* v___x_2885_; lean_object* v___x_2886_; lean_object* v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; lean_object* v___x_2890_; lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2894_; 
v_ref_2875_ = lean_ctor_get(v___y_2802_, 5);
v___x_2876_ = l_Lean_Syntax_getArg(v___x_2866_, v___x_2795_);
lean_dec(v___x_2866_);
v___x_2877_ = l_Lean_Syntax_getArg(v___x_2845_, v___x_2791_);
lean_dec(v___x_2845_);
v___x_2878_ = l_Lean_SourceInfo_fromRef(v_ref_2875_, v___x_2824_);
v___x_2879_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__2));
lean_inc_n(v___x_2878_, 8);
v___x_2880_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2880_, 0, v___x_2878_);
lean_ctor_set(v___x_2880_, 1, v___x_2879_);
v___x_2881_ = l_Lean_Syntax_node1(v___x_2878_, v___x_2835_, v___x_2837_);
v___x_2882_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2883_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_2884_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_2885_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2885_, 0, v___x_2878_);
lean_ctor_set(v___x_2885_, 1, v___x_2884_);
v___x_2886_ = l_Lean_Syntax_node2(v___x_2878_, v___x_2883_, v___x_2885_, v___x_2876_);
v___x_2887_ = l_Lean_Syntax_node1(v___x_2878_, v___x_2882_, v___x_2886_);
v___x_2888_ = l_Lean_Syntax_node2(v___x_2878_, v___x_2823_, v___x_2881_, v___x_2887_);
v___x_2889_ = l_Lean_Syntax_node1(v___x_2878_, v___x_2820_, v___x_2888_);
v___x_2890_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2891_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2891_, 0, v___x_2878_);
lean_ctor_set(v___x_2891_, 1, v___x_2890_);
v___x_2892_ = l_Lean_Syntax_node4(v___x_2878_, v___x_2817_, v___x_2880_, v___x_2889_, v___x_2891_, v___x_2877_);
if (v_isShared_2874_ == 0)
{
lean_ctor_set(v___x_2873_, 0, v___x_2892_);
v___x_2894_ = v___x_2873_;
goto v_reusejp_2893_;
}
else
{
lean_object* v_reuseFailAlloc_2895_; 
v_reuseFailAlloc_2895_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2895_, 0, v___x_2892_);
v___x_2894_ = v_reuseFailAlloc_2895_;
goto v_reusejp_2893_;
}
v_reusejp_2893_:
{
return v___x_2894_;
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
}
}
}
}
}
}
}
else
{
lean_object* v___x_2898_; lean_object* v___x_2899_; uint8_t v___x_2900_; 
v___x_2898_ = l_Lean_Syntax_getArg(v___x_2822_, v___x_2810_);
v___x_2899_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__5));
lean_inc(v___x_2898_);
v___x_2900_ = l_Lean_Syntax_isOfKind(v___x_2898_, v___x_2899_);
if (v___x_2900_ == 0)
{
lean_dec(v___x_2898_);
lean_dec(v___x_2822_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2901_; lean_object* v___x_2902_; uint8_t v___x_2903_; 
v___x_2901_ = l_Lean_Syntax_getArg(v___x_2898_, v___x_2810_);
lean_dec(v___x_2898_);
v___x_2902_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__4));
lean_inc(v___x_2901_);
v___x_2903_ = l_Lean_Syntax_isOfKind(v___x_2901_, v___x_2902_);
if (v___x_2903_ == 0)
{
lean_dec(v___x_2901_);
lean_dec(v___x_2822_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2904_; uint8_t v___x_2905_; 
v___x_2904_ = l_Lean_Syntax_getArg(v___x_2822_, v___x_2787_);
lean_dec(v___x_2822_);
v___x_2905_ = l_Lean_Syntax_matchesNull(v___x_2904_, v___x_2810_);
if (v___x_2905_ == 0)
{
lean_dec(v___x_2901_);
lean_dec(v_a_2816_);
return v___x_2815_;
}
else
{
lean_object* v___x_2906_; uint8_t v___x_2907_; 
v___x_2906_ = l_Lean_Syntax_getArg(v_a_2816_, v___x_2791_);
lean_dec(v_a_2816_);
lean_inc(v___x_2906_);
v___x_2907_ = l_Lean_Syntax_isOfKind(v___x_2906_, v___x_2817_);
if (v___x_2907_ == 0)
{
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2908_; uint8_t v___x_2909_; 
v___x_2908_ = l_Lean_Syntax_getArg(v___x_2906_, v___x_2787_);
lean_inc(v___x_2908_);
v___x_2909_ = l_Lean_Syntax_isOfKind(v___x_2908_, v___x_2820_);
if (v___x_2909_ == 0)
{
lean_dec(v___x_2908_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2910_; lean_object* v___x_2911_; uint8_t v___x_2912_; 
v___x_2910_ = l_Lean_Syntax_getArg(v___x_2908_, v___x_2810_);
lean_dec(v___x_2908_);
v___x_2911_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___lam__3___closed__8));
lean_inc(v___x_2910_);
v___x_2912_ = l_Lean_Syntax_isOfKind(v___x_2910_, v___x_2911_);
if (v___x_2912_ == 0)
{
lean_dec(v___x_2910_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2913_; uint8_t v___x_2914_; 
v___x_2913_ = l_Lean_Syntax_getArg(v___x_2910_, v___x_2810_);
lean_dec(v___x_2910_);
lean_inc(v___x_2913_);
v___x_2914_ = l_Lean_Syntax_matchesNull(v___x_2913_, v___x_2787_);
if (v___x_2914_ == 0)
{
lean_dec(v___x_2913_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2915_; lean_object* v___x_2916_; uint8_t v___x_2917_; 
v___x_2915_ = l_Lean_Syntax_getArg(v___x_2913_, v___x_2810_);
lean_dec(v___x_2913_);
v___x_2916_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__1));
lean_inc(v___x_2915_);
v___x_2917_ = l_Lean_Syntax_isOfKind(v___x_2915_, v___x_2916_);
if (v___x_2917_ == 0)
{
lean_dec(v___x_2915_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2918_; uint8_t v___x_2919_; 
v___x_2918_ = l_Lean_Syntax_getArg(v___x_2915_, v___x_2787_);
lean_dec(v___x_2915_);
lean_inc(v___x_2918_);
v___x_2919_ = l_Lean_Syntax_isOfKind(v___x_2918_, v___x_2823_);
if (v___x_2919_ == 0)
{
lean_dec(v___x_2918_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2920_; uint8_t v___x_2921_; 
v___x_2920_ = l_Lean_Syntax_getArg(v___x_2918_, v___x_2810_);
lean_inc(v___x_2920_);
v___x_2921_ = l_Lean_Syntax_isOfKind(v___x_2920_, v___x_2899_);
if (v___x_2921_ == 0)
{
lean_dec(v___x_2920_);
lean_dec(v___x_2918_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2922_; lean_object* v___x_2923_; uint8_t v___x_2924_; 
v___x_2922_ = l_Lean_Syntax_getArg(v___x_2920_, v___x_2810_);
lean_dec(v___x_2920_);
v___x_2923_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__10));
v___x_2924_ = l_Lean_Syntax_isOfKind(v___x_2922_, v___x_2923_);
if (v___x_2924_ == 0)
{
lean_dec(v___x_2918_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2925_; uint8_t v___x_2926_; 
v___x_2925_ = l_Lean_Syntax_getArg(v___x_2918_, v___x_2787_);
lean_dec(v___x_2918_);
lean_inc(v___x_2925_);
v___x_2926_ = l_Lean_Syntax_matchesNull(v___x_2925_, v___x_2787_);
if (v___x_2926_ == 0)
{
lean_dec(v___x_2925_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2927_; lean_object* v___x_2928_; uint8_t v___x_2929_; 
v___x_2927_ = l_Lean_Syntax_getArg(v___x_2925_, v___x_2810_);
lean_dec(v___x_2925_);
v___x_2928_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__0___closed__7));
lean_inc(v___x_2927_);
v___x_2929_ = l_Lean_Syntax_isOfKind(v___x_2927_, v___x_2928_);
if (v___x_2929_ == 0)
{
lean_dec(v___x_2927_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2930_; lean_object* v___x_2931_; uint8_t v___x_2932_; 
v___x_2930_ = l_Lean_Syntax_getArg(v___x_2927_, v___x_2787_);
lean_dec(v___x_2927_);
v___x_2931_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__6));
lean_inc(v___x_2930_);
v___x_2932_ = l_Lean_Syntax_isOfKind(v___x_2930_, v___x_2931_);
if (v___x_2932_ == 0)
{
lean_dec(v___x_2930_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2933_; uint8_t v___x_2934_; 
v___x_2933_ = l_Lean_Syntax_getArg(v___x_2930_, v___x_2810_);
lean_inc(v___x_2933_);
v___x_2934_ = l_Lean_Syntax_isOfKind(v___x_2933_, v___x_2902_);
if (v___x_2934_ == 0)
{
lean_dec(v___x_2933_);
lean_dec(v___x_2930_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
uint8_t v___x_2935_; 
v___x_2935_ = l_Lean_Syntax_structEq(v___x_2901_, v___x_2933_);
lean_dec(v___x_2933_);
if (v___x_2935_ == 0)
{
lean_dec(v___x_2930_);
lean_dec(v___x_2906_);
lean_dec(v___x_2901_);
return v___x_2815_;
}
else
{
lean_object* v___x_2937_; uint8_t v_isShared_2938_; uint8_t v_isSharedCheck_2961_; 
v_isSharedCheck_2961_ = !lean_is_exclusive(v___x_2815_);
if (v_isSharedCheck_2961_ == 0)
{
lean_object* v_unused_2962_; 
v_unused_2962_ = lean_ctor_get(v___x_2815_, 0);
lean_dec(v_unused_2962_);
v___x_2937_ = v___x_2815_;
v_isShared_2938_ = v_isSharedCheck_2961_;
goto v_resetjp_2936_;
}
else
{
lean_dec(v___x_2815_);
v___x_2937_ = lean_box(0);
v_isShared_2938_ = v_isSharedCheck_2961_;
goto v_resetjp_2936_;
}
v_resetjp_2936_:
{
lean_object* v_ref_2939_; lean_object* v___x_2940_; lean_object* v___x_2941_; uint8_t v___x_2942_; lean_object* v___x_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; lean_object* v___x_2947_; lean_object* v___x_2948_; lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; lean_object* v___x_2952_; lean_object* v___x_2953_; lean_object* v___x_2954_; lean_object* v___x_2955_; lean_object* v___x_2956_; lean_object* v___x_2957_; lean_object* v___x_2959_; 
v_ref_2939_ = lean_ctor_get(v___y_2802_, 5);
v___x_2940_ = l_Lean_Syntax_getArg(v___x_2930_, v___x_2795_);
lean_dec(v___x_2930_);
v___x_2941_ = l_Lean_Syntax_getArg(v___x_2906_, v___x_2791_);
lean_dec(v___x_2906_);
v___x_2942_ = 0;
v___x_2943_ = l_Lean_SourceInfo_fromRef(v_ref_2939_, v___x_2942_);
v___x_2944_ = ((lean_object*)(lp_mathlib_Set_term_u22c2___x2c___00__closed__2));
lean_inc_n(v___x_2943_, 8);
v___x_2945_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2945_, 0, v___x_2943_);
lean_ctor_set(v___x_2945_, 1, v___x_2944_);
v___x_2946_ = l_Lean_Syntax_node1(v___x_2943_, v___x_2899_, v___x_2901_);
v___x_2947_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__21));
v___x_2948_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__8));
v___x_2949_ = ((lean_object*)(lp_mathlib_iSup__delab___lam__2___closed__9));
v___x_2950_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2950_, 0, v___x_2943_);
lean_ctor_set(v___x_2950_, 1, v___x_2949_);
v___x_2951_ = l_Lean_Syntax_node2(v___x_2943_, v___x_2948_, v___x_2950_, v___x_2940_);
v___x_2952_ = l_Lean_Syntax_node1(v___x_2943_, v___x_2947_, v___x_2951_);
v___x_2953_ = l_Lean_Syntax_node2(v___x_2943_, v___x_2823_, v___x_2946_, v___x_2952_);
v___x_2954_ = l_Lean_Syntax_node1(v___x_2943_, v___x_2820_, v___x_2953_);
v___x_2955_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______macroRules__term_u2a06___x2c____1___closed__23));
v___x_2956_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2956_, 0, v___x_2943_);
lean_ctor_set(v___x_2956_, 1, v___x_2955_);
v___x_2957_ = l_Lean_Syntax_node4(v___x_2943_, v___x_2817_, v___x_2945_, v___x_2954_, v___x_2956_, v___x_2941_);
if (v_isShared_2938_ == 0)
{
lean_ctor_set(v___x_2937_, 0, v___x_2957_);
v___x_2959_ = v___x_2937_;
goto v_reusejp_2958_;
}
else
{
lean_object* v_reuseFailAlloc_2960_; 
v_reuseFailAlloc_2960_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2960_, 0, v___x_2957_);
v___x_2959_ = v_reuseFailAlloc_2960_;
goto v_reusejp_2958_;
}
v_reusejp_2958_:
{
return v___x_2959_;
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
}
}
}
}
}
else
{
return v___x_2815_;
}
}
else
{
lean_object* v_a_2963_; lean_object* v___x_2965_; uint8_t v_isShared_2966_; uint8_t v_isSharedCheck_2970_; 
lean_dec(v_a_2805_);
lean_dec(v___x_2796_);
v_a_2963_ = lean_ctor_get(v___x_2807_, 0);
v_isSharedCheck_2970_ = !lean_is_exclusive(v___x_2807_);
if (v_isSharedCheck_2970_ == 0)
{
v___x_2965_ = v___x_2807_;
v_isShared_2966_ = v_isSharedCheck_2970_;
goto v_resetjp_2964_;
}
else
{
lean_inc(v_a_2963_);
lean_dec(v___x_2807_);
v___x_2965_ = lean_box(0);
v_isShared_2966_ = v_isSharedCheck_2970_;
goto v_resetjp_2964_;
}
v_resetjp_2964_:
{
lean_object* v___x_2968_; 
if (v_isShared_2966_ == 0)
{
v___x_2968_ = v___x_2965_;
goto v_reusejp_2967_;
}
else
{
lean_object* v_reuseFailAlloc_2969_; 
v_reuseFailAlloc_2969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2969_, 0, v_a_2963_);
v___x_2968_ = v_reuseFailAlloc_2969_;
goto v_reusejp_2967_;
}
v_reusejp_2967_:
{
return v___x_2968_;
}
}
}
}
else
{
lean_object* v_a_2971_; lean_object* v___x_2973_; uint8_t v_isShared_2974_; uint8_t v_isSharedCheck_2978_; 
lean_dec(v___x_2796_);
v_a_2971_ = lean_ctor_get(v___x_2804_, 0);
v_isSharedCheck_2978_ = !lean_is_exclusive(v___x_2804_);
if (v_isSharedCheck_2978_ == 0)
{
v___x_2973_ = v___x_2804_;
v_isShared_2974_ = v_isSharedCheck_2978_;
goto v_resetjp_2972_;
}
else
{
lean_inc(v_a_2971_);
lean_dec(v___x_2804_);
v___x_2973_ = lean_box(0);
v_isShared_2974_ = v_isSharedCheck_2978_;
goto v_resetjp_2972_;
}
v_resetjp_2972_:
{
lean_object* v___x_2976_; 
if (v_isShared_2974_ == 0)
{
v___x_2976_ = v___x_2973_;
goto v_reusejp_2975_;
}
else
{
lean_object* v_reuseFailAlloc_2977_; 
v_reuseFailAlloc_2977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2977_, 0, v_a_2971_);
v___x_2976_ = v_reuseFailAlloc_2977_;
goto v_reusejp_2975_;
}
v_reusejp_2975_:
{
return v___x_2976_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___lam__2___boxed(lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_){
_start:
{
lean_object* v_res_2996_; 
v_res_2996_ = lp_mathlib_Set_sInter__delab___lam__2(v___y_2989_, v___y_2990_, v___y_2991_, v___y_2992_, v___y_2993_, v___y_2994_);
lean_dec(v___y_2994_);
lean_dec_ref(v___y_2993_);
lean_dec(v___y_2992_);
lean_dec_ref(v___y_2991_);
lean_dec(v___y_2990_);
lean_dec_ref(v___y_2989_);
return v_res_2996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab(lean_object* v_a_2998_, lean_object* v_a_2999_, lean_object* v_a_3000_, lean_object* v_a_3001_, lean_object* v_a_3002_, lean_object* v_a_3003_){
_start:
{
lean_object* v___f_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; 
v___f_3005_ = ((lean_object*)(lp_mathlib_Set_sInter__delab___closed__0));
v___x_3006_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__SetNotation______delab__app__term_u2a06___x2c____1___closed__3));
v___x_3007_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_3006_, v___f_3005_, v_a_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_, v_a_3003_);
return v___x_3007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_sInter__delab___boxed(lean_object* v_a_3008_, lean_object* v_a_3009_, lean_object* v_a_3010_, lean_object* v_a_3011_, lean_object* v_a_3012_, lean_object* v_a_3013_, lean_object* v_a_3014_){
_start:
{
lean_object* v_res_3015_; 
v_res_3015_ = lp_mathlib_Set_sInter__delab(v_a_3008_, v_a_3009_, v_a_3010_, v_a_3011_, v_a_3012_, v_a_3013_);
lean_dec(v_a_3013_);
lean_dec_ref(v_a_3012_);
lean_dec(v_a_3011_);
lean_dec_ref(v_a_3010_);
lean_dec(v_a_3009_);
lean_dec_ref(v_a_3008_);
return v_res_3015_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_Notation3(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Notation3(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_term_u2a06___x2c__ = _init_lp_mathlib_term_u2a06___x2c__();
lean_mark_persistent(lp_mathlib_term_u2a06___x2c__);
lp_mathlib_term_u2a05___x2c__ = _init_lp_mathlib_term_u2a05___x2c__();
lean_mark_persistent(lp_mathlib_term_u2a05___x2c__);
lp_mathlib_Set_term_u22c3___x2c__ = _init_lp_mathlib_Set_term_u22c3___x2c__();
lean_mark_persistent(lp_mathlib_Set_term_u22c3___x2c__);
lp_mathlib_Set_term_u22c2___x2c__ = _init_lp_mathlib_Set_term_u22c2___x2c__();
lean_mark_persistent(lp_mathlib_Set_term_u22c2___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_Notation3(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_Notation3(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_SetNotation(builtin);
}
#ifdef __cplusplus
}
#endif
