// Lean compiler output
// Module: Mathlib.Data.Set.Defs
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias public import Batteries.Util.ExtendedBinder public import Mathlib.Tactic.SetNotationForOrder public import Mathlib.Tactic.ToDual
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder;
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instMembership(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instLE(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instEmptyCollection(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "setBuilder"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(55, 252, 174, 2, 80, 49, 173, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_setBuilder___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " | "};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_setBuilder___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_setBuilder___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Meta_setBuilder___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_setBuilder___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_setBuilder___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_setBuilder___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_setBuilder___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_setBuilder;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__5_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__7_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__9_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Set.ofPred"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ofPred"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__17_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__18_value),LEAN_SCALAR_PTR_LITERAL(126, 61, 51, 114, 236, 177, 229, 37)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__22_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__26_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↦"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∧_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__30_value),LEAN_SCALAR_PTR_LITERAL(213, 224, 85, 99, 168, 124, 84, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termSatisfies_binder_pred%__"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__32_value),LEAN_SCALAR_PTR_LITERAL(35, 32, 166, 185, 227, 132, 228, 81)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "satisfies_binder_pred%"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∧"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__36_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__4_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term{_|_}"};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(16, 91, 238, 21, 109, 150, 73, 138)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term∃ᵉ_,_"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(224, 183, 129, 16, 236, 95, 122, 189)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∃ᵉ"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_=_"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(167, 251, 107, 62, 223, 239, 203, 78)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "="};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "macroPattSetBuilder"};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 153, 5, 7, 181, 54, 182, 232)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_macroPattSetBuilder = (const lean_object*)&lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "match"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(9, 208, 235, 82, 91, 230, 203, 159)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "matchDiscr"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(99, 51, 127, 238, 206, 239, 57, 130)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "matchAlts"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(193, 186, 26, 109, 82, 172, 197, 183)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "matchAlt"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(178, 0, 203, 112, 215, 49, 100, 229)}};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "term{_|_}_1"};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 186, 237, 85, 221, 221, 43, 160)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPredPatternMatchUnexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPredPatternMatchUnexpander___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instInsert(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instSingletonSet(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instUnion(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instInter(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instSDiff(lean_object*);
static const lean_string_object lp_mathlib_Set_term_U0001d4ab___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 6, .m_data = "term𝒫_"};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Set_term_U0001d4ab___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__17_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term_U0001d4ab___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(231, 19, 75, 28, 59, 157, 250, 16)}};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__1_value;
static const lean_string_object lp_mathlib_Set_term_U0001d4ab___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 2, .m_data = "𝒫 "};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Set_term_U0001d4ab___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__2_value)}};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Set_term_U0001d4ab___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__13_value),((lean_object*)(((size_t)(100) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__4 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Set_term_U0001d4ab___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_setBuilder___closed__5_value),((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__3_value),((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__4_value)}};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__5 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Set_term_U0001d4ab___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__1_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__5_value)}};
static const lean_object* lp_mathlib_Set_term_U0001d4ab___00__closed__6 = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_term_U0001d4ab__ = (const lean_object*)&lp_mathlib_Set_term_U0001d4ab___00__closed__6_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "powerset"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__1;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 26, 175, 181, 163, 8, 133, 48)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__17_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(169, 177, 191, 242, 228, 207, 232, 61)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______unexpand__Set__powerset__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______unexpand__Set__powerset__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Set_instFunctor_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
LEAN_EXPORT const lean_object* lp_mathlib_Set_instFunctor = (const lean_object*)&lp_mathlib_Set_instFunctor_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_instMembership(lean_object* v_00_u03b1_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instLE(lean_object* v_00_u03b1_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instEmptyCollection(lean_object* v_00_u03b1_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__8(void){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_20_ = lp_batteries_Batteries_ExtendedBinder_extBinder;
v___x_21_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__7));
v___x_22_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__5));
v___x_23_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_23_, 0, v___x_22_);
lean_ctor_set(v___x_23_, 1, v___x_21_);
lean_ctor_set(v___x_23_, 2, v___x_20_);
return v___x_23_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__11(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__10));
v___x_28_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_setBuilder___closed__8, &lp_mathlib_Mathlib_Meta_setBuilder___closed__8_once, _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__8);
v___x_29_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__5));
v___x_30_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___x_28_);
lean_ctor_set(v___x_30_, 2, v___x_27_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__15(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_37_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__14));
v___x_38_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_setBuilder___closed__11, &lp_mathlib_Mathlib_Meta_setBuilder___closed__11_once, _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__11);
v___x_39_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__5));
v___x_40_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___x_38_);
lean_ctor_set(v___x_40_, 2, v___x_37_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__18(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_44_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__17));
v___x_45_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_setBuilder___closed__15, &lp_mathlib_Mathlib_Meta_setBuilder___closed__15_once, _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__15);
v___x_46_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__5));
v___x_47_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_47_, 0, v___x_46_);
lean_ctor_set(v___x_47_, 1, v___x_45_);
lean_ctor_set(v___x_47_, 2, v___x_44_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__19(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_48_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_setBuilder___closed__18, &lp_mathlib_Mathlib_Meta_setBuilder___closed__18_once, _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__18);
v___x_49_ = lean_unsigned_to_nat(1024u);
v___x_50_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
v___x_51_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_51_, 0, v___x_50_);
lean_ctor_set(v___x_51_, 1, v___x_49_);
lean_ctor_set(v___x_51_, 2, v___x_48_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_setBuilder(void){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_setBuilder___closed__19, &lp_mathlib_Mathlib_Meta_setBuilder___closed__19_once, _init_lp_mathlib_Mathlib_Meta_setBuilder___closed__19);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_box(0);
v___x_54_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_55_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_55_, 0, v___x_54_);
lean_ctor_set(v___x_55_, 1, v___x_53_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg(){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___closed__0);
v___x_58_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg___boxed(lean_object* v___y_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0(lean_object* v_00_u03b1_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___boxed(lean_object* v_00_u03b1_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0(v_00_u03b1_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
return v_res_78_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_106_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__15));
v___x_107_ = l_String_toRawSubstring_x27(v___x_106_);
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28(void){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = l_Array_mkArray0(lean_box(0));
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder(lean_object* v_x_152_, lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_){
_start:
{
lean_object* v___x_161_; uint8_t v___x_162_; 
v___x_161_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
lean_inc(v_x_152_);
v___x_162_ = l_Lean_Syntax_isOfKind(v_x_152_, v___x_161_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; 
lean_dec(v_x_153_);
lean_dec(v_x_152_);
v___x_163_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v___x_163_;
}
else
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; 
v___x_164_ = lean_unsigned_to_nat(1u);
v___x_165_ = l_Lean_Syntax_getArg(v_x_152_, v___x_164_);
v___x_166_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3));
lean_inc(v___x_165_);
v___x_167_ = l_Lean_Syntax_isOfKind(v___x_165_, v___x_166_);
if (v___x_167_ == 0)
{
lean_object* v___x_168_; 
lean_dec(v___x_165_);
lean_dec(v_x_153_);
lean_dec(v_x_152_);
v___x_168_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v___x_168_;
}
else
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_169_ = lean_unsigned_to_nat(0u);
v___x_170_ = l_Lean_Syntax_getArg(v___x_165_, v___x_169_);
v___x_171_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6));
lean_inc(v___x_170_);
v___x_172_ = l_Lean_Syntax_isOfKind(v___x_170_, v___x_171_);
if (v___x_172_ == 0)
{
lean_object* v___x_173_; 
lean_dec(v___x_170_);
lean_dec(v___x_165_);
lean_dec(v_x_153_);
lean_dec(v_x_152_);
v___x_173_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v___x_173_;
}
else
{
lean_object* v___x_174_; lean_object* v___x_175_; uint8_t v___x_176_; 
v___x_174_ = l_Lean_Syntax_getArg(v___x_170_, v___x_169_);
lean_dec(v___x_170_);
v___x_175_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8));
lean_inc(v___x_174_);
v___x_176_ = l_Lean_Syntax_isOfKind(v___x_174_, v___x_175_);
if (v___x_176_ == 0)
{
lean_object* v___x_177_; 
lean_dec(v___x_174_);
lean_dec(v___x_165_);
lean_dec(v_x_153_);
lean_dec(v_x_152_);
v___x_177_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v___x_177_;
}
else
{
lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_178_ = l_Lean_Syntax_getArg(v___x_165_, v___x_164_);
lean_dec(v___x_165_);
lean_inc(v___x_178_);
v___x_179_ = l_Lean_Syntax_matchesNull(v___x_178_, v___x_169_);
if (v___x_179_ == 0)
{
uint8_t v___x_180_; 
lean_inc(v___x_178_);
v___x_180_ = l_Lean_Syntax_matchesNull(v___x_178_, v___x_164_);
if (v___x_180_ == 0)
{
lean_object* v___x_181_; 
lean_dec(v___x_178_);
lean_dec(v___x_174_);
lean_dec(v_x_153_);
lean_dec(v_x_152_);
v___x_181_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabSetBuilder_spec__0___redArg();
return v___x_181_;
}
else
{
lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_182_ = l_Lean_Syntax_getArg(v___x_178_, v___x_169_);
lean_dec(v___x_178_);
v___x_183_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__10));
lean_inc(v___x_182_);
v___x_184_ = l_Lean_Syntax_isOfKind(v___x_182_, v___x_183_);
if (v___x_184_ == 0)
{
lean_object* v_ref_185_; lean_object* v_quotContext_186_; lean_object* v_currMacroScope_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_ref_185_ = lean_ctor_get(v_a_158_, 5);
v_quotContext_186_ = lean_ctor_get(v_a_158_, 10);
v_currMacroScope_187_ = lean_ctor_get(v_a_158_, 11);
v___x_188_ = lean_unsigned_to_nat(3u);
v___x_189_ = l_Lean_Syntax_getArg(v_x_152_, v___x_188_);
lean_dec(v_x_152_);
v___x_190_ = l_Lean_SourceInfo_fromRef(v_ref_185_, v___x_184_);
v___x_191_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
v___x_192_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16);
v___x_193_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19));
lean_inc(v_currMacroScope_187_);
lean_inc(v_quotContext_186_);
v___x_194_ = l_Lean_addMacroScope(v_quotContext_186_, v___x_193_, v_currMacroScope_187_);
v___x_195_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__21));
lean_inc_n(v___x_190_, 12);
v___x_196_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_196_, 0, v___x_190_);
lean_ctor_set(v___x_196_, 1, v___x_192_);
lean_ctor_set(v___x_196_, 2, v___x_194_);
lean_ctor_set(v___x_196_, 3, v___x_195_);
v___x_197_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24));
v___x_199_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25));
v___x_200_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_190_);
lean_ctor_set(v___x_200_, 1, v___x_198_);
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27));
lean_inc(v___x_174_);
v___x_202_ = l_Lean_Syntax_node1(v___x_190_, v___x_197_, v___x_174_);
v___x_203_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28);
v___x_204_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_204_, 0, v___x_190_);
lean_ctor_set(v___x_204_, 1, v___x_197_);
lean_ctor_set(v___x_204_, 2, v___x_203_);
v___x_205_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__29));
v___x_206_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_190_);
lean_ctor_set(v___x_206_, 1, v___x_205_);
v___x_207_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__31));
v___x_208_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__33));
v___x_209_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__34));
v___x_210_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_210_, 0, v___x_190_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
v___x_211_ = l_Lean_Syntax_node3(v___x_190_, v___x_208_, v___x_210_, v___x_174_, v___x_182_);
v___x_212_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__35));
v___x_213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_190_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = l_Lean_Syntax_node3(v___x_190_, v___x_207_, v___x_211_, v___x_213_, v___x_189_);
v___x_215_ = l_Lean_Syntax_node4(v___x_190_, v___x_201_, v___x_202_, v___x_204_, v___x_206_, v___x_214_);
v___x_216_ = l_Lean_Syntax_node2(v___x_190_, v___x_199_, v___x_200_, v___x_215_);
v___x_217_ = l_Lean_Syntax_node1(v___x_190_, v___x_197_, v___x_216_);
v___x_218_ = l_Lean_Syntax_node2(v___x_190_, v___x_191_, v___x_196_, v___x_217_);
v___x_219_ = l_Lean_Elab_Term_elabTerm(v___x_218_, v_x_153_, v___x_176_, v___x_176_, v_a_154_, v_a_155_, v_a_156_, v_a_157_, v_a_158_, v_a_159_);
return v___x_219_;
}
else
{
lean_object* v_ref_220_; lean_object* v_quotContext_221_; lean_object* v_currMacroScope_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v_ref_220_ = lean_ctor_get(v_a_158_, 5);
v_quotContext_221_ = lean_ctor_get(v_a_158_, 10);
v_currMacroScope_222_ = lean_ctor_get(v_a_158_, 11);
v___x_223_ = l_Lean_Syntax_getArg(v___x_182_, v___x_164_);
lean_dec(v___x_182_);
v___x_224_ = lean_unsigned_to_nat(3u);
v___x_225_ = l_Lean_Syntax_getArg(v_x_152_, v___x_224_);
lean_dec(v_x_152_);
v___x_226_ = l_Lean_SourceInfo_fromRef(v_ref_220_, v___x_179_);
v___x_227_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
v___x_228_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16);
v___x_229_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19));
lean_inc(v_currMacroScope_222_);
lean_inc(v_quotContext_221_);
v___x_230_ = l_Lean_addMacroScope(v_quotContext_221_, v___x_229_, v_currMacroScope_222_);
v___x_231_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__21));
lean_inc_n(v___x_226_, 10);
v___x_232_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_232_, 0, v___x_226_);
lean_ctor_set(v___x_232_, 1, v___x_228_);
lean_ctor_set(v___x_232_, 2, v___x_230_);
lean_ctor_set(v___x_232_, 3, v___x_231_);
v___x_233_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_234_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24));
v___x_235_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25));
v___x_236_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_226_);
lean_ctor_set(v___x_236_, 1, v___x_234_);
v___x_237_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27));
v___x_238_ = l_Lean_Syntax_node1(v___x_226_, v___x_233_, v___x_174_);
v___x_239_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__37));
v___x_240_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38));
v___x_241_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_226_);
lean_ctor_set(v___x_241_, 1, v___x_240_);
v___x_242_ = l_Lean_Syntax_node2(v___x_226_, v___x_239_, v___x_241_, v___x_223_);
v___x_243_ = l_Lean_Syntax_node1(v___x_226_, v___x_233_, v___x_242_);
v___x_244_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__29));
v___x_245_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_226_);
lean_ctor_set(v___x_245_, 1, v___x_244_);
v___x_246_ = l_Lean_Syntax_node4(v___x_226_, v___x_237_, v___x_238_, v___x_243_, v___x_245_, v___x_225_);
v___x_247_ = l_Lean_Syntax_node2(v___x_226_, v___x_235_, v___x_236_, v___x_246_);
v___x_248_ = l_Lean_Syntax_node1(v___x_226_, v___x_233_, v___x_247_);
v___x_249_ = l_Lean_Syntax_node2(v___x_226_, v___x_227_, v___x_232_, v___x_248_);
v___x_250_ = l_Lean_Elab_Term_elabTerm(v___x_249_, v_x_153_, v___x_176_, v___x_176_, v_a_154_, v_a_155_, v_a_156_, v_a_157_, v_a_158_, v_a_159_);
return v___x_250_;
}
}
}
else
{
lean_object* v_ref_251_; lean_object* v_quotContext_252_; lean_object* v_currMacroScope_253_; lean_object* v___x_254_; lean_object* v___x_255_; uint8_t v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
lean_dec(v___x_178_);
v_ref_251_ = lean_ctor_get(v_a_158_, 5);
v_quotContext_252_ = lean_ctor_get(v_a_158_, 10);
v_currMacroScope_253_ = lean_ctor_get(v_a_158_, 11);
v___x_254_ = lean_unsigned_to_nat(3u);
v___x_255_ = l_Lean_Syntax_getArg(v_x_152_, v___x_254_);
lean_dec(v_x_152_);
v___x_256_ = 0;
v___x_257_ = l_Lean_SourceInfo_fromRef(v_ref_251_, v___x_256_);
v___x_258_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
v___x_259_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__16);
v___x_260_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__19));
lean_inc(v_currMacroScope_253_);
lean_inc(v_quotContext_252_);
v___x_261_ = l_Lean_addMacroScope(v_quotContext_252_, v___x_260_, v_currMacroScope_253_);
v___x_262_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__21));
lean_inc_n(v___x_257_, 8);
v___x_263_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_263_, 0, v___x_257_);
lean_ctor_set(v___x_263_, 1, v___x_259_);
lean_ctor_set(v___x_263_, 2, v___x_261_);
lean_ctor_set(v___x_263_, 3, v___x_262_);
v___x_264_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_265_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__24));
v___x_266_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25));
v___x_267_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_257_);
lean_ctor_set(v___x_267_, 1, v___x_265_);
v___x_268_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27));
v___x_269_ = l_Lean_Syntax_node1(v___x_257_, v___x_264_, v___x_174_);
v___x_270_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28);
v___x_271_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_271_, 0, v___x_257_);
lean_ctor_set(v___x_271_, 1, v___x_264_);
lean_ctor_set(v___x_271_, 2, v___x_270_);
v___x_272_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__29));
v___x_273_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_257_);
lean_ctor_set(v___x_273_, 1, v___x_272_);
v___x_274_ = l_Lean_Syntax_node4(v___x_257_, v___x_268_, v___x_269_, v___x_271_, v___x_273_, v___x_255_);
v___x_275_ = l_Lean_Syntax_node2(v___x_257_, v___x_266_, v___x_267_, v___x_274_);
v___x_276_ = l_Lean_Syntax_node1(v___x_257_, v___x_264_, v___x_275_);
v___x_277_ = l_Lean_Syntax_node2(v___x_257_, v___x_258_, v___x_263_, v___x_276_);
v___x_278_ = l_Lean_Elab_Term_elabTerm(v___x_277_, v_x_153_, v___x_176_, v___x_176_, v_a_154_, v_a_155_, v_a_156_, v_a_157_, v_a_158_, v_a_159_);
return v___x_278_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabSetBuilder___boxed(lean_object* v_x_279_, lean_object* v_x_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Mathlib_Meta_elabSetBuilder(v_x_279_, v_x_280_, v_a_281_, v_a_282_, v_a_283_, v_a_284_, v_a_285_, v_a_286_);
lean_dec(v_a_286_);
lean_dec_ref(v_a_285_);
lean_dec(v_a_284_);
lean_dec_ref(v_a_283_);
lean_dec(v_a_282_);
lean_dec_ref(v_a_281_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander(lean_object* v_x_305_, lean_object* v_a_306_, lean_object* v_a_307_){
_start:
{
lean_object* v___x_308_; uint8_t v___x_309_; 
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
lean_inc(v_x_305_);
v___x_309_ = l_Lean_Syntax_isOfKind(v_x_305_, v___x_308_);
if (v___x_309_ == 0)
{
lean_object* v___x_310_; lean_object* v___x_311_; 
lean_dec(v_x_305_);
v___x_310_ = lean_box(0);
v___x_311_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v_a_307_);
return v___x_311_;
}
else
{
lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v___x_312_ = lean_unsigned_to_nat(1u);
v___x_313_ = l_Lean_Syntax_getArg(v_x_305_, v___x_312_);
lean_dec(v_x_305_);
lean_inc(v___x_313_);
v___x_314_ = l_Lean_Syntax_matchesNull(v___x_313_, v___x_312_);
if (v___x_314_ == 0)
{
lean_object* v___x_315_; lean_object* v___x_316_; 
lean_dec(v___x_313_);
v___x_315_ = lean_box(0);
v___x_316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v_a_307_);
return v___x_316_;
}
else
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; uint8_t v___x_320_; 
v___x_317_ = lean_unsigned_to_nat(0u);
v___x_318_ = l_Lean_Syntax_getArg(v___x_313_, v___x_317_);
lean_dec(v___x_313_);
v___x_319_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25));
lean_inc(v___x_318_);
v___x_320_ = l_Lean_Syntax_isOfKind(v___x_318_, v___x_319_);
if (v___x_320_ == 0)
{
lean_object* v___x_321_; lean_object* v___x_322_; 
lean_dec(v___x_318_);
v___x_321_ = lean_box(0);
v___x_322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_321_);
lean_ctor_set(v___x_322_, 1, v_a_307_);
return v___x_322_;
}
else
{
lean_object* v___x_323_; lean_object* v___x_324_; uint8_t v___x_325_; 
v___x_323_ = l_Lean_Syntax_getArg(v___x_318_, v___x_312_);
lean_dec(v___x_318_);
v___x_324_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27));
lean_inc(v___x_323_);
v___x_325_ = l_Lean_Syntax_isOfKind(v___x_323_, v___x_324_);
if (v___x_325_ == 0)
{
lean_object* v___x_326_; lean_object* v___x_327_; 
lean_dec(v___x_323_);
v___x_326_ = lean_box(0);
v___x_327_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_327_, 0, v___x_326_);
lean_ctor_set(v___x_327_, 1, v_a_307_);
return v___x_327_;
}
else
{
lean_object* v___x_328_; uint8_t v___x_329_; 
v___x_328_ = l_Lean_Syntax_getArg(v___x_323_, v___x_317_);
lean_inc(v___x_328_);
v___x_329_ = l_Lean_Syntax_matchesNull(v___x_328_, v___x_312_);
if (v___x_329_ == 0)
{
lean_object* v___x_330_; lean_object* v___x_331_; 
lean_dec(v___x_328_);
lean_dec(v___x_323_);
v___x_330_ = lean_box(0);
v___x_331_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
lean_ctor_set(v___x_331_, 1, v_a_307_);
return v___x_331_;
}
else
{
lean_object* v___x_332_; lean_object* v___x_333_; uint8_t v___x_334_; 
v___x_332_ = l_Lean_Syntax_getArg(v___x_328_, v___x_317_);
lean_dec(v___x_328_);
v___x_333_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8));
lean_inc(v___x_332_);
v___x_334_ = l_Lean_Syntax_isOfKind(v___x_332_, v___x_333_);
if (v___x_334_ == 0)
{
lean_object* v___x_335_; uint8_t v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1));
lean_inc(v___x_332_);
v___x_336_ = l_Lean_Syntax_isOfKind(v___x_332_, v___x_335_);
if (v___x_336_ == 0)
{
lean_object* v___x_337_; lean_object* v___x_338_; 
lean_dec(v___x_332_);
lean_dec(v___x_323_);
v___x_337_ = lean_box(0);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v_a_307_);
return v___x_338_;
}
else
{
lean_object* v___x_339_; lean_object* v___x_340_; uint8_t v___x_341_; 
v___x_339_ = l_Lean_Syntax_getArg(v___x_332_, v___x_317_);
v___x_340_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3));
lean_inc(v___x_339_);
v___x_341_ = l_Lean_Syntax_isOfKind(v___x_339_, v___x_340_);
if (v___x_341_ == 0)
{
lean_object* v___x_342_; lean_object* v___x_343_; 
lean_dec(v___x_339_);
lean_dec(v___x_332_);
lean_dec(v___x_323_);
v___x_342_ = lean_box(0);
v___x_343_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_343_, 0, v___x_342_);
lean_ctor_set(v___x_343_, 1, v_a_307_);
return v___x_343_;
}
else
{
lean_object* v___x_344_; lean_object* v___x_345_; uint8_t v___x_346_; 
v___x_344_ = l_Lean_Syntax_getArg(v___x_339_, v___x_312_);
lean_dec(v___x_339_);
v___x_345_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__5));
lean_inc(v___x_344_);
v___x_346_ = l_Lean_Syntax_isOfKind(v___x_344_, v___x_345_);
if (v___x_346_ == 0)
{
lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec(v___x_344_);
lean_dec(v___x_332_);
lean_dec(v___x_323_);
v___x_347_ = lean_box(0);
v___x_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v_a_307_);
return v___x_348_;
}
else
{
lean_object* v___x_349_; lean_object* v___x_350_; uint8_t v___x_351_; 
v___x_349_ = l_Lean_Syntax_getArg(v___x_344_, v___x_317_);
lean_dec(v___x_344_);
v___x_350_ = lean_box(0);
v___x_351_ = l_Lean_Syntax_matchesIdent(v___x_349_, v___x_350_);
lean_dec(v___x_349_);
if (v___x_351_ == 0)
{
lean_object* v___x_352_; lean_object* v___x_353_; 
lean_dec(v___x_332_);
lean_dec(v___x_323_);
v___x_352_ = lean_box(0);
v___x_353_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_353_, 0, v___x_352_);
lean_ctor_set(v___x_353_, 1, v_a_307_);
return v___x_353_;
}
else
{
lean_object* v___x_354_; uint8_t v___x_355_; 
v___x_354_ = l_Lean_Syntax_getArg(v___x_332_, v___x_312_);
lean_inc(v___x_354_);
v___x_355_ = l_Lean_Syntax_isOfKind(v___x_354_, v___x_333_);
if (v___x_355_ == 0)
{
lean_object* v___x_356_; lean_object* v___x_357_; 
lean_dec(v___x_354_);
lean_dec(v___x_332_);
lean_dec(v___x_323_);
v___x_356_ = lean_box(0);
v___x_357_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
lean_ctor_set(v___x_357_, 1, v_a_307_);
return v___x_357_;
}
else
{
lean_object* v___x_358_; lean_object* v___x_359_; uint8_t v___x_360_; 
v___x_358_ = lean_unsigned_to_nat(3u);
v___x_359_ = l_Lean_Syntax_getArg(v___x_332_, v___x_358_);
lean_dec(v___x_332_);
lean_inc(v___x_359_);
v___x_360_ = l_Lean_Syntax_matchesNull(v___x_359_, v___x_312_);
if (v___x_360_ == 0)
{
lean_object* v___x_361_; lean_object* v___x_362_; 
lean_dec(v___x_359_);
lean_dec(v___x_354_);
lean_dec(v___x_323_);
v___x_361_ = lean_box(0);
v___x_362_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_361_);
lean_ctor_set(v___x_362_, 1, v_a_307_);
return v___x_362_;
}
else
{
lean_object* v___x_363_; uint8_t v___x_364_; 
v___x_363_ = l_Lean_Syntax_getArg(v___x_323_, v___x_312_);
v___x_364_ = l_Lean_Syntax_matchesNull(v___x_363_, v___x_317_);
if (v___x_364_ == 0)
{
lean_object* v___x_365_; lean_object* v___x_366_; 
lean_dec(v___x_359_);
lean_dec(v___x_354_);
lean_dec(v___x_323_);
v___x_365_ = lean_box(0);
v___x_366_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_366_, 0, v___x_365_);
lean_ctor_set(v___x_366_, 1, v_a_307_);
return v___x_366_;
}
else
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_367_ = l_Lean_Syntax_getArg(v___x_359_, v___x_317_);
lean_dec(v___x_359_);
v___x_368_ = l_Lean_Syntax_getArg(v___x_323_, v___x_358_);
lean_dec(v___x_323_);
v___x_369_ = l_Lean_SourceInfo_fromRef(v_a_306_, v___x_334_);
v___x_370_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_369_, 8);
v___x_372_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_372_, 0, v___x_369_);
lean_ctor_set(v___x_372_, 1, v___x_371_);
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3));
v___x_374_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6));
v___x_375_ = l_Lean_Syntax_node1(v___x_369_, v___x_374_, v___x_354_);
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_377_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__10));
v___x_378_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38));
v___x_379_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_369_);
lean_ctor_set(v___x_379_, 1, v___x_378_);
v___x_380_ = l_Lean_Syntax_node2(v___x_369_, v___x_377_, v___x_379_, v___x_367_);
v___x_381_ = l_Lean_Syntax_node1(v___x_369_, v___x_376_, v___x_380_);
v___x_382_ = l_Lean_Syntax_node2(v___x_369_, v___x_373_, v___x_375_, v___x_381_);
v___x_383_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_384_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_384_, 0, v___x_369_);
lean_ctor_set(v___x_384_, 1, v___x_383_);
v___x_385_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_386_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_369_);
lean_ctor_set(v___x_386_, 1, v___x_385_);
v___x_387_ = l_Lean_Syntax_node5(v___x_369_, v___x_370_, v___x_372_, v___x_382_, v___x_384_, v___x_368_, v___x_386_);
v___x_388_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_387_);
lean_ctor_set(v___x_388_, 1, v_a_307_);
return v___x_388_;
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
lean_object* v___x_389_; uint8_t v___x_390_; 
v___x_389_ = l_Lean_Syntax_getArg(v___x_323_, v___x_312_);
v___x_390_ = l_Lean_Syntax_matchesNull(v___x_389_, v___x_317_);
if (v___x_390_ == 0)
{
lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec(v___x_332_);
lean_dec(v___x_323_);
v___x_391_ = lean_box(0);
v___x_392_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
lean_ctor_set(v___x_392_, 1, v_a_307_);
return v___x_392_;
}
else
{
lean_object* v___x_393_; lean_object* v___x_394_; uint8_t v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_393_ = lean_unsigned_to_nat(3u);
v___x_394_ = l_Lean_Syntax_getArg(v___x_323_, v___x_393_);
lean_dec(v___x_323_);
v___x_395_ = 0;
v___x_396_ = l_Lean_SourceInfo_fromRef(v_a_306_, v___x_395_);
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
v___x_398_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_396_, 6);
v___x_399_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_396_);
lean_ctor_set(v___x_399_, 1, v___x_398_);
v___x_400_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3));
v___x_401_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6));
v___x_402_ = l_Lean_Syntax_node1(v___x_396_, v___x_401_, v___x_332_);
v___x_403_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_404_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28);
v___x_405_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_405_, 0, v___x_396_);
lean_ctor_set(v___x_405_, 1, v___x_403_);
lean_ctor_set(v___x_405_, 2, v___x_404_);
v___x_406_ = l_Lean_Syntax_node2(v___x_396_, v___x_400_, v___x_402_, v___x_405_);
v___x_407_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_408_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_396_);
lean_ctor_set(v___x_408_, 1, v___x_407_);
v___x_409_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_410_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_396_);
lean_ctor_set(v___x_410_, 1, v___x_409_);
v___x_411_ = l_Lean_Syntax_node5(v___x_396_, v___x_397_, v___x_399_, v___x_406_, v___x_408_, v___x_394_, v___x_410_);
v___x_412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_412_, 0, v___x_411_);
lean_ctor_set(v___x_412_, 1, v_a_307_);
return v___x_412_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPred_unexpander___boxed(lean_object* v_x_413_, lean_object* v_a_414_, lean_object* v_a_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Mathlib_Meta_ofPred_unexpander(v_x_413_, v_a_414_, v_a_415_);
lean_dec(v_a_414_);
return v_res_416_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__4(void){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_430_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_431_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__3));
v___x_432_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__5));
v___x_433_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
lean_ctor_set(v___x_433_, 1, v___x_431_);
lean_ctor_set(v___x_433_, 2, v___x_430_);
return v___x_433_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__5(void){
_start:
{
lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__17));
v___x_435_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__4, &lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__4_once, _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__4);
v___x_436_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__5));
v___x_437_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_437_, 0, v___x_436_);
lean_ctor_set(v___x_437_, 1, v___x_435_);
lean_ctor_set(v___x_437_, 2, v___x_434_);
return v___x_437_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__6(void){
_start:
{
lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_438_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__5, &lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__5_once, _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__5);
v___x_439_ = lean_unsigned_to_nat(1024u);
v___x_440_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1));
v___x_441_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_441_, 0, v___x_440_);
lean_ctor_set(v___x_441_, 1, v___x_439_);
lean_ctor_set(v___x_441_, 2, v___x_438_);
return v___x_441_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d(void){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__6, &lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__6_once, _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__6);
return v___x_442_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1(void){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_444_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__0));
v___x_445_ = l_String_toRawSubstring_x27(v___x_444_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1(lean_object* v_x_459_, lean_object* v_a_460_, lean_object* v_a_461_){
_start:
{
lean_object* v___x_462_; uint8_t v___x_463_; 
v___x_462_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d___closed__1));
lean_inc(v_x_459_);
v___x_463_ = l_Lean_Syntax_isOfKind(v_x_459_, v___x_462_);
if (v___x_463_ == 0)
{
lean_object* v___x_464_; lean_object* v___x_465_; 
lean_dec(v_x_459_);
v___x_464_ = lean_box(1);
v___x_465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
lean_ctor_set(v___x_465_, 1, v_a_461_);
return v___x_465_;
}
else
{
lean_object* v_quotContext_466_; lean_object* v_currMacroScope_467_; lean_object* v_ref_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; uint8_t v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
v_quotContext_466_ = lean_ctor_get(v_a_460_, 1);
v_currMacroScope_467_ = lean_ctor_get(v_a_460_, 2);
v_ref_468_ = lean_ctor_get(v_a_460_, 5);
v___x_469_ = lean_unsigned_to_nat(1u);
v___x_470_ = l_Lean_Syntax_getArg(v_x_459_, v___x_469_);
v___x_471_ = lean_unsigned_to_nat(3u);
v___x_472_ = l_Lean_Syntax_getArg(v_x_459_, v___x_471_);
lean_dec(v_x_459_);
v___x_473_ = 0;
v___x_474_ = l_Lean_SourceInfo_fromRef(v_ref_468_, v___x_473_);
v___x_475_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
v___x_476_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_474_, 12);
v___x_477_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_477_, 0, v___x_474_);
lean_ctor_set(v___x_477_, 1, v___x_476_);
v___x_478_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3));
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6));
v___x_480_ = lean_obj_once(&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1, &lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1);
v___x_481_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__2));
lean_inc(v_currMacroScope_467_);
lean_inc(v_quotContext_466_);
v___x_482_ = l_Lean_addMacroScope(v_quotContext_466_, v___x_481_, v_currMacroScope_467_);
v___x_483_ = lean_box(0);
v___x_484_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_484_, 0, v___x_474_);
lean_ctor_set(v___x_484_, 1, v___x_480_);
lean_ctor_set(v___x_484_, 2, v___x_482_);
lean_ctor_set(v___x_484_, 3, v___x_483_);
lean_inc_ref(v___x_484_);
v___x_485_ = l_Lean_Syntax_node1(v___x_474_, v___x_479_, v___x_484_);
v___x_486_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_487_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28);
v___x_488_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_488_, 0, v___x_474_);
lean_ctor_set(v___x_488_, 1, v___x_486_);
lean_ctor_set(v___x_488_, 2, v___x_487_);
v___x_489_ = l_Lean_Syntax_node2(v___x_474_, v___x_478_, v___x_485_, v___x_488_);
v___x_490_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_491_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_491_, 0, v___x_474_);
lean_ctor_set(v___x_491_, 1, v___x_490_);
v___x_492_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__4));
v___x_493_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__5));
v___x_494_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_474_);
lean_ctor_set(v___x_494_, 1, v___x_493_);
v___x_495_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__6));
v___x_496_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_496_, 0, v___x_474_);
lean_ctor_set(v___x_496_, 1, v___x_495_);
v___x_497_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__8));
v___x_498_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__9));
v___x_499_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_499_, 0, v___x_474_);
lean_ctor_set(v___x_499_, 1, v___x_498_);
v___x_500_ = l_Lean_Syntax_node3(v___x_474_, v___x_497_, v___x_470_, v___x_499_, v___x_484_);
v___x_501_ = l_Lean_Syntax_node4(v___x_474_, v___x_492_, v___x_494_, v___x_472_, v___x_496_, v___x_500_);
v___x_502_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_474_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
v___x_504_ = l_Lean_Syntax_node5(v___x_474_, v___x_475_, v___x_477_, v___x_489_, v___x_491_, v___x_501_, v___x_503_);
v___x_505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_505_, 0, v___x_504_);
lean_ctor_set(v___x_505_, 1, v_a_461_);
return v___x_505_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___boxed(lean_object* v_x_506_, lean_object* v_a_507_, lean_object* v_a_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1(v_x_506_, v_a_507_, v_a_508_);
lean_dec_ref(v_a_507_);
return v_res_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1(lean_object* v_x_569_, lean_object* v_a_570_, lean_object* v_a_571_){
_start:
{
lean_object* v___x_572_; uint8_t v___x_573_; 
v___x_572_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1));
lean_inc(v_x_569_);
v___x_573_ = l_Lean_Syntax_isOfKind(v_x_569_, v___x_572_);
if (v___x_573_ == 0)
{
lean_object* v___x_574_; lean_object* v___x_575_; 
lean_dec(v_x_569_);
v___x_574_ = lean_box(1);
v___x_575_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_575_, 0, v___x_574_);
lean_ctor_set(v___x_575_, 1, v_a_571_);
return v___x_575_;
}
else
{
lean_object* v_quotContext_576_; lean_object* v_currMacroScope_577_; lean_object* v_ref_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; uint8_t v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; 
v_quotContext_576_ = lean_ctor_get(v_a_570_, 1);
v_currMacroScope_577_ = lean_ctor_get(v_a_570_, 2);
v_ref_578_ = lean_ctor_get(v_a_570_, 5);
v___x_579_ = lean_unsigned_to_nat(1u);
v___x_580_ = l_Lean_Syntax_getArg(v_x_569_, v___x_579_);
v___x_581_ = lean_unsigned_to_nat(3u);
v___x_582_ = l_Lean_Syntax_getArg(v_x_569_, v___x_581_);
v___x_583_ = lean_unsigned_to_nat(5u);
v___x_584_ = l_Lean_Syntax_getArg(v_x_569_, v___x_583_);
lean_dec(v_x_569_);
v___x_585_ = 0;
v___x_586_ = l_Lean_SourceInfo_fromRef(v_ref_578_, v___x_585_);
v___x_587_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
v___x_588_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_586_, 21);
v___x_589_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_586_);
lean_ctor_set(v___x_589_, 1, v___x_588_);
v___x_590_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3));
v___x_591_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6));
v___x_592_ = lean_obj_once(&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1, &lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1);
v___x_593_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__2));
lean_inc(v_currMacroScope_577_);
lean_inc(v_quotContext_576_);
v___x_594_ = l_Lean_addMacroScope(v_quotContext_576_, v___x_593_, v_currMacroScope_577_);
v___x_595_ = lean_box(0);
v___x_596_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_596_, 0, v___x_586_);
lean_ctor_set(v___x_596_, 1, v___x_592_);
lean_ctor_set(v___x_596_, 2, v___x_594_);
lean_ctor_set(v___x_596_, 3, v___x_595_);
lean_inc_ref(v___x_596_);
v___x_597_ = l_Lean_Syntax_node1(v___x_586_, v___x_591_, v___x_596_);
v___x_598_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_599_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__10));
v___x_600_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38));
v___x_601_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_586_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = l_Lean_Syntax_node2(v___x_586_, v___x_599_, v___x_601_, v___x_582_);
v___x_603_ = l_Lean_Syntax_node1(v___x_586_, v___x_598_, v___x_602_);
v___x_604_ = l_Lean_Syntax_node2(v___x_586_, v___x_590_, v___x_597_, v___x_603_);
v___x_605_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_606_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_606_, 0, v___x_586_);
lean_ctor_set(v___x_606_, 1, v___x_605_);
v___x_607_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__0));
v___x_608_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1));
v___x_609_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_609_, 0, v___x_586_);
lean_ctor_set(v___x_609_, 1, v___x_607_);
v___x_610_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28);
v___x_611_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_611_, 0, v___x_586_);
lean_ctor_set(v___x_611_, 1, v___x_598_);
lean_ctor_set(v___x_611_, 2, v___x_610_);
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3));
lean_inc_ref_n(v___x_611_, 2);
v___x_613_ = l_Lean_Syntax_node2(v___x_586_, v___x_612_, v___x_611_, v___x_596_);
v___x_614_ = l_Lean_Syntax_node1(v___x_586_, v___x_598_, v___x_613_);
v___x_615_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__4));
v___x_616_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_586_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
v___x_617_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6));
v___x_618_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8));
v___x_619_ = l_Lean_Syntax_node1(v___x_586_, v___x_598_, v___x_580_);
v___x_620_ = l_Lean_Syntax_node1(v___x_586_, v___x_598_, v___x_619_);
v___x_621_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__9));
v___x_622_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_586_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
lean_inc_ref(v___x_606_);
v___x_623_ = l_Lean_Syntax_node4(v___x_586_, v___x_618_, v___x_606_, v___x_620_, v___x_622_, v___x_584_);
v___x_624_ = l_Lean_Syntax_node1(v___x_586_, v___x_598_, v___x_623_);
v___x_625_ = l_Lean_Syntax_node1(v___x_586_, v___x_617_, v___x_624_);
v___x_626_ = l_Lean_Syntax_node6(v___x_586_, v___x_608_, v___x_609_, v___x_611_, v___x_611_, v___x_614_, v___x_616_, v___x_625_);
v___x_627_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_628_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_586_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
v___x_629_ = l_Lean_Syntax_node5(v___x_586_, v___x_587_, v___x_589_, v___x_604_, v___x_606_, v___x_626_, v___x_628_);
v___x_630_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_629_);
lean_ctor_set(v___x_630_, 1, v_a_571_);
return v___x_630_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___boxed(lean_object* v_x_631_, lean_object* v_a_632_, lean_object* v_a_633_){
_start:
{
lean_object* v_res_634_; 
v_res_634_ = lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1(v_x_631_, v_a_632_, v_a_633_);
lean_dec_ref(v_a_632_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1__1(lean_object* v_x_653_, lean_object* v_a_654_, lean_object* v_a_655_){
_start:
{
lean_object* v___x_656_; uint8_t v___x_657_; 
v___x_656_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1));
lean_inc(v_x_653_);
v___x_657_ = l_Lean_Syntax_isOfKind(v_x_653_, v___x_656_);
if (v___x_657_ == 0)
{
lean_object* v___x_658_; lean_object* v___x_659_; 
lean_dec(v_x_653_);
v___x_658_ = lean_box(1);
v___x_659_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_659_, 0, v___x_658_);
lean_ctor_set(v___x_659_, 1, v_a_655_);
return v___x_659_;
}
else
{
lean_object* v_quotContext_660_; lean_object* v_currMacroScope_661_; lean_object* v_ref_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; uint8_t v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v_quotContext_660_ = lean_ctor_get(v_a_654_, 1);
v_currMacroScope_661_ = lean_ctor_get(v_a_654_, 2);
v_ref_662_ = lean_ctor_get(v_a_654_, 5);
v___x_663_ = lean_unsigned_to_nat(1u);
v___x_664_ = l_Lean_Syntax_getArg(v_x_653_, v___x_663_);
v___x_665_ = lean_unsigned_to_nat(3u);
v___x_666_ = l_Lean_Syntax_getArg(v_x_653_, v___x_665_);
lean_dec(v_x_653_);
v___x_667_ = 0;
v___x_668_ = l_Lean_SourceInfo_fromRef(v_ref_662_, v___x_667_);
v___x_669_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__3));
v___x_670_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_668_, 18);
v___x_671_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_671_, 0, v___x_668_);
lean_ctor_set(v___x_671_, 1, v___x_670_);
v___x_672_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__3));
v___x_673_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__6));
v___x_674_ = lean_obj_once(&lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1, &lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__1);
v___x_675_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1___closed__2));
lean_inc(v_currMacroScope_661_);
lean_inc(v_quotContext_660_);
v___x_676_ = l_Lean_addMacroScope(v_quotContext_660_, v___x_675_, v_currMacroScope_661_);
v___x_677_ = lean_box(0);
v___x_678_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_678_, 0, v___x_668_);
lean_ctor_set(v___x_678_, 1, v___x_674_);
lean_ctor_set(v___x_678_, 2, v___x_676_);
lean_ctor_set(v___x_678_, 3, v___x_677_);
lean_inc_ref(v___x_678_);
v___x_679_ = l_Lean_Syntax_node1(v___x_668_, v___x_673_, v___x_678_);
v___x_680_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_681_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28, &lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28_once, _init_lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__28);
v___x_682_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_682_, 0, v___x_668_);
lean_ctor_set(v___x_682_, 1, v___x_680_);
lean_ctor_set(v___x_682_, 2, v___x_681_);
lean_inc_ref_n(v___x_682_, 3);
v___x_683_ = l_Lean_Syntax_node2(v___x_668_, v___x_672_, v___x_679_, v___x_682_);
v___x_684_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_685_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_685_, 0, v___x_668_);
lean_ctor_set(v___x_685_, 1, v___x_684_);
v___x_686_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__0));
v___x_687_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1));
v___x_688_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_688_, 0, v___x_668_);
lean_ctor_set(v___x_688_, 1, v___x_686_);
v___x_689_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3));
v___x_690_ = l_Lean_Syntax_node2(v___x_668_, v___x_689_, v___x_682_, v___x_678_);
v___x_691_ = l_Lean_Syntax_node1(v___x_668_, v___x_680_, v___x_690_);
v___x_692_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__4));
v___x_693_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_693_, 0, v___x_668_);
lean_ctor_set(v___x_693_, 1, v___x_692_);
v___x_694_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6));
v___x_695_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8));
v___x_696_ = l_Lean_Syntax_node1(v___x_668_, v___x_680_, v___x_664_);
v___x_697_ = l_Lean_Syntax_node1(v___x_668_, v___x_680_, v___x_696_);
v___x_698_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__9));
v___x_699_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_699_, 0, v___x_668_);
lean_ctor_set(v___x_699_, 1, v___x_698_);
lean_inc_ref(v___x_685_);
v___x_700_ = l_Lean_Syntax_node4(v___x_668_, v___x_695_, v___x_685_, v___x_697_, v___x_699_, v___x_666_);
v___x_701_ = l_Lean_Syntax_node1(v___x_668_, v___x_680_, v___x_700_);
v___x_702_ = l_Lean_Syntax_node1(v___x_668_, v___x_694_, v___x_701_);
v___x_703_ = l_Lean_Syntax_node6(v___x_668_, v___x_687_, v___x_688_, v___x_682_, v___x_682_, v___x_691_, v___x_693_, v___x_702_);
v___x_704_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_705_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_668_);
lean_ctor_set(v___x_705_, 1, v___x_704_);
v___x_706_ = l_Lean_Syntax_node5(v___x_668_, v___x_669_, v___x_671_, v___x_683_, v___x_685_, v___x_703_, v___x_705_);
v___x_707_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
lean_ctor_set(v___x_707_, 1, v_a_655_);
return v___x_707_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1__1___boxed(lean_object* v_x_708_, lean_object* v_a_709_, lean_object* v_a_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__term_x7b___x7c___x7d__1__1(v_x_708_, v_a_709_, v_a_710_);
lean_dec_ref(v_a_709_);
return v_res_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPredPatternMatchUnexpander(lean_object* v_x_712_, lean_object* v_a_713_, lean_object* v_a_714_){
_start:
{
lean_object* v___x_715_; uint8_t v___x_716_; 
v___x_715_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
lean_inc(v_x_712_);
v___x_716_ = l_Lean_Syntax_isOfKind(v_x_712_, v___x_715_);
if (v___x_716_ == 0)
{
lean_object* v___x_717_; lean_object* v___x_718_; 
lean_dec(v_x_712_);
v___x_717_ = lean_box(0);
v___x_718_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_718_, 0, v___x_717_);
lean_ctor_set(v___x_718_, 1, v_a_714_);
return v___x_718_;
}
else
{
lean_object* v___x_719_; lean_object* v___x_720_; uint8_t v___x_721_; 
v___x_719_ = lean_unsigned_to_nat(1u);
v___x_720_ = l_Lean_Syntax_getArg(v_x_712_, v___x_719_);
lean_dec(v_x_712_);
lean_inc(v___x_720_);
v___x_721_ = l_Lean_Syntax_matchesNull(v___x_720_, v___x_719_);
if (v___x_721_ == 0)
{
lean_object* v___x_722_; lean_object* v___x_723_; 
lean_dec(v___x_720_);
v___x_722_ = lean_box(0);
v___x_723_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_722_);
lean_ctor_set(v___x_723_, 1, v_a_714_);
return v___x_723_;
}
else
{
lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; uint8_t v___x_727_; 
v___x_724_ = lean_unsigned_to_nat(0u);
v___x_725_ = l_Lean_Syntax_getArg(v___x_720_, v___x_724_);
lean_dec(v___x_720_);
v___x_726_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__25));
lean_inc(v___x_725_);
v___x_727_ = l_Lean_Syntax_isOfKind(v___x_725_, v___x_726_);
if (v___x_727_ == 0)
{
lean_object* v___x_728_; lean_object* v___x_729_; 
lean_dec(v___x_725_);
v___x_728_ = lean_box(0);
v___x_729_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_729_, 0, v___x_728_);
lean_ctor_set(v___x_729_, 1, v_a_714_);
return v___x_729_;
}
else
{
lean_object* v___x_730_; lean_object* v___x_731_; uint8_t v___x_732_; 
v___x_730_ = l_Lean_Syntax_getArg(v___x_725_, v___x_719_);
lean_dec(v___x_725_);
v___x_731_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__27));
lean_inc(v___x_730_);
v___x_732_ = l_Lean_Syntax_isOfKind(v___x_730_, v___x_731_);
if (v___x_732_ == 0)
{
lean_object* v___x_733_; lean_object* v___x_734_; 
lean_dec(v___x_730_);
v___x_733_ = lean_box(0);
v___x_734_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_733_);
lean_ctor_set(v___x_734_, 1, v_a_714_);
return v___x_734_;
}
else
{
lean_object* v___x_735_; uint8_t v___x_736_; 
v___x_735_ = l_Lean_Syntax_getArg(v___x_730_, v___x_724_);
lean_inc(v___x_735_);
v___x_736_ = l_Lean_Syntax_matchesNull(v___x_735_, v___x_719_);
if (v___x_736_ == 0)
{
lean_object* v___x_737_; lean_object* v___x_738_; 
lean_dec(v___x_735_);
lean_dec(v___x_730_);
v___x_737_ = lean_box(0);
v___x_738_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_738_, 0, v___x_737_);
lean_ctor_set(v___x_738_, 1, v_a_714_);
return v___x_738_;
}
else
{
lean_object* v___x_739_; lean_object* v___x_740_; uint8_t v___x_741_; 
v___x_739_ = l_Lean_Syntax_getArg(v___x_735_, v___x_724_);
lean_dec(v___x_735_);
v___x_740_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8));
lean_inc(v___x_739_);
v___x_741_ = l_Lean_Syntax_isOfKind(v___x_739_, v___x_740_);
if (v___x_741_ == 0)
{
lean_object* v___x_742_; uint8_t v___x_743_; 
v___x_742_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__1));
lean_inc(v___x_739_);
v___x_743_ = l_Lean_Syntax_isOfKind(v___x_739_, v___x_742_);
if (v___x_743_ == 0)
{
lean_object* v___x_744_; lean_object* v___x_745_; 
lean_dec(v___x_739_);
lean_dec(v___x_730_);
v___x_744_ = lean_box(0);
v___x_745_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
lean_ctor_set(v___x_745_, 1, v_a_714_);
return v___x_745_;
}
else
{
lean_object* v___x_746_; lean_object* v___x_747_; uint8_t v___x_748_; 
v___x_746_ = l_Lean_Syntax_getArg(v___x_739_, v___x_724_);
v___x_747_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__3));
lean_inc(v___x_746_);
v___x_748_ = l_Lean_Syntax_isOfKind(v___x_746_, v___x_747_);
if (v___x_748_ == 0)
{
lean_object* v___x_749_; lean_object* v___x_750_; 
lean_dec(v___x_746_);
lean_dec(v___x_739_);
lean_dec(v___x_730_);
v___x_749_ = lean_box(0);
v___x_750_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_750_, 0, v___x_749_);
lean_ctor_set(v___x_750_, 1, v_a_714_);
return v___x_750_;
}
else
{
lean_object* v___x_751_; lean_object* v___x_752_; uint8_t v___x_753_; 
v___x_751_ = l_Lean_Syntax_getArg(v___x_746_, v___x_719_);
lean_dec(v___x_746_);
v___x_752_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__5));
lean_inc(v___x_751_);
v___x_753_ = l_Lean_Syntax_isOfKind(v___x_751_, v___x_752_);
if (v___x_753_ == 0)
{
lean_object* v___x_754_; lean_object* v___x_755_; 
lean_dec(v___x_751_);
lean_dec(v___x_739_);
lean_dec(v___x_730_);
v___x_754_ = lean_box(0);
v___x_755_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_755_, 0, v___x_754_);
lean_ctor_set(v___x_755_, 1, v_a_714_);
return v___x_755_;
}
else
{
lean_object* v___x_756_; lean_object* v___x_757_; uint8_t v___x_758_; 
v___x_756_ = l_Lean_Syntax_getArg(v___x_751_, v___x_724_);
lean_dec(v___x_751_);
v___x_757_ = lean_box(0);
v___x_758_ = l_Lean_Syntax_matchesIdent(v___x_756_, v___x_757_);
lean_dec(v___x_756_);
if (v___x_758_ == 0)
{
lean_object* v___x_759_; lean_object* v___x_760_; 
lean_dec(v___x_739_);
lean_dec(v___x_730_);
v___x_759_ = lean_box(0);
v___x_760_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_760_, 0, v___x_759_);
lean_ctor_set(v___x_760_, 1, v_a_714_);
return v___x_760_;
}
else
{
lean_object* v___x_761_; uint8_t v___x_762_; 
v___x_761_ = l_Lean_Syntax_getArg(v___x_739_, v___x_719_);
lean_inc(v___x_761_);
v___x_762_ = l_Lean_Syntax_isOfKind(v___x_761_, v___x_740_);
if (v___x_762_ == 0)
{
lean_object* v___x_763_; lean_object* v___x_764_; 
lean_dec(v___x_761_);
lean_dec(v___x_739_);
lean_dec(v___x_730_);
v___x_763_ = lean_box(0);
v___x_764_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_764_, 0, v___x_763_);
lean_ctor_set(v___x_764_, 1, v_a_714_);
return v___x_764_;
}
else
{
lean_object* v___x_765_; lean_object* v___x_766_; uint8_t v___x_767_; 
v___x_765_ = lean_unsigned_to_nat(3u);
v___x_766_ = l_Lean_Syntax_getArg(v___x_739_, v___x_765_);
lean_dec(v___x_739_);
lean_inc(v___x_766_);
v___x_767_ = l_Lean_Syntax_matchesNull(v___x_766_, v___x_719_);
if (v___x_767_ == 0)
{
lean_object* v___x_768_; lean_object* v___x_769_; 
lean_dec(v___x_766_);
lean_dec(v___x_761_);
lean_dec(v___x_730_);
v___x_768_ = lean_box(0);
v___x_769_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_769_, 0, v___x_768_);
lean_ctor_set(v___x_769_, 1, v_a_714_);
return v___x_769_;
}
else
{
lean_object* v___x_770_; uint8_t v___x_771_; 
v___x_770_ = l_Lean_Syntax_getArg(v___x_730_, v___x_719_);
v___x_771_ = l_Lean_Syntax_matchesNull(v___x_770_, v___x_724_);
if (v___x_771_ == 0)
{
lean_object* v___x_772_; lean_object* v___x_773_; 
lean_dec(v___x_766_);
lean_dec(v___x_761_);
lean_dec(v___x_730_);
v___x_772_ = lean_box(0);
v___x_773_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_773_, 0, v___x_772_);
lean_ctor_set(v___x_773_, 1, v_a_714_);
return v___x_773_;
}
else
{
lean_object* v___x_774_; lean_object* v___x_775_; uint8_t v___x_776_; 
v___x_774_ = l_Lean_Syntax_getArg(v___x_730_, v___x_765_);
lean_dec(v___x_730_);
v___x_775_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1));
lean_inc(v___x_774_);
v___x_776_ = l_Lean_Syntax_isOfKind(v___x_774_, v___x_775_);
if (v___x_776_ == 0)
{
lean_object* v___x_777_; lean_object* v___x_778_; 
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_777_ = lean_box(0);
v___x_778_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_778_, 0, v___x_777_);
lean_ctor_set(v___x_778_, 1, v_a_714_);
return v___x_778_;
}
else
{
lean_object* v___x_779_; uint8_t v___x_780_; 
v___x_779_ = l_Lean_Syntax_getArg(v___x_774_, v___x_719_);
v___x_780_ = l_Lean_Syntax_matchesNull(v___x_779_, v___x_724_);
if (v___x_780_ == 0)
{
lean_object* v___x_781_; lean_object* v___x_782_; 
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_781_ = lean_box(0);
v___x_782_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_782_, 0, v___x_781_);
lean_ctor_set(v___x_782_, 1, v_a_714_);
return v___x_782_;
}
else
{
lean_object* v___x_783_; lean_object* v___x_784_; uint8_t v___x_785_; 
v___x_783_ = lean_unsigned_to_nat(2u);
v___x_784_ = l_Lean_Syntax_getArg(v___x_774_, v___x_783_);
v___x_785_ = l_Lean_Syntax_matchesNull(v___x_784_, v___x_724_);
if (v___x_785_ == 0)
{
lean_object* v___x_786_; lean_object* v___x_787_; 
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_786_ = lean_box(0);
v___x_787_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_787_, 0, v___x_786_);
lean_ctor_set(v___x_787_, 1, v_a_714_);
return v___x_787_;
}
else
{
lean_object* v___x_788_; uint8_t v___x_789_; 
v___x_788_ = l_Lean_Syntax_getArg(v___x_774_, v___x_765_);
lean_inc(v___x_788_);
v___x_789_ = l_Lean_Syntax_matchesNull(v___x_788_, v___x_719_);
if (v___x_789_ == 0)
{
lean_object* v___x_790_; lean_object* v___x_791_; 
lean_dec(v___x_788_);
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_790_ = lean_box(0);
v___x_791_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_791_, 0, v___x_790_);
lean_ctor_set(v___x_791_, 1, v_a_714_);
return v___x_791_;
}
else
{
lean_object* v___x_792_; lean_object* v___x_793_; uint8_t v___x_794_; 
v___x_792_ = l_Lean_Syntax_getArg(v___x_788_, v___x_724_);
lean_dec(v___x_788_);
v___x_793_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3));
lean_inc(v___x_792_);
v___x_794_ = l_Lean_Syntax_isOfKind(v___x_792_, v___x_793_);
if (v___x_794_ == 0)
{
lean_object* v___x_795_; lean_object* v___x_796_; 
lean_dec(v___x_792_);
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_795_ = lean_box(0);
v___x_796_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_796_, 0, v___x_795_);
lean_ctor_set(v___x_796_, 1, v_a_714_);
return v___x_796_;
}
else
{
lean_object* v___x_797_; uint8_t v___x_798_; 
v___x_797_ = l_Lean_Syntax_getArg(v___x_792_, v___x_724_);
v___x_798_ = l_Lean_Syntax_matchesNull(v___x_797_, v___x_724_);
if (v___x_798_ == 0)
{
lean_object* v___x_799_; lean_object* v___x_800_; 
lean_dec(v___x_792_);
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_799_ = lean_box(0);
v___x_800_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_800_, 0, v___x_799_);
lean_ctor_set(v___x_800_, 1, v_a_714_);
return v___x_800_;
}
else
{
lean_object* v___x_801_; uint8_t v___x_802_; 
v___x_801_ = l_Lean_Syntax_getArg(v___x_792_, v___x_719_);
lean_dec(v___x_792_);
lean_inc(v___x_801_);
v___x_802_ = l_Lean_Syntax_isOfKind(v___x_801_, v___x_740_);
if (v___x_802_ == 0)
{
lean_object* v___x_803_; lean_object* v___x_804_; 
lean_dec(v___x_801_);
lean_dec(v___x_774_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_803_ = lean_box(0);
v___x_804_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_804_, 0, v___x_803_);
lean_ctor_set(v___x_804_, 1, v_a_714_);
return v___x_804_;
}
else
{
lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; uint8_t v___x_808_; 
v___x_805_ = lean_unsigned_to_nat(5u);
v___x_806_ = l_Lean_Syntax_getArg(v___x_774_, v___x_805_);
lean_dec(v___x_774_);
v___x_807_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6));
lean_inc(v___x_806_);
v___x_808_ = l_Lean_Syntax_isOfKind(v___x_806_, v___x_807_);
if (v___x_808_ == 0)
{
lean_object* v___x_809_; lean_object* v___x_810_; 
lean_dec(v___x_806_);
lean_dec(v___x_801_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_809_ = lean_box(0);
v___x_810_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_810_, 0, v___x_809_);
lean_ctor_set(v___x_810_, 1, v_a_714_);
return v___x_810_;
}
else
{
lean_object* v___x_811_; uint8_t v___x_812_; 
v___x_811_ = l_Lean_Syntax_getArg(v___x_806_, v___x_724_);
lean_dec(v___x_806_);
lean_inc(v___x_811_);
v___x_812_ = l_Lean_Syntax_matchesNull(v___x_811_, v___x_719_);
if (v___x_812_ == 0)
{
lean_object* v___x_813_; lean_object* v___x_814_; 
lean_dec(v___x_811_);
lean_dec(v___x_801_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_813_ = lean_box(0);
v___x_814_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_814_, 0, v___x_813_);
lean_ctor_set(v___x_814_, 1, v_a_714_);
return v___x_814_;
}
else
{
lean_object* v___x_815_; lean_object* v___x_816_; uint8_t v___x_817_; 
v___x_815_ = l_Lean_Syntax_getArg(v___x_811_, v___x_724_);
lean_dec(v___x_811_);
v___x_816_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8));
lean_inc(v___x_815_);
v___x_817_ = l_Lean_Syntax_isOfKind(v___x_815_, v___x_816_);
if (v___x_817_ == 0)
{
lean_object* v___x_818_; lean_object* v___x_819_; 
lean_dec(v___x_815_);
lean_dec(v___x_801_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_818_ = lean_box(0);
v___x_819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
lean_ctor_set(v___x_819_, 1, v_a_714_);
return v___x_819_;
}
else
{
lean_object* v___x_820_; uint8_t v___x_821_; 
v___x_820_ = l_Lean_Syntax_getArg(v___x_815_, v___x_719_);
lean_inc(v___x_820_);
v___x_821_ = l_Lean_Syntax_matchesNull(v___x_820_, v___x_719_);
if (v___x_821_ == 0)
{
lean_object* v___x_822_; lean_object* v___x_823_; 
lean_dec(v___x_820_);
lean_dec(v___x_815_);
lean_dec(v___x_801_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_822_ = lean_box(0);
v___x_823_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_823_, 0, v___x_822_);
lean_ctor_set(v___x_823_, 1, v_a_714_);
return v___x_823_;
}
else
{
lean_object* v___x_824_; uint8_t v___x_825_; 
v___x_824_ = l_Lean_Syntax_getArg(v___x_820_, v___x_724_);
lean_dec(v___x_820_);
lean_inc(v___x_824_);
v___x_825_ = l_Lean_Syntax_matchesNull(v___x_824_, v___x_719_);
if (v___x_825_ == 0)
{
lean_object* v___x_826_; lean_object* v___x_827_; 
lean_dec(v___x_824_);
lean_dec(v___x_815_);
lean_dec(v___x_801_);
lean_dec(v___x_766_);
lean_dec(v___x_761_);
v___x_826_ = lean_box(0);
v___x_827_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_827_, 0, v___x_826_);
lean_ctor_set(v___x_827_, 1, v_a_714_);
return v___x_827_;
}
else
{
uint8_t v___x_828_; 
v___x_828_ = l_Lean_Syntax_structEq(v___x_761_, v___x_801_);
lean_dec(v___x_801_);
lean_dec(v___x_761_);
if (v___x_828_ == 0)
{
lean_object* v___x_829_; lean_object* v___x_830_; 
lean_dec(v___x_824_);
lean_dec(v___x_815_);
lean_dec(v___x_766_);
v___x_829_ = lean_box(0);
v___x_830_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_830_, 0, v___x_829_);
lean_ctor_set(v___x_830_, 1, v_a_714_);
return v___x_830_;
}
else
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; 
v___x_831_ = l_Lean_Syntax_getArg(v___x_766_, v___x_724_);
lean_dec(v___x_766_);
v___x_832_ = l_Lean_Syntax_getArg(v___x_824_, v___x_724_);
lean_dec(v___x_824_);
v___x_833_ = l_Lean_Syntax_getArg(v___x_815_, v___x_765_);
lean_dec(v___x_815_);
v___x_834_ = l_Lean_SourceInfo_fromRef(v_a_713_, v___x_741_);
v___x_835_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_macroPattSetBuilder___closed__1));
v___x_836_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_834_, 4);
v___x_837_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_837_, 0, v___x_834_);
lean_ctor_set(v___x_837_, 1, v___x_836_);
v___x_838_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__38));
v___x_839_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_839_, 0, v___x_834_);
lean_ctor_set(v___x_839_, 1, v___x_838_);
v___x_840_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_841_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_841_, 0, v___x_834_);
lean_ctor_set(v___x_841_, 1, v___x_840_);
v___x_842_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_843_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_843_, 0, v___x_834_);
lean_ctor_set(v___x_843_, 1, v___x_842_);
v___x_844_ = l_Lean_Syntax_node7(v___x_834_, v___x_835_, v___x_837_, v___x_832_, v___x_839_, v___x_831_, v___x_841_, v___x_833_, v___x_843_);
v___x_845_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_845_, 0, v___x_844_);
lean_ctor_set(v___x_845_, 1, v_a_714_);
return v___x_845_;
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
lean_object* v___x_846_; uint8_t v___x_847_; 
v___x_846_ = l_Lean_Syntax_getArg(v___x_730_, v___x_719_);
v___x_847_ = l_Lean_Syntax_matchesNull(v___x_846_, v___x_724_);
if (v___x_847_ == 0)
{
lean_object* v___x_848_; lean_object* v___x_849_; 
lean_dec(v___x_739_);
lean_dec(v___x_730_);
v___x_848_ = lean_box(0);
v___x_849_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
lean_ctor_set(v___x_849_, 1, v_a_714_);
return v___x_849_;
}
else
{
lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; uint8_t v___x_853_; 
v___x_850_ = lean_unsigned_to_nat(3u);
v___x_851_ = l_Lean_Syntax_getArg(v___x_730_, v___x_850_);
lean_dec(v___x_730_);
v___x_852_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__1));
lean_inc(v___x_851_);
v___x_853_ = l_Lean_Syntax_isOfKind(v___x_851_, v___x_852_);
if (v___x_853_ == 0)
{
lean_object* v___x_854_; lean_object* v___x_855_; 
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_854_ = lean_box(0);
v___x_855_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_855_, 0, v___x_854_);
lean_ctor_set(v___x_855_, 1, v_a_714_);
return v___x_855_;
}
else
{
lean_object* v___x_856_; uint8_t v___x_857_; 
v___x_856_ = l_Lean_Syntax_getArg(v___x_851_, v___x_719_);
v___x_857_ = l_Lean_Syntax_matchesNull(v___x_856_, v___x_724_);
if (v___x_857_ == 0)
{
lean_object* v___x_858_; lean_object* v___x_859_; 
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_858_ = lean_box(0);
v___x_859_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_859_, 0, v___x_858_);
lean_ctor_set(v___x_859_, 1, v_a_714_);
return v___x_859_;
}
else
{
lean_object* v___x_860_; lean_object* v___x_861_; uint8_t v___x_862_; 
v___x_860_ = lean_unsigned_to_nat(2u);
v___x_861_ = l_Lean_Syntax_getArg(v___x_851_, v___x_860_);
v___x_862_ = l_Lean_Syntax_matchesNull(v___x_861_, v___x_724_);
if (v___x_862_ == 0)
{
lean_object* v___x_863_; lean_object* v___x_864_; 
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_863_ = lean_box(0);
v___x_864_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_864_, 0, v___x_863_);
lean_ctor_set(v___x_864_, 1, v_a_714_);
return v___x_864_;
}
else
{
lean_object* v___x_865_; uint8_t v___x_866_; 
v___x_865_ = l_Lean_Syntax_getArg(v___x_851_, v___x_850_);
lean_inc(v___x_865_);
v___x_866_ = l_Lean_Syntax_matchesNull(v___x_865_, v___x_719_);
if (v___x_866_ == 0)
{
lean_object* v___x_867_; lean_object* v___x_868_; 
lean_dec(v___x_865_);
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_867_ = lean_box(0);
v___x_868_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_868_, 0, v___x_867_);
lean_ctor_set(v___x_868_, 1, v_a_714_);
return v___x_868_;
}
else
{
lean_object* v___x_869_; lean_object* v___x_870_; uint8_t v___x_871_; 
v___x_869_ = l_Lean_Syntax_getArg(v___x_865_, v___x_724_);
lean_dec(v___x_865_);
v___x_870_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__3));
lean_inc(v___x_869_);
v___x_871_ = l_Lean_Syntax_isOfKind(v___x_869_, v___x_870_);
if (v___x_871_ == 0)
{
lean_object* v___x_872_; lean_object* v___x_873_; 
lean_dec(v___x_869_);
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_872_ = lean_box(0);
v___x_873_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_873_, 0, v___x_872_);
lean_ctor_set(v___x_873_, 1, v_a_714_);
return v___x_873_;
}
else
{
lean_object* v___x_874_; uint8_t v___x_875_; 
v___x_874_ = l_Lean_Syntax_getArg(v___x_869_, v___x_724_);
v___x_875_ = l_Lean_Syntax_matchesNull(v___x_874_, v___x_724_);
if (v___x_875_ == 0)
{
lean_object* v___x_876_; lean_object* v___x_877_; 
lean_dec(v___x_869_);
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_876_ = lean_box(0);
v___x_877_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_877_, 0, v___x_876_);
lean_ctor_set(v___x_877_, 1, v_a_714_);
return v___x_877_;
}
else
{
lean_object* v___x_878_; uint8_t v___x_879_; 
v___x_878_ = l_Lean_Syntax_getArg(v___x_869_, v___x_719_);
lean_dec(v___x_869_);
lean_inc(v___x_878_);
v___x_879_ = l_Lean_Syntax_isOfKind(v___x_878_, v___x_740_);
if (v___x_879_ == 0)
{
lean_object* v___x_880_; lean_object* v___x_881_; 
lean_dec(v___x_878_);
lean_dec(v___x_851_);
lean_dec(v___x_739_);
v___x_880_ = lean_box(0);
v___x_881_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_881_, 0, v___x_880_);
lean_ctor_set(v___x_881_, 1, v_a_714_);
return v___x_881_;
}
else
{
lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; uint8_t v___x_885_; 
v___x_882_ = lean_unsigned_to_nat(5u);
v___x_883_ = l_Lean_Syntax_getArg(v___x_851_, v___x_882_);
lean_dec(v___x_851_);
v___x_884_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__6));
lean_inc(v___x_883_);
v___x_885_ = l_Lean_Syntax_isOfKind(v___x_883_, v___x_884_);
if (v___x_885_ == 0)
{
lean_object* v___x_886_; lean_object* v___x_887_; 
lean_dec(v___x_883_);
lean_dec(v___x_878_);
lean_dec(v___x_739_);
v___x_886_ = lean_box(0);
v___x_887_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_887_, 0, v___x_886_);
lean_ctor_set(v___x_887_, 1, v_a_714_);
return v___x_887_;
}
else
{
lean_object* v___x_888_; uint8_t v___x_889_; 
v___x_888_ = l_Lean_Syntax_getArg(v___x_883_, v___x_724_);
lean_dec(v___x_883_);
lean_inc(v___x_888_);
v___x_889_ = l_Lean_Syntax_matchesNull(v___x_888_, v___x_719_);
if (v___x_889_ == 0)
{
lean_object* v___x_890_; lean_object* v___x_891_; 
lean_dec(v___x_888_);
lean_dec(v___x_878_);
lean_dec(v___x_739_);
v___x_890_ = lean_box(0);
v___x_891_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_891_, 0, v___x_890_);
lean_ctor_set(v___x_891_, 1, v_a_714_);
return v___x_891_;
}
else
{
lean_object* v___x_892_; lean_object* v___x_893_; uint8_t v___x_894_; 
v___x_892_ = l_Lean_Syntax_getArg(v___x_888_, v___x_724_);
lean_dec(v___x_888_);
v___x_893_ = ((lean_object*)(lp_mathlib_Mathlib_Meta___aux__Mathlib__Data__Set__Defs______macroRules__Mathlib__Meta__macroPattSetBuilder__1___closed__8));
lean_inc(v___x_892_);
v___x_894_ = l_Lean_Syntax_isOfKind(v___x_892_, v___x_893_);
if (v___x_894_ == 0)
{
lean_object* v___x_895_; lean_object* v___x_896_; 
lean_dec(v___x_892_);
lean_dec(v___x_878_);
lean_dec(v___x_739_);
v___x_895_ = lean_box(0);
v___x_896_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_896_, 0, v___x_895_);
lean_ctor_set(v___x_896_, 1, v_a_714_);
return v___x_896_;
}
else
{
lean_object* v___x_897_; uint8_t v___x_898_; 
v___x_897_ = l_Lean_Syntax_getArg(v___x_892_, v___x_719_);
lean_inc(v___x_897_);
v___x_898_ = l_Lean_Syntax_matchesNull(v___x_897_, v___x_719_);
if (v___x_898_ == 0)
{
lean_object* v___x_899_; lean_object* v___x_900_; 
lean_dec(v___x_897_);
lean_dec(v___x_892_);
lean_dec(v___x_878_);
lean_dec(v___x_739_);
v___x_899_ = lean_box(0);
v___x_900_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_900_, 0, v___x_899_);
lean_ctor_set(v___x_900_, 1, v_a_714_);
return v___x_900_;
}
else
{
lean_object* v___x_901_; uint8_t v___x_902_; 
v___x_901_ = l_Lean_Syntax_getArg(v___x_897_, v___x_724_);
lean_dec(v___x_897_);
lean_inc(v___x_901_);
v___x_902_ = l_Lean_Syntax_matchesNull(v___x_901_, v___x_719_);
if (v___x_902_ == 0)
{
lean_object* v___x_903_; lean_object* v___x_904_; 
lean_dec(v___x_901_);
lean_dec(v___x_892_);
lean_dec(v___x_878_);
lean_dec(v___x_739_);
v___x_903_ = lean_box(0);
v___x_904_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_904_, 0, v___x_903_);
lean_ctor_set(v___x_904_, 1, v_a_714_);
return v___x_904_;
}
else
{
uint8_t v___x_905_; 
v___x_905_ = l_Lean_Syntax_structEq(v___x_739_, v___x_878_);
lean_dec(v___x_878_);
lean_dec(v___x_739_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; lean_object* v___x_907_; 
lean_dec(v___x_901_);
lean_dec(v___x_892_);
v___x_906_ = lean_box(0);
v___x_907_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_907_, 0, v___x_906_);
lean_ctor_set(v___x_907_, 1, v_a_714_);
return v___x_907_;
}
else
{
lean_object* v___x_908_; lean_object* v___x_909_; uint8_t v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; 
v___x_908_ = l_Lean_Syntax_getArg(v___x_901_, v___x_724_);
lean_dec(v___x_901_);
v___x_909_ = l_Lean_Syntax_getArg(v___x_892_, v___x_850_);
lean_dec(v___x_892_);
v___x_910_ = 0;
v___x_911_ = l_Lean_SourceInfo_fromRef(v_a_713_, v___x_910_);
v___x_912_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d__1___closed__1));
v___x_913_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__6));
lean_inc_n(v___x_911_, 3);
v___x_914_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_914_, 0, v___x_911_);
lean_ctor_set(v___x_914_, 1, v___x_913_);
v___x_915_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_ofPred_unexpander___closed__6));
v___x_916_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_916_, 0, v___x_911_);
lean_ctor_set(v___x_916_, 1, v___x_915_);
v___x_917_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_setBuilder___closed__16));
v___x_918_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_918_, 0, v___x_911_);
lean_ctor_set(v___x_918_, 1, v___x_917_);
v___x_919_ = l_Lean_Syntax_node5(v___x_911_, v___x_912_, v___x_914_, v___x_908_, v___x_916_, v___x_909_, v___x_918_);
v___x_920_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_920_, 0, v___x_919_);
lean_ctor_set(v___x_920_, 1, v_a_714_);
return v___x_920_;
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_ofPredPatternMatchUnexpander___boxed(lean_object* v_x_921_, lean_object* v_a_922_, lean_object* v_a_923_){
_start:
{
lean_object* v_res_924_; 
v_res_924_ = lp_mathlib_Mathlib_Meta_ofPredPatternMatchUnexpander(v_x_921_, v_a_922_, v_a_923_);
lean_dec(v_a_922_);
return v_res_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instInsert(lean_object* v_00_u03b1_925_){
_start:
{
lean_object* v___x_926_; 
v___x_926_ = lean_box(0);
return v___x_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instSingletonSet(lean_object* v_00_u03b1_927_){
_start:
{
lean_object* v___x_928_; 
v___x_928_ = lean_box(0);
return v___x_928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instUnion(lean_object* v_00_u03b1_929_){
_start:
{
lean_object* v___x_930_; 
v___x_930_ = lean_box(0);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instInter(lean_object* v_00_u03b1_931_){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = lean_box(0);
return v___x_932_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instSDiff(lean_object* v_00_u03b1_933_){
_start:
{
lean_object* v___x_934_; 
v___x_934_ = lean_box(0);
return v___x_934_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__1(void){
_start:
{
lean_object* v___x_955_; lean_object* v___x_956_; 
v___x_955_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__0));
v___x_956_ = l_String_toRawSubstring_x27(v___x_955_);
return v___x_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1(lean_object* v_x_968_, lean_object* v_a_969_, lean_object* v_a_970_){
_start:
{
lean_object* v___x_971_; uint8_t v___x_972_; 
v___x_971_ = ((lean_object*)(lp_mathlib_Set_term_U0001d4ab___00__closed__1));
lean_inc(v_x_968_);
v___x_972_ = l_Lean_Syntax_isOfKind(v_x_968_, v___x_971_);
if (v___x_972_ == 0)
{
lean_object* v___x_973_; lean_object* v___x_974_; 
lean_dec(v_x_968_);
v___x_973_ = lean_box(1);
v___x_974_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_974_, 0, v___x_973_);
lean_ctor_set(v___x_974_, 1, v_a_970_);
return v___x_974_;
}
else
{
lean_object* v_quotContext_975_; lean_object* v_currMacroScope_976_; lean_object* v_ref_977_; lean_object* v___x_978_; lean_object* v___x_979_; uint8_t v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; 
v_quotContext_975_ = lean_ctor_get(v_a_969_, 1);
v_currMacroScope_976_ = lean_ctor_get(v_a_969_, 2);
v_ref_977_ = lean_ctor_get(v_a_969_, 5);
v___x_978_ = lean_unsigned_to_nat(1u);
v___x_979_ = l_Lean_Syntax_getArg(v_x_968_, v___x_978_);
lean_dec(v_x_968_);
v___x_980_ = 0;
v___x_981_ = l_Lean_SourceInfo_fromRef(v_ref_977_, v___x_980_);
v___x_982_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
v___x_983_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__1, &lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__1_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__1);
v___x_984_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__2));
lean_inc(v_currMacroScope_976_);
lean_inc(v_quotContext_975_);
v___x_985_ = l_Lean_addMacroScope(v_quotContext_975_, v___x_984_, v_currMacroScope_976_);
v___x_986_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___closed__5));
lean_inc_n(v___x_981_, 2);
v___x_987_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_987_, 0, v___x_981_);
lean_ctor_set(v___x_987_, 1, v___x_983_);
lean_ctor_set(v___x_987_, 2, v___x_985_);
lean_ctor_set(v___x_987_, 3, v___x_986_);
v___x_988_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__23));
v___x_989_ = l_Lean_Syntax_node1(v___x_981_, v___x_988_, v___x_979_);
v___x_990_ = l_Lean_Syntax_node2(v___x_981_, v___x_982_, v___x_987_, v___x_989_);
v___x_991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_991_, 0, v___x_990_);
lean_ctor_set(v___x_991_, 1, v_a_970_);
return v___x_991_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1___boxed(lean_object* v_x_992_, lean_object* v_a_993_, lean_object* v_a_994_){
_start:
{
lean_object* v_res_995_; 
v_res_995_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______macroRules__Set__term_U0001d4ab____1(v_x_992_, v_a_993_, v_a_994_);
lean_dec_ref(v_a_993_);
return v_res_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______unexpand__Set__powerset__1(lean_object* v_x_996_, lean_object* v_a_997_, lean_object* v_a_998_){
_start:
{
lean_object* v___x_999_; uint8_t v___x_1000_; 
v___x_999_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__14));
lean_inc(v_x_996_);
v___x_1000_ = l_Lean_Syntax_isOfKind(v_x_996_, v___x_999_);
if (v___x_1000_ == 0)
{
lean_object* v___x_1001_; lean_object* v___x_1002_; 
lean_dec(v_x_996_);
v___x_1001_ = lean_box(0);
v___x_1002_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1001_);
lean_ctor_set(v___x_1002_, 1, v_a_998_);
return v___x_1002_;
}
else
{
lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; uint8_t v___x_1006_; 
v___x_1003_ = lean_unsigned_to_nat(0u);
v___x_1004_ = l_Lean_Syntax_getArg(v_x_996_, v___x_1003_);
v___x_1005_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabSetBuilder___closed__8));
lean_inc(v___x_1004_);
v___x_1006_ = l_Lean_Syntax_isOfKind(v___x_1004_, v___x_1005_);
if (v___x_1006_ == 0)
{
lean_object* v___x_1007_; lean_object* v___x_1008_; 
lean_dec(v___x_1004_);
lean_dec(v_x_996_);
v___x_1007_ = lean_box(0);
v___x_1008_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1008_, 0, v___x_1007_);
lean_ctor_set(v___x_1008_, 1, v_a_998_);
return v___x_1008_;
}
else
{
lean_object* v___x_1009_; lean_object* v___x_1010_; uint8_t v___x_1011_; 
v___x_1009_ = lean_unsigned_to_nat(1u);
v___x_1010_ = l_Lean_Syntax_getArg(v_x_996_, v___x_1009_);
lean_dec(v_x_996_);
lean_inc(v___x_1010_);
v___x_1011_ = l_Lean_Syntax_matchesNull(v___x_1010_, v___x_1009_);
if (v___x_1011_ == 0)
{
lean_object* v___x_1012_; lean_object* v___x_1013_; 
lean_dec(v___x_1010_);
lean_dec(v___x_1004_);
v___x_1012_ = lean_box(0);
v___x_1013_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1013_, 0, v___x_1012_);
lean_ctor_set(v___x_1013_, 1, v_a_998_);
return v___x_1013_;
}
else
{
lean_object* v___x_1014_; lean_object* v_ref_1015_; uint8_t v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; 
v___x_1014_ = l_Lean_Syntax_getArg(v___x_1010_, v___x_1003_);
lean_dec(v___x_1010_);
v_ref_1015_ = l_Lean_replaceRef(v___x_1004_, v_a_997_);
lean_dec(v___x_1004_);
v___x_1016_ = 0;
v___x_1017_ = l_Lean_SourceInfo_fromRef(v_ref_1015_, v___x_1016_);
lean_dec(v_ref_1015_);
v___x_1018_ = ((lean_object*)(lp_mathlib_Set_term_U0001d4ab___00__closed__1));
v___x_1019_ = ((lean_object*)(lp_mathlib_Set_term_U0001d4ab___00__closed__2));
lean_inc(v___x_1017_);
v___x_1020_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1020_, 0, v___x_1017_);
lean_ctor_set(v___x_1020_, 1, v___x_1019_);
v___x_1021_ = l_Lean_Syntax_node2(v___x_1017_, v___x_1018_, v___x_1020_, v___x_1014_);
v___x_1022_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1022_, 0, v___x_1021_);
lean_ctor_set(v___x_1022_, 1, v_a_998_);
return v___x_1022_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______unexpand__Set__powerset__1___boxed(lean_object* v_x_1023_, lean_object* v_a_1024_, lean_object* v_a_1025_){
_start:
{
lean_object* v_res_1026_; 
v_res_1026_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Defs______unexpand__Set__powerset__1(v_x_1023_, v_a_1024_, v_a_1025_);
lean_dec(v_a_1024_);
return v_res_1026_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_ExtendedBinder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SetNotationForOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_ExtendedBinder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SetNotationForOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Meta_setBuilder = _init_lp_mathlib_Mathlib_Meta_setBuilder();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_setBuilder);
lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d = _init_lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_term_x7b___x7c___x7d);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_ExtendedBinder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SetNotationForOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_ExtendedBinder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SetNotationForOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
