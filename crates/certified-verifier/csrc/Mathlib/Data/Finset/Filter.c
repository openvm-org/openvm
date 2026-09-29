// Lean compiler output
// Module: Mathlib.Data.Finset.Filter
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Empty public import Mathlib.Data.Multiset.Filter
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "setBuilder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__2_value),LEAN_SCALAR_PTR_LITERAL(55, 252, 174, 2, 80, 49, 173, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__6_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∈_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__13_value),LEAN_SCALAR_PTR_LITERAL(150, 164, 254, 63, 76, 57, 126, 92)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__17_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Finset.filter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "filter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__2_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__21_value),LEAN_SCALAR_PTR_LITERAL(88, 243, 224, 152, 142, 113, 169, 220)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__25_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__27_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__29_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__32_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__34_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__35;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__38_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__42_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__43_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__42_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__45_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__42_value),LEAN_SCALAR_PTR_LITERAL(228, 185, 96, 51, 222, 54, 124, 240)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__47_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__47_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__49_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__50_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__51_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__52_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__53_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subtype"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__54_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__55_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__56_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Multiset"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__57_value),LEAN_SCALAR_PTR_LITERAL(23, 131, 115, 119, 79, 192, 198, 77)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__58_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__59_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__56_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__60_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__53_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__50_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__62_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__48_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__63_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__64_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__46_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__64_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__65_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__44_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__65_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__37_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__66_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__41_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__67_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__39_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__68_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__69_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__37_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__69_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__70_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__71_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__71_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__15_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__16_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__73_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__75;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↦"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__76_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__77_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter___redArg(lean_object* v_inst_1_, lean_object* v_s_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_Multiset_filter___redArg(v_inst_1_, v_s_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filter(lean_object* v_00_u03b1_4_, lean_object* v_p_5_, lean_object* v_inst_6_, lean_object* v_s_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Multiset_filter___redArg(v_inst_6_, v_s_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_box(0);
v___x_10_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_11_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___x_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg(){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___closed__0);
v___x_14_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg___boxed(lean_object* v___y_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0(lean_object* v_00_u03b1_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___boxed(lean_object* v_00_u03b1_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0(v_00_u03b1_26_, v___y_27_, v___y_28_, v___y_29_, v___y_30_, v___y_31_, v___y_32_);
lean_dec(v___y_32_);
lean_dec_ref(v___y_31_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
lean_dec(v___y_28_);
lean_dec_ref(v___y_27_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(lean_object* v_expectedType_x3f_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_, lean_object* v_a_46_, lean_object* v_a_47_){
_start:
{
if (lean_obj_tag(v_expectedType_x3f_41_) == 0)
{
uint8_t v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = 0;
v___x_54_ = lean_box(v___x_53_);
v___x_55_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_55_, 0, v___x_54_);
return v___x_55_;
}
else
{
lean_object* v_val_56_; lean_object* v___x_58_; uint8_t v_isShared_59_; uint8_t v_isSharedCheck_72_; 
v_val_56_ = lean_ctor_get(v_expectedType_x3f_41_, 0);
v_isSharedCheck_72_ = !lean_is_exclusive(v_expectedType_x3f_41_);
if (v_isSharedCheck_72_ == 0)
{
v___x_58_ = v_expectedType_x3f_41_;
v_isShared_59_ = v_isSharedCheck_72_;
goto v_resetjp_57_;
}
else
{
lean_inc(v_val_56_);
lean_dec(v_expectedType_x3f_41_);
v___x_58_ = lean_box(0);
v_isShared_59_ = v_isSharedCheck_72_;
goto v_resetjp_57_;
}
v_resetjp_57_:
{
lean_object* v___x_60_; uint8_t v___x_61_; 
v___x_60_ = l_Lean_Expr_cleanupAnnotations(v_val_56_);
v___x_61_ = l_Lean_Expr_isApp(v___x_60_);
if (v___x_61_ == 0)
{
lean_dec_ref(v___x_60_);
lean_del_object(v___x_58_);
goto v___jp_49_;
}
else
{
lean_object* v___x_62_; lean_object* v___x_63_; uint8_t v___x_64_; 
v___x_62_ = l_Lean_Expr_appFnCleanup___redArg(v___x_60_);
v___x_63_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__1));
v___x_64_ = l_Lean_Expr_isConstOf(v___x_62_, v___x_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__3));
v___x_66_ = l_Lean_Expr_isConstOf(v___x_62_, v___x_65_);
lean_dec_ref(v___x_62_);
if (v___x_66_ == 0)
{
lean_del_object(v___x_58_);
goto v___jp_49_;
}
else
{
lean_object* v___x_67_; lean_object* v___x_69_; 
v___x_67_ = lean_box(v___x_66_);
if (v_isShared_59_ == 0)
{
lean_ctor_set_tag(v___x_58_, 0);
lean_ctor_set(v___x_58_, 0, v___x_67_);
v___x_69_ = v___x_58_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v___x_67_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
return v___x_69_;
}
}
}
else
{
lean_object* v___x_71_; 
lean_dec_ref(v___x_62_);
lean_del_object(v___x_58_);
v___x_71_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_71_;
}
}
}
}
v___jp_49_:
{
uint8_t v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_50_ = 0;
v___x_51_ = lean_box(v___x_50_);
v___x_52_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
return v___x_52_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___boxed(lean_object* v_expectedType_x3f_73_, lean_object* v_a_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_, lean_object* v_a_78_, lean_object* v_a_79_, lean_object* v_a_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_expectedType_x3f_73_, v_a_74_, v_a_75_, v_a_76_, v_a_77_, v_a_78_, v_a_79_);
lean_dec(v_a_79_);
lean_dec_ref(v_a_78_);
lean_dec(v_a_77_);
lean_dec_ref(v_a_76_);
lean_dec(v_a_75_);
lean_dec_ref(v_a_74_);
return v_res_81_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__20(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__19));
v___x_118_ = l_String_toRawSubstring_x27(v___x_117_);
return v___x_118_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__35(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__34));
v___x_150_ = l_String_toRawSubstring_x27(v___x_149_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__75(void){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = l_Array_mkArray0(lean_box(0));
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep(lean_object* v_x_250_, lean_object* v_x_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_, lean_object* v_a_255_, lean_object* v_a_256_, lean_object* v_a_257_){
_start:
{
lean_object* v___x_259_; uint8_t v___x_260_; 
v___x_259_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__3));
lean_inc(v_x_250_);
v___x_260_ = l_Lean_Syntax_isOfKind(v_x_250_, v___x_259_);
if (v___x_260_ == 0)
{
lean_object* v___x_261_; 
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v___x_261_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_261_;
}
else
{
lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; uint8_t v___x_265_; 
v___x_262_ = lean_unsigned_to_nat(1u);
v___x_263_ = l_Lean_Syntax_getArg(v_x_250_, v___x_262_);
v___x_264_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__7));
lean_inc(v___x_263_);
v___x_265_ = l_Lean_Syntax_isOfKind(v___x_263_, v___x_264_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; 
lean_dec(v___x_263_);
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v___x_266_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_266_;
}
else
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; uint8_t v___x_270_; 
v___x_267_ = lean_unsigned_to_nat(0u);
v___x_268_ = l_Lean_Syntax_getArg(v___x_263_, v___x_267_);
v___x_269_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__10));
lean_inc(v___x_268_);
v___x_270_ = l_Lean_Syntax_isOfKind(v___x_268_, v___x_269_);
if (v___x_270_ == 0)
{
lean_object* v___x_271_; 
lean_dec(v___x_268_);
lean_dec(v___x_263_);
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v___x_271_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_271_;
}
else
{
lean_object* v___x_272_; lean_object* v___x_273_; uint8_t v___x_274_; 
v___x_272_ = l_Lean_Syntax_getArg(v___x_268_, v___x_267_);
lean_dec(v___x_268_);
v___x_273_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__12));
lean_inc(v___x_272_);
v___x_274_ = l_Lean_Syntax_isOfKind(v___x_272_, v___x_273_);
if (v___x_274_ == 0)
{
lean_object* v___x_275_; 
lean_dec(v___x_272_);
lean_dec(v___x_263_);
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v___x_275_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_275_;
}
else
{
lean_object* v___x_276_; uint8_t v___x_277_; 
v___x_276_ = l_Lean_Syntax_getArg(v___x_263_, v___x_262_);
lean_dec(v___x_263_);
lean_inc(v___x_276_);
v___x_277_ = l_Lean_Syntax_matchesNull(v___x_276_, v___x_262_);
if (v___x_277_ == 0)
{
lean_object* v___x_278_; 
lean_dec(v___x_276_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v___x_278_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_278_;
}
else
{
lean_object* v___x_279_; lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_279_ = l_Lean_Syntax_getArg(v___x_276_, v___x_267_);
lean_dec(v___x_276_);
v___x_280_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__14));
lean_inc(v___x_279_);
v___x_281_ = l_Lean_Syntax_isOfKind(v___x_279_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; 
lean_dec(v___x_279_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v___x_282_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_282_;
}
else
{
lean_object* v___x_283_; 
lean_inc(v_x_251_);
v___x_283_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_251_, v_a_252_, v_a_253_, v_a_254_, v_a_255_, v_a_256_, v_a_257_);
if (lean_obj_tag(v___x_283_) == 0)
{
lean_object* v_a_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___y_289_; lean_object* v___y_290_; lean_object* v___y_291_; lean_object* v___y_292_; lean_object* v___y_293_; lean_object* v___y_294_; lean_object* v___y_337_; lean_object* v___y_338_; lean_object* v___y_339_; lean_object* v___y_340_; lean_object* v___y_341_; lean_object* v___y_342_; lean_object* v_a_353_; lean_object* v___y_362_; uint8_t v___y_363_; lean_object* v___y_366_; uint8_t v___x_371_; 
v_a_284_ = lean_ctor_get(v___x_283_, 0);
lean_inc(v_a_284_);
lean_dec_ref_known(v___x_283_, 1);
v___x_285_ = l_Lean_Syntax_getArg(v___x_279_, v___x_262_);
lean_dec(v___x_279_);
v___x_286_ = lean_unsigned_to_nat(3u);
v___x_287_ = l_Lean_Syntax_getArg(v_x_250_, v___x_286_);
lean_dec(v_x_250_);
v___x_371_ = lean_unbox(v_a_284_);
lean_dec(v_a_284_);
if (v___x_371_ == 0)
{
lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_372_ = lean_box(0);
lean_inc(v___x_285_);
v___x_373_ = l_Lean_Elab_Term_elabTerm(v___x_285_, v___x_372_, v___x_281_, v___x_281_, v_a_252_, v_a_253_, v_a_254_, v_a_255_, v_a_256_, v_a_257_);
if (lean_obj_tag(v___x_373_) == 0)
{
lean_object* v_a_374_; lean_object* v___x_375_; 
v_a_374_ = lean_ctor_get(v___x_373_, 0);
lean_inc(v_a_374_);
lean_dec_ref_known(v___x_373_, 1);
lean_inc(v_a_257_);
lean_inc_ref(v_a_256_);
lean_inc(v_a_255_);
lean_inc_ref(v_a_254_);
v___x_375_ = lean_infer_type(v_a_374_, v_a_254_, v_a_255_, v_a_256_, v_a_257_);
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v_a_376_; lean_object* v___x_377_; 
v_a_376_ = lean_ctor_get(v___x_375_, 0);
lean_inc(v_a_376_);
lean_dec_ref_known(v___x_375_, 1);
v___x_377_ = l_Lean_Meta_whnfR(v_a_376_, v_a_254_, v_a_255_, v_a_256_, v_a_257_);
v___y_366_ = v___x_377_;
goto v___jp_365_;
}
else
{
v___y_366_ = v___x_375_;
goto v___jp_365_;
}
}
else
{
v___y_366_ = v___x_373_;
goto v___jp_365_;
}
}
else
{
v___y_289_ = v_a_252_;
v___y_290_ = v_a_253_;
v___y_291_ = v_a_254_;
v___y_292_ = v_a_255_;
v___y_293_ = v_a_256_;
v___y_294_ = v_a_257_;
goto v___jp_288_;
}
v___jp_288_:
{
lean_object* v_ref_295_; lean_object* v_quotContext_296_; lean_object* v_currMacroScope_297_; uint8_t v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v_ref_295_ = lean_ctor_get(v___y_293_, 5);
v_quotContext_296_ = lean_ctor_get(v___y_293_, 10);
v_currMacroScope_297_ = lean_ctor_get(v___y_293_, 11);
v___x_298_ = 0;
v___x_299_ = l_Lean_SourceInfo_fromRef(v_ref_295_, v___x_298_);
v___x_300_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__18));
v___x_301_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__20, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__20_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__20);
v___x_302_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__22));
lean_inc_n(v_currMacroScope_297_, 2);
lean_inc_n(v_quotContext_296_, 2);
v___x_303_ = l_Lean_addMacroScope(v_quotContext_296_, v___x_302_, v_currMacroScope_297_);
v___x_304_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__24));
lean_inc_n(v___x_299_, 14);
v___x_305_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_305_, 0, v___x_299_);
lean_ctor_set(v___x_305_, 1, v___x_301_);
lean_ctor_set(v___x_305_, 2, v___x_303_);
lean_ctor_set(v___x_305_, 3, v___x_304_);
v___x_306_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__26));
v___x_307_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__28));
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__30));
v___x_309_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__31));
v___x_310_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_299_);
lean_ctor_set(v___x_310_, 1, v___x_309_);
v___x_311_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__33));
v___x_312_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__35, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__35_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__35);
v___x_313_ = lean_box(0);
v___x_314_ = l_Lean_addMacroScope(v_quotContext_296_, v___x_313_, v_currMacroScope_297_);
v___x_315_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__70));
v___x_316_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_316_, 0, v___x_299_);
lean_ctor_set(v___x_316_, 1, v___x_312_);
lean_ctor_set(v___x_316_, 2, v___x_314_);
lean_ctor_set(v___x_316_, 3, v___x_315_);
v___x_317_ = l_Lean_Syntax_node1(v___x_299_, v___x_311_, v___x_316_);
v___x_318_ = l_Lean_Syntax_node2(v___x_299_, v___x_308_, v___x_310_, v___x_317_);
v___x_319_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__71));
v___x_320_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__72));
v___x_321_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_299_);
lean_ctor_set(v___x_321_, 1, v___x_319_);
v___x_322_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__74));
v___x_323_ = l_Lean_Syntax_node1(v___x_299_, v___x_306_, v___x_272_);
v___x_324_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__75, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__75_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__75);
v___x_325_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_325_, 0, v___x_299_);
lean_ctor_set(v___x_325_, 1, v___x_306_);
lean_ctor_set(v___x_325_, 2, v___x_324_);
v___x_326_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__76));
v___x_327_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_327_, 0, v___x_299_);
lean_ctor_set(v___x_327_, 1, v___x_326_);
v___x_328_ = l_Lean_Syntax_node4(v___x_299_, v___x_322_, v___x_323_, v___x_325_, v___x_327_, v___x_287_);
v___x_329_ = l_Lean_Syntax_node2(v___x_299_, v___x_320_, v___x_321_, v___x_328_);
v___x_330_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___closed__77));
v___x_331_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_331_, 0, v___x_299_);
lean_ctor_set(v___x_331_, 1, v___x_330_);
v___x_332_ = l_Lean_Syntax_node3(v___x_299_, v___x_307_, v___x_318_, v___x_329_, v___x_331_);
v___x_333_ = l_Lean_Syntax_node2(v___x_299_, v___x_306_, v___x_332_, v___x_285_);
v___x_334_ = l_Lean_Syntax_node2(v___x_299_, v___x_300_, v___x_305_, v___x_333_);
v___x_335_ = l_Lean_Elab_Term_elabTerm(v___x_334_, v_x_251_, v___x_281_, v___x_281_, v___y_289_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_);
return v___x_335_;
}
v___jp_336_:
{
lean_object* v___x_343_; lean_object* v_a_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_351_; 
v___x_343_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
v_a_344_ = lean_ctor_get(v___x_343_, 0);
v_isSharedCheck_351_ = !lean_is_exclusive(v___x_343_);
if (v_isSharedCheck_351_ == 0)
{
v___x_346_ = v___x_343_;
v_isShared_347_ = v_isSharedCheck_351_;
goto v_resetjp_345_;
}
else
{
lean_inc(v_a_344_);
lean_dec(v___x_343_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_351_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_349_; 
if (v_isShared_347_ == 0)
{
v___x_349_ = v___x_346_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v_a_344_);
v___x_349_ = v_reuseFailAlloc_350_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
return v___x_349_;
}
}
}
v___jp_352_:
{
lean_object* v___x_354_; 
v___x_354_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_353_, v_a_255_);
if (lean_obj_tag(v___x_354_) == 0)
{
lean_object* v_a_355_; lean_object* v___x_356_; uint8_t v___x_357_; 
v_a_355_ = lean_ctor_get(v___x_354_, 0);
lean_inc(v_a_355_);
lean_dec_ref_known(v___x_354_, 1);
v___x_356_ = l_Lean_Expr_cleanupAnnotations(v_a_355_);
v___x_357_ = l_Lean_Expr_isApp(v___x_356_);
if (v___x_357_ == 0)
{
lean_dec_ref(v___x_356_);
lean_dec(v___x_287_);
lean_dec(v___x_285_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
v___y_337_ = v_a_252_;
v___y_338_ = v_a_253_;
v___y_339_ = v_a_254_;
v___y_340_ = v_a_255_;
v___y_341_ = v_a_256_;
v___y_342_ = v_a_257_;
goto v___jp_336_;
}
else
{
lean_object* v___x_358_; lean_object* v___x_359_; uint8_t v___x_360_; 
v___x_358_ = l_Lean_Expr_appFnCleanup___redArg(v___x_356_);
v___x_359_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet___closed__3));
v___x_360_ = l_Lean_Expr_isConstOf(v___x_358_, v___x_359_);
lean_dec_ref(v___x_358_);
if (v___x_360_ == 0)
{
lean_dec(v___x_287_);
lean_dec(v___x_285_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
v___y_337_ = v_a_252_;
v___y_338_ = v_a_253_;
v___y_339_ = v_a_254_;
v___y_340_ = v_a_255_;
v___y_341_ = v_a_256_;
v___y_342_ = v_a_257_;
goto v___jp_336_;
}
else
{
v___y_289_ = v_a_252_;
v___y_290_ = v_a_253_;
v___y_291_ = v_a_254_;
v___y_292_ = v_a_255_;
v___y_293_ = v_a_256_;
v___y_294_ = v_a_257_;
goto v___jp_288_;
}
}
}
else
{
lean_dec(v___x_287_);
lean_dec(v___x_285_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
return v___x_354_;
}
}
v___jp_361_:
{
if (v___y_363_ == 0)
{
lean_object* v___x_364_; 
lean_dec_ref(v___y_362_);
v___x_364_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_knownToBeFinsetNotSet_spec__0___redArg();
return v___x_364_;
}
else
{
return v___y_362_;
}
}
v___jp_365_:
{
if (lean_obj_tag(v___y_366_) == 0)
{
lean_object* v_a_367_; 
v_a_367_ = lean_ctor_get(v___y_366_, 0);
lean_inc(v_a_367_);
lean_dec_ref_known(v___y_366_, 1);
v_a_353_ = v_a_367_;
goto v___jp_352_;
}
else
{
lean_object* v_a_368_; uint8_t v___x_369_; 
lean_dec(v___x_287_);
lean_dec(v___x_285_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
v_a_368_ = lean_ctor_get(v___y_366_, 0);
v___x_369_ = l_Lean_Exception_isInterrupt(v_a_368_);
if (v___x_369_ == 0)
{
uint8_t v___x_370_; 
lean_inc(v_a_368_);
v___x_370_ = l_Lean_Exception_isRuntime(v_a_368_);
v___y_362_ = v___y_366_;
v___y_363_ = v___x_370_;
goto v___jp_361_;
}
else
{
v___y_362_ = v___y_366_;
v___y_363_ = v___x_369_;
goto v___jp_361_;
}
}
}
}
else
{
lean_object* v_a_378_; lean_object* v___x_380_; uint8_t v_isShared_381_; uint8_t v_isSharedCheck_385_; 
lean_dec(v___x_279_);
lean_dec(v___x_272_);
lean_dec(v_x_251_);
lean_dec(v_x_250_);
v_a_378_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_385_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_385_ == 0)
{
v___x_380_ = v___x_283_;
v_isShared_381_ = v_isSharedCheck_385_;
goto v_resetjp_379_;
}
else
{
lean_inc(v_a_378_);
lean_dec(v___x_283_);
v___x_380_ = lean_box(0);
v_isShared_381_ = v_isSharedCheck_385_;
goto v_resetjp_379_;
}
v_resetjp_379_:
{
lean_object* v___x_383_; 
if (v_isShared_381_ == 0)
{
v___x_383_ = v___x_380_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_384_; 
v_reuseFailAlloc_384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_384_, 0, v_a_378_);
v___x_383_ = v_reuseFailAlloc_384_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
return v___x_383_;
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep___boxed(lean_object* v_x_386_, lean_object* v_x_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_Mathlib_Meta_elabFinsetBuilderSep(v_x_386_, v_x_387_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_, v_a_393_);
lean_dec(v_a_393_);
lean_dec_ref(v_a_392_);
lean_dec(v_a_391_);
lean_dec_ref(v_a_390_);
lean_dec(v_a_389_);
lean_dec_ref(v_a_388_);
return v_res_395_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Empty(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Empty(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
}
#ifdef __cplusplus
}
#endif
