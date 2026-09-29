// Lean compiler output
// Module: Mathlib.Tactic.Trace
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.ElabTerm public meta import Lean.Meta.Eval
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
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_evalExpr___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__3 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__3_value),LEAN_SCALAR_PTR_LITERAL(13, 15, 119, 24, 0, 228, 142, 141)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__4 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__5 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__6 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "trace "};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__7 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__8 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Parser_Tactic_trace___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__9 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__9_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__10 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__11 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__6_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__8_value),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__11_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__12 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Tactic_trace___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Tactic_trace___closed__13 = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Parser_Tactic_trace = (const lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "String"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 130, 56, 8, 41, 104, 134, 43)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__1_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Tactic_trace___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toString"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__3_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__4;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(47, 79, 177, 134, 210, 33, 7, 227)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ToString"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(30, 202, 174, 203, 16, 186, 159, 168)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(206, 210, 39, 124, 69, 192, 37, 107)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__11_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__13;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__2(void){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_35_ = lean_box(0);
v___x_36_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__1));
v___x_37_ = l_Lean_mkConst(v___x_36_, v___x_35_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1(lean_object* v_e_38_, lean_object* v_a_39_, lean_object* v_a_40_, lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
lean_object* v___x_44_; uint8_t v___x_45_; uint8_t v___x_46_; lean_object* v___x_47_; 
v___x_44_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__2, &lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__2);
v___x_45_ = 1;
v___x_46_ = 1;
v___x_47_ = l_Lean_Meta_evalExpr___redArg(v___x_44_, v_e_38_, v___x_45_, v___x_46_, v_a_39_, v_a_40_, v_a_41_, v_a_42_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___boxed(lean_object* v_e_48_, lean_object* v_a_49_, lean_object* v_a_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1(v_e_48_, v_a_49_, v_a_50_, v_a_51_, v_a_52_);
lean_dec(v_a_52_);
lean_dec_ref(v_a_51_);
lean_dec(v_a_50_);
lean_dec_ref(v_a_49_);
return v_res_54_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_55_ = lean_box(0);
v___x_56_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_57_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
lean_ctor_set(v___x_57_, 1, v___x_55_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg(){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___closed__0);
v___x_60_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg___boxed(lean_object* v___y_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg();
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0(lean_object* v_00_u03b1_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg();
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___boxed(lean_object* v_00_u03b1_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0(v_00_u03b1_74_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
lean_dec(v___y_82_);
lean_dec_ref(v___y_81_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
lean_dec(v___y_76_);
lean_dec_ref(v___y_75_);
return v_res_84_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__3(lean_object* v_opts_85_, lean_object* v_opt_86_){
_start:
{
lean_object* v_name_87_; lean_object* v_defValue_88_; lean_object* v_map_89_; lean_object* v___x_90_; 
v_name_87_ = lean_ctor_get(v_opt_86_, 0);
v_defValue_88_ = lean_ctor_get(v_opt_86_, 1);
v_map_89_ = lean_ctor_get(v_opts_85_, 0);
v___x_90_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_89_, v_name_87_);
if (lean_obj_tag(v___x_90_) == 0)
{
uint8_t v___x_91_; 
v___x_91_ = lean_unbox(v_defValue_88_);
return v___x_91_;
}
else
{
lean_object* v_val_92_; 
v_val_92_ = lean_ctor_get(v___x_90_, 0);
lean_inc(v_val_92_);
lean_dec_ref_known(v___x_90_, 1);
if (lean_obj_tag(v_val_92_) == 1)
{
uint8_t v_v_93_; 
v_v_93_ = lean_ctor_get_uint8(v_val_92_, 0);
lean_dec_ref_known(v_val_92_, 0);
return v_v_93_;
}
else
{
uint8_t v___x_94_; 
lean_dec(v_val_92_);
v___x_94_ = lean_unbox(v_defValue_88_);
return v___x_94_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__3___boxed(lean_object* v_opts_95_, lean_object* v_opt_96_){
_start:
{
uint8_t v_res_97_; lean_object* v_r_98_; 
v_res_97_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__3(v_opts_95_, v_opt_96_);
lean_dec_ref(v_opt_96_);
lean_dec_ref(v_opts_95_);
v_r_98_ = lean_box(v_res_97_);
return v_r_98_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0(uint8_t v___y_105_, uint8_t v_suppressElabErrors_106_, lean_object* v_x_107_){
_start:
{
if (lean_obj_tag(v_x_107_) == 1)
{
lean_object* v_pre_108_; 
v_pre_108_ = lean_ctor_get(v_x_107_, 0);
switch(lean_obj_tag(v_pre_108_))
{
case 1:
{
lean_object* v_pre_109_; 
v_pre_109_ = lean_ctor_get(v_pre_108_, 0);
switch(lean_obj_tag(v_pre_109_))
{
case 0:
{
lean_object* v_str_110_; lean_object* v_str_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
v_str_110_ = lean_ctor_get(v_x_107_, 1);
v_str_111_ = lean_ctor_get(v_pre_108_, 1);
v___x_112_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__0));
v___x_113_ = lean_string_dec_eq(v_str_111_, v___x_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; uint8_t v___x_115_; 
v___x_114_ = ((lean_object*)(lp_mathlib_Lean_Parser_Tactic_trace___closed__2));
v___x_115_ = lean_string_dec_eq(v_str_111_, v___x_114_);
if (v___x_115_ == 0)
{
return v___y_105_;
}
else
{
lean_object* v___x_116_; uint8_t v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__1));
v___x_117_ = lean_string_dec_eq(v_str_110_, v___x_116_);
if (v___x_117_ == 0)
{
return v___y_105_;
}
else
{
return v_suppressElabErrors_106_;
}
}
}
else
{
lean_object* v___x_118_; uint8_t v___x_119_; 
v___x_118_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__2));
v___x_119_ = lean_string_dec_eq(v_str_110_, v___x_118_);
if (v___x_119_ == 0)
{
return v___y_105_;
}
else
{
return v_suppressElabErrors_106_;
}
}
}
case 1:
{
lean_object* v_pre_120_; 
v_pre_120_ = lean_ctor_get(v_pre_109_, 0);
if (lean_obj_tag(v_pre_120_) == 0)
{
lean_object* v_str_121_; lean_object* v_str_122_; lean_object* v_str_123_; lean_object* v___x_124_; uint8_t v___x_125_; 
v_str_121_ = lean_ctor_get(v_x_107_, 1);
v_str_122_ = lean_ctor_get(v_pre_108_, 1);
v_str_123_ = lean_ctor_get(v_pre_109_, 1);
v___x_124_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__3));
v___x_125_ = lean_string_dec_eq(v_str_123_, v___x_124_);
if (v___x_125_ == 0)
{
return v___y_105_;
}
else
{
lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_126_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__4));
v___x_127_ = lean_string_dec_eq(v_str_122_, v___x_126_);
if (v___x_127_ == 0)
{
return v___y_105_;
}
else
{
lean_object* v___x_128_; uint8_t v___x_129_; 
v___x_128_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___closed__5));
v___x_129_ = lean_string_dec_eq(v_str_121_, v___x_128_);
if (v___x_129_ == 0)
{
return v___y_105_;
}
else
{
return v_suppressElabErrors_106_;
}
}
}
}
else
{
return v___y_105_;
}
}
default: 
{
return v___y_105_;
}
}
}
case 0:
{
lean_object* v_str_130_; lean_object* v___x_131_; uint8_t v___x_132_; 
v_str_130_ = lean_ctor_get(v_x_107_, 1);
v___x_131_ = ((lean_object*)(lp_mathlib_Lean_Parser_Tactic_trace___closed__3));
v___x_132_ = lean_string_dec_eq(v_str_130_, v___x_131_);
if (v___x_132_ == 0)
{
return v___y_105_;
}
else
{
return v_suppressElabErrors_106_;
}
}
default: 
{
return v___y_105_;
}
}
}
else
{
return v___y_105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___boxed(lean_object* v___y_133_, lean_object* v_suppressElabErrors_134_, lean_object* v_x_135_){
_start:
{
uint8_t v___y_6860__boxed_136_; uint8_t v_suppressElabErrors_boxed_137_; uint8_t v_res_138_; lean_object* v_r_139_; 
v___y_6860__boxed_136_ = lean_unbox(v___y_133_);
v_suppressElabErrors_boxed_137_ = lean_unbox(v_suppressElabErrors_134_);
v_res_138_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0(v___y_6860__boxed_136_, v_suppressElabErrors_boxed_137_, v_x_135_);
lean_dec(v_x_135_);
v_r_139_ = lean_box(v_res_138_);
return v_r_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__2(lean_object* v_msgData_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_){
_start:
{
lean_object* v___x_146_; lean_object* v_env_147_; lean_object* v___x_148_; lean_object* v_mctx_149_; lean_object* v_lctx_150_; lean_object* v_options_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_146_ = lean_st_ref_get(v___y_144_);
v_env_147_ = lean_ctor_get(v___x_146_, 0);
lean_inc_ref(v_env_147_);
lean_dec(v___x_146_);
v___x_148_ = lean_st_ref_get(v___y_142_);
v_mctx_149_ = lean_ctor_get(v___x_148_, 0);
lean_inc_ref(v_mctx_149_);
lean_dec(v___x_148_);
v_lctx_150_ = lean_ctor_get(v___y_141_, 2);
v_options_151_ = lean_ctor_get(v___y_143_, 2);
lean_inc_ref(v_options_151_);
lean_inc_ref(v_lctx_150_);
v___x_152_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_152_, 0, v_env_147_);
lean_ctor_set(v___x_152_, 1, v_mctx_149_);
lean_ctor_set(v___x_152_, 2, v_lctx_150_);
lean_ctor_set(v___x_152_, 3, v_options_151_);
v___x_153_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v_msgData_140_);
v___x_154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__2(v_msgData_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_158_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg(lean_object* v_ref_163_, lean_object* v_msgData_164_, uint8_t v_severity_165_, uint8_t v_isSilent_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
lean_object* v___y_173_; uint8_t v___y_174_; lean_object* v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; uint8_t v___y_178_; lean_object* v___y_179_; lean_object* v___y_180_; lean_object* v___y_181_; lean_object* v___y_209_; lean_object* v___y_210_; uint8_t v___y_211_; lean_object* v___y_212_; uint8_t v___y_213_; lean_object* v___y_214_; uint8_t v___y_215_; lean_object* v___y_216_; lean_object* v___y_234_; lean_object* v___y_235_; uint8_t v___y_236_; lean_object* v___y_237_; lean_object* v___y_238_; uint8_t v___y_239_; uint8_t v___y_240_; lean_object* v___y_241_; lean_object* v___y_245_; lean_object* v___y_246_; uint8_t v___y_247_; lean_object* v___y_248_; uint8_t v___y_249_; lean_object* v___y_250_; uint8_t v___y_251_; uint8_t v___x_256_; lean_object* v___y_258_; lean_object* v___y_259_; lean_object* v___y_260_; uint8_t v___y_261_; lean_object* v___y_262_; uint8_t v___y_263_; uint8_t v___y_264_; uint8_t v___y_266_; uint8_t v___x_281_; 
v___x_256_ = 2;
v___x_281_ = l_Lean_instBEqMessageSeverity_beq(v_severity_165_, v___x_256_);
if (v___x_281_ == 0)
{
v___y_266_ = v___x_281_;
goto v___jp_265_;
}
else
{
uint8_t v___x_282_; 
lean_inc_ref(v_msgData_164_);
v___x_282_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_164_);
v___y_266_ = v___x_282_;
goto v___jp_265_;
}
v___jp_172_:
{
lean_object* v___x_182_; lean_object* v_currNamespace_183_; lean_object* v_openDecls_184_; lean_object* v_env_185_; lean_object* v_nextMacroScope_186_; lean_object* v_ngen_187_; lean_object* v_auxDeclNGen_188_; lean_object* v_traceState_189_; lean_object* v_cache_190_; lean_object* v_messages_191_; lean_object* v_infoState_192_; lean_object* v_snapshotTasks_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_207_; 
v___x_182_ = lean_st_ref_take(v___y_181_);
v_currNamespace_183_ = lean_ctor_get(v___y_180_, 6);
v_openDecls_184_ = lean_ctor_get(v___y_180_, 7);
v_env_185_ = lean_ctor_get(v___x_182_, 0);
v_nextMacroScope_186_ = lean_ctor_get(v___x_182_, 1);
v_ngen_187_ = lean_ctor_get(v___x_182_, 2);
v_auxDeclNGen_188_ = lean_ctor_get(v___x_182_, 3);
v_traceState_189_ = lean_ctor_get(v___x_182_, 4);
v_cache_190_ = lean_ctor_get(v___x_182_, 5);
v_messages_191_ = lean_ctor_get(v___x_182_, 6);
v_infoState_192_ = lean_ctor_get(v___x_182_, 7);
v_snapshotTasks_193_ = lean_ctor_get(v___x_182_, 8);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_207_ == 0)
{
v___x_195_ = v___x_182_;
v_isShared_196_ = v_isSharedCheck_207_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_snapshotTasks_193_);
lean_inc(v_infoState_192_);
lean_inc(v_messages_191_);
lean_inc(v_cache_190_);
lean_inc(v_traceState_189_);
lean_inc(v_auxDeclNGen_188_);
lean_inc(v_ngen_187_);
lean_inc(v_nextMacroScope_186_);
lean_inc(v_env_185_);
lean_dec(v___x_182_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_207_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_202_; 
lean_inc(v_openDecls_184_);
lean_inc(v_currNamespace_183_);
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v_currNamespace_183_);
lean_ctor_set(v___x_197_, 1, v_openDecls_184_);
v___x_198_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set(v___x_198_, 1, v___y_179_);
lean_inc_ref(v___y_175_);
lean_inc_ref(v___y_177_);
v___x_199_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_199_, 0, v___y_177_);
lean_ctor_set(v___x_199_, 1, v___y_173_);
lean_ctor_set(v___x_199_, 2, v___y_176_);
lean_ctor_set(v___x_199_, 3, v___y_175_);
lean_ctor_set(v___x_199_, 4, v___x_198_);
lean_ctor_set_uint8(v___x_199_, sizeof(void*)*5, v___y_178_);
lean_ctor_set_uint8(v___x_199_, sizeof(void*)*5 + 1, v___y_174_);
lean_ctor_set_uint8(v___x_199_, sizeof(void*)*5 + 2, v_isSilent_166_);
v___x_200_ = l_Lean_MessageLog_add(v___x_199_, v_messages_191_);
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 6, v___x_200_);
v___x_202_ = v___x_195_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_env_185_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v_nextMacroScope_186_);
lean_ctor_set(v_reuseFailAlloc_206_, 2, v_ngen_187_);
lean_ctor_set(v_reuseFailAlloc_206_, 3, v_auxDeclNGen_188_);
lean_ctor_set(v_reuseFailAlloc_206_, 4, v_traceState_189_);
lean_ctor_set(v_reuseFailAlloc_206_, 5, v_cache_190_);
lean_ctor_set(v_reuseFailAlloc_206_, 6, v___x_200_);
lean_ctor_set(v_reuseFailAlloc_206_, 7, v_infoState_192_);
lean_ctor_set(v_reuseFailAlloc_206_, 8, v_snapshotTasks_193_);
v___x_202_ = v_reuseFailAlloc_206_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_203_ = lean_st_ref_set(v___y_181_, v___x_202_);
v___x_204_ = lean_box(0);
v___x_205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
return v___x_205_;
}
}
}
v___jp_208_:
{
lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v_a_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_232_; 
v___x_217_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_164_);
v___x_218_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__2(v___x_217_, v___y_167_, v___y_168_, v___y_169_, v___y_170_);
v_a_219_ = lean_ctor_get(v___x_218_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_232_ == 0)
{
v___x_221_ = v___x_218_;
v_isShared_222_ = v_isSharedCheck_232_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_a_219_);
lean_dec(v___x_218_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_232_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
lean_inc_ref_n(v___y_212_, 2);
v___x_223_ = l_Lean_FileMap_toPosition(v___y_212_, v___y_210_);
lean_dec(v___y_210_);
v___x_224_ = l_Lean_FileMap_toPosition(v___y_212_, v___y_216_);
lean_dec(v___y_216_);
v___x_225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
v___x_226_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___closed__0));
if (v___y_213_ == 0)
{
lean_del_object(v___x_221_);
lean_dec_ref(v___y_209_);
v___y_173_ = v___x_223_;
v___y_174_ = v___y_211_;
v___y_175_ = v___x_226_;
v___y_176_ = v___x_225_;
v___y_177_ = v___y_214_;
v___y_178_ = v___y_215_;
v___y_179_ = v_a_219_;
v___y_180_ = v___y_169_;
v___y_181_ = v___y_170_;
goto v___jp_172_;
}
else
{
uint8_t v___x_227_; 
lean_inc(v_a_219_);
v___x_227_ = l_Lean_MessageData_hasTag(v___y_209_, v_a_219_);
if (v___x_227_ == 0)
{
lean_object* v___x_228_; lean_object* v___x_230_; 
lean_dec_ref_known(v___x_225_, 1);
lean_dec_ref(v___x_223_);
lean_dec(v_a_219_);
v___x_228_ = lean_box(0);
if (v_isShared_222_ == 0)
{
lean_ctor_set(v___x_221_, 0, v___x_228_);
v___x_230_ = v___x_221_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v___x_228_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
else
{
lean_del_object(v___x_221_);
v___y_173_ = v___x_223_;
v___y_174_ = v___y_211_;
v___y_175_ = v___x_226_;
v___y_176_ = v___x_225_;
v___y_177_ = v___y_214_;
v___y_178_ = v___y_215_;
v___y_179_ = v_a_219_;
v___y_180_ = v___y_169_;
v___y_181_ = v___y_170_;
goto v___jp_172_;
}
}
}
}
v___jp_233_:
{
lean_object* v___x_242_; 
v___x_242_ = l_Lean_Syntax_getTailPos_x3f(v___y_235_, v___y_240_);
lean_dec(v___y_235_);
if (lean_obj_tag(v___x_242_) == 0)
{
lean_inc(v___y_241_);
v___y_209_ = v___y_234_;
v___y_210_ = v___y_241_;
v___y_211_ = v___y_236_;
v___y_212_ = v___y_237_;
v___y_213_ = v___y_239_;
v___y_214_ = v___y_238_;
v___y_215_ = v___y_240_;
v___y_216_ = v___y_241_;
goto v___jp_208_;
}
else
{
lean_object* v_val_243_; 
v_val_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_val_243_);
lean_dec_ref_known(v___x_242_, 1);
v___y_209_ = v___y_234_;
v___y_210_ = v___y_241_;
v___y_211_ = v___y_236_;
v___y_212_ = v___y_237_;
v___y_213_ = v___y_239_;
v___y_214_ = v___y_238_;
v___y_215_ = v___y_240_;
v___y_216_ = v_val_243_;
goto v___jp_208_;
}
}
v___jp_244_:
{
lean_object* v_ref_252_; lean_object* v___x_253_; 
v_ref_252_ = l_Lean_replaceRef(v_ref_163_, v___y_250_);
v___x_253_ = l_Lean_Syntax_getPos_x3f(v_ref_252_, v___y_249_);
if (lean_obj_tag(v___x_253_) == 0)
{
lean_object* v___x_254_; 
v___x_254_ = lean_unsigned_to_nat(0u);
v___y_234_ = v___y_245_;
v___y_235_ = v_ref_252_;
v___y_236_ = v___y_251_;
v___y_237_ = v___y_246_;
v___y_238_ = v___y_248_;
v___y_239_ = v___y_247_;
v___y_240_ = v___y_249_;
v___y_241_ = v___x_254_;
goto v___jp_233_;
}
else
{
lean_object* v_val_255_; 
v_val_255_ = lean_ctor_get(v___x_253_, 0);
lean_inc(v_val_255_);
lean_dec_ref_known(v___x_253_, 1);
v___y_234_ = v___y_245_;
v___y_235_ = v_ref_252_;
v___y_236_ = v___y_251_;
v___y_237_ = v___y_246_;
v___y_238_ = v___y_248_;
v___y_239_ = v___y_247_;
v___y_240_ = v___y_249_;
v___y_241_ = v_val_255_;
goto v___jp_233_;
}
}
v___jp_257_:
{
if (v___y_264_ == 0)
{
v___y_245_ = v___y_258_;
v___y_246_ = v___y_259_;
v___y_247_ = v___y_261_;
v___y_248_ = v___y_260_;
v___y_249_ = v___y_263_;
v___y_250_ = v___y_262_;
v___y_251_ = v_severity_165_;
goto v___jp_244_;
}
else
{
v___y_245_ = v___y_258_;
v___y_246_ = v___y_259_;
v___y_247_ = v___y_261_;
v___y_248_ = v___y_260_;
v___y_249_ = v___y_263_;
v___y_250_ = v___y_262_;
v___y_251_ = v___x_256_;
goto v___jp_244_;
}
}
v___jp_265_:
{
if (v___y_266_ == 0)
{
lean_object* v_fileName_267_; lean_object* v_fileMap_268_; lean_object* v_options_269_; lean_object* v_ref_270_; uint8_t v_suppressElabErrors_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___f_274_; uint8_t v___x_275_; uint8_t v___x_276_; 
v_fileName_267_ = lean_ctor_get(v___y_169_, 0);
v_fileMap_268_ = lean_ctor_get(v___y_169_, 1);
v_options_269_ = lean_ctor_get(v___y_169_, 2);
v_ref_270_ = lean_ctor_get(v___y_169_, 5);
v_suppressElabErrors_271_ = lean_ctor_get_uint8(v___y_169_, sizeof(void*)*14 + 1);
v___x_272_ = lean_box(v___y_266_);
v___x_273_ = lean_box(v_suppressElabErrors_271_);
v___f_274_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_274_, 0, v___x_272_);
lean_closure_set(v___f_274_, 1, v___x_273_);
v___x_275_ = 1;
v___x_276_ = l_Lean_instBEqMessageSeverity_beq(v_severity_165_, v___x_275_);
if (v___x_276_ == 0)
{
v___y_258_ = v___f_274_;
v___y_259_ = v_fileMap_268_;
v___y_260_ = v_fileName_267_;
v___y_261_ = v_suppressElabErrors_271_;
v___y_262_ = v_ref_270_;
v___y_263_ = v___y_266_;
v___y_264_ = v___x_276_;
goto v___jp_257_;
}
else
{
lean_object* v___x_277_; uint8_t v___x_278_; 
v___x_277_ = l_Lean_warningAsError;
v___x_278_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1_spec__3(v_options_269_, v___x_277_);
v___y_258_ = v___f_274_;
v___y_259_ = v_fileMap_268_;
v___y_260_ = v_fileName_267_;
v___y_261_ = v_suppressElabErrors_271_;
v___y_262_ = v_ref_270_;
v___y_263_ = v___y_266_;
v___y_264_ = v___x_278_;
goto v___jp_257_;
}
}
else
{
lean_object* v___x_279_; lean_object* v___x_280_; 
lean_dec_ref(v_msgData_164_);
v___x_279_ = lean_box(0);
v___x_280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_280_, 0, v___x_279_);
return v___x_280_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg___boxed(lean_object* v_ref_283_, lean_object* v_msgData_284_, lean_object* v_severity_285_, lean_object* v_isSilent_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
uint8_t v_severity_boxed_292_; uint8_t v_isSilent_boxed_293_; lean_object* v_res_294_; 
v_severity_boxed_292_ = lean_unbox(v_severity_285_);
v_isSilent_boxed_293_ = lean_unbox(v_isSilent_286_);
v_res_294_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg(v_ref_283_, v_msgData_284_, v_severity_boxed_292_, v_isSilent_boxed_293_, v___y_287_, v___y_288_, v___y_289_, v___y_290_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v_ref_283_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1(lean_object* v_ref_295_, lean_object* v_msgData_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
uint8_t v___x_306_; uint8_t v___x_307_; lean_object* v___x_308_; 
v___x_306_ = 0;
v___x_307_ = 0;
v___x_308_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg(v_ref_295_, v_msgData_296_, v___x_306_, v___x_307_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1___boxed(lean_object* v_ref_309_, lean_object* v_msgData_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1(v_ref_309_, v_msgData_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_);
lean_dec(v___y_318_);
lean_dec_ref(v___y_317_);
lean_dec(v___y_316_);
lean_dec_ref(v___y_315_);
lean_dec(v___y_314_);
lean_dec_ref(v___y_313_);
lean_dec(v___y_312_);
lean_dec_ref(v___y_311_);
lean_dec(v_ref_309_);
return v_res_320_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__4(void){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_329_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__3));
v___x_330_ = l_String_toRawSubstring_x27(v___x_329_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12(void){
_start:
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_346_ = lean_box(0);
v___x_347_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Trace_0____aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_unsafe__1___closed__1));
v___x_348_ = l_Lean_mkConst(v___x_347_, v___x_346_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__13(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_349_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12, &lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12_once, _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12);
v___x_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1(lean_object* v_x_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_, lean_object* v_a_359_){
_start:
{
lean_object* v___x_361_; uint8_t v___x_362_; 
v___x_361_ = ((lean_object*)(lp_mathlib_Lean_Parser_Tactic_trace___closed__4));
lean_inc(v_x_351_);
v___x_362_ = l_Lean_Syntax_isOfKind(v_x_351_, v___x_361_);
if (v___x_362_ == 0)
{
lean_object* v___x_363_; 
lean_dec(v_x_351_);
v___x_363_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__0___redArg();
return v___x_363_;
}
else
{
lean_object* v_ref_364_; lean_object* v_quotContext_365_; lean_object* v_currMacroScope_366_; lean_object* v___x_367_; lean_object* v___x_368_; uint8_t v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v_ref_364_ = lean_ctor_get(v_a_358_, 5);
v_quotContext_365_ = lean_ctor_get(v_a_358_, 10);
v_currMacroScope_366_ = lean_ctor_get(v_a_358_, 11);
v___x_367_ = lean_unsigned_to_nat(1u);
v___x_368_ = l_Lean_Syntax_getArg(v_x_351_, v___x_367_);
v___x_369_ = 0;
v___x_370_ = l_Lean_SourceInfo_fromRef(v_ref_364_, v___x_369_);
v___x_371_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__2));
v___x_372_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__4, &lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__4_once, _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__4);
v___x_373_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__5));
lean_inc(v_currMacroScope_366_);
lean_inc(v_quotContext_365_);
v___x_374_ = l_Lean_addMacroScope(v_quotContext_365_, v___x_373_, v_currMacroScope_366_);
v___x_375_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__9));
lean_inc_n(v___x_370_, 2);
v___x_376_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_376_, 0, v___x_370_);
lean_ctor_set(v___x_376_, 1, v___x_372_);
lean_ctor_set(v___x_376_, 2, v___x_374_);
lean_ctor_set(v___x_376_, 3, v___x_375_);
v___x_377_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__11));
v___x_378_ = l_Lean_Syntax_node1(v___x_370_, v___x_377_, v___x_368_);
v___x_379_ = l_Lean_Syntax_node2(v___x_370_, v___x_371_, v___x_376_, v___x_378_);
v___x_380_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12, &lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12_once, _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__12);
v___x_381_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__13, &lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__13_once, _init_lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___closed__13);
v___x_382_ = l_Lean_Elab_Tactic_elabTerm(v___x_379_, v___x_381_, v___x_369_, v_a_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_, v_a_357_, v_a_358_, v_a_359_);
if (lean_obj_tag(v___x_382_) == 0)
{
lean_object* v_a_383_; uint8_t v___x_384_; lean_object* v___x_385_; 
v_a_383_ = lean_ctor_get(v___x_382_, 0);
lean_inc(v_a_383_);
lean_dec_ref_known(v___x_382_, 1);
v___x_384_ = 1;
v___x_385_ = l_Lean_Meta_evalExpr___redArg(v___x_380_, v_a_383_, v___x_384_, v___x_362_, v_a_356_, v_a_357_, v_a_358_, v_a_359_);
if (lean_obj_tag(v___x_385_) == 0)
{
lean_object* v_a_386_; lean_object* v___x_387_; lean_object* v_tk_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v_a_386_ = lean_ctor_get(v___x_385_, 0);
lean_inc(v_a_386_);
lean_dec_ref_known(v___x_385_, 1);
v___x_387_ = lean_unsigned_to_nat(0u);
v_tk_388_ = l_Lean_Syntax_getArg(v_x_351_, v___x_387_);
lean_dec(v_x_351_);
v___x_389_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_389_, 0, v_a_386_);
v___x_390_ = l_Lean_MessageData_ofFormat(v___x_389_);
v___x_391_ = lp_mathlib_Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1(v_tk_388_, v___x_390_, v_a_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_, v_a_357_, v_a_358_, v_a_359_);
lean_dec(v_tk_388_);
return v___x_391_;
}
else
{
lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_399_; 
lean_dec(v_x_351_);
v_a_392_ = lean_ctor_get(v___x_385_, 0);
v_isSharedCheck_399_ = !lean_is_exclusive(v___x_385_);
if (v_isSharedCheck_399_ == 0)
{
v___x_394_ = v___x_385_;
v_isShared_395_ = v_isSharedCheck_399_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_385_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_399_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_397_; 
if (v_isShared_395_ == 0)
{
v___x_397_ = v___x_394_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_a_392_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
}
else
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec(v_x_351_);
v_a_400_ = lean_ctor_get(v___x_382_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_382_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_382_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_382_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_405_; 
if (v_isShared_403_ == 0)
{
v___x_405_ = v___x_402_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_406_; 
v_reuseFailAlloc_406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_406_, 0, v_a_400_);
v___x_405_ = v_reuseFailAlloc_406_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
return v___x_405_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1___boxed(lean_object* v_x_408_, lean_object* v_a_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib___aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1(v_x_408_, v_a_409_, v_a_410_, v_a_411_, v_a_412_, v_a_413_, v_a_414_, v_a_415_, v_a_416_);
lean_dec(v_a_416_);
lean_dec_ref(v_a_415_);
lean_dec(v_a_414_);
lean_dec_ref(v_a_413_);
lean_dec(v_a_412_);
lean_dec_ref(v_a_411_);
lean_dec(v_a_410_);
lean_dec_ref(v_a_409_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1(lean_object* v_ref_419_, lean_object* v_msgData_420_, uint8_t v_severity_421_, uint8_t v_isSilent_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___redArg(v_ref_419_, v_msgData_420_, v_severity_421_, v_isSilent_422_, v___y_427_, v___y_428_, v___y_429_, v___y_430_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1___boxed(lean_object* v_ref_433_, lean_object* v_msgData_434_, lean_object* v_severity_435_, lean_object* v_isSilent_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
uint8_t v_severity_boxed_446_; uint8_t v_isSilent_boxed_447_; lean_object* v_res_448_; 
v_severity_boxed_446_ = lean_unbox(v_severity_435_);
v_isSilent_boxed_447_ = lean_unbox(v_isSilent_436_);
v_res_448_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__Mathlib__Tactic__Trace______elabRules__Lean__Parser__Tactic__trace__1_spec__1_spec__1(v_ref_433_, v_msgData_434_, v_severity_boxed_446_, v_isSilent_boxed_447_, v___y_437_, v___y_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
lean_dec(v_ref_433_);
return v_res_448_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Trace(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Eval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Trace(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
lean_object* initialize_Lean_Meta_Eval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Trace(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Trace(builtin);
}
#ifdef __cplusplus
}
#endif
