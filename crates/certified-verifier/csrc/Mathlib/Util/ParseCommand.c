// Lean compiler output
// Module: Mathlib.Util.ParseCommand
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Mathlib.Tactic.Linter.Header public import Lean.Parser.Term
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Parser_whitespace(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_tacticSeq;
lean_object* l_Lean_Parser_andthenFn(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_Parser_mkInputContext___redArg(lean_object*, lean_object*, uint8_t, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Parser_getTokenTable(lean_object*);
lean_object* l_Lean_Parser_mkParserState(lean_object*);
lean_object* l_Lean_Parser_ParserFn_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Parser_instBEqError_beq(lean_object*, lean_object*);
lean_object* l_Lean_Parser_ParserState_toErrorMsg(lean_object*, lean_object*);
uint8_t lean_string_utf8_at_end(lean_object*, lean_object*);
lean_object* l_Lean_Parser_ParserState_mkError(lean_object*, lean_object*);
lean_object* l_Lean_Parser_SyntaxStack_back(lean_object*);
lean_object* l_Lean_Parser_ParserState_allErrors(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Parser_InputContext_atEnd(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_captureException___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<input>"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_captureException___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_captureException___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_captureException___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "end of input"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_captureException___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_captureException___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions_captureException(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Mathlib_GuardExceptions_parseAsTacticSeq_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Mathlib_GuardExceptions_parseAsTacticSeq_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_whitespace, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "GuardExceptions"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "parseCmd"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__1_value),LEAN_SCALAR_PTR_LITERAL(179, 135, 170, 224, 69, 68, 2, 85)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__2_value),LEAN_SCALAR_PTR_LITERAL(4, 209, 88, 227, 163, 150, 190, 75)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "#parse "};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__15_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__19_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_GuardExceptions_parseCmd = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__19_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "runCmd"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(65, 158, 215, 209, 131, 110, 142, 142)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "run_cmd"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "doSeqIndent"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(93, 115, 138, 230, 225, 195, 43, 46)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "doSeqItem"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(10, 94, 50, 120, 46, 251, 13, 13)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doNested"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(220, 154, 41, 109, 103, 76, 110, 63)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "do"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "doLetArrow"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(155, 105, 77, 168, 26, 188, 17, 34)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "let"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__18;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(5, 186, 227, 151, 19, 40, 136, 241)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "doIdDecl"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(41, 95, 84, 160, 28, 70, 78, 179)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "exc"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__24;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(52, 72, 34, 199, 237, 175, 120, 101)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "doExpr"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(130, 168, 60, 255, 153, 218, 88, 77)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_<|_"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(152, 38, 96, 140, 215, 46, 31, 82)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.ofExcept"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__32;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ofExcept"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(138, 236, 204, 116, 34, 116, 116, 175)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__36_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__35_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "<|"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__39_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__40_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "captureException"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__42_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__43;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(9, 29, 169, 31, 173, 129, 147, 75)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__1_value),LEAN_SCALAR_PTR_LITERAL(179, 135, 170, 224, 69, 68, 2, 85)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(177, 126, 74, 54, 126, 236, 255, 139)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__45_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__46_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__48_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__50_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__52_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__53_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__54_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__55_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__56;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__57_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__57_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__1_value),LEAN_SCALAR_PTR_LITERAL(179, 135, 170, 224, 69, 68, 2, 85)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__57_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__58_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__59_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__60_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__61_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__62_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__59_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__63_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__64_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__65_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__65_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__62_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__65_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__65_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__66_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__67_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__67_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__67_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__69_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__69_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__70_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__71_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__68_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__71_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__66_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__72_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__64_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__73_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__74_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__61_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__74_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__75_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__58_value),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__75_value)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__76_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "nestedAction"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__77_value),LEAN_SCALAR_PTR_LITERAL(115, 27, 24, 243, 204, 49, 153, 202)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "getEnv"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__79_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__80_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__80;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__79_value),LEAN_SCALAR_PTR_LITERAL(72, 61, 95, 191, 209, 198, 184, 155)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__81_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "MonadEnv"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__82_value),LEAN_SCALAR_PTR_LITERAL(39, 26, 63, 127, 255, 220, 234, 209)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__79_value),LEAN_SCALAR_PTR_LITERAL(196, 158, 45, 33, 7, 23, 65, 211)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__83_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__84_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__84_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__85_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__86_value;
static const lean_string_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "logInfo"};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__87_value;
static lean_once_cell_t lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__88_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__88;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__87_value),LEAN_SCALAR_PTR_LITERAL(202, 91, 142, 254, 66, 53, 122, 238)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__89_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__90_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__90_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__87_value),LEAN_SCALAR_PTR_LITERAL(203, 37, 56, 83, 4, 68, 47, 204)}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__90_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__90_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__91_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__92_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions_captureException(lean_object* v_env_3_, lean_object* v_s_4_, lean_object* v_input_5_){
_start:
{
lean_object* v___x_6_; uint8_t v___x_7_; lean_object* v___x_8_; lean_object* v_ictx_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v_s_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; uint8_t v___x_20_; 
v___x_6_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions_captureException___closed__0));
v___x_7_ = 1;
v___x_8_ = lean_string_utf8_byte_size(v_input_5_);
lean_inc_ref(v_input_5_);
v_ictx_9_ = l_Lean_Parser_mkInputContext___redArg(v_input_5_, v___x_6_, v___x_7_, v___x_8_);
v___x_10_ = l_Lean_Options_empty;
v___x_11_ = lean_box(0);
v___x_12_ = lean_box(0);
lean_inc_ref(v_env_3_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_3_);
lean_ctor_set(v___x_13_, 1, v___x_10_);
lean_ctor_set(v___x_13_, 2, v___x_11_);
lean_ctor_set(v___x_13_, 3, v___x_12_);
v___x_14_ = l_Lean_Parser_getTokenTable(v_env_3_);
v___x_15_ = l_Lean_Parser_mkParserState(v_input_5_);
lean_dec_ref(v_input_5_);
lean_inc_ref(v_ictx_9_);
v_s_16_ = l_Lean_Parser_ParserFn_run(v_s_4_, v_ictx_9_, v___x_13_, v___x_14_, v___x_15_);
lean_inc_ref(v_s_16_);
v___x_17_ = l_Lean_Parser_ParserState_allErrors(v_s_16_);
v___x_18_ = lean_array_get_size(v___x_17_);
lean_dec_ref(v___x_17_);
v___x_19_ = lean_unsigned_to_nat(0u);
v___x_20_ = lean_nat_dec_eq(v___x_18_, v___x_19_);
if (v___x_20_ == 0)
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = l_Lean_Parser_ParserState_toErrorMsg(v_ictx_9_, v_s_16_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
return v___x_22_;
}
else
{
lean_object* v_stxStack_23_; lean_object* v_pos_24_; uint8_t v___x_25_; 
v_stxStack_23_ = lean_ctor_get(v_s_16_, 0);
lean_inc_ref(v_stxStack_23_);
v_pos_24_ = lean_ctor_get(v_s_16_, 2);
lean_inc(v_pos_24_);
v___x_25_ = l_Lean_Parser_InputContext_atEnd(v_ictx_9_, v_pos_24_);
lean_dec(v_pos_24_);
if (v___x_25_ == 0)
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
lean_dec_ref(v_stxStack_23_);
v___x_26_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions_captureException___closed__1));
v___x_27_ = l_Lean_Parser_ParserState_mkError(v_s_16_, v___x_26_);
v___x_28_ = l_Lean_Parser_ParserState_toErrorMsg(v_ictx_9_, v___x_27_);
v___x_29_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_29_, 0, v___x_28_);
return v___x_29_;
}
else
{
lean_object* v___x_30_; lean_object* v___x_31_; 
lean_dec_ref(v_s_16_);
lean_dec_ref(v_ictx_9_);
v___x_30_ = l_Lean_Parser_SyntaxStack_back(v_stxStack_23_);
lean_dec_ref(v_stxStack_23_);
v___x_31_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_31_, 0, v___x_30_);
return v___x_31_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Mathlib_GuardExceptions_parseAsTacticSeq_spec__0(lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
if (lean_obj_tag(v_x_32_) == 0)
{
if (lean_obj_tag(v_x_33_) == 0)
{
uint8_t v___x_34_; 
v___x_34_ = 1;
return v___x_34_;
}
else
{
uint8_t v___x_35_; 
v___x_35_ = 0;
return v___x_35_;
}
}
else
{
if (lean_obj_tag(v_x_33_) == 0)
{
uint8_t v___x_36_; 
v___x_36_ = 0;
return v___x_36_;
}
else
{
lean_object* v_val_37_; lean_object* v_val_38_; uint8_t v___x_39_; 
v_val_37_ = lean_ctor_get(v_x_32_, 0);
v_val_38_ = lean_ctor_get(v_x_33_, 0);
v___x_39_ = l_Lean_Parser_instBEqError_beq(v_val_37_, v_val_38_);
return v___x_39_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Mathlib_GuardExceptions_parseAsTacticSeq_spec__0___boxed(lean_object* v_x_40_, lean_object* v_x_41_){
_start:
{
uint8_t v_res_42_; lean_object* v_r_43_; 
v_res_42_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_GuardExceptions_parseAsTacticSeq_spec__0(v_x_40_, v_x_41_);
lean_dec(v_x_41_);
lean_dec(v_x_40_);
v_r_43_ = lean_box(v_res_42_);
return v_r_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq(lean_object* v_env_45_, lean_object* v_input_46_, lean_object* v_fileName_47_){
_start:
{
lean_object* v___x_48_; lean_object* v_fn_49_; lean_object* v___x_50_; lean_object* v_p_51_; uint8_t v___x_52_; lean_object* v___x_53_; lean_object* v_ictx_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v_s_61_; lean_object* v_stxStack_62_; lean_object* v_pos_63_; lean_object* v_errorMsg_64_; lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_48_ = l_Lean_Parser_Tactic_tacticSeq;
v_fn_49_ = lean_ctor_get(v___x_48_, 1);
v___x_50_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions_parseAsTacticSeq___closed__0));
lean_inc_ref(v_fn_49_);
v_p_51_ = lean_alloc_closure((void*)(l_Lean_Parser_andthenFn), 4, 2);
lean_closure_set(v_p_51_, 0, v___x_50_);
lean_closure_set(v_p_51_, 1, v_fn_49_);
v___x_52_ = 1;
v___x_53_ = lean_string_utf8_byte_size(v_input_46_);
lean_inc_ref(v_input_46_);
v_ictx_54_ = l_Lean_Parser_mkInputContext___redArg(v_input_46_, v_fileName_47_, v___x_52_, v___x_53_);
v___x_55_ = l_Lean_Options_empty;
v___x_56_ = lean_box(0);
v___x_57_ = lean_box(0);
lean_inc_ref(v_env_45_);
v___x_58_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_58_, 0, v_env_45_);
lean_ctor_set(v___x_58_, 1, v___x_55_);
lean_ctor_set(v___x_58_, 2, v___x_56_);
lean_ctor_set(v___x_58_, 3, v___x_57_);
v___x_59_ = l_Lean_Parser_getTokenTable(v_env_45_);
v___x_60_ = l_Lean_Parser_mkParserState(v_input_46_);
lean_inc_ref(v_ictx_54_);
v_s_61_ = l_Lean_Parser_ParserFn_run(v_p_51_, v_ictx_54_, v___x_58_, v___x_59_, v___x_60_);
v_stxStack_62_ = lean_ctor_get(v_s_61_, 0);
lean_inc_ref(v_stxStack_62_);
v_pos_63_ = lean_ctor_get(v_s_61_, 2);
lean_inc(v_pos_63_);
v_errorMsg_64_ = lean_ctor_get(v_s_61_, 4);
lean_inc(v_errorMsg_64_);
v___x_65_ = lean_box(0);
v___x_66_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_GuardExceptions_parseAsTacticSeq_spec__0(v_errorMsg_64_, v___x_65_);
lean_dec(v_errorMsg_64_);
if (v___x_66_ == 0)
{
lean_object* v___x_67_; lean_object* v___x_68_; 
lean_dec(v_pos_63_);
lean_dec_ref(v_stxStack_62_);
lean_dec_ref(v_input_46_);
v___x_67_ = l_Lean_Parser_ParserState_toErrorMsg(v_ictx_54_, v_s_61_);
v___x_68_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
return v___x_68_;
}
else
{
uint8_t v___x_69_; 
v___x_69_ = lean_string_utf8_at_end(v_input_46_, v_pos_63_);
lean_dec(v_pos_63_);
lean_dec_ref(v_input_46_);
if (v___x_69_ == 0)
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
lean_dec_ref(v_stxStack_62_);
v___x_70_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions_captureException___closed__1));
v___x_71_ = l_Lean_Parser_ParserState_mkError(v_s_61_, v___x_70_);
v___x_72_ = l_Lean_Parser_ParserState_toErrorMsg(v_ictx_54_, v___x_71_);
v___x_73_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
return v___x_73_;
}
else
{
lean_object* v___x_74_; lean_object* v___x_75_; 
lean_dec_ref(v_s_61_);
lean_dec_ref(v_ictx_54_);
v___x_74_ = l_Lean_Parser_SyntaxStack_back(v_stxStack_62_);
lean_dec_ref(v_stxStack_62_);
v___x_75_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
return v___x_75_;
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_119_ = lean_box(0);
v___x_120_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_121_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v___x_119_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg(){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_123_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___closed__0);
v___x_124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_124_, 0, v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg___boxed(lean_object* v___y_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg();
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0(lean_object* v_00_u03b1_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg();
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___boxed(lean_object* v_00_u03b1_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0(v_00_u03b1_132_, v___y_133_, v___y_134_);
lean_dec(v___y_134_);
lean_dec_ref(v___y_133_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg(lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; lean_object* v_env_140_; lean_object* v___x_141_; lean_object* v_mainModule_142_; lean_object* v___x_143_; 
v___x_139_ = lean_st_ref_get(v___y_137_);
v_env_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc_ref(v_env_140_);
lean_dec(v___x_139_);
v___x_141_ = l_Lean_Environment_header(v_env_140_);
lean_dec_ref(v_env_140_);
v_mainModule_142_ = lean_ctor_get(v___x_141_, 0);
lean_inc(v_mainModule_142_);
lean_dec_ref(v___x_141_);
v___x_143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_143_, 0, v_mainModule_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg___boxed(lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg(v___y_144_);
lean_dec(v___y_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1(lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg(v___y_148_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___boxed(lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1(v___y_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_154_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__18(void){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = l_Array_mkArray0(lean_box(0));
return v___x_192_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__24(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__23));
v___x_207_ = l_String_toRawSubstring_x27(v___x_206_);
return v___x_207_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__32(void){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_221_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__31));
v___x_222_ = l_String_toRawSubstring_x27(v___x_221_);
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__43(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__42));
v___x_247_ = l_String_toRawSubstring_x27(v___x_246_);
return v___x_247_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__56(void){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__55));
v___x_278_ = l_String_toRawSubstring_x27(v___x_277_);
return v___x_278_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__80(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__79));
v___x_338_ = l_String_toRawSubstring_x27(v___x_337_);
return v___x_338_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__88(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_354_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__87));
v___x_355_ = l_String_toRawSubstring_x27(v___x_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1(lean_object* v_x_367_, lean_object* v_a_368_, lean_object* v_a_369_){
_start:
{
lean_object* v___x_371_; uint8_t v___x_372_; 
v___x_371_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions_parseCmd___closed__3));
lean_inc(v_x_367_);
v___x_372_ = l_Lean_Syntax_isOfKind(v_x_367_, v___x_371_);
if (v___x_372_ == 0)
{
lean_object* v___x_373_; 
lean_dec(v_x_367_);
v___x_373_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__0___redArg();
return v___x_373_;
}
else
{
lean_object* v___x_374_; 
v___x_374_ = l_Lean_Elab_Command_getRef___redArg(v_a_368_);
if (lean_obj_tag(v___x_374_) == 0)
{
lean_object* v_a_375_; lean_object* v___x_376_; 
v_a_375_ = lean_ctor_get(v___x_374_, 0);
lean_inc(v_a_375_);
lean_dec_ref_known(v___x_374_, 1);
v___x_376_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v_a_368_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v_quotContext_x3f_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; uint8_t v___x_383_; lean_object* v___x_384_; lean_object* v_a_386_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_a_377_);
lean_dec_ref_known(v___x_376_, 1);
v_quotContext_x3f_378_ = lean_ctor_get(v_a_368_, 5);
v___x_379_ = lean_unsigned_to_nat(1u);
v___x_380_ = l_Lean_Syntax_getArg(v_x_367_, v___x_379_);
v___x_381_ = lean_unsigned_to_nat(3u);
v___x_382_ = l_Lean_Syntax_getArg(v_x_367_, v___x_381_);
lean_dec(v_x_367_);
v___x_383_ = 0;
v___x_384_ = l_Lean_SourceInfo_fromRef(v_a_375_, v___x_383_);
lean_dec(v_a_375_);
if (lean_obj_tag(v_quotContext_x3f_378_) == 0)
{
lean_object* v___x_473_; lean_object* v_a_474_; 
v___x_473_ = lp_mathlib_Lean_getMainModule___at___00Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1_spec__1___redArg(v_a_369_);
v_a_474_ = lean_ctor_get(v___x_473_, 0);
lean_inc(v_a_474_);
lean_dec_ref(v___x_473_);
v_a_386_ = v_a_474_;
goto v___jp_385_;
}
else
{
lean_object* v_val_475_; 
v_val_475_ = lean_ctor_get(v_quotContext_x3f_378_, 0);
lean_inc(v_val_475_);
v_a_386_ = v_val_475_;
goto v___jp_385_;
}
v___jp_385_:
{
lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_387_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__2));
v___x_388_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__3));
lean_inc_n(v___x_384_, 37);
v___x_389_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_389_, 0, v___x_384_);
lean_ctor_set(v___x_389_, 1, v___x_388_);
v___x_390_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__7));
v___x_391_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__9));
v___x_392_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__11));
v___x_393_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__13));
v___x_394_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__14));
v___x_395_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_384_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v___x_396_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__16));
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__17));
v___x_398_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_398_, 0, v___x_384_);
lean_ctor_set(v___x_398_, 1, v___x_397_);
v___x_399_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__18, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__18_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__18);
v___x_400_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_400_, 0, v___x_384_);
lean_ctor_set(v___x_400_, 1, v___x_391_);
lean_ctor_set(v___x_400_, 2, v___x_399_);
v___x_401_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__20));
lean_inc_ref_n(v___x_400_, 5);
v___x_402_ = l_Lean_Syntax_node1(v___x_384_, v___x_401_, v___x_400_);
v___x_403_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__22));
v___x_404_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__24, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__24_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__24);
v___x_405_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__25));
lean_inc_n(v_a_377_, 5);
lean_inc_n(v_a_386_, 5);
v___x_406_ = l_Lean_addMacroScope(v_a_386_, v___x_405_, v_a_377_);
v___x_407_ = lean_box(0);
v___x_408_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_408_, 0, v___x_384_);
lean_ctor_set(v___x_408_, 1, v___x_404_);
lean_ctor_set(v___x_408_, 2, v___x_406_);
lean_ctor_set(v___x_408_, 3, v___x_407_);
v___x_409_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__26));
v___x_410_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_384_);
lean_ctor_set(v___x_410_, 1, v___x_409_);
v___x_411_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__28));
v___x_412_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__30));
v___x_413_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__32, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__32_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__32);
v___x_414_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__34));
v___x_415_ = l_Lean_addMacroScope(v_a_386_, v___x_414_, v_a_377_);
v___x_416_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__38));
v___x_417_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_417_, 0, v___x_384_);
lean_ctor_set(v___x_417_, 1, v___x_413_);
lean_ctor_set(v___x_417_, 2, v___x_415_);
lean_ctor_set(v___x_417_, 3, v___x_416_);
v___x_418_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__39));
v___x_419_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_384_);
lean_ctor_set(v___x_419_, 1, v___x_418_);
v___x_420_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__41));
v___x_421_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__43, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__43_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__43);
v___x_422_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__44));
v___x_423_ = l_Lean_addMacroScope(v_a_386_, v___x_422_, v_a_377_);
v___x_424_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__47));
v___x_425_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_425_, 0, v___x_384_);
lean_ctor_set(v___x_425_, 1, v___x_421_);
lean_ctor_set(v___x_425_, 2, v___x_423_);
lean_ctor_set(v___x_425_, 3, v___x_424_);
v___x_426_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__49));
v___x_427_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__51));
v___x_428_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__52));
v___x_429_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_384_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
v___x_430_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__54));
v___x_431_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__56, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__56_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__56);
v___x_432_ = lean_box(0);
v___x_433_ = l_Lean_addMacroScope(v_a_386_, v___x_432_, v_a_377_);
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__76));
v___x_435_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_435_, 0, v___x_384_);
lean_ctor_set(v___x_435_, 1, v___x_431_);
lean_ctor_set(v___x_435_, 2, v___x_433_);
lean_ctor_set(v___x_435_, 3, v___x_434_);
v___x_436_ = l_Lean_Syntax_node1(v___x_384_, v___x_430_, v___x_435_);
v___x_437_ = l_Lean_Syntax_node2(v___x_384_, v___x_427_, v___x_429_, v___x_436_);
v___x_438_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__78));
v___x_439_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__80, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__80_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__80);
v___x_440_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__81));
v___x_441_ = l_Lean_addMacroScope(v_a_386_, v___x_440_, v_a_377_);
v___x_442_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__85));
v___x_443_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_443_, 0, v___x_384_);
lean_ctor_set(v___x_443_, 1, v___x_439_);
lean_ctor_set(v___x_443_, 2, v___x_441_);
lean_ctor_set(v___x_443_, 3, v___x_442_);
v___x_444_ = l_Lean_Syntax_node1(v___x_384_, v___x_411_, v___x_443_);
lean_inc_ref(v___x_410_);
v___x_445_ = l_Lean_Syntax_node2(v___x_384_, v___x_438_, v___x_410_, v___x_444_);
v___x_446_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__86));
v___x_447_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_447_, 0, v___x_384_);
lean_ctor_set(v___x_447_, 1, v___x_446_);
v___x_448_ = l_Lean_Syntax_node3(v___x_384_, v___x_426_, v___x_437_, v___x_445_, v___x_447_);
lean_inc(v___x_382_);
v___x_449_ = l_Lean_Syntax_node3(v___x_384_, v___x_391_, v___x_448_, v___x_380_, v___x_382_);
v___x_450_ = l_Lean_Syntax_node2(v___x_384_, v___x_420_, v___x_425_, v___x_449_);
v___x_451_ = l_Lean_Syntax_node3(v___x_384_, v___x_412_, v___x_417_, v___x_419_, v___x_450_);
v___x_452_ = l_Lean_Syntax_node1(v___x_384_, v___x_411_, v___x_451_);
v___x_453_ = l_Lean_Syntax_node4(v___x_384_, v___x_403_, v___x_408_, v___x_400_, v___x_410_, v___x_452_);
v___x_454_ = l_Lean_Syntax_node4(v___x_384_, v___x_396_, v___x_398_, v___x_400_, v___x_402_, v___x_453_);
v___x_455_ = l_Lean_Syntax_node2(v___x_384_, v___x_392_, v___x_454_, v___x_400_);
v___x_456_ = lean_obj_once(&lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__88, &lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__88_once, _init_lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__88);
v___x_457_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__89));
v___x_458_ = l_Lean_addMacroScope(v_a_386_, v___x_457_, v_a_377_);
v___x_459_ = ((lean_object*)(lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___closed__92));
v___x_460_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_460_, 0, v___x_384_);
lean_ctor_set(v___x_460_, 1, v___x_456_);
lean_ctor_set(v___x_460_, 2, v___x_458_);
lean_ctor_set(v___x_460_, 3, v___x_459_);
v___x_461_ = l_Lean_Syntax_node1(v___x_384_, v___x_391_, v___x_382_);
v___x_462_ = l_Lean_Syntax_node2(v___x_384_, v___x_420_, v___x_460_, v___x_461_);
v___x_463_ = l_Lean_Syntax_node1(v___x_384_, v___x_411_, v___x_462_);
v___x_464_ = l_Lean_Syntax_node2(v___x_384_, v___x_392_, v___x_463_, v___x_400_);
v___x_465_ = l_Lean_Syntax_node2(v___x_384_, v___x_391_, v___x_455_, v___x_464_);
v___x_466_ = l_Lean_Syntax_node1(v___x_384_, v___x_390_, v___x_465_);
v___x_467_ = l_Lean_Syntax_node2(v___x_384_, v___x_393_, v___x_395_, v___x_466_);
v___x_468_ = l_Lean_Syntax_node2(v___x_384_, v___x_392_, v___x_467_, v___x_400_);
v___x_469_ = l_Lean_Syntax_node1(v___x_384_, v___x_391_, v___x_468_);
v___x_470_ = l_Lean_Syntax_node1(v___x_384_, v___x_390_, v___x_469_);
v___x_471_ = l_Lean_Syntax_node2(v___x_384_, v___x_387_, v___x_389_, v___x_470_);
v___x_472_ = l_Lean_Elab_Command_elabCommand(v___x_471_, v_a_368_, v_a_369_);
return v___x_472_;
}
}
else
{
lean_object* v_a_476_; lean_object* v___x_478_; uint8_t v_isShared_479_; uint8_t v_isSharedCheck_483_; 
lean_dec(v_a_375_);
lean_dec(v_x_367_);
v_a_476_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_483_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_483_ == 0)
{
v___x_478_ = v___x_376_;
v_isShared_479_ = v_isSharedCheck_483_;
goto v_resetjp_477_;
}
else
{
lean_inc(v_a_476_);
lean_dec(v___x_376_);
v___x_478_ = lean_box(0);
v_isShared_479_ = v_isSharedCheck_483_;
goto v_resetjp_477_;
}
v_resetjp_477_:
{
lean_object* v___x_481_; 
if (v_isShared_479_ == 0)
{
v___x_481_ = v___x_478_;
goto v_reusejp_480_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v_a_476_);
v___x_481_ = v_reuseFailAlloc_482_;
goto v_reusejp_480_;
}
v_reusejp_480_:
{
return v___x_481_;
}
}
}
}
else
{
lean_object* v_a_484_; lean_object* v___x_486_; uint8_t v_isShared_487_; uint8_t v_isSharedCheck_491_; 
lean_dec(v_x_367_);
v_a_484_ = lean_ctor_get(v___x_374_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_374_);
if (v_isSharedCheck_491_ == 0)
{
v___x_486_ = v___x_374_;
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
else
{
lean_inc(v_a_484_);
lean_dec(v___x_374_);
v___x_486_ = lean_box(0);
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
v_resetjp_485_:
{
lean_object* v___x_489_; 
if (v_isShared_487_ == 0)
{
v___x_489_ = v___x_486_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v_a_484_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1___boxed(lean_object* v_x_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_Mathlib_GuardExceptions___aux__Mathlib__Util__ParseCommand______elabRules__Mathlib__GuardExceptions__parseCmd__1(v_x_492_, v_a_493_, v_a_494_);
lean_dec(v_a_494_);
lean_dec_ref(v_a_493_);
return v_res_496_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Term(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_ParseCommand(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_ParseCommand(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Parser_Term(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_ParseCommand(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_ParseCommand(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_ParseCommand(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_ParseCommand(builtin);
}
#ifdef __cplusplus
}
#endif
