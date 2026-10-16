// Lean compiler output
// Module: Mathlib.Util.Delaborators
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.PrettyPrinter.Delaborator.Builtins public import Mathlib.Util.PPOptions
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
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delabForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkAntiquot(lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Parser_termParser_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* lp_mathlib_Mathlib_getPPBinderPredicates___boxed(lean_object*);
uint8_t l_Lean_Expr_isLambda(lean_object*);
lean_object* l_Lean_Parser_Term_bracketedBinder_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_Term_binderIdent_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Parenthesizer_orelse_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_unicodeSymbol_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_symbol_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Parenthesizer_andthen_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_Term_optType_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_ppSpace_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_many1_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_leadPrec;
lean_object* l_Lean_PrettyPrinter_Parenthesizer_leadingNode_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_unicodeSymbol___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_termParser(lean_object*);
lean_object* l_Lean_Parser_unicodeSymbol_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_PrettyPrinter_Delaborator_delabFailureId;
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Parser_symbol(lean_object*);
lean_object* l_Lean_Parser_andthen(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Term_optType;
lean_object* l_Lean_Parser_Term_bracketedBinder(uint8_t);
extern lean_object* l_Lean_Parser_Term_binderIdent;
lean_object* l_Lean_Parser_orelse(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_skip;
lean_object* l_Lean_Parser_many1(lean_object*);
lean_object* l_Lean_Parser_leadingNode(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_withAntiquot(lean_object*, lean_object*);
lean_object* l_Lean_Parser_withCache(lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_Term_bracketedBinder_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Macro_throwUnsupported___redArg(lean_object*);
lean_object* l_Lean_Parser_termParser_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_symbol_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_andthen_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_Term_optType_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_Term_binderIdent_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_orelse_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_pushLine___redArg(lean_object*);
lean_object* l_Lean_Parser_many1_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Parser_leadingNode_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_orelse_formatter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_PiNotation_piNotation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "PiNotation"};
static const lean_object* lp_mathlib_PiNotation_piNotation___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_piNotation___closed__0_value;
static const lean_string_object lp_mathlib_PiNotation_piNotation___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "piNotation"};
static const lean_object* lp_mathlib_PiNotation_piNotation___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_piNotation___closed__1_value;
static const lean_ctor_object lp_mathlib_PiNotation_piNotation___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(47, 155, 205, 42, 138, 170, 201, 52)}};
static const lean_ctor_object lp_mathlib_PiNotation_piNotation___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__2_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__1_value),LEAN_SCALAR_PTR_LITERAL(38, 196, 146, 61, 175, 56, 76, 59)}};
static const lean_object* lp_mathlib_PiNotation_piNotation___closed__2 = (const lean_object*)&lp_mathlib_PiNotation_piNotation___closed__2_value;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__3;
static const lean_string_object lp_mathlib_PiNotation_piNotation___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "Π"};
static const lean_object* lp_mathlib_PiNotation_piNotation___closed__4 = (const lean_object*)&lp_mathlib_PiNotation_piNotation___closed__4_value;
static const lean_string_object lp_mathlib_PiNotation_piNotation___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "PiType"};
static const lean_object* lp_mathlib_PiNotation_piNotation___closed__5 = (const lean_object*)&lp_mathlib_PiNotation_piNotation___closed__5_value;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__6;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__7;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__8;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__9;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__10;
static const lean_string_object lp_mathlib_PiNotation_piNotation___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_PiNotation_piNotation___closed__11 = (const lean_object*)&lp_mathlib_PiNotation_piNotation___closed__11_value;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__12;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__13;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__14;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__15;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__16;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__17;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__18;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__19;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation___closed__20;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PiNotation_piNotation_formatter___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__0_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_formatter___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__1_value),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__2_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__1_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_unicodeSymbol_formatter___boxed, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__4_value),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__2 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__2_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_Term_binderIdent_formatter___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__3 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__3_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_Term_bracketedBinder_formatter___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__4 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__4_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Formatter_orelse_formatter___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__3_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__4_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__5 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__5_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Formatter_andthen_formatter___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__0_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__5_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__6 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__6_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_many1_formatter___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__6_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__7 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__7_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_Term_optType_formatter___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__8 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__8_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_symbol_formatter___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__11_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__9 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__9_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_termParser_formatter___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__10 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__10_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Formatter_andthen_formatter___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__9_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__10_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__11 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__11_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Formatter_andthen_formatter___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__8_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__11_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__12 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__12_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Formatter_andthen_formatter___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__7_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__12_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__13 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__13_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_formatter___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Formatter_andthen_formatter___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__2_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__13_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__14 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_formatter___closed__14_value;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation_formatter___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation_formatter___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_parenthesizer___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__1_value),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__2_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__0_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_unicodeSymbol_parenthesizer___boxed, .m_arity = 8, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__4_value),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__1_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_ppSpace_parenthesizer___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__2 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__2_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_Term_binderIdent_parenthesizer___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__3 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__3_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_Term_bracketedBinder_parenthesizer___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__4 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__4_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Parenthesizer_orelse_parenthesizer___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__3_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__4_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__5 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__5_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Parenthesizer_andthen_parenthesizer___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__2_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__5_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__6 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__6_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_many1_parenthesizer___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__6_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__7 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__7_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_Term_optType_parenthesizer___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__8 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__8_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_symbol_parenthesizer___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__11_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__9 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__9_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_termParser_parenthesizer___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__10 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__10_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Parenthesizer_andthen_parenthesizer___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__9_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__10_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__11 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__11_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Parenthesizer_andthen_parenthesizer___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__8_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__11_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__12 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__12_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Parenthesizer_andthen_parenthesizer___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__7_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__12_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__13 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__13_value;
static const lean_closure_object lp_mathlib_PiNotation_piNotation_parenthesizer___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Parenthesizer_andthen_parenthesizer___boxed, .m_arity = 7, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__1_value),((lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__13_value)} };
static const lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__14 = (const lean_object*)&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__14_value;
static lean_once_cell_t lp_mathlib_PiNotation_piNotation_parenthesizer___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 9, .m_data = "termΠ__,_"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__0 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__0_value),LEAN_SCALAR_PTR_LITERAL(47, 155, 205, 42, 138, 170, 201, 52)}};
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(177, 106, 61, 160, 31, 200, 169, 189)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__2 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 2, .m_data = "Π "};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__4 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__4_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__5 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__5_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__9 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__9_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value_aux_1),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value_aux_2),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(140, 254, 143, 37, 104, 151, 100, 109)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__10_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__11 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__5_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__11_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__12 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__12_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "binderPred"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__13 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__13_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(218, 134, 142, 164, 134, 201, 62, 191)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__14 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__14_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__15 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__15_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__12_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__15_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__16 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__16_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_piNotation___closed__11_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__17 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__17_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__16_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__17_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__18 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__18_value;
static const lean_string_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__19 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__19_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__19_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__20 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__20_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__21 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__21_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__3_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__18_value),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__21_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__22 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__22_value;
static const lean_ctor_object lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__22_value)}};
static const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__23 = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__23_value;
LEAN_EXPORT const lean_object* lp_mathlib_PiNotation_term_u03a0_____x2c__ = (const lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__23_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__0 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__0_value;
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__2 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__4 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__4_value;
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "x"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__6 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__7;
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(243, 101, 181, 186, 114, 114, 131, 189)}};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__8 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__8_value;
static lean_once_cell_t lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "arrow"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__11 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__11_value;
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(182, 146, 143, 73, 122, 115, 5, 207)}};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termSatisfies_binder_pred%__"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__13 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(35, 32, 166, 185, 227, 132, 228, 81)}};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "satisfies_binder_pred%"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__15 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__15_value;
static const lean_string_object lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "→"};
static const lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__16 = (const lean_object*)&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_PiNotation_replacePiNotation___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "forall"};
static const lean_object* lp_mathlib_PiNotation_replacePiNotation___redArg___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_replacePiNotation___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib_PiNotation_replacePiNotation___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(195, 142, 115, 15, 55, 103, 31, 115)}};
static const lean_object* lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_replacePiNotation___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_replacePiNotation(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_replacePiNotation___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "explicitBinder"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 119, 193, 23, 170, 93, 183, 238)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∈_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__2 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(145, 149, 102, 29, 65, 152, 113, 144)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__3 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∈_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__4 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(150, 164, 254, 63, 76, 57, 126, 92)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__5 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∈"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__6 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_>_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__7 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(21, 139, 32, 245, 79, 44, 200, 27)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__8 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term_<_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__9 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__9_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(192, 242, 106, 74, 199, 131, 133, 95)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__10 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__10_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≥_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__11 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__11_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__11_value),LEAN_SCALAR_PTR_LITERAL(58, 65, 30, 214, 7, 203, 184, 211)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__12 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__12_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≤_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__13 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(111, 3, 61, 112, 38, 138, 106, 121)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__14 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__14_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∉_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__15 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__15_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(241, 191, 47, 82, 105, 120, 162, 72)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__16 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__16_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊆_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__17 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__17_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(17, 202, 90, 218, 225, 73, 214, 71)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__18 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__18_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊂_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__19 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__19_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__19_value),LEAN_SCALAR_PTR_LITERAL(168, 36, 104, 26, 7, 158, 117, 91)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__20 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__20_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊇_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__21 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__21_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__21_value),LEAN_SCALAR_PTR_LITERAL(126, 48, 9, 251, 76, 50, 57, 116)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__22 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__22_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⊃_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__23 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__23_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__23_value),LEAN_SCALAR_PTR_LITERAL(50, 217, 255, 107, 39, 224, 209, 40)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__24 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__24_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term∀__,_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__25 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__25_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__26_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__25_value),LEAN_SCALAR_PTR_LITERAL(124, 16, 227, 203, 159, 8, 82, 19)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__26 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__26_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∀"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__27 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__27_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__28_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__28 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__28_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred⊃_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__29 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__29_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__30_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__29_value),LEAN_SCALAR_PTR_LITERAL(227, 228, 155, 154, 179, 5, 240, 181)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__30 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__30_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊃"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__31 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__31_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred⊇_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__32 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__32_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__33_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__32_value),LEAN_SCALAR_PTR_LITERAL(121, 205, 109, 198, 229, 195, 21, 21)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__33 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__33_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊇"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__34 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__34_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred⊂_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__35 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__35_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__36_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__35_value),LEAN_SCALAR_PTR_LITERAL(43, 93, 221, 41, 188, 55, 153, 107)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__36 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__36_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊂"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__37 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__37_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred⊆_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__38 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__38_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__39_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__38_value),LEAN_SCALAR_PTR_LITERAL(177, 23, 43, 221, 193, 123, 248, 62)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__39 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__39_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⊆"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__40 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__40_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∉_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__41 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__41_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__42_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__41_value),LEAN_SCALAR_PTR_LITERAL(147, 253, 164, 249, 200, 108, 121, 70)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__42 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__42_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∉"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__43 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__43_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≤_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__44 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__44_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__45_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__44_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 52, 209, 56, 53, 218, 188)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__45 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__45_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≤"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__46 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__46_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≥_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__47 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__47_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__48_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__47_value),LEAN_SCALAR_PTR_LITERAL(212, 139, 67, 46, 49, 133, 157, 246)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__48 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__48_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≥"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__49 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__49_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderPred<_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__50 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__50_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__51_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__50_value),LEAN_SCALAR_PTR_LITERAL(87, 122, 58, 108, 39, 32, 195, 29)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__51 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__51_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "<"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__52 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__52_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderPred>_"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__53 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__53_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi___lam__0___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__54_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__53_value),LEAN_SCALAR_PTR_LITERAL(140, 244, 246, 184, 111, 78, 213, 47)}};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__54 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__54_value;
static const lean_string_object lp_mathlib_PiNotation_delabPi___lam__0___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ">"};
static const lean_object* lp_mathlib_PiNotation_delabPi___lam__0___closed__55 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___lam__0___closed__55_value;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PiNotation_delabPi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PiNotation_delabPi___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_delabPi___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___closed__0_value;
static const lean_closure_object lp_mathlib_PiNotation_delabPi___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_getPPBinderPredicates___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_delabPi___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___closed__1_value;
static const lean_closure_object lp_mathlib_PiNotation_delabPi___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_delabPi___closed__2 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___closed__2_value;
static const lean_closure_object lp_mathlib_PiNotation_delabPi___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi___closed__2_value),((lean_object*)&lp_mathlib_PiNotation_delabPi___closed__0_value)} };
static const lean_object* lp_mathlib_PiNotation_delabPi___closed__3 = (const lean_object*)&lp_mathlib_PiNotation_delabPi___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "depArrow"};
static const lean_object* lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(115, 137, 180, 163, 158, 211, 191, 168)}};
static const lean_object* lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1 = (const lean_object*)&lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PiNotation_delabPi_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PiNotation_delabPi_x27___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PiNotation_delabPi_x27___closed__0 = (const lean_object*)&lp_mathlib_PiNotation_delabPi_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term∃_,_"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__0 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 105, 219, 112, 166, 139, 167, 161)}};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__1 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∃"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__2 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "explicitBinders"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__3 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(167, 149, 127, 13, 202, 239, 226, 94)}};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__4 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "bracketedExplicitBinders"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__5 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(22, 65, 7, 186, 44, 89, 152, 79)}};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__6 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__7 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__8 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__8_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__9 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "unbracketedExplicitBinders"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__10 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__10_value;
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_exists__delab___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__11_value_aux_0),((lean_object*)&lp_mathlib_exists__delab___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 220, 119, 82, 242, 112, 119, 200)}};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__11 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__11_value;
static const lean_string_object lp_mathlib_exists__delab___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_exists__delab___lam__0___closed__12 = (const lean_object*)&lp_mathlib_exists__delab___lam__0___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__0(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__1(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_exists__delab___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_exists__delab___lam__2___closed__0;
static const lean_closure_object lp_mathlib_exists__delab___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPFunBinderTypes___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_exists__delab___lam__2___closed__1 = (const lean_object*)&lp_mathlib_exists__delab___lam__2___closed__1_value;
static const lean_closure_object lp_mathlib_exists__delab___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_exists__delab___lam__2___closed__2 = (const lean_object*)&lp_mathlib_exists__delab___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_exists__delab___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∧_"};
static const lean_object* lp_mathlib_exists__delab___lam__2___closed__3 = (const lean_object*)&lp_mathlib_exists__delab___lam__2___closed__3_value;
static const lean_ctor_object lp_mathlib_exists__delab___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_exists__delab___lam__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(213, 224, 85, 99, 168, 124, 84, 223)}};
static const lean_object* lp_mathlib_exists__delab___lam__2___closed__4 = (const lean_object*)&lp_mathlib_exists__delab___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_exists__delab___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term∃__,_"};
static const lean_object* lp_mathlib_exists__delab___lam__2___closed__5 = (const lean_object*)&lp_mathlib_exists__delab___lam__2___closed__5_value;
static const lean_ctor_object lp_mathlib_exists__delab___lam__2___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_exists__delab___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_exists__delab___lam__2___closed__6_value_aux_0),((lean_object*)&lp_mathlib_exists__delab___lam__2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(25, 165, 221, 98, 134, 231, 221, 237)}};
static const lean_object* lp_mathlib_exists__delab___lam__2___closed__6 = (const lean_object*)&lp_mathlib_exists__delab___lam__2___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_exists__delab___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_exists__delab___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_exists__delab___closed__0 = (const lean_object*)&lp_mathlib_exists__delab___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_exists__delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_delabNotIn___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_exists__delab___lam__2___closed__2_value)} };
static const lean_object* lp_mathlib_delabNotIn___lam__0___closed__0 = (const lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__0_value;
static const lean_closure_object lp_mathlib_delabNotIn___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_mathlib_exists__delab___lam__2___closed__2_value)} };
static const lean_object* lp_mathlib_delabNotIn___lam__0___closed__1 = (const lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_delabNotIn___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Membership"};
static const lean_object* lp_mathlib_delabNotIn___lam__0___closed__2 = (const lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_delabNotIn___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mem"};
static const lean_object* lp_mathlib_delabNotIn___lam__0___closed__3 = (const lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_delabNotIn___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 217, 109, 94, 255, 55, 82, 109)}};
static const lean_ctor_object lp_mathlib_delabNotIn___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(224, 90, 126, 237, 128, 148, 153, 69)}};
static const lean_object* lp_mathlib_delabNotIn___lam__0___closed__4 = (const lean_object*)&lp_mathlib_delabNotIn___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_delabNotIn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_delabNotIn___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_delabNotIn___closed__0 = (const lean_object*)&lp_mathlib_delabNotIn___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__3(void){
_start:
{
uint8_t v___x_6_; uint8_t v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_6_ = 0;
v___x_7_ = 1;
v___x_8_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_9_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__1));
v___x_10_ = l_Lean_Parser_mkAntiquot(v___x_9_, v___x_8_, v___x_7_, v___x_6_);
return v___x_10_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__6(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_13_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__5));
v___x_14_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__4));
v___x_15_ = l_Lean_Parser_unicodeSymbol___redArg(v___x_14_, v___x_13_);
return v___x_15_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__7(void){
_start:
{
uint8_t v___x_16_; lean_object* v___x_17_; 
v___x_16_ = 0;
v___x_17_ = l_Lean_Parser_Term_bracketedBinder(v___x_16_);
return v___x_17_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__8(void){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_18_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__7, &lp_mathlib_PiNotation_piNotation___closed__7_once, _init_lp_mathlib_PiNotation_piNotation___closed__7);
v___x_19_ = l_Lean_Parser_Term_binderIdent;
v___x_20_ = l_Lean_Parser_orelse(v___x_19_, v___x_18_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__9(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_21_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__8, &lp_mathlib_PiNotation_piNotation___closed__8_once, _init_lp_mathlib_PiNotation_piNotation___closed__8);
v___x_22_ = l_Lean_Parser_skip;
v___x_23_ = l_Lean_Parser_andthen(v___x_22_, v___x_21_);
return v___x_23_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__10(void){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__9, &lp_mathlib_PiNotation_piNotation___closed__9_once, _init_lp_mathlib_PiNotation_piNotation___closed__9);
v___x_25_ = l_Lean_Parser_many1(v___x_24_);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__11));
v___x_28_ = l_Lean_Parser_symbol(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lean_unsigned_to_nat(0u);
v___x_30_ = l_Lean_Parser_termParser(v___x_29_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__14(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_31_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__13, &lp_mathlib_PiNotation_piNotation___closed__13_once, _init_lp_mathlib_PiNotation_piNotation___closed__13);
v___x_32_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__12, &lp_mathlib_PiNotation_piNotation___closed__12_once, _init_lp_mathlib_PiNotation_piNotation___closed__12);
v___x_33_ = l_Lean_Parser_andthen(v___x_32_, v___x_31_);
return v___x_33_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__15(void){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_34_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__14, &lp_mathlib_PiNotation_piNotation___closed__14_once, _init_lp_mathlib_PiNotation_piNotation___closed__14);
v___x_35_ = l_Lean_Parser_Term_optType;
v___x_36_ = l_Lean_Parser_andthen(v___x_35_, v___x_34_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__16(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__15, &lp_mathlib_PiNotation_piNotation___closed__15_once, _init_lp_mathlib_PiNotation_piNotation___closed__15);
v___x_38_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__10, &lp_mathlib_PiNotation_piNotation___closed__10_once, _init_lp_mathlib_PiNotation_piNotation___closed__10);
v___x_39_ = l_Lean_Parser_andthen(v___x_38_, v___x_37_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__17(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_40_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__16, &lp_mathlib_PiNotation_piNotation___closed__16_once, _init_lp_mathlib_PiNotation_piNotation___closed__16);
v___x_41_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__6, &lp_mathlib_PiNotation_piNotation___closed__6_once, _init_lp_mathlib_PiNotation_piNotation___closed__6);
v___x_42_ = l_Lean_Parser_andthen(v___x_41_, v___x_40_);
return v___x_42_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__18(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_43_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__17, &lp_mathlib_PiNotation_piNotation___closed__17_once, _init_lp_mathlib_PiNotation_piNotation___closed__17);
v___x_44_ = l_Lean_Parser_leadPrec;
v___x_45_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_46_ = l_Lean_Parser_leadingNode(v___x_45_, v___x_44_, v___x_43_);
return v___x_46_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__19(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_47_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__18, &lp_mathlib_PiNotation_piNotation___closed__18_once, _init_lp_mathlib_PiNotation_piNotation___closed__18);
v___x_48_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__3, &lp_mathlib_PiNotation_piNotation___closed__3_once, _init_lp_mathlib_PiNotation_piNotation___closed__3);
v___x_49_ = l_Lean_Parser_withAntiquot(v___x_48_, v___x_47_);
return v___x_49_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation___closed__20(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_50_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__19, &lp_mathlib_PiNotation_piNotation___closed__19_once, _init_lp_mathlib_PiNotation_piNotation___closed__19);
v___x_51_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_52_ = l_Lean_Parser_withCache(v___x_51_, v___x_50_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation(void){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation___closed__20, &lp_mathlib_PiNotation_piNotation___closed__20_once, _init_lp_mathlib_PiNotation_piNotation___closed__20);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter___lam__0(lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = l_Lean_PrettyPrinter_Formatter_pushLine___redArg(v___y_55_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter___lam__0___boxed(lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_PiNotation_piNotation_formatter___lam__0(v___y_60_, v___y_61_, v___y_62_, v___y_63_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
lean_dec(v___y_61_);
lean_dec_ref(v___y_60_);
return v_res_65_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation_formatter___closed__15(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_108_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation_formatter___closed__14));
v___x_109_ = l_Lean_Parser_leadPrec;
v___x_110_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_111_ = lean_alloc_closure((void*)(l_Lean_Parser_leadingNode_formatter___boxed), 8, 3);
lean_closure_set(v___x_111_, 0, v___x_110_);
lean_closure_set(v___x_111_, 1, v___x_109_);
lean_closure_set(v___x_111_, 2, v___x_108_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter(lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation_formatter___closed__1));
v___x_118_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation_formatter___closed__15, &lp_mathlib_PiNotation_piNotation_formatter___closed__15_once, _init_lp_mathlib_PiNotation_piNotation_formatter___closed__15);
v___x_119_ = l_Lean_PrettyPrinter_Formatter_orelse_formatter(v___x_117_, v___x_118_, v_a_112_, v_a_113_, v_a_114_, v_a_115_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_formatter___boxed(lean_object* v_a_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_PiNotation_piNotation_formatter(v_a_120_, v_a_121_, v_a_122_, v_a_123_);
lean_dec(v_a_123_);
lean_dec_ref(v_a_122_);
lean_dec(v_a_121_);
lean_dec_ref(v_a_120_);
return v_res_125_;
}
}
static lean_object* _init_lp_mathlib_PiNotation_piNotation_parenthesizer___closed__15(void){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_168_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation_parenthesizer___closed__14));
v___x_169_ = l_Lean_Parser_leadPrec;
v___x_170_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_171_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_Parenthesizer_leadingNode_parenthesizer___boxed), 8, 3);
lean_closure_set(v___x_171_, 0, v___x_170_);
lean_closure_set(v___x_171_, 1, v___x_169_);
lean_closure_set(v___x_171_, 2, v___x_168_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer(lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_177_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation_parenthesizer___closed__0));
v___x_178_ = lean_obj_once(&lp_mathlib_PiNotation_piNotation_parenthesizer___closed__15, &lp_mathlib_PiNotation_piNotation_parenthesizer___closed__15_once, _init_lp_mathlib_PiNotation_piNotation_parenthesizer___closed__15);
v___x_179_ = l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(v___x_177_, v___x_178_, v_a_172_, v_a_173_, v_a_174_, v_a_175_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_piNotation_parenthesizer___boxed(lean_object* v_a_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_PiNotation_piNotation_parenthesizer(v_a_180_, v_a_181_, v_a_182_, v_a_183_);
lean_dec(v_a_183_);
lean_dec_ref(v_a_182_);
lean_dec(v_a_181_);
lean_dec_ref(v_a_180_);
return v_res_185_;
}
}
static lean_object* _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__7(void){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_255_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__6));
v___x_256_ = l_String_toRawSubstring_x27(v___x_255_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9(void){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = l_Array_mkArray0(lean_box(0));
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1(lean_object* v_x_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v___x_276_; uint8_t v___x_277_; 
v___x_276_ = ((lean_object*)(lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1));
lean_inc(v_x_273_);
v___x_277_ = l_Lean_Syntax_isOfKind(v_x_273_, v___x_276_);
if (v___x_277_ == 0)
{
lean_object* v___x_278_; lean_object* v___x_279_; 
lean_dec(v_x_273_);
v___x_278_ = lean_box(1);
v___x_279_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_278_);
lean_ctor_set(v___x_279_, 1, v_a_275_);
return v___x_279_;
}
else
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; uint8_t v___x_283_; 
v___x_280_ = lean_unsigned_to_nat(1u);
v___x_281_ = l_Lean_Syntax_getArg(v_x_273_, v___x_280_);
v___x_282_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1));
lean_inc(v___x_281_);
v___x_283_ = l_Lean_Syntax_isOfKind(v___x_281_, v___x_282_);
if (v___x_283_ == 0)
{
lean_object* v___x_284_; uint8_t v___x_285_; 
v___x_284_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3));
v___x_285_ = l_Lean_Syntax_isOfKind(v___x_281_, v___x_284_);
if (v___x_285_ == 0)
{
lean_object* v___x_286_; lean_object* v___x_287_; 
lean_dec(v_x_273_);
v___x_286_ = lean_box(1);
v___x_287_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
lean_ctor_set(v___x_287_, 1, v_a_275_);
return v___x_287_;
}
else
{
lean_object* v_quotContext_288_; lean_object* v_currMacroScope_289_; lean_object* v_ref_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v_quotContext_288_ = lean_ctor_get(v_a_274_, 1);
v_currMacroScope_289_ = lean_ctor_get(v_a_274_, 2);
v_ref_290_ = lean_ctor_get(v_a_274_, 5);
v___x_291_ = lean_unsigned_to_nat(2u);
v___x_292_ = l_Lean_Syntax_getArg(v_x_273_, v___x_291_);
v___x_293_ = lean_unsigned_to_nat(4u);
v___x_294_ = l_Lean_Syntax_getArg(v_x_273_, v___x_293_);
lean_dec(v_x_273_);
v___x_295_ = l_Lean_SourceInfo_fromRef(v_ref_290_, v___x_283_);
v___x_296_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_297_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__4));
lean_inc_n(v___x_295_, 9);
v___x_298_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_298_, 0, v___x_295_);
lean_ctor_set(v___x_298_, 1, v___x_297_);
v___x_299_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_300_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__7, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__7_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__7);
v___x_301_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__8));
lean_inc(v_currMacroScope_289_);
lean_inc(v_quotContext_288_);
v___x_302_ = l_Lean_addMacroScope(v_quotContext_288_, v___x_301_, v_currMacroScope_289_);
v___x_303_ = lean_box(0);
v___x_304_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_304_, 0, v___x_295_);
lean_ctor_set(v___x_304_, 1, v___x_300_);
lean_ctor_set(v___x_304_, 2, v___x_302_);
lean_ctor_set(v___x_304_, 3, v___x_303_);
lean_inc_ref(v___x_304_);
v___x_305_ = l_Lean_Syntax_node1(v___x_295_, v___x_299_, v___x_304_);
v___x_306_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9);
v___x_307_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_307_, 0, v___x_295_);
lean_ctor_set(v___x_307_, 1, v___x_299_);
lean_ctor_set(v___x_307_, 2, v___x_306_);
v___x_308_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_309_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_295_);
lean_ctor_set(v___x_309_, 1, v___x_308_);
v___x_310_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12));
v___x_311_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14));
v___x_312_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__15));
v___x_313_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_295_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
v___x_314_ = l_Lean_Syntax_node3(v___x_295_, v___x_311_, v___x_313_, v___x_304_, v___x_292_);
v___x_315_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__16));
v___x_316_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_295_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
v___x_317_ = l_Lean_Syntax_node3(v___x_295_, v___x_310_, v___x_314_, v___x_316_, v___x_294_);
v___x_318_ = l_Lean_Syntax_node5(v___x_295_, v___x_296_, v___x_298_, v___x_305_, v___x_307_, v___x_309_, v___x_317_);
v___x_319_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_319_, 0, v___x_318_);
lean_ctor_set(v___x_319_, 1, v_a_275_);
return v___x_319_;
}
}
else
{
lean_object* v_ref_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; uint8_t v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v_ref_320_ = lean_ctor_get(v_a_274_, 5);
v___x_321_ = lean_unsigned_to_nat(2u);
v___x_322_ = l_Lean_Syntax_getArg(v_x_273_, v___x_321_);
v___x_323_ = lean_unsigned_to_nat(4u);
v___x_324_ = l_Lean_Syntax_getArg(v_x_273_, v___x_323_);
lean_dec(v_x_273_);
v___x_325_ = 0;
v___x_326_ = l_Lean_SourceInfo_fromRef(v_ref_320_, v___x_325_);
v___x_327_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_328_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__4));
lean_inc_n(v___x_326_, 8);
v___x_329_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_326_);
lean_ctor_set(v___x_329_, 1, v___x_328_);
v___x_330_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
lean_inc(v___x_281_);
v___x_331_ = l_Lean_Syntax_node1(v___x_326_, v___x_330_, v___x_281_);
v___x_332_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9);
v___x_333_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_333_, 0, v___x_326_);
lean_ctor_set(v___x_333_, 1, v___x_330_);
lean_ctor_set(v___x_333_, 2, v___x_332_);
v___x_334_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_335_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_335_, 0, v___x_326_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
v___x_336_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12));
v___x_337_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__14));
v___x_338_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__15));
v___x_339_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_326_);
lean_ctor_set(v___x_339_, 1, v___x_338_);
v___x_340_ = l_Lean_Syntax_node3(v___x_326_, v___x_337_, v___x_339_, v___x_281_, v___x_322_);
v___x_341_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__16));
v___x_342_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_326_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = l_Lean_Syntax_node3(v___x_326_, v___x_336_, v___x_340_, v___x_342_, v___x_324_);
v___x_344_ = l_Lean_Syntax_node5(v___x_326_, v___x_327_, v___x_329_, v___x_331_, v___x_333_, v___x_335_, v___x_343_);
v___x_345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_344_);
lean_ctor_set(v___x_345_, 1, v_a_275_);
return v___x_345_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___boxed(lean_object* v_x_346_, lean_object* v_a_347_, lean_object* v_a_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1(v_x_346_, v_a_347_, v_a_348_);
lean_dec_ref(v_a_347_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_replacePiNotation___redArg(lean_object* v_x_356_, lean_object* v_a_357_){
_start:
{
if (lean_obj_tag(v_x_356_) == 1)
{
lean_object* v_info_358_; lean_object* v_args_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_368_; 
v_info_358_ = lean_ctor_get(v_x_356_, 0);
v_args_359_ = lean_ctor_get(v_x_356_, 2);
v_isSharedCheck_368_ = !lean_is_exclusive(v_x_356_);
if (v_isSharedCheck_368_ == 0)
{
lean_object* v_unused_369_; 
v_unused_369_ = lean_ctor_get(v_x_356_, 1);
lean_dec(v_unused_369_);
v___x_361_ = v_x_356_;
v_isShared_362_ = v_isSharedCheck_368_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_args_359_);
lean_inc(v_info_358_);
lean_dec(v_x_356_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_368_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v___x_363_; lean_object* v___x_365_; 
v___x_363_ = ((lean_object*)(lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1));
if (v_isShared_362_ == 0)
{
lean_ctor_set(v___x_361_, 1, v___x_363_);
v___x_365_ = v___x_361_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_info_358_);
lean_ctor_set(v_reuseFailAlloc_367_, 1, v___x_363_);
lean_ctor_set(v_reuseFailAlloc_367_, 2, v_args_359_);
v___x_365_ = v_reuseFailAlloc_367_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
lean_object* v___x_366_; 
v___x_366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_366_, 0, v___x_365_);
lean_ctor_set(v___x_366_, 1, v_a_357_);
return v___x_366_;
}
}
}
else
{
lean_object* v___x_370_; 
lean_dec(v_x_356_);
v___x_370_ = l_Lean_Macro_throwUnsupported___redArg(v_a_357_);
return v___x_370_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_replacePiNotation(lean_object* v_x_371_, lean_object* v_a_372_, lean_object* v_a_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_mathlib_PiNotation_replacePiNotation___redArg(v_x_371_, v_a_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_replacePiNotation___boxed(lean_object* v_x_375_, lean_object* v_a_376_, lean_object* v_a_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_mathlib_PiNotation_replacePiNotation(v_x_375_, v_a_376_, v_a_377_);
lean_dec_ref(v_a_376_);
return v_res_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi___lam__0(lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = l_Lean_PrettyPrinter_Delaborator_delabForall(v___y_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_);
if (lean_obj_tag(v___x_480_) == 0)
{
lean_object* v_a_481_; lean_object* v___x_482_; uint8_t v___x_483_; 
v_a_481_ = lean_ctor_get(v___x_480_, 0);
lean_inc_n(v_a_481_, 2);
v___x_482_ = ((lean_object*)(lp_mathlib_PiNotation_replacePiNotation___redArg___closed__1));
v___x_483_ = l_Lean_Syntax_isOfKind(v_a_481_, v___x_482_);
if (v___x_483_ == 0)
{
lean_object* v___x_484_; uint8_t v___x_485_; 
v___x_484_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
lean_inc(v_a_481_);
v___x_485_ = l_Lean_Syntax_isOfKind(v_a_481_, v___x_484_);
if (v___x_485_ == 0)
{
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_486_; lean_object* v___x_487_; uint8_t v___x_488_; 
v___x_486_ = lean_unsigned_to_nat(1u);
v___x_487_ = l_Lean_Syntax_getArg(v_a_481_, v___x_486_);
lean_inc(v___x_487_);
v___x_488_ = l_Lean_Syntax_matchesNull(v___x_487_, v___x_486_);
if (v___x_488_ == 0)
{
lean_dec(v___x_487_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; uint8_t v___x_492_; 
v___x_489_ = lean_unsigned_to_nat(0u);
v___x_490_ = l_Lean_Syntax_getArg(v___x_487_, v___x_489_);
lean_dec(v___x_487_);
v___x_491_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__1));
lean_inc(v___x_490_);
v___x_492_ = l_Lean_Syntax_isOfKind(v___x_490_, v___x_491_);
if (v___x_492_ == 0)
{
lean_dec(v___x_490_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_493_; uint8_t v___x_494_; 
v___x_493_ = l_Lean_Syntax_getArg(v___x_490_, v___x_486_);
lean_inc(v___x_493_);
v___x_494_ = l_Lean_Syntax_matchesNull(v___x_493_, v___x_486_);
if (v___x_494_ == 0)
{
lean_dec(v___x_493_);
lean_dec(v___x_490_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_495_; lean_object* v___x_496_; uint8_t v___x_497_; 
v___x_495_ = l_Lean_Syntax_getArg(v___x_493_, v___x_489_);
lean_dec(v___x_493_);
v___x_496_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1));
lean_inc(v___x_495_);
v___x_497_ = l_Lean_Syntax_isOfKind(v___x_495_, v___x_496_);
if (v___x_497_ == 0)
{
lean_dec(v___x_495_);
lean_dec(v___x_490_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_498_; lean_object* v___x_499_; uint8_t v___x_500_; 
v___x_498_ = lean_unsigned_to_nat(2u);
v___x_499_ = l_Lean_Syntax_getArg(v___x_490_, v___x_498_);
v___x_500_ = l_Lean_Syntax_matchesNull(v___x_499_, v___x_498_);
if (v___x_500_ == 0)
{
lean_dec(v___x_495_);
lean_dec(v___x_490_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_501_; lean_object* v___x_502_; uint8_t v___x_503_; 
v___x_501_ = lean_unsigned_to_nat(3u);
v___x_502_ = l_Lean_Syntax_getArg(v___x_490_, v___x_501_);
lean_dec(v___x_490_);
v___x_503_ = l_Lean_Syntax_matchesNull(v___x_502_, v___x_489_);
if (v___x_503_ == 0)
{
lean_dec(v___x_495_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_504_; uint8_t v___x_505_; 
v___x_504_ = l_Lean_Syntax_getArg(v_a_481_, v___x_498_);
v___x_505_ = l_Lean_Syntax_matchesNull(v___x_504_, v___x_489_);
if (v___x_505_ == 0)
{
lean_dec(v___x_495_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; uint8_t v___x_509_; 
v___x_506_ = lean_unsigned_to_nat(4u);
v___x_507_ = l_Lean_Syntax_getArg(v_a_481_, v___x_506_);
lean_dec(v_a_481_);
v___x_508_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12));
lean_inc(v___x_507_);
v___x_509_ = l_Lean_Syntax_isOfKind(v___x_507_, v___x_508_);
if (v___x_509_ == 0)
{
lean_dec(v___x_507_);
lean_dec(v___x_495_);
return v___x_480_;
}
else
{
lean_object* v___x_510_; lean_object* v___x_511_; uint8_t v___x_512_; 
v___x_510_ = l_Lean_Syntax_getArg(v___x_507_, v___x_489_);
v___x_511_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__3));
lean_inc(v___x_510_);
v___x_512_ = l_Lean_Syntax_isOfKind(v___x_510_, v___x_511_);
if (v___x_512_ == 0)
{
lean_dec(v___x_510_);
lean_dec(v___x_507_);
lean_dec(v___x_495_);
return v___x_480_;
}
else
{
lean_object* v___x_513_; uint8_t v___x_514_; 
v___x_513_ = l_Lean_Syntax_getArg(v___x_510_, v___x_489_);
lean_inc(v___x_513_);
v___x_514_ = l_Lean_Syntax_isOfKind(v___x_513_, v___x_496_);
if (v___x_514_ == 0)
{
lean_dec(v___x_513_);
lean_dec(v___x_510_);
lean_dec(v___x_507_);
lean_dec(v___x_495_);
return v___x_480_;
}
else
{
uint8_t v___x_515_; 
v___x_515_ = l_Lean_Syntax_structEq(v___x_495_, v___x_513_);
lean_dec(v___x_513_);
if (v___x_515_ == 0)
{
lean_dec(v___x_510_);
lean_dec(v___x_507_);
lean_dec(v___x_495_);
return v___x_480_;
}
else
{
lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_536_; 
v_isSharedCheck_536_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; 
v_unused_537_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_537_);
v___x_517_ = v___x_480_;
v_isShared_518_ = v_isSharedCheck_536_;
goto v_resetjp_516_;
}
else
{
lean_dec(v___x_480_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_536_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v_ref_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_534_; 
v_ref_519_ = lean_ctor_get(v___y_477_, 5);
v___x_520_ = l_Lean_Syntax_getArg(v___x_510_, v___x_498_);
lean_dec(v___x_510_);
v___x_521_ = l_Lean_Syntax_getArg(v___x_507_, v___x_498_);
lean_dec(v___x_507_);
v___x_522_ = l_Lean_SourceInfo_fromRef(v_ref_519_, v___x_483_);
v___x_523_ = ((lean_object*)(lp_mathlib_PiNotation_term_u03a0_____x2c___00__closed__1));
v___x_524_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__4));
lean_inc_n(v___x_522_, 4);
v___x_525_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_522_);
lean_ctor_set(v___x_525_, 1, v___x_524_);
v___x_526_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__5));
v___x_527_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__6));
v___x_528_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_528_, 0, v___x_522_);
lean_ctor_set(v___x_528_, 1, v___x_527_);
v___x_529_ = l_Lean_Syntax_node2(v___x_522_, v___x_526_, v___x_528_, v___x_520_);
v___x_530_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_531_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_522_);
lean_ctor_set(v___x_531_, 1, v___x_530_);
v___x_532_ = l_Lean_Syntax_node5(v___x_522_, v___x_523_, v___x_525_, v___x_495_, v___x_529_, v___x_531_, v___x_521_);
if (v_isShared_518_ == 0)
{
lean_ctor_set(v___x_517_, 0, v___x_532_);
v___x_534_ = v___x_517_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_535_; 
v_reuseFailAlloc_535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_535_, 0, v___x_532_);
v___x_534_ = v_reuseFailAlloc_535_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
return v___x_534_;
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
lean_object* v___x_538_; lean_object* v___x_539_; uint8_t v___x_540_; 
v___x_538_ = lean_unsigned_to_nat(1u);
v___x_539_ = l_Lean_Syntax_getArg(v_a_481_, v___x_538_);
lean_inc(v___x_539_);
v___x_540_ = l_Lean_Syntax_matchesNull(v___x_539_, v___x_538_);
if (v___x_540_ == 0)
{
lean_dec(v___x_539_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; uint8_t v___x_544_; 
v___x_541_ = lean_unsigned_to_nat(0u);
v___x_542_ = l_Lean_Syntax_getArg(v___x_539_, v___x_541_);
lean_dec(v___x_539_);
v___x_543_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__1));
lean_inc(v___x_542_);
v___x_544_ = l_Lean_Syntax_isOfKind(v___x_542_, v___x_543_);
if (v___x_544_ == 0)
{
lean_dec(v___x_542_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_545_; uint8_t v___x_546_; 
v___x_545_ = l_Lean_Syntax_getArg(v___x_542_, v___x_538_);
lean_inc(v___x_545_);
v___x_546_ = l_Lean_Syntax_matchesNull(v___x_545_, v___x_538_);
if (v___x_546_ == 0)
{
lean_dec(v___x_545_);
lean_dec(v___x_542_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_547_; lean_object* v___x_548_; uint8_t v___x_549_; 
v___x_547_ = l_Lean_Syntax_getArg(v___x_545_, v___x_541_);
lean_dec(v___x_545_);
v___x_548_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1));
lean_inc(v___x_547_);
v___x_549_ = l_Lean_Syntax_isOfKind(v___x_547_, v___x_548_);
if (v___x_549_ == 0)
{
lean_dec(v___x_547_);
lean_dec(v___x_542_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_550_; lean_object* v___x_551_; uint8_t v___x_552_; 
v___x_550_ = lean_unsigned_to_nat(2u);
v___x_551_ = l_Lean_Syntax_getArg(v___x_542_, v___x_550_);
v___x_552_ = l_Lean_Syntax_matchesNull(v___x_551_, v___x_550_);
if (v___x_552_ == 0)
{
lean_dec(v___x_547_);
lean_dec(v___x_542_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_553_; lean_object* v___x_554_; uint8_t v___x_555_; 
v___x_553_ = lean_unsigned_to_nat(3u);
v___x_554_ = l_Lean_Syntax_getArg(v___x_542_, v___x_553_);
lean_dec(v___x_542_);
v___x_555_ = l_Lean_Syntax_matchesNull(v___x_554_, v___x_541_);
if (v___x_555_ == 0)
{
lean_dec(v___x_547_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_556_; uint8_t v___x_557_; 
v___x_556_ = l_Lean_Syntax_getArg(v_a_481_, v___x_550_);
v___x_557_ = l_Lean_Syntax_matchesNull(v___x_556_, v___x_541_);
if (v___x_557_ == 0)
{
lean_dec(v___x_547_);
lean_dec(v_a_481_);
return v___x_480_;
}
else
{
lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; uint8_t v___x_561_; 
v___x_558_ = lean_unsigned_to_nat(4u);
v___x_559_ = l_Lean_Syntax_getArg(v_a_481_, v___x_558_);
lean_dec(v_a_481_);
v___x_560_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__12));
lean_inc(v___x_559_);
v___x_561_ = l_Lean_Syntax_isOfKind(v___x_559_, v___x_560_);
if (v___x_561_ == 0)
{
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_562_; lean_object* v___x_563_; uint8_t v___x_564_; 
v___x_562_ = l_Lean_Syntax_getArg(v___x_559_, v___x_541_);
v___x_563_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__3));
lean_inc(v___x_562_);
v___x_564_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_563_);
if (v___x_564_ == 0)
{
lean_object* v___x_565_; uint8_t v___x_566_; 
v___x_565_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__8));
lean_inc(v___x_562_);
v___x_566_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_565_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; uint8_t v___x_568_; 
v___x_567_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__10));
lean_inc(v___x_562_);
v___x_568_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_567_);
if (v___x_568_ == 0)
{
lean_object* v___x_569_; uint8_t v___x_570_; 
v___x_569_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__12));
lean_inc(v___x_562_);
v___x_570_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_569_);
if (v___x_570_ == 0)
{
lean_object* v___x_571_; uint8_t v___x_572_; 
v___x_571_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__14));
lean_inc(v___x_562_);
v___x_572_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; uint8_t v___x_574_; 
v___x_573_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__16));
lean_inc(v___x_562_);
v___x_574_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_573_);
if (v___x_574_ == 0)
{
lean_object* v___x_575_; uint8_t v___x_576_; 
v___x_575_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__18));
lean_inc(v___x_562_);
v___x_576_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_575_);
if (v___x_576_ == 0)
{
lean_object* v___x_577_; uint8_t v___x_578_; 
v___x_577_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__20));
lean_inc(v___x_562_);
v___x_578_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_577_);
if (v___x_578_ == 0)
{
lean_object* v___x_579_; uint8_t v___x_580_; 
v___x_579_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__22));
lean_inc(v___x_562_);
v___x_580_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_579_);
if (v___x_580_ == 0)
{
lean_object* v___x_581_; uint8_t v___x_582_; 
v___x_581_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__24));
lean_inc(v___x_562_);
v___x_582_ = l_Lean_Syntax_isOfKind(v___x_562_, v___x_581_);
if (v___x_582_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_583_; uint8_t v___x_584_; 
v___x_583_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_583_);
v___x_584_ = l_Lean_Syntax_isOfKind(v___x_583_, v___x_548_);
if (v___x_584_ == 0)
{
lean_dec(v___x_583_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_585_; 
v___x_585_ = l_Lean_Syntax_structEq(v___x_547_, v___x_583_);
lean_dec(v___x_583_);
if (v___x_585_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_608_; 
v_isSharedCheck_608_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_608_ == 0)
{
lean_object* v_unused_609_; 
v_unused_609_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_609_);
v___x_587_ = v___x_480_;
v_isShared_588_ = v_isSharedCheck_608_;
goto v_resetjp_586_;
}
else
{
lean_dec(v___x_480_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_608_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v_ref_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_606_; 
v_ref_589_ = lean_ctor_get(v___y_477_, 5);
v___x_590_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_591_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_592_ = l_Lean_SourceInfo_fromRef(v_ref_589_, v___x_580_);
v___x_593_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_594_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_592_, 5);
v___x_595_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_592_);
lean_ctor_set(v___x_595_, 1, v___x_594_);
v___x_596_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_597_ = l_Lean_Syntax_node1(v___x_592_, v___x_596_, v___x_547_);
v___x_598_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__30));
v___x_599_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__31));
v___x_600_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_600_, 0, v___x_592_);
lean_ctor_set(v___x_600_, 1, v___x_599_);
v___x_601_ = l_Lean_Syntax_node2(v___x_592_, v___x_598_, v___x_600_, v___x_590_);
v___x_602_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_603_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_603_, 0, v___x_592_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
v___x_604_ = l_Lean_Syntax_node5(v___x_592_, v___x_593_, v___x_595_, v___x_597_, v___x_601_, v___x_603_, v___x_591_);
if (v_isShared_588_ == 0)
{
lean_ctor_set(v___x_587_, 0, v___x_604_);
v___x_606_ = v___x_587_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_604_);
v___x_606_ = v_reuseFailAlloc_607_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
return v___x_606_;
}
}
}
}
}
}
else
{
lean_object* v___x_610_; uint8_t v___x_611_; 
v___x_610_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_610_);
v___x_611_ = l_Lean_Syntax_isOfKind(v___x_610_, v___x_548_);
if (v___x_611_ == 0)
{
lean_dec(v___x_610_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_612_; 
v___x_612_ = l_Lean_Syntax_structEq(v___x_547_, v___x_610_);
lean_dec(v___x_610_);
if (v___x_612_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_614_; uint8_t v_isShared_615_; uint8_t v_isSharedCheck_635_; 
v_isSharedCheck_635_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_635_ == 0)
{
lean_object* v_unused_636_; 
v_unused_636_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_636_);
v___x_614_ = v___x_480_;
v_isShared_615_ = v_isSharedCheck_635_;
goto v_resetjp_613_;
}
else
{
lean_dec(v___x_480_);
v___x_614_ = lean_box(0);
v_isShared_615_ = v_isSharedCheck_635_;
goto v_resetjp_613_;
}
v_resetjp_613_:
{
lean_object* v_ref_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_633_; 
v_ref_616_ = lean_ctor_get(v___y_477_, 5);
v___x_617_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_618_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_619_ = l_Lean_SourceInfo_fromRef(v_ref_616_, v___x_578_);
v___x_620_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_621_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_619_, 5);
v___x_622_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_619_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_624_ = l_Lean_Syntax_node1(v___x_619_, v___x_623_, v___x_547_);
v___x_625_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__33));
v___x_626_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__34));
v___x_627_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_627_, 0, v___x_619_);
lean_ctor_set(v___x_627_, 1, v___x_626_);
v___x_628_ = l_Lean_Syntax_node2(v___x_619_, v___x_625_, v___x_627_, v___x_617_);
v___x_629_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_630_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_619_);
lean_ctor_set(v___x_630_, 1, v___x_629_);
v___x_631_ = l_Lean_Syntax_node5(v___x_619_, v___x_620_, v___x_622_, v___x_624_, v___x_628_, v___x_630_, v___x_618_);
if (v_isShared_615_ == 0)
{
lean_ctor_set(v___x_614_, 0, v___x_631_);
v___x_633_ = v___x_614_;
goto v_reusejp_632_;
}
else
{
lean_object* v_reuseFailAlloc_634_; 
v_reuseFailAlloc_634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_634_, 0, v___x_631_);
v___x_633_ = v_reuseFailAlloc_634_;
goto v_reusejp_632_;
}
v_reusejp_632_:
{
return v___x_633_;
}
}
}
}
}
}
else
{
lean_object* v___x_637_; uint8_t v___x_638_; 
v___x_637_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_637_);
v___x_638_ = l_Lean_Syntax_isOfKind(v___x_637_, v___x_548_);
if (v___x_638_ == 0)
{
lean_dec(v___x_637_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_639_; 
v___x_639_ = l_Lean_Syntax_structEq(v___x_547_, v___x_637_);
lean_dec(v___x_637_);
if (v___x_639_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_641_; uint8_t v_isShared_642_; uint8_t v_isSharedCheck_662_; 
v_isSharedCheck_662_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_662_ == 0)
{
lean_object* v_unused_663_; 
v_unused_663_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_663_);
v___x_641_ = v___x_480_;
v_isShared_642_ = v_isSharedCheck_662_;
goto v_resetjp_640_;
}
else
{
lean_dec(v___x_480_);
v___x_641_ = lean_box(0);
v_isShared_642_ = v_isSharedCheck_662_;
goto v_resetjp_640_;
}
v_resetjp_640_:
{
lean_object* v_ref_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_660_; 
v_ref_643_ = lean_ctor_get(v___y_477_, 5);
v___x_644_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_645_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_646_ = l_Lean_SourceInfo_fromRef(v_ref_643_, v___x_576_);
v___x_647_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_648_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_646_, 5);
v___x_649_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_649_, 0, v___x_646_);
lean_ctor_set(v___x_649_, 1, v___x_648_);
v___x_650_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_651_ = l_Lean_Syntax_node1(v___x_646_, v___x_650_, v___x_547_);
v___x_652_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__36));
v___x_653_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__37));
v___x_654_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_654_, 0, v___x_646_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = l_Lean_Syntax_node2(v___x_646_, v___x_652_, v___x_654_, v___x_644_);
v___x_656_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_657_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_646_);
lean_ctor_set(v___x_657_, 1, v___x_656_);
v___x_658_ = l_Lean_Syntax_node5(v___x_646_, v___x_647_, v___x_649_, v___x_651_, v___x_655_, v___x_657_, v___x_645_);
if (v_isShared_642_ == 0)
{
lean_ctor_set(v___x_641_, 0, v___x_658_);
v___x_660_ = v___x_641_;
goto v_reusejp_659_;
}
else
{
lean_object* v_reuseFailAlloc_661_; 
v_reuseFailAlloc_661_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_661_, 0, v___x_658_);
v___x_660_ = v_reuseFailAlloc_661_;
goto v_reusejp_659_;
}
v_reusejp_659_:
{
return v___x_660_;
}
}
}
}
}
}
else
{
lean_object* v___x_664_; uint8_t v___x_665_; 
v___x_664_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_664_);
v___x_665_ = l_Lean_Syntax_isOfKind(v___x_664_, v___x_548_);
if (v___x_665_ == 0)
{
lean_dec(v___x_664_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_666_; 
v___x_666_ = l_Lean_Syntax_structEq(v___x_547_, v___x_664_);
lean_dec(v___x_664_);
if (v___x_666_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_689_; 
v_isSharedCheck_689_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_689_ == 0)
{
lean_object* v_unused_690_; 
v_unused_690_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_690_);
v___x_668_ = v___x_480_;
v_isShared_669_ = v_isSharedCheck_689_;
goto v_resetjp_667_;
}
else
{
lean_dec(v___x_480_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_689_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v_ref_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_687_; 
v_ref_670_ = lean_ctor_get(v___y_477_, 5);
v___x_671_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_672_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_673_ = l_Lean_SourceInfo_fromRef(v_ref_670_, v___x_574_);
v___x_674_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_675_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_673_, 5);
v___x_676_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_676_, 0, v___x_673_);
lean_ctor_set(v___x_676_, 1, v___x_675_);
v___x_677_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_678_ = l_Lean_Syntax_node1(v___x_673_, v___x_677_, v___x_547_);
v___x_679_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__39));
v___x_680_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__40));
v___x_681_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_673_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
v___x_682_ = l_Lean_Syntax_node2(v___x_673_, v___x_679_, v___x_681_, v___x_671_);
v___x_683_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_684_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_684_, 0, v___x_673_);
lean_ctor_set(v___x_684_, 1, v___x_683_);
v___x_685_ = l_Lean_Syntax_node5(v___x_673_, v___x_674_, v___x_676_, v___x_678_, v___x_682_, v___x_684_, v___x_672_);
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 0, v___x_685_);
v___x_687_ = v___x_668_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v___x_685_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
}
}
}
}
else
{
lean_object* v___x_691_; uint8_t v___x_692_; 
v___x_691_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_691_);
v___x_692_ = l_Lean_Syntax_isOfKind(v___x_691_, v___x_548_);
if (v___x_692_ == 0)
{
lean_dec(v___x_691_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_693_; 
v___x_693_ = l_Lean_Syntax_structEq(v___x_547_, v___x_691_);
lean_dec(v___x_691_);
if (v___x_693_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_716_; 
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_716_ == 0)
{
lean_object* v_unused_717_; 
v_unused_717_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_717_);
v___x_695_ = v___x_480_;
v_isShared_696_ = v_isSharedCheck_716_;
goto v_resetjp_694_;
}
else
{
lean_dec(v___x_480_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_716_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v_ref_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_714_; 
v_ref_697_ = lean_ctor_get(v___y_477_, 5);
v___x_698_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_699_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_700_ = l_Lean_SourceInfo_fromRef(v_ref_697_, v___x_572_);
v___x_701_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_702_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_700_, 5);
v___x_703_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_703_, 0, v___x_700_);
lean_ctor_set(v___x_703_, 1, v___x_702_);
v___x_704_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_705_ = l_Lean_Syntax_node1(v___x_700_, v___x_704_, v___x_547_);
v___x_706_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__42));
v___x_707_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__43));
v___x_708_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_708_, 0, v___x_700_);
lean_ctor_set(v___x_708_, 1, v___x_707_);
v___x_709_ = l_Lean_Syntax_node2(v___x_700_, v___x_706_, v___x_708_, v___x_698_);
v___x_710_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_711_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_711_, 0, v___x_700_);
lean_ctor_set(v___x_711_, 1, v___x_710_);
v___x_712_ = l_Lean_Syntax_node5(v___x_700_, v___x_701_, v___x_703_, v___x_705_, v___x_709_, v___x_711_, v___x_699_);
if (v_isShared_696_ == 0)
{
lean_ctor_set(v___x_695_, 0, v___x_712_);
v___x_714_ = v___x_695_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v___x_712_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
}
}
}
else
{
lean_object* v___x_718_; uint8_t v___x_719_; 
v___x_718_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_718_);
v___x_719_ = l_Lean_Syntax_isOfKind(v___x_718_, v___x_548_);
if (v___x_719_ == 0)
{
lean_dec(v___x_718_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_720_; 
v___x_720_ = l_Lean_Syntax_structEq(v___x_547_, v___x_718_);
lean_dec(v___x_718_);
if (v___x_720_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_722_; uint8_t v_isShared_723_; uint8_t v_isSharedCheck_743_; 
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_743_ == 0)
{
lean_object* v_unused_744_; 
v_unused_744_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_744_);
v___x_722_ = v___x_480_;
v_isShared_723_ = v_isSharedCheck_743_;
goto v_resetjp_721_;
}
else
{
lean_dec(v___x_480_);
v___x_722_ = lean_box(0);
v_isShared_723_ = v_isSharedCheck_743_;
goto v_resetjp_721_;
}
v_resetjp_721_:
{
lean_object* v_ref_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_741_; 
v_ref_724_ = lean_ctor_get(v___y_477_, 5);
v___x_725_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_726_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_727_ = l_Lean_SourceInfo_fromRef(v_ref_724_, v___x_570_);
v___x_728_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_729_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_727_, 5);
v___x_730_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_727_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_732_ = l_Lean_Syntax_node1(v___x_727_, v___x_731_, v___x_547_);
v___x_733_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__45));
v___x_734_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__46));
v___x_735_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_735_, 0, v___x_727_);
lean_ctor_set(v___x_735_, 1, v___x_734_);
v___x_736_ = l_Lean_Syntax_node2(v___x_727_, v___x_733_, v___x_735_, v___x_725_);
v___x_737_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_738_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_738_, 0, v___x_727_);
lean_ctor_set(v___x_738_, 1, v___x_737_);
v___x_739_ = l_Lean_Syntax_node5(v___x_727_, v___x_728_, v___x_730_, v___x_732_, v___x_736_, v___x_738_, v___x_726_);
if (v_isShared_723_ == 0)
{
lean_ctor_set(v___x_722_, 0, v___x_739_);
v___x_741_ = v___x_722_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v___x_739_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
}
}
else
{
lean_object* v___x_745_; uint8_t v___x_746_; 
v___x_745_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_745_);
v___x_746_ = l_Lean_Syntax_isOfKind(v___x_745_, v___x_548_);
if (v___x_746_ == 0)
{
lean_dec(v___x_745_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_747_; 
v___x_747_ = l_Lean_Syntax_structEq(v___x_547_, v___x_745_);
lean_dec(v___x_745_);
if (v___x_747_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_770_; 
v_isSharedCheck_770_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_770_ == 0)
{
lean_object* v_unused_771_; 
v_unused_771_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_771_);
v___x_749_ = v___x_480_;
v_isShared_750_ = v_isSharedCheck_770_;
goto v_resetjp_748_;
}
else
{
lean_dec(v___x_480_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_770_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v_ref_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_768_; 
v_ref_751_ = lean_ctor_get(v___y_477_, 5);
v___x_752_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_753_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_754_ = l_Lean_SourceInfo_fromRef(v_ref_751_, v___x_568_);
v___x_755_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_756_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_754_, 5);
v___x_757_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_757_, 0, v___x_754_);
lean_ctor_set(v___x_757_, 1, v___x_756_);
v___x_758_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_759_ = l_Lean_Syntax_node1(v___x_754_, v___x_758_, v___x_547_);
v___x_760_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__48));
v___x_761_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__49));
v___x_762_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_762_, 0, v___x_754_);
lean_ctor_set(v___x_762_, 1, v___x_761_);
v___x_763_ = l_Lean_Syntax_node2(v___x_754_, v___x_760_, v___x_762_, v___x_752_);
v___x_764_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_765_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_765_, 0, v___x_754_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
v___x_766_ = l_Lean_Syntax_node5(v___x_754_, v___x_755_, v___x_757_, v___x_759_, v___x_763_, v___x_765_, v___x_753_);
if (v_isShared_750_ == 0)
{
lean_ctor_set(v___x_749_, 0, v___x_766_);
v___x_768_ = v___x_749_;
goto v_reusejp_767_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v___x_766_);
v___x_768_ = v_reuseFailAlloc_769_;
goto v_reusejp_767_;
}
v_reusejp_767_:
{
return v___x_768_;
}
}
}
}
}
}
else
{
lean_object* v___x_772_; uint8_t v___x_773_; 
v___x_772_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_772_);
v___x_773_ = l_Lean_Syntax_isOfKind(v___x_772_, v___x_548_);
if (v___x_773_ == 0)
{
lean_dec(v___x_772_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_774_; 
v___x_774_ = l_Lean_Syntax_structEq(v___x_547_, v___x_772_);
lean_dec(v___x_772_);
if (v___x_774_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_797_; 
v_isSharedCheck_797_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_797_ == 0)
{
lean_object* v_unused_798_; 
v_unused_798_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_798_);
v___x_776_ = v___x_480_;
v_isShared_777_ = v_isSharedCheck_797_;
goto v_resetjp_775_;
}
else
{
lean_dec(v___x_480_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_797_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v_ref_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_795_; 
v_ref_778_ = lean_ctor_get(v___y_477_, 5);
v___x_779_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_780_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_781_ = l_Lean_SourceInfo_fromRef(v_ref_778_, v___x_566_);
v___x_782_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_783_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_781_, 5);
v___x_784_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_784_, 0, v___x_781_);
lean_ctor_set(v___x_784_, 1, v___x_783_);
v___x_785_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_786_ = l_Lean_Syntax_node1(v___x_781_, v___x_785_, v___x_547_);
v___x_787_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__51));
v___x_788_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__52));
v___x_789_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_781_);
lean_ctor_set(v___x_789_, 1, v___x_788_);
v___x_790_ = l_Lean_Syntax_node2(v___x_781_, v___x_787_, v___x_789_, v___x_779_);
v___x_791_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_792_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_792_, 0, v___x_781_);
lean_ctor_set(v___x_792_, 1, v___x_791_);
v___x_793_ = l_Lean_Syntax_node5(v___x_781_, v___x_782_, v___x_784_, v___x_786_, v___x_790_, v___x_792_, v___x_780_);
if (v_isShared_777_ == 0)
{
lean_ctor_set(v___x_776_, 0, v___x_793_);
v___x_795_ = v___x_776_;
goto v_reusejp_794_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v___x_793_);
v___x_795_ = v_reuseFailAlloc_796_;
goto v_reusejp_794_;
}
v_reusejp_794_:
{
return v___x_795_;
}
}
}
}
}
}
else
{
lean_object* v___x_799_; uint8_t v___x_800_; 
v___x_799_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_799_);
v___x_800_ = l_Lean_Syntax_isOfKind(v___x_799_, v___x_548_);
if (v___x_800_ == 0)
{
lean_dec(v___x_799_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_801_; 
v___x_801_ = l_Lean_Syntax_structEq(v___x_547_, v___x_799_);
lean_dec(v___x_799_);
if (v___x_801_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_824_; 
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_824_ == 0)
{
lean_object* v_unused_825_; 
v_unused_825_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_825_);
v___x_803_ = v___x_480_;
v_isShared_804_ = v_isSharedCheck_824_;
goto v_resetjp_802_;
}
else
{
lean_dec(v___x_480_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_824_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v_ref_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_822_; 
v_ref_805_ = lean_ctor_get(v___y_477_, 5);
v___x_806_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_807_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_808_ = l_Lean_SourceInfo_fromRef(v_ref_805_, v___x_564_);
v___x_809_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_810_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_808_, 5);
v___x_811_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_811_, 0, v___x_808_);
lean_ctor_set(v___x_811_, 1, v___x_810_);
v___x_812_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_813_ = l_Lean_Syntax_node1(v___x_808_, v___x_812_, v___x_547_);
v___x_814_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__54));
v___x_815_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__55));
v___x_816_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_816_, 0, v___x_808_);
lean_ctor_set(v___x_816_, 1, v___x_815_);
v___x_817_ = l_Lean_Syntax_node2(v___x_808_, v___x_814_, v___x_816_, v___x_806_);
v___x_818_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_819_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_819_, 0, v___x_808_);
lean_ctor_set(v___x_819_, 1, v___x_818_);
v___x_820_ = l_Lean_Syntax_node5(v___x_808_, v___x_809_, v___x_811_, v___x_813_, v___x_817_, v___x_819_, v___x_807_);
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 0, v___x_820_);
v___x_822_ = v___x_803_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_820_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
}
}
}
else
{
lean_object* v___x_826_; uint8_t v___x_827_; 
v___x_826_ = l_Lean_Syntax_getArg(v___x_562_, v___x_541_);
lean_inc(v___x_826_);
v___x_827_ = l_Lean_Syntax_isOfKind(v___x_826_, v___x_548_);
if (v___x_827_ == 0)
{
lean_dec(v___x_826_);
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
uint8_t v___x_828_; 
v___x_828_ = l_Lean_Syntax_structEq(v___x_547_, v___x_826_);
lean_dec(v___x_826_);
if (v___x_828_ == 0)
{
lean_dec(v___x_562_);
lean_dec(v___x_559_);
lean_dec(v___x_547_);
return v___x_480_;
}
else
{
lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_852_; 
v_isSharedCheck_852_ = !lean_is_exclusive(v___x_480_);
if (v_isSharedCheck_852_ == 0)
{
lean_object* v_unused_853_; 
v_unused_853_ = lean_ctor_get(v___x_480_, 0);
lean_dec(v_unused_853_);
v___x_830_ = v___x_480_;
v_isShared_831_ = v_isSharedCheck_852_;
goto v_resetjp_829_;
}
else
{
lean_dec(v___x_480_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_852_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
lean_object* v_ref_832_; lean_object* v___x_833_; lean_object* v___x_834_; uint8_t v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_850_; 
v_ref_832_ = lean_ctor_get(v___y_477_, 5);
v___x_833_ = l_Lean_Syntax_getArg(v___x_562_, v___x_550_);
lean_dec(v___x_562_);
v___x_834_ = l_Lean_Syntax_getArg(v___x_559_, v___x_550_);
lean_dec(v___x_559_);
v___x_835_ = 0;
v___x_836_ = l_Lean_SourceInfo_fromRef(v_ref_832_, v___x_835_);
v___x_837_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__26));
v___x_838_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__27));
lean_inc_n(v___x_836_, 5);
v___x_839_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_839_, 0, v___x_836_);
lean_ctor_set(v___x_839_, 1, v___x_838_);
v___x_840_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_841_ = l_Lean_Syntax_node1(v___x_836_, v___x_840_, v___x_547_);
v___x_842_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__5));
v___x_843_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__6));
v___x_844_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_844_, 0, v___x_836_);
lean_ctor_set(v___x_844_, 1, v___x_843_);
v___x_845_ = l_Lean_Syntax_node2(v___x_836_, v___x_842_, v___x_844_, v___x_833_);
v___x_846_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_847_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_847_, 0, v___x_836_);
lean_ctor_set(v___x_847_, 1, v___x_846_);
v___x_848_ = l_Lean_Syntax_node5(v___x_836_, v___x_837_, v___x_839_, v___x_841_, v___x_845_, v___x_847_, v___x_834_);
if (v_isShared_831_ == 0)
{
lean_ctor_set(v___x_830_, 0, v___x_848_);
v___x_850_ = v___x_830_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v___x_848_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
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
return v___x_480_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi___lam__0___boxed(lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_){
_start:
{
lean_object* v_res_861_; 
v_res_861_ = lp_mathlib_PiNotation_delabPi___lam__0(v___y_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_);
lean_dec(v___y_859_);
lean_dec_ref(v___y_858_);
lean_dec(v___y_857_);
lean_dec_ref(v___y_856_);
lean_dec(v___y_855_);
lean_dec_ref(v___y_854_);
return v_res_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi(lean_object* v_a_868_, lean_object* v_a_869_, lean_object* v_a_870_, lean_object* v_a_871_, lean_object* v_a_872_, lean_object* v_a_873_){
_start:
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; 
v___x_875_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___closed__1));
v___x_876_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___closed__3));
v___x_877_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_875_, v___x_876_, v_a_868_, v_a_869_, v_a_870_, v_a_871_, v_a_872_, v_a_873_);
return v___x_877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi___boxed(lean_object* v_a_878_, lean_object* v_a_879_, lean_object* v_a_880_, lean_object* v_a_881_, lean_object* v_a_882_, lean_object* v_a_883_, lean_object* v_a_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_mathlib_PiNotation_delabPi(v_a_878_, v_a_879_, v_a_880_, v_a_881_, v_a_882_, v_a_883_);
lean_dec(v_a_883_);
lean_dec_ref(v_a_882_);
lean_dec(v_a_881_);
lean_dec_ref(v_a_880_);
lean_dec(v_a_879_);
lean_dec_ref(v_a_878_);
return v_res_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27___lam__0(lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
lean_object* v_stx_900_; lean_object* v___y_940_; lean_object* v___x_961_; 
v___x_961_ = lp_mathlib_PiNotation_delabPi(v___y_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
if (lean_obj_tag(v___x_961_) == 0)
{
v___y_940_ = v___x_961_;
goto v___jp_939_;
}
else
{
lean_object* v_a_962_; lean_object* v___x_963_; uint8_t v___y_965_; uint8_t v___x_969_; 
v_a_962_ = lean_ctor_get(v___x_961_, 0);
lean_inc(v_a_962_);
v___x_963_ = l_Lean_PrettyPrinter_Delaborator_delabFailureId;
v___x_969_ = l_Lean_Exception_isInterrupt(v_a_962_);
if (v___x_969_ == 0)
{
uint8_t v___x_970_; 
lean_inc(v_a_962_);
v___x_970_ = l_Lean_Exception_isRuntime(v_a_962_);
v___y_965_ = v___x_970_;
goto v___jp_964_;
}
else
{
v___y_965_ = v___x_969_;
goto v___jp_964_;
}
v___jp_964_:
{
if (v___y_965_ == 0)
{
if (lean_obj_tag(v_a_962_) == 0)
{
lean_dec_ref_known(v_a_962_, 2);
v___y_940_ = v___x_961_;
goto v___jp_939_;
}
else
{
lean_object* v_id_966_; uint8_t v___x_967_; 
v_id_966_ = lean_ctor_get(v_a_962_, 0);
lean_inc(v_id_966_);
lean_dec_ref_known(v_a_962_, 2);
v___x_967_ = l_Lean_instBEqInternalExceptionId_beq(v___x_963_, v_id_966_);
lean_dec(v_id_966_);
if (v___x_967_ == 0)
{
v___y_940_ = v___x_961_;
goto v___jp_939_;
}
else
{
lean_object* v___x_968_; 
lean_dec_ref_known(v___x_961_, 1);
v___x_968_ = l_Lean_PrettyPrinter_Delaborator_delabForall(v___y_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
v___y_940_ = v___x_968_;
goto v___jp_939_;
}
}
}
else
{
lean_dec(v_a_962_);
v___y_940_ = v___x_961_;
goto v___jp_939_;
}
}
}
v___jp_899_:
{
lean_object* v___x_901_; uint8_t v___x_902_; 
v___x_901_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
lean_inc(v_stx_900_);
v___x_902_ = l_Lean_Syntax_isOfKind(v_stx_900_, v___x_901_);
if (v___x_902_ == 0)
{
lean_object* v___x_903_; 
v___x_903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_903_, 0, v_stx_900_);
return v___x_903_;
}
else
{
lean_object* v___x_904_; lean_object* v___x_905_; uint8_t v___x_906_; 
v___x_904_ = lean_unsigned_to_nat(1u);
v___x_905_ = l_Lean_Syntax_getArg(v_stx_900_, v___x_904_);
lean_inc(v___x_905_);
v___x_906_ = l_Lean_Syntax_matchesNull(v___x_905_, v___x_904_);
if (v___x_906_ == 0)
{
lean_object* v___x_907_; 
lean_dec(v___x_905_);
v___x_907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_907_, 0, v_stx_900_);
return v___x_907_;
}
else
{
lean_object* v___x_908_; lean_object* v___x_909_; lean_object* v___x_910_; uint8_t v___x_911_; 
v___x_908_ = lean_unsigned_to_nat(0u);
v___x_909_ = lean_unsigned_to_nat(2u);
v___x_910_ = l_Lean_Syntax_getArg(v_stx_900_, v___x_909_);
v___x_911_ = l_Lean_Syntax_matchesNull(v___x_910_, v___x_908_);
if (v___x_911_ == 0)
{
lean_object* v___x_912_; 
lean_dec(v___x_905_);
v___x_912_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_912_, 0, v_stx_900_);
return v___x_912_;
}
else
{
lean_object* v___x_913_; lean_object* v___x_914_; uint8_t v___x_915_; 
v___x_913_ = lean_unsigned_to_nat(4u);
v___x_914_ = l_Lean_Syntax_getArg(v_stx_900_, v___x_913_);
lean_inc(v___x_914_);
v___x_915_ = l_Lean_Syntax_isOfKind(v___x_914_, v___x_901_);
if (v___x_915_ == 0)
{
lean_object* v___x_916_; 
lean_dec(v___x_914_);
lean_dec(v___x_905_);
v___x_916_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_916_, 0, v_stx_900_);
return v___x_916_;
}
else
{
lean_object* v___x_917_; uint8_t v___x_918_; 
v___x_917_ = l_Lean_Syntax_getArg(v___x_914_, v___x_909_);
v___x_918_ = l_Lean_Syntax_matchesNull(v___x_917_, v___x_908_);
if (v___x_918_ == 0)
{
lean_object* v___x_919_; 
lean_dec(v___x_914_);
lean_dec(v___x_905_);
v___x_919_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_919_, 0, v_stx_900_);
return v___x_919_;
}
else
{
lean_object* v_ref_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v_groups_924_; uint8_t v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
lean_dec(v_stx_900_);
v_ref_920_ = lean_ctor_get(v___y_896_, 5);
v___x_921_ = l_Lean_Syntax_getArg(v___x_905_, v___x_908_);
lean_dec(v___x_905_);
v___x_922_ = l_Lean_Syntax_getArg(v___x_914_, v___x_904_);
v___x_923_ = l_Lean_Syntax_getArg(v___x_914_, v___x_913_);
lean_dec(v___x_914_);
v_groups_924_ = l_Lean_Syntax_getArgs(v___x_922_);
lean_dec(v___x_922_);
v___x_925_ = 0;
v___x_926_ = l_Lean_SourceInfo_fromRef(v_ref_920_, v___x_925_);
v___x_927_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__4));
lean_inc_n(v___x_926_, 4);
v___x_928_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_928_, 0, v___x_926_);
lean_ctor_set(v___x_928_, 1, v___x_927_);
v___x_929_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_930_ = l_Array_mkArray1___redArg(v___x_921_);
v___x_931_ = l_Array_append___redArg(v___x_930_, v_groups_924_);
lean_dec_ref(v_groups_924_);
v___x_932_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_932_, 0, v___x_926_);
lean_ctor_set(v___x_932_, 1, v___x_929_);
lean_ctor_set(v___x_932_, 2, v___x_931_);
v___x_933_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9);
v___x_934_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_934_, 0, v___x_926_);
lean_ctor_set(v___x_934_, 1, v___x_929_);
lean_ctor_set(v___x_934_, 2, v___x_933_);
v___x_935_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_936_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_936_, 0, v___x_926_);
lean_ctor_set(v___x_936_, 1, v___x_935_);
v___x_937_ = l_Lean_Syntax_node5(v___x_926_, v___x_901_, v___x_928_, v___x_932_, v___x_934_, v___x_936_, v___x_923_);
v___x_938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_938_, 0, v___x_937_);
return v___x_938_;
}
}
}
}
}
}
v___jp_939_:
{
if (lean_obj_tag(v___y_940_) == 0)
{
lean_object* v_a_941_; lean_object* v___x_942_; uint8_t v___x_943_; 
v_a_941_ = lean_ctor_get(v___y_940_, 0);
lean_inc_n(v_a_941_, 2);
lean_dec_ref_known(v___y_940_, 1);
v___x_942_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi_x27___lam__0___closed__1));
v___x_943_ = l_Lean_Syntax_isOfKind(v_a_941_, v___x_942_);
if (v___x_943_ == 0)
{
v_stx_900_ = v_a_941_;
goto v___jp_899_;
}
else
{
lean_object* v_ref_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; uint8_t v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; 
v_ref_944_ = lean_ctor_get(v___y_896_, 5);
v___x_945_ = lean_unsigned_to_nat(0u);
v___x_946_ = l_Lean_Syntax_getArg(v_a_941_, v___x_945_);
v___x_947_ = lean_unsigned_to_nat(2u);
v___x_948_ = l_Lean_Syntax_getArg(v_a_941_, v___x_947_);
lean_dec(v_a_941_);
v___x_949_ = 0;
v___x_950_ = l_Lean_SourceInfo_fromRef(v_ref_944_, v___x_949_);
v___x_951_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__2));
v___x_952_ = ((lean_object*)(lp_mathlib_PiNotation_piNotation___closed__4));
lean_inc_n(v___x_950_, 4);
v___x_953_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_953_, 0, v___x_950_);
lean_ctor_set(v___x_953_, 1, v___x_952_);
v___x_954_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_955_ = l_Lean_Syntax_node1(v___x_950_, v___x_954_, v___x_946_);
v___x_956_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9);
v___x_957_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_957_, 0, v___x_950_);
lean_ctor_set(v___x_957_, 1, v___x_954_);
lean_ctor_set(v___x_957_, 2, v___x_956_);
v___x_958_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_959_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_959_, 0, v___x_950_);
lean_ctor_set(v___x_959_, 1, v___x_958_);
v___x_960_ = l_Lean_Syntax_node5(v___x_950_, v___x_951_, v___x_953_, v___x_955_, v___x_957_, v___x_959_, v___x_948_);
v_stx_900_ = v___x_960_;
goto v___jp_899_;
}
}
else
{
return v___y_940_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27___lam__0___boxed(lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_PiNotation_delabPi_x27___lam__0(v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_, v___y_976_);
lean_dec(v___y_976_);
lean_dec_ref(v___y_975_);
lean_dec(v___y_974_);
lean_dec_ref(v___y_973_);
lean_dec(v___y_972_);
lean_dec_ref(v___y_971_);
return v_res_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27(lean_object* v_a_980_, lean_object* v_a_981_, lean_object* v_a_982_, lean_object* v_a_983_, lean_object* v_a_984_, lean_object* v_a_985_){
_start:
{
lean_object* v___f_987_; lean_object* v___x_988_; lean_object* v___x_989_; 
v___f_987_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi_x27___closed__0));
v___x_988_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___closed__2));
v___x_989_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_988_, v___f_987_, v_a_980_, v_a_981_, v_a_982_, v_a_983_, v_a_984_, v_a_985_);
return v___x_989_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PiNotation_delabPi_x27___boxed(lean_object* v_a_990_, lean_object* v_a_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_, lean_object* v_a_996_){
_start:
{
lean_object* v_res_997_; 
v_res_997_ = lp_mathlib_PiNotation_delabPi_x27(v_a_990_, v_a_991_, v_a_992_, v_a_993_, v_a_994_, v_a_995_);
lean_dec(v_a_995_);
lean_dec_ref(v_a_994_);
lean_dec(v_a_993_);
lean_dec_ref(v_a_992_);
lean_dec(v_a_991_);
lean_dec_ref(v_a_990_);
return v_res_997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(lean_object* v___y_998_){
_start:
{
lean_object* v_subExpr_1000_; lean_object* v_expr_1001_; lean_object* v___x_1002_; 
v_subExpr_1000_ = lean_ctor_get(v___y_998_, 3);
v_expr_1001_ = lean_ctor_get(v_subExpr_1000_, 0);
lean_inc_ref(v_expr_1001_);
v___x_1002_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1002_, 0, v_expr_1001_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg___boxed(lean_object* v___y_1003_, lean_object* v___y_1004_){
_start:
{
lean_object* v_res_1005_; 
v_res_1005_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_1003_);
lean_dec_ref(v___y_1003_);
return v_res_1005_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0(lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_){
_start:
{
lean_object* v___x_1013_; 
v___x_1013_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_1006_);
return v___x_1013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___boxed(lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v_res_1021_; 
v_res_1021_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0(v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v___y_1015_);
lean_dec_ref(v___y_1014_);
return v_res_1021_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__0(lean_object* v_a_1042_, uint8_t v_a_1043_, uint8_t v_a_1044_, uint8_t v___x_1045_, lean_object* v_x_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_){
_start:
{
lean_object* v___x_1054_; 
v___x_1054_ = l_Lean_PrettyPrinter_Delaborator_delab(v___y_1047_, v___y_1048_, v___y_1049_, v___y_1050_, v___y_1051_, v___y_1052_);
if (lean_obj_tag(v___x_1054_) == 0)
{
lean_object* v_a_1055_; lean_object* v___x_1057_; uint8_t v_isShared_1058_; uint8_t v_isSharedCheck_1137_; 
v_a_1055_ = lean_ctor_get(v___x_1054_, 0);
v_isSharedCheck_1137_ = !lean_is_exclusive(v___x_1054_);
if (v_isSharedCheck_1137_ == 0)
{
v___x_1057_ = v___x_1054_;
v_isShared_1058_ = v_isSharedCheck_1137_;
goto v_resetjp_1056_;
}
else
{
lean_inc(v_a_1055_);
lean_dec(v___x_1054_);
v___x_1057_ = lean_box(0);
v_isShared_1058_ = v_isSharedCheck_1137_;
goto v_resetjp_1056_;
}
v_resetjp_1056_:
{
uint8_t v___y_1060_; uint8_t v___y_1088_; 
if (v_a_1043_ == 0)
{
v___y_1088_ = v_a_1043_;
goto v___jp_1087_;
}
else
{
if (v___x_1045_ == 0)
{
lean_object* v_ref_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; 
lean_del_object(v___x_1057_);
lean_dec(v_x_1046_);
v_ref_1108_ = lean_ctor_get(v___y_1051_, 5);
v___x_1109_ = l_Lean_SourceInfo_fromRef(v_ref_1108_, v___x_1045_);
v___x_1110_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__1));
v___x_1111_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1109_, 12);
v___x_1112_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1112_, 0, v___x_1109_);
lean_ctor_set(v___x_1112_, 1, v___x_1111_);
v___x_1113_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__4));
v___x_1114_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_1115_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__6));
v___x_1116_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__7));
v___x_1117_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1117_, 0, v___x_1109_);
lean_ctor_set(v___x_1117_, 1, v___x_1116_);
v___x_1118_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_1119_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__3));
v___x_1120_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__12));
v___x_1121_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1121_, 0, v___x_1109_);
lean_ctor_set(v___x_1121_, 1, v___x_1120_);
v___x_1122_ = l_Lean_Syntax_node1(v___x_1109_, v___x_1119_, v___x_1121_);
v___x_1123_ = l_Lean_Syntax_node1(v___x_1109_, v___x_1118_, v___x_1122_);
v___x_1124_ = l_Lean_Syntax_node1(v___x_1109_, v___x_1114_, v___x_1123_);
v___x_1125_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__8));
v___x_1126_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1126_, 0, v___x_1109_);
lean_ctor_set(v___x_1126_, 1, v___x_1125_);
v___x_1127_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__9));
v___x_1128_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1128_, 0, v___x_1109_);
lean_ctor_set(v___x_1128_, 1, v___x_1127_);
v___x_1129_ = l_Lean_Syntax_node5(v___x_1109_, v___x_1115_, v___x_1117_, v___x_1124_, v___x_1126_, v_a_1042_, v___x_1128_);
v___x_1130_ = l_Lean_Syntax_node1(v___x_1109_, v___x_1114_, v___x_1129_);
v___x_1131_ = l_Lean_Syntax_node1(v___x_1109_, v___x_1113_, v___x_1130_);
v___x_1132_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1133_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1109_);
lean_ctor_set(v___x_1133_, 1, v___x_1132_);
v___x_1134_ = l_Lean_Syntax_node4(v___x_1109_, v___x_1110_, v___x_1112_, v___x_1131_, v___x_1133_, v_a_1055_);
v___x_1135_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1135_, 0, v___x_1134_);
return v___x_1135_;
}
else
{
uint8_t v___x_1136_; 
v___x_1136_ = 0;
v___y_1088_ = v___x_1136_;
goto v___jp_1087_;
}
}
v___jp_1059_:
{
lean_object* v_ref_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1085_; 
v_ref_1061_ = lean_ctor_get(v___y_1051_, 5);
v___x_1062_ = l_Lean_SourceInfo_fromRef(v_ref_1061_, v___y_1060_);
v___x_1063_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__1));
v___x_1064_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1062_, 10);
v___x_1065_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1062_);
lean_ctor_set(v___x_1065_, 1, v___x_1064_);
v___x_1066_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__4));
v___x_1067_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_1068_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__6));
v___x_1069_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__7));
v___x_1070_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1070_, 0, v___x_1062_);
lean_ctor_set(v___x_1070_, 1, v___x_1069_);
v___x_1071_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_1072_ = l_Lean_Syntax_node1(v___x_1062_, v___x_1071_, v_x_1046_);
v___x_1073_ = l_Lean_Syntax_node1(v___x_1062_, v___x_1067_, v___x_1072_);
v___x_1074_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__8));
v___x_1075_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1075_, 0, v___x_1062_);
lean_ctor_set(v___x_1075_, 1, v___x_1074_);
v___x_1076_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__9));
v___x_1077_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1077_, 0, v___x_1062_);
lean_ctor_set(v___x_1077_, 1, v___x_1076_);
v___x_1078_ = l_Lean_Syntax_node5(v___x_1062_, v___x_1068_, v___x_1070_, v___x_1073_, v___x_1075_, v_a_1042_, v___x_1077_);
v___x_1079_ = l_Lean_Syntax_node1(v___x_1062_, v___x_1067_, v___x_1078_);
v___x_1080_ = l_Lean_Syntax_node1(v___x_1062_, v___x_1066_, v___x_1079_);
v___x_1081_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1082_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1082_, 0, v___x_1062_);
lean_ctor_set(v___x_1082_, 1, v___x_1081_);
v___x_1083_ = l_Lean_Syntax_node4(v___x_1062_, v___x_1063_, v___x_1065_, v___x_1080_, v___x_1082_, v_a_1055_);
if (v_isShared_1058_ == 0)
{
lean_ctor_set(v___x_1057_, 0, v___x_1083_);
v___x_1085_ = v___x_1057_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v___x_1083_);
v___x_1085_ = v_reuseFailAlloc_1086_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
return v___x_1085_;
}
}
v___jp_1087_:
{
if (v_a_1043_ == 0)
{
if (v_a_1044_ == 0)
{
lean_object* v_ref_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; 
lean_del_object(v___x_1057_);
lean_dec(v_a_1042_);
v_ref_1089_ = lean_ctor_get(v___y_1051_, 5);
v___x_1090_ = l_Lean_SourceInfo_fromRef(v_ref_1089_, v_a_1044_);
v___x_1091_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__1));
v___x_1092_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1090_, 7);
v___x_1093_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1093_, 0, v___x_1090_);
lean_ctor_set(v___x_1093_, 1, v___x_1092_);
v___x_1094_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__4));
v___x_1095_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__11));
v___x_1096_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_1097_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
v___x_1098_ = l_Lean_Syntax_node1(v___x_1090_, v___x_1097_, v_x_1046_);
v___x_1099_ = l_Lean_Syntax_node1(v___x_1090_, v___x_1096_, v___x_1098_);
v___x_1100_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9);
v___x_1101_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1101_, 0, v___x_1090_);
lean_ctor_set(v___x_1101_, 1, v___x_1096_);
lean_ctor_set(v___x_1101_, 2, v___x_1100_);
v___x_1102_ = l_Lean_Syntax_node2(v___x_1090_, v___x_1095_, v___x_1099_, v___x_1101_);
v___x_1103_ = l_Lean_Syntax_node1(v___x_1090_, v___x_1094_, v___x_1102_);
v___x_1104_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1105_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1105_, 0, v___x_1090_);
lean_ctor_set(v___x_1105_, 1, v___x_1104_);
v___x_1106_ = l_Lean_Syntax_node4(v___x_1090_, v___x_1091_, v___x_1093_, v___x_1103_, v___x_1105_, v_a_1055_);
v___x_1107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1107_, 0, v___x_1106_);
return v___x_1107_;
}
else
{
v___y_1060_ = v___y_1088_;
goto v___jp_1059_;
}
}
else
{
v___y_1060_ = v___y_1088_;
goto v___jp_1059_;
}
}
}
}
else
{
lean_dec(v_x_1046_);
lean_dec(v_a_1042_);
return v___x_1054_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__0___boxed(lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_, lean_object* v___x_1141_, lean_object* v_x_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_){
_start:
{
uint8_t v_a_245118__boxed_1150_; uint8_t v_a_245119__boxed_1151_; uint8_t v___x_245120__boxed_1152_; lean_object* v_res_1153_; 
v_a_245118__boxed_1150_ = lean_unbox(v_a_1139_);
v_a_245119__boxed_1151_ = lean_unbox(v_a_1140_);
v___x_245120__boxed_1152_ = lean_unbox(v___x_1141_);
v_res_1153_ = lp_mathlib_exists__delab___lam__0(v_a_1138_, v_a_245118__boxed_1150_, v_a_245119__boxed_1151_, v___x_245120__boxed_1152_, v_x_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_, v___y_1148_);
lean_dec(v___y_1148_);
lean_dec_ref(v___y_1147_);
lean_dec(v___y_1146_);
lean_dec_ref(v___y_1145_);
lean_dec(v___y_1144_);
lean_dec_ref(v___y_1143_);
return v_res_1153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg(lean_object* v_child_1154_, lean_object* v_childIdx_1155_, lean_object* v_x_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_){
_start:
{
lean_object* v_subExpr_1164_; lean_object* v_optionsPerPos_1165_; lean_object* v_currNamespace_1166_; lean_object* v_openDecls_1167_; uint8_t v_inPattern_1168_; lean_object* v_depth_1169_; lean_object* v_lctxInitIndices_1170_; lean_object* v_pos_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; 
v_subExpr_1164_ = lean_ctor_get(v___y_1157_, 3);
v_optionsPerPos_1165_ = lean_ctor_get(v___y_1157_, 0);
v_currNamespace_1166_ = lean_ctor_get(v___y_1157_, 1);
v_openDecls_1167_ = lean_ctor_get(v___y_1157_, 2);
v_inPattern_1168_ = lean_ctor_get_uint8(v___y_1157_, sizeof(void*)*6);
v_depth_1169_ = lean_ctor_get(v___y_1157_, 4);
v_lctxInitIndices_1170_ = lean_ctor_get(v___y_1157_, 5);
v_pos_1171_ = lean_ctor_get(v_subExpr_1164_, 1);
v___x_1172_ = l_Lean_SubExpr_Pos_push(v_pos_1171_, v_childIdx_1155_);
v___x_1173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1173_, 0, v_child_1154_);
lean_ctor_set(v___x_1173_, 1, v___x_1172_);
lean_inc(v_lctxInitIndices_1170_);
lean_inc(v_depth_1169_);
lean_inc(v_openDecls_1167_);
lean_inc(v_currNamespace_1166_);
lean_inc(v_optionsPerPos_1165_);
v___x_1174_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_1174_, 0, v_optionsPerPos_1165_);
lean_ctor_set(v___x_1174_, 1, v_currNamespace_1166_);
lean_ctor_set(v___x_1174_, 2, v_openDecls_1167_);
lean_ctor_set(v___x_1174_, 3, v___x_1173_);
lean_ctor_set(v___x_1174_, 4, v_depth_1169_);
lean_ctor_set(v___x_1174_, 5, v_lctxInitIndices_1170_);
lean_ctor_set_uint8(v___x_1174_, sizeof(void*)*6, v_inPattern_1168_);
lean_inc(v___y_1162_);
lean_inc_ref(v___y_1161_);
lean_inc(v___y_1160_);
lean_inc_ref(v___y_1159_);
lean_inc(v___y_1158_);
v___x_1175_ = lean_apply_7(v_x_1156_, v___x_1174_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, lean_box(0));
return v___x_1175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg___boxed(lean_object* v_child_1176_, lean_object* v_childIdx_1177_, lean_object* v_x_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_){
_start:
{
lean_object* v_res_1186_; 
v_res_1186_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg(v_child_1176_, v_childIdx_1177_, v_x_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_, v___y_1184_);
lean_dec(v___y_1184_);
lean_dec_ref(v___y_1183_);
lean_dec(v___y_1182_);
lean_dec_ref(v___y_1181_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
return v_res_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg(lean_object* v_x_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_){
_start:
{
lean_object* v___x_1195_; lean_object* v_a_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v___x_1195_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_1188_);
v_a_1196_ = lean_ctor_get(v___x_1195_, 0);
lean_inc(v_a_1196_);
lean_dec_ref(v___x_1195_);
v___x_1197_ = l_Lean_Expr_bindingDomain_x21(v_a_1196_);
lean_dec(v_a_1196_);
v___x_1198_ = lean_unsigned_to_nat(0u);
v___x_1199_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg(v___x_1197_, v___x_1198_, v_x_1187_, v___y_1188_, v___y_1189_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_);
return v___x_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg___boxed(lean_object* v_x_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v_res_1208_; 
v_res_1208_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg(v_x_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_);
lean_dec(v___y_1206_);
lean_dec_ref(v___y_1205_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
return v_res_1208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__1(lean_object* v___x_1209_, uint8_t v_a_1210_, uint8_t v_a_1211_, uint8_t v___x_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_){
_start:
{
lean_object* v___x_1220_; 
v___x_1220_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg(v___x_1209_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_, v___y_1218_);
if (lean_obj_tag(v___x_1220_) == 0)
{
lean_object* v_a_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___f_1225_; uint8_t v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; 
v_a_1221_ = lean_ctor_get(v___x_1220_, 0);
lean_inc(v_a_1221_);
lean_dec_ref_known(v___x_1220_, 1);
v___x_1222_ = lean_box(v_a_1210_);
v___x_1223_ = lean_box(v_a_1211_);
v___x_1224_ = lean_box(v___x_1212_);
v___f_1225_ = lean_alloc_closure((void*)(lp_mathlib_exists__delab___lam__0___boxed), 12, 4);
lean_closure_set(v___f_1225_, 0, v_a_1221_);
lean_closure_set(v___f_1225_, 1, v___x_1222_);
lean_closure_set(v___f_1225_, 2, v___x_1223_);
lean_closure_set(v___f_1225_, 3, v___x_1224_);
v___x_1226_ = 0;
v___x_1227_ = l_Lean_NameSet_empty;
v___x_1228_ = l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___redArg(v___f_1225_, v___x_1226_, v___x_1227_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_, v___y_1218_);
return v___x_1228_;
}
else
{
return v___x_1220_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__1___boxed(lean_object* v___x_1229_, lean_object* v_a_1230_, lean_object* v_a_1231_, lean_object* v___x_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_){
_start:
{
uint8_t v_a_245395__boxed_1240_; uint8_t v_a_245396__boxed_1241_; uint8_t v___x_245397__boxed_1242_; lean_object* v_res_1243_; 
v_a_245395__boxed_1240_ = lean_unbox(v_a_1230_);
v_a_245396__boxed_1241_ = lean_unbox(v_a_1231_);
v___x_245397__boxed_1242_ = lean_unbox(v___x_1232_);
v_res_1243_ = lp_mathlib_exists__delab___lam__1(v___x_1229_, v_a_245395__boxed_1240_, v_a_245396__boxed_1241_, v___x_245397__boxed_1242_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_);
lean_dec(v___y_1238_);
lean_dec_ref(v___y_1237_);
lean_dec(v___y_1236_);
lean_dec_ref(v___y_1235_);
lean_dec(v___y_1234_);
lean_dec_ref(v___y_1233_);
return v_res_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__3(size_t v_sz_1244_, size_t v_i_1245_, lean_object* v_bs_1246_){
_start:
{
uint8_t v___x_1247_; 
v___x_1247_ = lean_usize_dec_lt(v_i_1245_, v_sz_1244_);
if (v___x_1247_ == 0)
{
lean_object* v___x_1248_; 
v___x_1248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1248_, 0, v_bs_1246_);
return v___x_1248_;
}
else
{
lean_object* v_v_1249_; lean_object* v___x_1250_; uint8_t v___x_1251_; 
v_v_1249_ = lean_array_uget(v_bs_1246_, v_i_1245_);
v___x_1250_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__6));
lean_inc(v_v_1249_);
v___x_1251_ = l_Lean_Syntax_isOfKind(v_v_1249_, v___x_1250_);
if (v___x_1251_ == 0)
{
lean_object* v___x_1252_; 
lean_dec(v_v_1249_);
lean_dec_ref(v_bs_1246_);
v___x_1252_ = lean_box(0);
return v___x_1252_;
}
else
{
lean_object* v___x_1253_; lean_object* v_bs_x27_1254_; size_t v___x_1255_; size_t v___x_1256_; lean_object* v___x_1257_; 
v___x_1253_ = lean_unsigned_to_nat(0u);
v_bs_x27_1254_ = lean_array_uset(v_bs_1246_, v_i_1245_, v___x_1253_);
v___x_1255_ = ((size_t)1ULL);
v___x_1256_ = lean_usize_add(v_i_1245_, v___x_1255_);
v___x_1257_ = lean_array_uset(v_bs_x27_1254_, v_i_1245_, v_v_1249_);
v_i_1245_ = v___x_1256_;
v_bs_1246_ = v___x_1257_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__3___boxed(lean_object* v_sz_1259_, lean_object* v_i_1260_, lean_object* v_bs_1261_){
_start:
{
size_t v_sz_boxed_1262_; size_t v_i_boxed_1263_; lean_object* v_res_1264_; 
v_sz_boxed_1262_ = lean_unbox_usize(v_sz_1259_);
lean_dec(v_sz_1259_);
v_i_boxed_1263_ = lean_unbox_usize(v_i_1260_);
lean_dec(v_i_1260_);
v_res_1264_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__3(v_sz_boxed_1262_, v_i_boxed_1263_, v_bs_1261_);
return v_res_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__1(size_t v_sz_1265_, size_t v_i_1266_, lean_object* v_bs_1267_){
_start:
{
uint8_t v___x_1268_; 
v___x_1268_ = lean_usize_dec_lt(v_i_1266_, v_sz_1265_);
if (v___x_1268_ == 0)
{
lean_object* v___x_1269_; 
v___x_1269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1269_, 0, v_bs_1267_);
return v___x_1269_;
}
else
{
lean_object* v_v_1270_; lean_object* v___x_1271_; uint8_t v___x_1272_; 
v_v_1270_ = lean_array_uget(v_bs_1267_, v_i_1266_);
v___x_1271_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
lean_inc(v_v_1270_);
v___x_1272_ = l_Lean_Syntax_isOfKind(v_v_1270_, v___x_1271_);
if (v___x_1272_ == 0)
{
lean_object* v___x_1273_; 
lean_dec(v_v_1270_);
lean_dec_ref(v_bs_1267_);
v___x_1273_ = lean_box(0);
return v___x_1273_;
}
else
{
lean_object* v___x_1274_; lean_object* v_bs_x27_1275_; size_t v___x_1276_; size_t v___x_1277_; lean_object* v___x_1278_; 
v___x_1274_ = lean_unsigned_to_nat(0u);
v_bs_x27_1275_ = lean_array_uset(v_bs_1267_, v_i_1266_, v___x_1274_);
v___x_1276_ = ((size_t)1ULL);
v___x_1277_ = lean_usize_add(v_i_1266_, v___x_1276_);
v___x_1278_ = lean_array_uset(v_bs_x27_1275_, v_i_1266_, v_v_1270_);
v_i_1266_ = v___x_1277_;
v_bs_1267_ = v___x_1278_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__1___boxed(lean_object* v_sz_1280_, lean_object* v_i_1281_, lean_object* v_bs_1282_){
_start:
{
size_t v_sz_boxed_1283_; size_t v_i_boxed_1284_; lean_object* v_res_1285_; 
v_sz_boxed_1283_ = lean_unbox_usize(v_sz_1280_);
lean_dec(v_sz_1280_);
v_i_boxed_1284_ = lean_unbox_usize(v_i_1281_);
lean_dec(v_i_1281_);
v_res_1285_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__1(v_sz_boxed_1283_, v_i_boxed_1284_, v_bs_1282_);
return v_res_1285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__2(size_t v_sz_1286_, size_t v_i_1287_, lean_object* v_bs_1288_){
_start:
{
uint8_t v___x_1289_; 
v___x_1289_ = lean_usize_dec_lt(v_i_1287_, v_sz_1286_);
if (v___x_1289_ == 0)
{
return v_bs_1288_;
}
else
{
lean_object* v_v_1290_; lean_object* v___x_1291_; lean_object* v_bs_x27_1292_; size_t v___x_1293_; size_t v___x_1294_; lean_object* v___x_1295_; 
v_v_1290_ = lean_array_uget(v_bs_1288_, v_i_1287_);
v___x_1291_ = lean_unsigned_to_nat(0u);
v_bs_x27_1292_ = lean_array_uset(v_bs_1288_, v_i_1287_, v___x_1291_);
v___x_1293_ = ((size_t)1ULL);
v___x_1294_ = lean_usize_add(v_i_1287_, v___x_1293_);
v___x_1295_ = lean_array_uset(v_bs_x27_1292_, v_i_1287_, v_v_1290_);
v_i_1287_ = v___x_1294_;
v_bs_1288_ = v___x_1295_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__2___boxed(lean_object* v_sz_1297_, lean_object* v_i_1298_, lean_object* v_bs_1299_){
_start:
{
size_t v_sz_boxed_1300_; size_t v_i_boxed_1301_; lean_object* v_res_1302_; 
v_sz_boxed_1300_ = lean_unbox_usize(v_sz_1297_);
lean_dec(v_sz_1297_);
v_i_boxed_1301_ = lean_unbox_usize(v_i_1298_);
lean_dec(v_i_1298_);
v_res_1302_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__2(v_sz_boxed_1300_, v_i_boxed_1301_, v_bs_1299_);
return v_res_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(lean_object* v_x_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_){
_start:
{
lean_object* v___x_1311_; lean_object* v_a_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; 
v___x_1311_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_1304_);
v_a_1312_ = lean_ctor_get(v___x_1311_, 0);
lean_inc(v_a_1312_);
lean_dec_ref(v___x_1311_);
v___x_1313_ = l_Lean_Expr_appArg_x21(v_a_1312_);
lean_dec(v_a_1312_);
v___x_1314_ = lean_unsigned_to_nat(1u);
v___x_1315_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg(v___x_1313_, v___x_1314_, v_x_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_);
return v___x_1315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg___boxed(lean_object* v_x_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_){
_start:
{
lean_object* v_res_1324_; 
v_res_1324_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(v_x_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
lean_dec(v___y_1322_);
lean_dec_ref(v___y_1321_);
lean_dec(v___y_1320_);
lean_dec_ref(v___y_1319_);
lean_dec(v___y_1318_);
lean_dec_ref(v___y_1317_);
return v_res_1324_;
}
}
static lean_object* _init_lp_mathlib_exists__delab___lam__2___closed__0(void){
_start:
{
lean_object* v___x_1325_; lean_object* v_dummy_1326_; 
v___x_1325_ = lean_box(0);
v_dummy_1326_ = l_Lean_Expr_sort___override(v___x_1325_);
return v_dummy_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__2(lean_object* v___y_1336_, lean_object* v___y_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v___x_1343_; lean_object* v_a_1344_; lean_object* v___x_1346_; uint8_t v_isShared_1347_; uint8_t v_isSharedCheck_1990_; 
v___x_1343_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_1336_);
v_a_1344_ = lean_ctor_get(v___x_1343_, 0);
v_isSharedCheck_1990_ = !lean_is_exclusive(v___x_1343_);
if (v_isSharedCheck_1990_ == 0)
{
v___x_1346_ = v___x_1343_;
v_isShared_1347_ = v_isSharedCheck_1990_;
goto v_resetjp_1345_;
}
else
{
lean_inc(v_a_1344_);
lean_dec(v___x_1343_);
v___x_1346_ = lean_box(0);
v_isShared_1347_ = v_isSharedCheck_1990_;
goto v_resetjp_1345_;
}
v_resetjp_1345_:
{
lean_object* v_dummy_1348_; lean_object* v_nargs_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; uint8_t v___x_1356_; 
v_dummy_1348_ = lean_obj_once(&lp_mathlib_exists__delab___lam__2___closed__0, &lp_mathlib_exists__delab___lam__2___closed__0_once, _init_lp_mathlib_exists__delab___lam__2___closed__0);
v_nargs_1349_ = l_Lean_Expr_getAppNumArgs(v_a_1344_);
lean_inc(v_nargs_1349_);
v___x_1350_ = lean_mk_array(v_nargs_1349_, v_dummy_1348_);
v___x_1351_ = lean_unsigned_to_nat(1u);
v___x_1352_ = lean_nat_sub(v_nargs_1349_, v___x_1351_);
lean_dec(v_nargs_1349_);
v___x_1353_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_1344_, v___x_1350_, v___x_1352_);
v___x_1354_ = lean_array_get_size(v___x_1353_);
v___x_1355_ = lean_unsigned_to_nat(2u);
v___x_1356_ = lean_nat_dec_eq(v___x_1354_, v___x_1355_);
if (v___x_1356_ == 0)
{
lean_object* v___x_1357_; 
lean_dec_ref(v___x_1353_);
lean_del_object(v___x_1346_);
v___x_1357_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_1357_;
}
else
{
lean_object* v___x_1358_; lean_object* v_stx_1360_; lean_object* v___y_1361_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___y_1493_; lean_object* v___y_1494_; lean_object* v___y_1495_; lean_object* v___y_1496_; lean_object* v___y_1497_; lean_object* v___y_1498_; uint8_t v___x_1980_; 
v___x_1358_ = lean_unsigned_to_nat(0u);
v___x_1490_ = lean_array_fget(v___x_1353_, v___x_1358_);
v___x_1491_ = lean_array_fget(v___x_1353_, v___x_1351_);
lean_dec_ref(v___x_1353_);
v___x_1980_ = l_Lean_Expr_isLambda(v___x_1491_);
if (v___x_1980_ == 0)
{
lean_object* v___x_1981_; 
v___x_1981_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1981_) == 0)
{
lean_dec_ref_known(v___x_1981_, 1);
v___y_1493_ = v___y_1336_;
v___y_1494_ = v___y_1337_;
v___y_1495_ = v___y_1338_;
v___y_1496_ = v___y_1339_;
v___y_1497_ = v___y_1340_;
v___y_1498_ = v___y_1341_;
goto v___jp_1492_;
}
else
{
lean_object* v_a_1982_; lean_object* v___x_1984_; uint8_t v_isShared_1985_; uint8_t v_isSharedCheck_1989_; 
lean_dec(v___x_1491_);
lean_dec(v___x_1490_);
lean_del_object(v___x_1346_);
v_a_1982_ = lean_ctor_get(v___x_1981_, 0);
v_isSharedCheck_1989_ = !lean_is_exclusive(v___x_1981_);
if (v_isSharedCheck_1989_ == 0)
{
v___x_1984_ = v___x_1981_;
v_isShared_1985_ = v_isSharedCheck_1989_;
goto v_resetjp_1983_;
}
else
{
lean_inc(v_a_1982_);
lean_dec(v___x_1981_);
v___x_1984_ = lean_box(0);
v_isShared_1985_ = v_isSharedCheck_1989_;
goto v_resetjp_1983_;
}
v_resetjp_1983_:
{
lean_object* v___x_1987_; 
if (v_isShared_1985_ == 0)
{
v___x_1987_ = v___x_1984_;
goto v_reusejp_1986_;
}
else
{
lean_object* v_reuseFailAlloc_1988_; 
v_reuseFailAlloc_1988_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1988_, 0, v_a_1982_);
v___x_1987_ = v_reuseFailAlloc_1988_;
goto v_reusejp_1986_;
}
v_reusejp_1986_:
{
return v___x_1987_;
}
}
}
}
else
{
v___y_1493_ = v___y_1336_;
v___y_1494_ = v___y_1337_;
v___y_1495_ = v___y_1338_;
v___y_1496_ = v___y_1339_;
v___y_1497_ = v___y_1340_;
v___y_1498_ = v___y_1341_;
goto v___jp_1492_;
}
v___jp_1359_:
{
lean_object* v___x_1362_; uint8_t v___x_1363_; 
v___x_1362_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__1));
lean_inc(v_stx_1360_);
v___x_1363_ = l_Lean_Syntax_isOfKind(v_stx_1360_, v___x_1362_);
if (v___x_1363_ == 0)
{
lean_object* v___x_1365_; 
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1365_ = v___x_1346_;
goto v_reusejp_1364_;
}
else
{
lean_object* v_reuseFailAlloc_1366_; 
v_reuseFailAlloc_1366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1366_, 0, v_stx_1360_);
v___x_1365_ = v_reuseFailAlloc_1366_;
goto v_reusejp_1364_;
}
v_reusejp_1364_:
{
return v___x_1365_;
}
}
else
{
lean_object* v___x_1367_; lean_object* v___x_1368_; uint8_t v___x_1369_; 
v___x_1367_ = l_Lean_Syntax_getArg(v_stx_1360_, v___x_1351_);
v___x_1368_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__4));
lean_inc(v___x_1367_);
v___x_1369_ = l_Lean_Syntax_isOfKind(v___x_1367_, v___x_1368_);
if (v___x_1369_ == 0)
{
lean_object* v___x_1371_; 
lean_dec(v___x_1367_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1371_ = v___x_1346_;
goto v_reusejp_1370_;
}
else
{
lean_object* v_reuseFailAlloc_1372_; 
v_reuseFailAlloc_1372_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1372_, 0, v_stx_1360_);
v___x_1371_ = v_reuseFailAlloc_1372_;
goto v_reusejp_1370_;
}
v_reusejp_1370_:
{
return v___x_1371_;
}
}
else
{
lean_object* v___x_1373_; uint8_t v___x_1374_; 
v___x_1373_ = l_Lean_Syntax_getArg(v___x_1367_, v___x_1358_);
lean_dec(v___x_1367_);
lean_inc(v___x_1373_);
v___x_1374_ = l_Lean_Syntax_matchesNull(v___x_1373_, v___x_1351_);
if (v___x_1374_ == 0)
{
lean_object* v___x_1375_; uint8_t v___x_1376_; 
v___x_1375_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__11));
lean_inc(v___x_1373_);
v___x_1376_ = l_Lean_Syntax_isOfKind(v___x_1373_, v___x_1375_);
if (v___x_1376_ == 0)
{
lean_object* v___x_1378_; 
lean_dec(v___x_1373_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1378_ = v___x_1346_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v_stx_1360_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
else
{
lean_object* v___x_1380_; uint8_t v___x_1381_; 
v___x_1380_ = l_Lean_Syntax_getArg(v___x_1373_, v___x_1358_);
lean_inc(v___x_1380_);
v___x_1381_ = l_Lean_Syntax_matchesNull(v___x_1380_, v___x_1351_);
if (v___x_1381_ == 0)
{
lean_object* v___x_1383_; 
lean_dec(v___x_1380_);
lean_dec(v___x_1373_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1383_ = v___x_1346_;
goto v_reusejp_1382_;
}
else
{
lean_object* v_reuseFailAlloc_1384_; 
v_reuseFailAlloc_1384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1384_, 0, v_stx_1360_);
v___x_1383_ = v_reuseFailAlloc_1384_;
goto v_reusejp_1382_;
}
v_reusejp_1382_:
{
return v___x_1383_;
}
}
else
{
lean_object* v___x_1385_; lean_object* v___x_1386_; uint8_t v___x_1387_; 
v___x_1385_ = l_Lean_Syntax_getArg(v___x_1380_, v___x_1358_);
lean_dec(v___x_1380_);
v___x_1386_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
lean_inc(v___x_1385_);
v___x_1387_ = l_Lean_Syntax_isOfKind(v___x_1385_, v___x_1386_);
if (v___x_1387_ == 0)
{
lean_object* v___x_1389_; 
lean_dec(v___x_1385_);
lean_dec(v___x_1373_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1389_ = v___x_1346_;
goto v_reusejp_1388_;
}
else
{
lean_object* v_reuseFailAlloc_1390_; 
v_reuseFailAlloc_1390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1390_, 0, v_stx_1360_);
v___x_1389_ = v_reuseFailAlloc_1390_;
goto v_reusejp_1388_;
}
v_reusejp_1388_:
{
return v___x_1389_;
}
}
else
{
lean_object* v___x_1391_; uint8_t v___x_1392_; 
v___x_1391_ = l_Lean_Syntax_getArg(v___x_1373_, v___x_1351_);
lean_dec(v___x_1373_);
v___x_1392_ = l_Lean_Syntax_matchesNull(v___x_1391_, v___x_1358_);
if (v___x_1392_ == 0)
{
lean_object* v___x_1394_; 
lean_dec(v___x_1385_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1394_ = v___x_1346_;
goto v_reusejp_1393_;
}
else
{
lean_object* v_reuseFailAlloc_1395_; 
v_reuseFailAlloc_1395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1395_, 0, v_stx_1360_);
v___x_1394_ = v_reuseFailAlloc_1395_;
goto v_reusejp_1393_;
}
v_reusejp_1393_:
{
return v___x_1394_;
}
}
else
{
lean_object* v___x_1396_; lean_object* v___x_1397_; uint8_t v___x_1398_; 
v___x_1396_ = lean_unsigned_to_nat(3u);
v___x_1397_ = l_Lean_Syntax_getArg(v_stx_1360_, v___x_1396_);
lean_inc(v___x_1397_);
v___x_1398_ = l_Lean_Syntax_isOfKind(v___x_1397_, v___x_1362_);
if (v___x_1398_ == 0)
{
lean_object* v___x_1400_; 
lean_dec(v___x_1397_);
lean_dec(v___x_1385_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1400_ = v___x_1346_;
goto v_reusejp_1399_;
}
else
{
lean_object* v_reuseFailAlloc_1401_; 
v_reuseFailAlloc_1401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1401_, 0, v_stx_1360_);
v___x_1400_ = v_reuseFailAlloc_1401_;
goto v_reusejp_1399_;
}
v_reusejp_1399_:
{
return v___x_1400_;
}
}
else
{
lean_object* v___x_1402_; uint8_t v___x_1403_; 
v___x_1402_ = l_Lean_Syntax_getArg(v___x_1397_, v___x_1351_);
lean_inc(v___x_1402_);
v___x_1403_ = l_Lean_Syntax_isOfKind(v___x_1402_, v___x_1368_);
if (v___x_1403_ == 0)
{
lean_object* v___x_1405_; 
lean_dec(v___x_1402_);
lean_dec(v___x_1397_);
lean_dec(v___x_1385_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1405_ = v___x_1346_;
goto v_reusejp_1404_;
}
else
{
lean_object* v_reuseFailAlloc_1406_; 
v_reuseFailAlloc_1406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1406_, 0, v_stx_1360_);
v___x_1405_ = v_reuseFailAlloc_1406_;
goto v_reusejp_1404_;
}
v_reusejp_1404_:
{
return v___x_1405_;
}
}
else
{
lean_object* v___x_1407_; uint8_t v___x_1408_; 
v___x_1407_ = l_Lean_Syntax_getArg(v___x_1402_, v___x_1358_);
lean_dec(v___x_1402_);
lean_inc(v___x_1407_);
v___x_1408_ = l_Lean_Syntax_isOfKind(v___x_1407_, v___x_1375_);
if (v___x_1408_ == 0)
{
lean_object* v___x_1410_; 
lean_dec(v___x_1407_);
lean_dec(v___x_1397_);
lean_dec(v___x_1385_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1410_ = v___x_1346_;
goto v_reusejp_1409_;
}
else
{
lean_object* v_reuseFailAlloc_1411_; 
v_reuseFailAlloc_1411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1411_, 0, v_stx_1360_);
v___x_1410_ = v_reuseFailAlloc_1411_;
goto v_reusejp_1409_;
}
v_reusejp_1409_:
{
return v___x_1410_;
}
}
else
{
lean_object* v___x_1412_; lean_object* v___x_1413_; size_t v_sz_1414_; size_t v___x_1415_; lean_object* v___x_1416_; 
v___x_1412_ = l_Lean_Syntax_getArg(v___x_1407_, v___x_1358_);
v___x_1413_ = l_Lean_Syntax_getArgs(v___x_1412_);
lean_dec(v___x_1412_);
v_sz_1414_ = lean_array_size(v___x_1413_);
v___x_1415_ = ((size_t)0ULL);
v___x_1416_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__1(v_sz_1414_, v___x_1415_, v___x_1413_);
if (lean_obj_tag(v___x_1416_) == 0)
{
lean_object* v___x_1418_; 
lean_dec(v___x_1407_);
lean_dec(v___x_1397_);
lean_dec(v___x_1385_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1418_ = v___x_1346_;
goto v_reusejp_1417_;
}
else
{
lean_object* v_reuseFailAlloc_1419_; 
v_reuseFailAlloc_1419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1419_, 0, v_stx_1360_);
v___x_1418_ = v_reuseFailAlloc_1419_;
goto v_reusejp_1417_;
}
v_reusejp_1417_:
{
return v___x_1418_;
}
}
else
{
lean_object* v_val_1420_; lean_object* v___x_1421_; uint8_t v___x_1422_; 
v_val_1420_ = lean_ctor_get(v___x_1416_, 0);
lean_inc(v_val_1420_);
lean_dec_ref_known(v___x_1416_, 1);
v___x_1421_ = l_Lean_Syntax_getArg(v___x_1407_, v___x_1351_);
lean_dec(v___x_1407_);
v___x_1422_ = l_Lean_Syntax_matchesNull(v___x_1421_, v___x_1358_);
if (v___x_1422_ == 0)
{
lean_object* v___x_1424_; 
lean_dec(v_val_1420_);
lean_dec(v___x_1397_);
lean_dec(v___x_1385_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1424_ = v___x_1346_;
goto v_reusejp_1423_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v_stx_1360_);
v___x_1424_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1423_;
}
v_reusejp_1423_:
{
return v___x_1424_;
}
}
else
{
lean_object* v_ref_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; size_t v_sz_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1445_; 
lean_dec(v_stx_1360_);
v_ref_1426_ = lean_ctor_get(v___y_1361_, 5);
v___x_1427_ = l_Lean_Syntax_getArg(v___x_1397_, v___x_1396_);
lean_dec(v___x_1397_);
v___x_1428_ = l_Lean_SourceInfo_fromRef(v_ref_1426_, v___x_1374_);
v___x_1429_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1428_, 6);
v___x_1430_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1428_);
lean_ctor_set(v___x_1430_, 1, v___x_1429_);
v___x_1431_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_1432_ = l_Array_mkArray1___redArg(v___x_1385_);
v_sz_1433_ = lean_array_size(v_val_1420_);
v___x_1434_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__2(v_sz_1433_, v___x_1415_, v_val_1420_);
v___x_1435_ = l_Array_append___redArg(v___x_1432_, v___x_1434_);
lean_dec_ref(v___x_1434_);
v___x_1436_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1436_, 0, v___x_1428_);
lean_ctor_set(v___x_1436_, 1, v___x_1431_);
lean_ctor_set(v___x_1436_, 2, v___x_1435_);
v___x_1437_ = lean_obj_once(&lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9, &lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9_once, _init_lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__9);
v___x_1438_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1438_, 0, v___x_1428_);
lean_ctor_set(v___x_1438_, 1, v___x_1431_);
lean_ctor_set(v___x_1438_, 2, v___x_1437_);
v___x_1439_ = l_Lean_Syntax_node2(v___x_1428_, v___x_1375_, v___x_1436_, v___x_1438_);
v___x_1440_ = l_Lean_Syntax_node1(v___x_1428_, v___x_1368_, v___x_1439_);
v___x_1441_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1442_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1442_, 0, v___x_1428_);
lean_ctor_set(v___x_1442_, 1, v___x_1441_);
v___x_1443_ = l_Lean_Syntax_node4(v___x_1428_, v___x_1362_, v___x_1430_, v___x_1440_, v___x_1442_, v___x_1427_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v___x_1443_);
v___x_1445_ = v___x_1346_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v___x_1443_);
v___x_1445_ = v_reuseFailAlloc_1446_;
goto v_reusejp_1444_;
}
v_reusejp_1444_:
{
return v___x_1445_;
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
lean_object* v___x_1447_; lean_object* v___x_1448_; uint8_t v___x_1449_; 
v___x_1447_ = l_Lean_Syntax_getArg(v___x_1373_, v___x_1358_);
lean_dec(v___x_1373_);
v___x_1448_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__6));
lean_inc(v___x_1447_);
v___x_1449_ = l_Lean_Syntax_isOfKind(v___x_1447_, v___x_1448_);
if (v___x_1449_ == 0)
{
lean_object* v___x_1451_; 
lean_dec(v___x_1447_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1451_ = v___x_1346_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_stx_1360_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
else
{
lean_object* v___x_1453_; lean_object* v___x_1454_; uint8_t v___x_1455_; 
v___x_1453_ = lean_unsigned_to_nat(3u);
v___x_1454_ = l_Lean_Syntax_getArg(v_stx_1360_, v___x_1453_);
lean_inc(v___x_1454_);
v___x_1455_ = l_Lean_Syntax_isOfKind(v___x_1454_, v___x_1362_);
if (v___x_1455_ == 0)
{
lean_object* v___x_1457_; 
lean_dec(v___x_1454_);
lean_dec(v___x_1447_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1457_ = v___x_1346_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1458_; 
v_reuseFailAlloc_1458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1458_, 0, v_stx_1360_);
v___x_1457_ = v_reuseFailAlloc_1458_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
return v___x_1457_;
}
}
else
{
lean_object* v___x_1459_; uint8_t v___x_1460_; 
v___x_1459_ = l_Lean_Syntax_getArg(v___x_1454_, v___x_1351_);
lean_inc(v___x_1459_);
v___x_1460_ = l_Lean_Syntax_isOfKind(v___x_1459_, v___x_1368_);
if (v___x_1460_ == 0)
{
lean_object* v___x_1462_; 
lean_dec(v___x_1459_);
lean_dec(v___x_1454_);
lean_dec(v___x_1447_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1462_ = v___x_1346_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1463_; 
v_reuseFailAlloc_1463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1463_, 0, v_stx_1360_);
v___x_1462_ = v_reuseFailAlloc_1463_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
return v___x_1462_;
}
}
else
{
lean_object* v___x_1464_; lean_object* v___x_1465_; size_t v_sz_1466_; size_t v___x_1467_; lean_object* v___x_1468_; 
v___x_1464_ = l_Lean_Syntax_getArg(v___x_1459_, v___x_1358_);
lean_dec(v___x_1459_);
v___x_1465_ = l_Lean_Syntax_getArgs(v___x_1464_);
lean_dec(v___x_1464_);
v_sz_1466_ = lean_array_size(v___x_1465_);
v___x_1467_ = ((size_t)0ULL);
v___x_1468_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00exists__delab_spec__3(v_sz_1466_, v___x_1467_, v___x_1465_);
if (lean_obj_tag(v___x_1468_) == 0)
{
lean_object* v___x_1470_; 
lean_dec(v___x_1454_);
lean_dec(v___x_1447_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v_stx_1360_);
v___x_1470_ = v___x_1346_;
goto v_reusejp_1469_;
}
else
{
lean_object* v_reuseFailAlloc_1471_; 
v_reuseFailAlloc_1471_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1471_, 0, v_stx_1360_);
v___x_1470_ = v_reuseFailAlloc_1471_;
goto v_reusejp_1469_;
}
v_reusejp_1469_:
{
return v___x_1470_;
}
}
else
{
lean_object* v_val_1472_; lean_object* v_ref_1473_; lean_object* v___x_1474_; uint8_t v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1488_; 
lean_dec(v_stx_1360_);
v_val_1472_ = lean_ctor_get(v___x_1468_, 0);
lean_inc(v_val_1472_);
lean_dec_ref_known(v___x_1468_, 1);
v_ref_1473_ = lean_ctor_get(v___y_1361_, 5);
v___x_1474_ = l_Lean_Syntax_getArg(v___x_1454_, v___x_1453_);
lean_dec(v___x_1454_);
v___x_1475_ = 0;
v___x_1476_ = l_Lean_SourceInfo_fromRef(v_ref_1473_, v___x_1475_);
v___x_1477_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1476_, 4);
v___x_1478_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1478_, 0, v___x_1476_);
lean_ctor_set(v___x_1478_, 1, v___x_1477_);
v___x_1479_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__5));
v___x_1480_ = l_Array_mkArray1___redArg(v___x_1447_);
v___x_1481_ = l_Array_append___redArg(v___x_1480_, v_val_1472_);
lean_dec(v_val_1472_);
v___x_1482_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1482_, 0, v___x_1476_);
lean_ctor_set(v___x_1482_, 1, v___x_1479_);
lean_ctor_set(v___x_1482_, 2, v___x_1481_);
v___x_1483_ = l_Lean_Syntax_node1(v___x_1476_, v___x_1368_, v___x_1482_);
v___x_1484_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1485_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1485_, 0, v___x_1476_);
lean_ctor_set(v___x_1485_, 1, v___x_1484_);
v___x_1486_ = l_Lean_Syntax_node4(v___x_1476_, v___x_1362_, v___x_1478_, v___x_1483_, v___x_1485_, v___x_1474_);
if (v_isShared_1347_ == 0)
{
lean_ctor_set(v___x_1346_, 0, v___x_1486_);
v___x_1488_ = v___x_1346_;
goto v_reusejp_1487_;
}
else
{
lean_object* v_reuseFailAlloc_1489_; 
v_reuseFailAlloc_1489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1489_, 0, v___x_1486_);
v___x_1488_ = v_reuseFailAlloc_1489_;
goto v_reusejp_1487_;
}
v_reusejp_1487_:
{
return v___x_1488_;
}
}
}
}
}
}
}
}
}
v___jp_1492_:
{
lean_object* v___x_1499_; 
v___x_1499_ = l_Lean_Meta_isProp(v___x_1490_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_);
if (lean_obj_tag(v___x_1499_) == 0)
{
lean_object* v_a_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; 
v_a_1500_ = lean_ctor_get(v___x_1499_, 0);
lean_inc(v_a_1500_);
lean_dec_ref_known(v___x_1499_, 1);
v___x_1501_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__1));
v___x_1502_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_1501_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_);
if (lean_obj_tag(v___x_1502_) == 0)
{
lean_object* v_a_1503_; lean_object* v___x_1504_; uint8_t v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___f_1508_; lean_object* v___x_1509_; 
v_a_1503_ = lean_ctor_get(v___x_1502_, 0);
lean_inc(v_a_1503_);
lean_dec_ref_known(v___x_1502_, 1);
v___x_1504_ = l_Lean_Expr_bindingBody_x21(v___x_1491_);
lean_dec(v___x_1491_);
v___x_1505_ = lean_expr_has_loose_bvar(v___x_1504_, v___x_1358_);
lean_dec_ref(v___x_1504_);
v___x_1506_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__2));
v___x_1507_ = lean_box(v___x_1505_);
v___f_1508_ = lean_alloc_closure((void*)(lp_mathlib_exists__delab___lam__1___boxed), 11, 4);
lean_closure_set(v___f_1508_, 0, v___x_1506_);
lean_closure_set(v___f_1508_, 1, v_a_1500_);
lean_closure_set(v___f_1508_, 2, v_a_1503_);
lean_closure_set(v___f_1508_, 3, v___x_1507_);
v___x_1509_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(v___f_1508_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_);
if (lean_obj_tag(v___x_1509_) == 0)
{
lean_object* v_a_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; 
v_a_1510_ = lean_ctor_get(v___x_1509_, 0);
lean_inc(v_a_1510_);
lean_dec_ref_known(v___x_1509_, 1);
v___x_1511_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___closed__1));
v___x_1512_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_1511_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_);
if (lean_obj_tag(v___x_1512_) == 0)
{
lean_object* v_a_1513_; uint8_t v___x_1514_; 
v_a_1513_ = lean_ctor_get(v___x_1512_, 0);
lean_inc(v_a_1513_);
lean_dec_ref_known(v___x_1512_, 1);
v___x_1514_ = lean_unbox(v_a_1513_);
lean_dec(v_a_1513_);
if (v___x_1514_ == 0)
{
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1515_; uint8_t v___x_1516_; 
v___x_1515_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__1));
lean_inc(v_a_1510_);
v___x_1516_ = l_Lean_Syntax_isOfKind(v_a_1510_, v___x_1515_);
if (v___x_1516_ == 0)
{
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1517_; lean_object* v___x_1518_; uint8_t v___x_1519_; 
v___x_1517_ = l_Lean_Syntax_getArg(v_a_1510_, v___x_1351_);
v___x_1518_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__4));
lean_inc(v___x_1517_);
v___x_1519_ = l_Lean_Syntax_isOfKind(v___x_1517_, v___x_1518_);
if (v___x_1519_ == 0)
{
lean_dec(v___x_1517_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1520_; lean_object* v___x_1521_; uint8_t v___x_1522_; 
v___x_1520_ = l_Lean_Syntax_getArg(v___x_1517_, v___x_1358_);
lean_dec(v___x_1517_);
v___x_1521_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__11));
lean_inc(v___x_1520_);
v___x_1522_ = l_Lean_Syntax_isOfKind(v___x_1520_, v___x_1521_);
if (v___x_1522_ == 0)
{
uint8_t v___x_1523_; 
lean_inc(v___x_1520_);
v___x_1523_ = l_Lean_Syntax_matchesNull(v___x_1520_, v___x_1351_);
if (v___x_1523_ == 0)
{
lean_dec(v___x_1520_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1524_; lean_object* v___x_1525_; uint8_t v___x_1526_; 
v___x_1524_ = l_Lean_Syntax_getArg(v___x_1520_, v___x_1358_);
lean_dec(v___x_1520_);
v___x_1525_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__6));
lean_inc(v___x_1524_);
v___x_1526_ = l_Lean_Syntax_isOfKind(v___x_1524_, v___x_1525_);
if (v___x_1526_ == 0)
{
lean_dec(v___x_1524_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1527_; uint8_t v___x_1528_; 
v___x_1527_ = l_Lean_Syntax_getArg(v___x_1524_, v___x_1351_);
lean_dec(v___x_1524_);
lean_inc(v___x_1527_);
v___x_1528_ = l_Lean_Syntax_matchesNull(v___x_1527_, v___x_1351_);
if (v___x_1528_ == 0)
{
lean_dec(v___x_1527_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1529_; lean_object* v___x_1530_; uint8_t v___x_1531_; 
v___x_1529_ = l_Lean_Syntax_getArg(v___x_1527_, v___x_1358_);
lean_dec(v___x_1527_);
v___x_1530_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
lean_inc(v___x_1529_);
v___x_1531_ = l_Lean_Syntax_isOfKind(v___x_1529_, v___x_1530_);
if (v___x_1531_ == 0)
{
lean_dec(v___x_1529_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1532_; lean_object* v___x_1533_; uint8_t v___x_1534_; 
v___x_1532_ = l_Lean_Syntax_getArg(v___x_1529_, v___x_1358_);
lean_dec(v___x_1529_);
v___x_1533_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1));
lean_inc(v___x_1532_);
v___x_1534_ = l_Lean_Syntax_isOfKind(v___x_1532_, v___x_1533_);
if (v___x_1534_ == 0)
{
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; uint8_t v___x_1538_; 
v___x_1535_ = lean_unsigned_to_nat(3u);
v___x_1536_ = l_Lean_Syntax_getArg(v_a_1510_, v___x_1535_);
v___x_1537_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__4));
lean_inc(v___x_1536_);
v___x_1538_ = l_Lean_Syntax_isOfKind(v___x_1536_, v___x_1537_);
if (v___x_1538_ == 0)
{
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1539_; lean_object* v___x_1540_; uint8_t v___x_1541_; 
v___x_1539_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1358_);
v___x_1540_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__3));
lean_inc(v___x_1539_);
v___x_1541_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1540_);
if (v___x_1541_ == 0)
{
lean_object* v___x_1542_; uint8_t v___x_1543_; 
v___x_1542_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__8));
lean_inc(v___x_1539_);
v___x_1543_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1542_);
if (v___x_1543_ == 0)
{
lean_object* v___x_1544_; uint8_t v___x_1545_; 
v___x_1544_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__10));
lean_inc(v___x_1539_);
v___x_1545_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1544_);
if (v___x_1545_ == 0)
{
lean_object* v___x_1546_; uint8_t v___x_1547_; 
v___x_1546_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__12));
lean_inc(v___x_1539_);
v___x_1547_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1546_);
if (v___x_1547_ == 0)
{
lean_object* v___x_1548_; uint8_t v___x_1549_; 
v___x_1548_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__14));
lean_inc(v___x_1539_);
v___x_1549_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1548_);
if (v___x_1549_ == 0)
{
lean_object* v___x_1550_; uint8_t v___x_1551_; 
v___x_1550_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__16));
lean_inc(v___x_1539_);
v___x_1551_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1550_);
if (v___x_1551_ == 0)
{
lean_object* v___x_1552_; uint8_t v___x_1553_; 
v___x_1552_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__18));
lean_inc(v___x_1539_);
v___x_1553_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1552_);
if (v___x_1553_ == 0)
{
lean_object* v___x_1554_; uint8_t v___x_1555_; 
v___x_1554_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__20));
lean_inc(v___x_1539_);
v___x_1555_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1554_);
if (v___x_1555_ == 0)
{
lean_object* v___x_1556_; uint8_t v___x_1557_; 
v___x_1556_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__22));
lean_inc(v___x_1539_);
v___x_1557_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1556_);
if (v___x_1557_ == 0)
{
lean_object* v___x_1558_; uint8_t v___x_1559_; 
v___x_1558_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__24));
lean_inc(v___x_1539_);
v___x_1559_ = l_Lean_Syntax_isOfKind(v___x_1539_, v___x_1558_);
if (v___x_1559_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1560_; uint8_t v___x_1561_; 
v___x_1560_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1560_);
v___x_1561_ = l_Lean_Syntax_isOfKind(v___x_1560_, v___x_1533_);
if (v___x_1561_ == 0)
{
lean_dec(v___x_1560_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1562_; 
v___x_1562_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1560_);
lean_dec(v___x_1560_);
if (v___x_1562_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; 
lean_dec(v_a_1510_);
v_ref_1563_ = lean_ctor_get(v___y_1497_, 5);
v___x_1564_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1565_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1566_ = l_Lean_SourceInfo_fromRef(v_ref_1563_, v___x_1557_);
v___x_1567_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1568_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1566_, 5);
v___x_1569_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1569_, 0, v___x_1566_);
lean_ctor_set(v___x_1569_, 1, v___x_1568_);
v___x_1570_ = l_Lean_Syntax_node1(v___x_1566_, v___x_1530_, v___x_1532_);
v___x_1571_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__30));
v___x_1572_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__31));
v___x_1573_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1573_, 0, v___x_1566_);
lean_ctor_set(v___x_1573_, 1, v___x_1572_);
v___x_1574_ = l_Lean_Syntax_node2(v___x_1566_, v___x_1571_, v___x_1573_, v___x_1564_);
v___x_1575_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1576_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1566_);
lean_ctor_set(v___x_1576_, 1, v___x_1575_);
v___x_1577_ = l_Lean_Syntax_node5(v___x_1566_, v___x_1567_, v___x_1569_, v___x_1570_, v___x_1574_, v___x_1576_, v___x_1565_);
v_stx_1360_ = v___x_1577_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1578_; uint8_t v___x_1579_; 
v___x_1578_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1578_);
v___x_1579_ = l_Lean_Syntax_isOfKind(v___x_1578_, v___x_1533_);
if (v___x_1579_ == 0)
{
lean_dec(v___x_1578_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1580_; 
v___x_1580_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1578_);
lean_dec(v___x_1578_);
if (v___x_1580_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; 
lean_dec(v_a_1510_);
v_ref_1581_ = lean_ctor_get(v___y_1497_, 5);
v___x_1582_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1583_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1584_ = l_Lean_SourceInfo_fromRef(v_ref_1581_, v___x_1555_);
v___x_1585_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1586_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1584_, 5);
v___x_1587_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1587_, 0, v___x_1584_);
lean_ctor_set(v___x_1587_, 1, v___x_1586_);
v___x_1588_ = l_Lean_Syntax_node1(v___x_1584_, v___x_1530_, v___x_1532_);
v___x_1589_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__33));
v___x_1590_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__34));
v___x_1591_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1591_, 0, v___x_1584_);
lean_ctor_set(v___x_1591_, 1, v___x_1590_);
v___x_1592_ = l_Lean_Syntax_node2(v___x_1584_, v___x_1589_, v___x_1591_, v___x_1582_);
v___x_1593_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1594_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1594_, 0, v___x_1584_);
lean_ctor_set(v___x_1594_, 1, v___x_1593_);
v___x_1595_ = l_Lean_Syntax_node5(v___x_1584_, v___x_1585_, v___x_1587_, v___x_1588_, v___x_1592_, v___x_1594_, v___x_1583_);
v_stx_1360_ = v___x_1595_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1596_; uint8_t v___x_1597_; 
v___x_1596_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1596_);
v___x_1597_ = l_Lean_Syntax_isOfKind(v___x_1596_, v___x_1533_);
if (v___x_1597_ == 0)
{
lean_dec(v___x_1596_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1598_; 
v___x_1598_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1596_);
lean_dec(v___x_1596_);
if (v___x_1598_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; 
lean_dec(v_a_1510_);
v_ref_1599_ = lean_ctor_get(v___y_1497_, 5);
v___x_1600_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1601_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1602_ = l_Lean_SourceInfo_fromRef(v_ref_1599_, v___x_1553_);
v___x_1603_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1604_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1602_, 5);
v___x_1605_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1605_, 0, v___x_1602_);
lean_ctor_set(v___x_1605_, 1, v___x_1604_);
v___x_1606_ = l_Lean_Syntax_node1(v___x_1602_, v___x_1530_, v___x_1532_);
v___x_1607_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__36));
v___x_1608_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__37));
v___x_1609_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1609_, 0, v___x_1602_);
lean_ctor_set(v___x_1609_, 1, v___x_1608_);
v___x_1610_ = l_Lean_Syntax_node2(v___x_1602_, v___x_1607_, v___x_1609_, v___x_1600_);
v___x_1611_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1612_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1602_);
lean_ctor_set(v___x_1612_, 1, v___x_1611_);
v___x_1613_ = l_Lean_Syntax_node5(v___x_1602_, v___x_1603_, v___x_1605_, v___x_1606_, v___x_1610_, v___x_1612_, v___x_1601_);
v_stx_1360_ = v___x_1613_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1614_; uint8_t v___x_1615_; 
v___x_1614_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1614_);
v___x_1615_ = l_Lean_Syntax_isOfKind(v___x_1614_, v___x_1533_);
if (v___x_1615_ == 0)
{
lean_dec(v___x_1614_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1616_; 
v___x_1616_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1614_);
lean_dec(v___x_1614_);
if (v___x_1616_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; 
lean_dec(v_a_1510_);
v_ref_1617_ = lean_ctor_get(v___y_1497_, 5);
v___x_1618_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1619_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1620_ = l_Lean_SourceInfo_fromRef(v_ref_1617_, v___x_1551_);
v___x_1621_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1622_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1620_, 5);
v___x_1623_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1623_, 0, v___x_1620_);
lean_ctor_set(v___x_1623_, 1, v___x_1622_);
v___x_1624_ = l_Lean_Syntax_node1(v___x_1620_, v___x_1530_, v___x_1532_);
v___x_1625_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__39));
v___x_1626_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__40));
v___x_1627_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1627_, 0, v___x_1620_);
lean_ctor_set(v___x_1627_, 1, v___x_1626_);
v___x_1628_ = l_Lean_Syntax_node2(v___x_1620_, v___x_1625_, v___x_1627_, v___x_1618_);
v___x_1629_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1630_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1630_, 0, v___x_1620_);
lean_ctor_set(v___x_1630_, 1, v___x_1629_);
v___x_1631_ = l_Lean_Syntax_node5(v___x_1620_, v___x_1621_, v___x_1623_, v___x_1624_, v___x_1628_, v___x_1630_, v___x_1619_);
v_stx_1360_ = v___x_1631_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1632_; uint8_t v___x_1633_; 
v___x_1632_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1632_);
v___x_1633_ = l_Lean_Syntax_isOfKind(v___x_1632_, v___x_1533_);
if (v___x_1633_ == 0)
{
lean_dec(v___x_1632_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1634_; 
v___x_1634_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1632_);
lean_dec(v___x_1632_);
if (v___x_1634_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; 
lean_dec(v_a_1510_);
v_ref_1635_ = lean_ctor_get(v___y_1497_, 5);
v___x_1636_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1637_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1638_ = l_Lean_SourceInfo_fromRef(v_ref_1635_, v___x_1549_);
v___x_1639_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1640_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1638_, 5);
v___x_1641_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1641_, 0, v___x_1638_);
lean_ctor_set(v___x_1641_, 1, v___x_1640_);
v___x_1642_ = l_Lean_Syntax_node1(v___x_1638_, v___x_1530_, v___x_1532_);
v___x_1643_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__42));
v___x_1644_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__43));
v___x_1645_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1645_, 0, v___x_1638_);
lean_ctor_set(v___x_1645_, 1, v___x_1644_);
v___x_1646_ = l_Lean_Syntax_node2(v___x_1638_, v___x_1643_, v___x_1645_, v___x_1636_);
v___x_1647_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1648_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1648_, 0, v___x_1638_);
lean_ctor_set(v___x_1648_, 1, v___x_1647_);
v___x_1649_ = l_Lean_Syntax_node5(v___x_1638_, v___x_1639_, v___x_1641_, v___x_1642_, v___x_1646_, v___x_1648_, v___x_1637_);
v_stx_1360_ = v___x_1649_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1650_; uint8_t v___x_1651_; 
v___x_1650_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1650_);
v___x_1651_ = l_Lean_Syntax_isOfKind(v___x_1650_, v___x_1533_);
if (v___x_1651_ == 0)
{
lean_dec(v___x_1650_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1652_; 
v___x_1652_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1650_);
lean_dec(v___x_1650_);
if (v___x_1652_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; 
lean_dec(v_a_1510_);
v_ref_1653_ = lean_ctor_get(v___y_1497_, 5);
v___x_1654_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1655_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1656_ = l_Lean_SourceInfo_fromRef(v_ref_1653_, v___x_1547_);
v___x_1657_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1658_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1656_, 5);
v___x_1659_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1659_, 0, v___x_1656_);
lean_ctor_set(v___x_1659_, 1, v___x_1658_);
v___x_1660_ = l_Lean_Syntax_node1(v___x_1656_, v___x_1530_, v___x_1532_);
v___x_1661_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__45));
v___x_1662_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__46));
v___x_1663_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1663_, 0, v___x_1656_);
lean_ctor_set(v___x_1663_, 1, v___x_1662_);
v___x_1664_ = l_Lean_Syntax_node2(v___x_1656_, v___x_1661_, v___x_1663_, v___x_1654_);
v___x_1665_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1666_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1666_, 0, v___x_1656_);
lean_ctor_set(v___x_1666_, 1, v___x_1665_);
v___x_1667_ = l_Lean_Syntax_node5(v___x_1656_, v___x_1657_, v___x_1659_, v___x_1660_, v___x_1664_, v___x_1666_, v___x_1655_);
v_stx_1360_ = v___x_1667_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1668_; uint8_t v___x_1669_; 
v___x_1668_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1668_);
v___x_1669_ = l_Lean_Syntax_isOfKind(v___x_1668_, v___x_1533_);
if (v___x_1669_ == 0)
{
lean_dec(v___x_1668_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1670_; 
v___x_1670_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1668_);
lean_dec(v___x_1668_);
if (v___x_1670_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; 
lean_dec(v_a_1510_);
v_ref_1671_ = lean_ctor_get(v___y_1497_, 5);
v___x_1672_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1673_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1674_ = l_Lean_SourceInfo_fromRef(v_ref_1671_, v___x_1545_);
v___x_1675_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1676_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1674_, 5);
v___x_1677_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1677_, 0, v___x_1674_);
lean_ctor_set(v___x_1677_, 1, v___x_1676_);
v___x_1678_ = l_Lean_Syntax_node1(v___x_1674_, v___x_1530_, v___x_1532_);
v___x_1679_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__48));
v___x_1680_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__49));
v___x_1681_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1681_, 0, v___x_1674_);
lean_ctor_set(v___x_1681_, 1, v___x_1680_);
v___x_1682_ = l_Lean_Syntax_node2(v___x_1674_, v___x_1679_, v___x_1681_, v___x_1672_);
v___x_1683_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1684_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1684_, 0, v___x_1674_);
lean_ctor_set(v___x_1684_, 1, v___x_1683_);
v___x_1685_ = l_Lean_Syntax_node5(v___x_1674_, v___x_1675_, v___x_1677_, v___x_1678_, v___x_1682_, v___x_1684_, v___x_1673_);
v_stx_1360_ = v___x_1685_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1686_; uint8_t v___x_1687_; 
v___x_1686_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1686_);
v___x_1687_ = l_Lean_Syntax_isOfKind(v___x_1686_, v___x_1533_);
if (v___x_1687_ == 0)
{
lean_dec(v___x_1686_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1688_; 
v___x_1688_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1686_);
lean_dec(v___x_1686_);
if (v___x_1688_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1689_; lean_object* v___x_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; 
lean_dec(v_a_1510_);
v_ref_1689_ = lean_ctor_get(v___y_1497_, 5);
v___x_1690_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1691_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1692_ = l_Lean_SourceInfo_fromRef(v_ref_1689_, v___x_1543_);
v___x_1693_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1694_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1692_, 5);
v___x_1695_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1695_, 0, v___x_1692_);
lean_ctor_set(v___x_1695_, 1, v___x_1694_);
v___x_1696_ = l_Lean_Syntax_node1(v___x_1692_, v___x_1530_, v___x_1532_);
v___x_1697_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__51));
v___x_1698_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__52));
v___x_1699_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1699_, 0, v___x_1692_);
lean_ctor_set(v___x_1699_, 1, v___x_1698_);
v___x_1700_ = l_Lean_Syntax_node2(v___x_1692_, v___x_1697_, v___x_1699_, v___x_1690_);
v___x_1701_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1702_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1702_, 0, v___x_1692_);
lean_ctor_set(v___x_1702_, 1, v___x_1701_);
v___x_1703_ = l_Lean_Syntax_node5(v___x_1692_, v___x_1693_, v___x_1695_, v___x_1696_, v___x_1700_, v___x_1702_, v___x_1691_);
v_stx_1360_ = v___x_1703_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1704_; uint8_t v___x_1705_; 
v___x_1704_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1704_);
v___x_1705_ = l_Lean_Syntax_isOfKind(v___x_1704_, v___x_1533_);
if (v___x_1705_ == 0)
{
lean_dec(v___x_1704_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1706_; 
v___x_1706_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1704_);
lean_dec(v___x_1704_);
if (v___x_1706_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; 
lean_dec(v_a_1510_);
v_ref_1707_ = lean_ctor_get(v___y_1497_, 5);
v___x_1708_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1709_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1710_ = l_Lean_SourceInfo_fromRef(v_ref_1707_, v___x_1541_);
v___x_1711_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1712_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1710_, 5);
v___x_1713_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1710_);
lean_ctor_set(v___x_1713_, 1, v___x_1712_);
v___x_1714_ = l_Lean_Syntax_node1(v___x_1710_, v___x_1530_, v___x_1532_);
v___x_1715_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__54));
v___x_1716_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__55));
v___x_1717_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1717_, 0, v___x_1710_);
lean_ctor_set(v___x_1717_, 1, v___x_1716_);
v___x_1718_ = l_Lean_Syntax_node2(v___x_1710_, v___x_1715_, v___x_1717_, v___x_1708_);
v___x_1719_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1720_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1720_, 0, v___x_1710_);
lean_ctor_set(v___x_1720_, 1, v___x_1719_);
v___x_1721_ = l_Lean_Syntax_node5(v___x_1710_, v___x_1711_, v___x_1713_, v___x_1714_, v___x_1718_, v___x_1720_, v___x_1709_);
v_stx_1360_ = v___x_1721_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1722_; uint8_t v___x_1723_; 
v___x_1722_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1358_);
lean_inc(v___x_1722_);
v___x_1723_ = l_Lean_Syntax_isOfKind(v___x_1722_, v___x_1533_);
if (v___x_1723_ == 0)
{
lean_dec(v___x_1722_);
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1724_; 
v___x_1724_ = l_Lean_Syntax_structEq(v___x_1532_, v___x_1722_);
lean_dec(v___x_1722_);
if (v___x_1724_ == 0)
{
lean_dec(v___x_1539_);
lean_dec(v___x_1536_);
lean_dec(v___x_1532_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; 
lean_dec(v_a_1510_);
v_ref_1725_ = lean_ctor_get(v___y_1497_, 5);
v___x_1726_ = l_Lean_Syntax_getArg(v___x_1539_, v___x_1355_);
lean_dec(v___x_1539_);
v___x_1727_ = l_Lean_Syntax_getArg(v___x_1536_, v___x_1355_);
lean_dec(v___x_1536_);
v___x_1728_ = l_Lean_SourceInfo_fromRef(v_ref_1725_, v___x_1522_);
v___x_1729_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1730_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1728_, 5);
v___x_1731_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1731_, 0, v___x_1728_);
lean_ctor_set(v___x_1731_, 1, v___x_1730_);
v___x_1732_ = l_Lean_Syntax_node1(v___x_1728_, v___x_1530_, v___x_1532_);
v___x_1733_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__5));
v___x_1734_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__6));
v___x_1735_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1735_, 0, v___x_1728_);
lean_ctor_set(v___x_1735_, 1, v___x_1734_);
v___x_1736_ = l_Lean_Syntax_node2(v___x_1728_, v___x_1733_, v___x_1735_, v___x_1726_);
v___x_1737_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1738_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1738_, 0, v___x_1728_);
lean_ctor_set(v___x_1738_, 1, v___x_1737_);
v___x_1739_ = l_Lean_Syntax_node5(v___x_1728_, v___x_1729_, v___x_1731_, v___x_1732_, v___x_1736_, v___x_1738_, v___x_1727_);
v_stx_1360_ = v___x_1739_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
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
lean_object* v___x_1740_; uint8_t v___x_1741_; 
v___x_1740_ = l_Lean_Syntax_getArg(v___x_1520_, v___x_1358_);
lean_inc(v___x_1740_);
v___x_1741_ = l_Lean_Syntax_matchesNull(v___x_1740_, v___x_1351_);
if (v___x_1741_ == 0)
{
lean_dec(v___x_1740_);
lean_dec(v___x_1520_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1742_; lean_object* v___x_1743_; uint8_t v___x_1744_; 
v___x_1742_ = l_Lean_Syntax_getArg(v___x_1740_, v___x_1358_);
lean_dec(v___x_1740_);
v___x_1743_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__28));
lean_inc(v___x_1742_);
v___x_1744_ = l_Lean_Syntax_isOfKind(v___x_1742_, v___x_1743_);
if (v___x_1744_ == 0)
{
lean_dec(v___x_1742_);
lean_dec(v___x_1520_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1745_; lean_object* v___x_1746_; uint8_t v___x_1747_; 
v___x_1745_ = l_Lean_Syntax_getArg(v___x_1742_, v___x_1358_);
lean_dec(v___x_1742_);
v___x_1746_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__1));
lean_inc(v___x_1745_);
v___x_1747_ = l_Lean_Syntax_isOfKind(v___x_1745_, v___x_1746_);
if (v___x_1747_ == 0)
{
lean_dec(v___x_1745_);
lean_dec(v___x_1520_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1748_; uint8_t v___x_1749_; 
v___x_1748_ = l_Lean_Syntax_getArg(v___x_1520_, v___x_1351_);
lean_dec(v___x_1520_);
v___x_1749_ = l_Lean_Syntax_matchesNull(v___x_1748_, v___x_1358_);
if (v___x_1749_ == 0)
{
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; uint8_t v___x_1753_; 
v___x_1750_ = lean_unsigned_to_nat(3u);
v___x_1751_ = l_Lean_Syntax_getArg(v_a_1510_, v___x_1750_);
v___x_1752_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__4));
lean_inc(v___x_1751_);
v___x_1753_ = l_Lean_Syntax_isOfKind(v___x_1751_, v___x_1752_);
if (v___x_1753_ == 0)
{
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1754_; lean_object* v___x_1755_; uint8_t v___x_1756_; 
v___x_1754_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1358_);
v___x_1755_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__3));
lean_inc(v___x_1754_);
v___x_1756_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1755_);
if (v___x_1756_ == 0)
{
lean_object* v___x_1757_; uint8_t v___x_1758_; 
v___x_1757_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__8));
lean_inc(v___x_1754_);
v___x_1758_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1757_);
if (v___x_1758_ == 0)
{
lean_object* v___x_1759_; uint8_t v___x_1760_; 
v___x_1759_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__10));
lean_inc(v___x_1754_);
v___x_1760_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1759_);
if (v___x_1760_ == 0)
{
lean_object* v___x_1761_; uint8_t v___x_1762_; 
v___x_1761_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__12));
lean_inc(v___x_1754_);
v___x_1762_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1761_);
if (v___x_1762_ == 0)
{
lean_object* v___x_1763_; uint8_t v___x_1764_; 
v___x_1763_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__14));
lean_inc(v___x_1754_);
v___x_1764_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1763_);
if (v___x_1764_ == 0)
{
lean_object* v___x_1765_; uint8_t v___x_1766_; 
v___x_1765_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__16));
lean_inc(v___x_1754_);
v___x_1766_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1765_);
if (v___x_1766_ == 0)
{
lean_object* v___x_1767_; uint8_t v___x_1768_; 
v___x_1767_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__18));
lean_inc(v___x_1754_);
v___x_1768_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1767_);
if (v___x_1768_ == 0)
{
lean_object* v___x_1769_; uint8_t v___x_1770_; 
v___x_1769_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__20));
lean_inc(v___x_1754_);
v___x_1770_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1769_);
if (v___x_1770_ == 0)
{
lean_object* v___x_1771_; uint8_t v___x_1772_; 
v___x_1771_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__22));
lean_inc(v___x_1754_);
v___x_1772_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1771_);
if (v___x_1772_ == 0)
{
lean_object* v___x_1773_; uint8_t v___x_1774_; 
v___x_1773_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__24));
lean_inc(v___x_1754_);
v___x_1774_ = l_Lean_Syntax_isOfKind(v___x_1754_, v___x_1773_);
if (v___x_1774_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v___x_1775_; uint8_t v___x_1776_; 
v___x_1775_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1775_);
v___x_1776_ = l_Lean_Syntax_isOfKind(v___x_1775_, v___x_1746_);
if (v___x_1776_ == 0)
{
lean_dec(v___x_1775_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1777_; 
v___x_1777_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1775_);
lean_dec(v___x_1775_);
if (v___x_1777_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; 
lean_dec(v_a_1510_);
v_ref_1778_ = lean_ctor_get(v___y_1497_, 5);
v___x_1779_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1780_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1781_ = l_Lean_SourceInfo_fromRef(v_ref_1778_, v___x_1772_);
v___x_1782_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1783_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1781_, 5);
v___x_1784_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1784_, 0, v___x_1781_);
lean_ctor_set(v___x_1784_, 1, v___x_1783_);
v___x_1785_ = l_Lean_Syntax_node1(v___x_1781_, v___x_1743_, v___x_1745_);
v___x_1786_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__30));
v___x_1787_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__31));
v___x_1788_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1788_, 0, v___x_1781_);
lean_ctor_set(v___x_1788_, 1, v___x_1787_);
v___x_1789_ = l_Lean_Syntax_node2(v___x_1781_, v___x_1786_, v___x_1788_, v___x_1779_);
v___x_1790_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1791_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1791_, 0, v___x_1781_);
lean_ctor_set(v___x_1791_, 1, v___x_1790_);
v___x_1792_ = l_Lean_Syntax_node5(v___x_1781_, v___x_1782_, v___x_1784_, v___x_1785_, v___x_1789_, v___x_1791_, v___x_1780_);
v_stx_1360_ = v___x_1792_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1793_; uint8_t v___x_1794_; 
v___x_1793_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1793_);
v___x_1794_ = l_Lean_Syntax_isOfKind(v___x_1793_, v___x_1746_);
if (v___x_1794_ == 0)
{
lean_dec(v___x_1793_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1795_; 
v___x_1795_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1793_);
lean_dec(v___x_1793_);
if (v___x_1795_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; 
lean_dec(v_a_1510_);
v_ref_1796_ = lean_ctor_get(v___y_1497_, 5);
v___x_1797_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1798_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1799_ = l_Lean_SourceInfo_fromRef(v_ref_1796_, v___x_1770_);
v___x_1800_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1801_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1799_, 5);
v___x_1802_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1802_, 0, v___x_1799_);
lean_ctor_set(v___x_1802_, 1, v___x_1801_);
v___x_1803_ = l_Lean_Syntax_node1(v___x_1799_, v___x_1743_, v___x_1745_);
v___x_1804_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__33));
v___x_1805_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__34));
v___x_1806_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1806_, 0, v___x_1799_);
lean_ctor_set(v___x_1806_, 1, v___x_1805_);
v___x_1807_ = l_Lean_Syntax_node2(v___x_1799_, v___x_1804_, v___x_1806_, v___x_1797_);
v___x_1808_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1809_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1809_, 0, v___x_1799_);
lean_ctor_set(v___x_1809_, 1, v___x_1808_);
v___x_1810_ = l_Lean_Syntax_node5(v___x_1799_, v___x_1800_, v___x_1802_, v___x_1803_, v___x_1807_, v___x_1809_, v___x_1798_);
v_stx_1360_ = v___x_1810_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1811_; uint8_t v___x_1812_; 
v___x_1811_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1811_);
v___x_1812_ = l_Lean_Syntax_isOfKind(v___x_1811_, v___x_1746_);
if (v___x_1812_ == 0)
{
lean_dec(v___x_1811_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1813_; 
v___x_1813_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1811_);
lean_dec(v___x_1811_);
if (v___x_1813_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; 
lean_dec(v_a_1510_);
v_ref_1814_ = lean_ctor_get(v___y_1497_, 5);
v___x_1815_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1816_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1817_ = l_Lean_SourceInfo_fromRef(v_ref_1814_, v___x_1768_);
v___x_1818_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1819_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1817_, 5);
v___x_1820_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1820_, 0, v___x_1817_);
lean_ctor_set(v___x_1820_, 1, v___x_1819_);
v___x_1821_ = l_Lean_Syntax_node1(v___x_1817_, v___x_1743_, v___x_1745_);
v___x_1822_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__36));
v___x_1823_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__37));
v___x_1824_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1824_, 0, v___x_1817_);
lean_ctor_set(v___x_1824_, 1, v___x_1823_);
v___x_1825_ = l_Lean_Syntax_node2(v___x_1817_, v___x_1822_, v___x_1824_, v___x_1815_);
v___x_1826_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1827_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1827_, 0, v___x_1817_);
lean_ctor_set(v___x_1827_, 1, v___x_1826_);
v___x_1828_ = l_Lean_Syntax_node5(v___x_1817_, v___x_1818_, v___x_1820_, v___x_1821_, v___x_1825_, v___x_1827_, v___x_1816_);
v_stx_1360_ = v___x_1828_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1829_; uint8_t v___x_1830_; 
v___x_1829_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1829_);
v___x_1830_ = l_Lean_Syntax_isOfKind(v___x_1829_, v___x_1746_);
if (v___x_1830_ == 0)
{
lean_dec(v___x_1829_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1831_; 
v___x_1831_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1829_);
lean_dec(v___x_1829_);
if (v___x_1831_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; 
lean_dec(v_a_1510_);
v_ref_1832_ = lean_ctor_get(v___y_1497_, 5);
v___x_1833_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1834_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1835_ = l_Lean_SourceInfo_fromRef(v_ref_1832_, v___x_1766_);
v___x_1836_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1837_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1835_, 5);
v___x_1838_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1838_, 0, v___x_1835_);
lean_ctor_set(v___x_1838_, 1, v___x_1837_);
v___x_1839_ = l_Lean_Syntax_node1(v___x_1835_, v___x_1743_, v___x_1745_);
v___x_1840_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__39));
v___x_1841_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__40));
v___x_1842_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1842_, 0, v___x_1835_);
lean_ctor_set(v___x_1842_, 1, v___x_1841_);
v___x_1843_ = l_Lean_Syntax_node2(v___x_1835_, v___x_1840_, v___x_1842_, v___x_1833_);
v___x_1844_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1845_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1845_, 0, v___x_1835_);
lean_ctor_set(v___x_1845_, 1, v___x_1844_);
v___x_1846_ = l_Lean_Syntax_node5(v___x_1835_, v___x_1836_, v___x_1838_, v___x_1839_, v___x_1843_, v___x_1845_, v___x_1834_);
v_stx_1360_ = v___x_1846_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1847_; uint8_t v___x_1848_; 
v___x_1847_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1847_);
v___x_1848_ = l_Lean_Syntax_isOfKind(v___x_1847_, v___x_1746_);
if (v___x_1848_ == 0)
{
lean_dec(v___x_1847_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1849_; 
v___x_1849_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1847_);
lean_dec(v___x_1847_);
if (v___x_1849_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; 
lean_dec(v_a_1510_);
v_ref_1850_ = lean_ctor_get(v___y_1497_, 5);
v___x_1851_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1852_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1853_ = l_Lean_SourceInfo_fromRef(v_ref_1850_, v___x_1764_);
v___x_1854_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1855_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1853_, 5);
v___x_1856_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1856_, 0, v___x_1853_);
lean_ctor_set(v___x_1856_, 1, v___x_1855_);
v___x_1857_ = l_Lean_Syntax_node1(v___x_1853_, v___x_1743_, v___x_1745_);
v___x_1858_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__42));
v___x_1859_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__43));
v___x_1860_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1860_, 0, v___x_1853_);
lean_ctor_set(v___x_1860_, 1, v___x_1859_);
v___x_1861_ = l_Lean_Syntax_node2(v___x_1853_, v___x_1858_, v___x_1860_, v___x_1851_);
v___x_1862_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1863_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1863_, 0, v___x_1853_);
lean_ctor_set(v___x_1863_, 1, v___x_1862_);
v___x_1864_ = l_Lean_Syntax_node5(v___x_1853_, v___x_1854_, v___x_1856_, v___x_1857_, v___x_1861_, v___x_1863_, v___x_1852_);
v_stx_1360_ = v___x_1864_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1865_; uint8_t v___x_1866_; 
v___x_1865_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1865_);
v___x_1866_ = l_Lean_Syntax_isOfKind(v___x_1865_, v___x_1746_);
if (v___x_1866_ == 0)
{
lean_dec(v___x_1865_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1867_; 
v___x_1867_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1865_);
lean_dec(v___x_1865_);
if (v___x_1867_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; 
lean_dec(v_a_1510_);
v_ref_1868_ = lean_ctor_get(v___y_1497_, 5);
v___x_1869_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1870_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1871_ = l_Lean_SourceInfo_fromRef(v_ref_1868_, v___x_1762_);
v___x_1872_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1873_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1871_, 5);
v___x_1874_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1874_, 0, v___x_1871_);
lean_ctor_set(v___x_1874_, 1, v___x_1873_);
v___x_1875_ = l_Lean_Syntax_node1(v___x_1871_, v___x_1743_, v___x_1745_);
v___x_1876_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__45));
v___x_1877_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__46));
v___x_1878_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1878_, 0, v___x_1871_);
lean_ctor_set(v___x_1878_, 1, v___x_1877_);
v___x_1879_ = l_Lean_Syntax_node2(v___x_1871_, v___x_1876_, v___x_1878_, v___x_1869_);
v___x_1880_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1881_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1881_, 0, v___x_1871_);
lean_ctor_set(v___x_1881_, 1, v___x_1880_);
v___x_1882_ = l_Lean_Syntax_node5(v___x_1871_, v___x_1872_, v___x_1874_, v___x_1875_, v___x_1879_, v___x_1881_, v___x_1870_);
v_stx_1360_ = v___x_1882_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1883_; uint8_t v___x_1884_; 
v___x_1883_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1883_);
v___x_1884_ = l_Lean_Syntax_isOfKind(v___x_1883_, v___x_1746_);
if (v___x_1884_ == 0)
{
lean_dec(v___x_1883_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1885_; 
v___x_1885_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1883_);
lean_dec(v___x_1883_);
if (v___x_1885_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; 
lean_dec(v_a_1510_);
v_ref_1886_ = lean_ctor_get(v___y_1497_, 5);
v___x_1887_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1888_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1889_ = l_Lean_SourceInfo_fromRef(v_ref_1886_, v___x_1760_);
v___x_1890_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1891_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1889_, 5);
v___x_1892_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1892_, 0, v___x_1889_);
lean_ctor_set(v___x_1892_, 1, v___x_1891_);
v___x_1893_ = l_Lean_Syntax_node1(v___x_1889_, v___x_1743_, v___x_1745_);
v___x_1894_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__48));
v___x_1895_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__49));
v___x_1896_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1896_, 0, v___x_1889_);
lean_ctor_set(v___x_1896_, 1, v___x_1895_);
v___x_1897_ = l_Lean_Syntax_node2(v___x_1889_, v___x_1894_, v___x_1896_, v___x_1887_);
v___x_1898_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1899_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1899_, 0, v___x_1889_);
lean_ctor_set(v___x_1899_, 1, v___x_1898_);
v___x_1900_ = l_Lean_Syntax_node5(v___x_1889_, v___x_1890_, v___x_1892_, v___x_1893_, v___x_1897_, v___x_1899_, v___x_1888_);
v_stx_1360_ = v___x_1900_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1901_; uint8_t v___x_1902_; 
v___x_1901_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1901_);
v___x_1902_ = l_Lean_Syntax_isOfKind(v___x_1901_, v___x_1746_);
if (v___x_1902_ == 0)
{
lean_dec(v___x_1901_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1903_; 
v___x_1903_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1901_);
lean_dec(v___x_1901_);
if (v___x_1903_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v___x_1918_; 
lean_dec(v_a_1510_);
v_ref_1904_ = lean_ctor_get(v___y_1497_, 5);
v___x_1905_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1906_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1907_ = l_Lean_SourceInfo_fromRef(v_ref_1904_, v___x_1758_);
v___x_1908_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1909_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1907_, 5);
v___x_1910_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1910_, 0, v___x_1907_);
lean_ctor_set(v___x_1910_, 1, v___x_1909_);
v___x_1911_ = l_Lean_Syntax_node1(v___x_1907_, v___x_1743_, v___x_1745_);
v___x_1912_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__51));
v___x_1913_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__52));
v___x_1914_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1914_, 0, v___x_1907_);
lean_ctor_set(v___x_1914_, 1, v___x_1913_);
v___x_1915_ = l_Lean_Syntax_node2(v___x_1907_, v___x_1912_, v___x_1914_, v___x_1905_);
v___x_1916_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1917_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1917_, 0, v___x_1907_);
lean_ctor_set(v___x_1917_, 1, v___x_1916_);
v___x_1918_ = l_Lean_Syntax_node5(v___x_1907_, v___x_1908_, v___x_1910_, v___x_1911_, v___x_1915_, v___x_1917_, v___x_1906_);
v_stx_1360_ = v___x_1918_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1919_; uint8_t v___x_1920_; 
v___x_1919_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1919_);
v___x_1920_ = l_Lean_Syntax_isOfKind(v___x_1919_, v___x_1746_);
if (v___x_1920_ == 0)
{
lean_dec(v___x_1919_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1921_; 
v___x_1921_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1919_);
lean_dec(v___x_1919_);
if (v___x_1921_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; 
lean_dec(v_a_1510_);
v_ref_1922_ = lean_ctor_get(v___y_1497_, 5);
v___x_1923_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1924_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1925_ = l_Lean_SourceInfo_fromRef(v_ref_1922_, v___x_1756_);
v___x_1926_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1927_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1925_, 5);
v___x_1928_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1928_, 0, v___x_1925_);
lean_ctor_set(v___x_1928_, 1, v___x_1927_);
v___x_1929_ = l_Lean_Syntax_node1(v___x_1925_, v___x_1743_, v___x_1745_);
v___x_1930_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__54));
v___x_1931_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__55));
v___x_1932_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1932_, 0, v___x_1925_);
lean_ctor_set(v___x_1932_, 1, v___x_1931_);
v___x_1933_ = l_Lean_Syntax_node2(v___x_1925_, v___x_1930_, v___x_1932_, v___x_1923_);
v___x_1934_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1935_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1935_, 0, v___x_1925_);
lean_ctor_set(v___x_1935_, 1, v___x_1934_);
v___x_1936_ = l_Lean_Syntax_node5(v___x_1925_, v___x_1926_, v___x_1928_, v___x_1929_, v___x_1933_, v___x_1935_, v___x_1924_);
v_stx_1360_ = v___x_1936_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
}
}
}
else
{
lean_object* v___x_1937_; uint8_t v___x_1938_; 
v___x_1937_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1358_);
lean_inc(v___x_1937_);
v___x_1938_ = l_Lean_Syntax_isOfKind(v___x_1937_, v___x_1746_);
if (v___x_1938_ == 0)
{
lean_dec(v___x_1937_);
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
uint8_t v___x_1939_; 
v___x_1939_ = l_Lean_Syntax_structEq(v___x_1745_, v___x_1937_);
lean_dec(v___x_1937_);
if (v___x_1939_ == 0)
{
lean_dec(v___x_1754_);
lean_dec(v___x_1751_);
lean_dec(v___x_1745_);
v_stx_1360_ = v_a_1510_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
}
else
{
lean_object* v_ref_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; uint8_t v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; 
lean_dec(v_a_1510_);
v_ref_1940_ = lean_ctor_get(v___y_1497_, 5);
v___x_1941_ = l_Lean_Syntax_getArg(v___x_1754_, v___x_1355_);
lean_dec(v___x_1754_);
v___x_1942_ = l_Lean_Syntax_getArg(v___x_1751_, v___x_1355_);
lean_dec(v___x_1751_);
v___x_1943_ = 0;
v___x_1944_ = l_Lean_SourceInfo_fromRef(v_ref_1940_, v___x_1943_);
v___x_1945_ = ((lean_object*)(lp_mathlib_exists__delab___lam__2___closed__6));
v___x_1946_ = ((lean_object*)(lp_mathlib_exists__delab___lam__0___closed__2));
lean_inc_n(v___x_1944_, 5);
v___x_1947_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1947_, 0, v___x_1944_);
lean_ctor_set(v___x_1947_, 1, v___x_1946_);
v___x_1948_ = l_Lean_Syntax_node1(v___x_1944_, v___x_1743_, v___x_1745_);
v___x_1949_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__5));
v___x_1950_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__6));
v___x_1951_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1951_, 0, v___x_1944_);
lean_ctor_set(v___x_1951_, 1, v___x_1950_);
v___x_1952_ = l_Lean_Syntax_node2(v___x_1944_, v___x_1949_, v___x_1951_, v___x_1941_);
v___x_1953_ = ((lean_object*)(lp_mathlib_PiNotation___aux__Mathlib__Util__Delaborators______macroRules__PiNotation__term_u03a0_____x2c____1___closed__10));
v___x_1954_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1954_, 0, v___x_1944_);
lean_ctor_set(v___x_1954_, 1, v___x_1953_);
v___x_1955_ = l_Lean_Syntax_node5(v___x_1944_, v___x_1945_, v___x_1947_, v___x_1948_, v___x_1952_, v___x_1954_, v___x_1942_);
v_stx_1360_ = v___x_1955_;
v___y_1361_ = v___y_1497_;
goto v___jp_1359_;
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
lean_object* v_a_1956_; lean_object* v___x_1958_; uint8_t v_isShared_1959_; uint8_t v_isSharedCheck_1963_; 
lean_dec(v_a_1510_);
lean_del_object(v___x_1346_);
v_a_1956_ = lean_ctor_get(v___x_1512_, 0);
v_isSharedCheck_1963_ = !lean_is_exclusive(v___x_1512_);
if (v_isSharedCheck_1963_ == 0)
{
v___x_1958_ = v___x_1512_;
v_isShared_1959_ = v_isSharedCheck_1963_;
goto v_resetjp_1957_;
}
else
{
lean_inc(v_a_1956_);
lean_dec(v___x_1512_);
v___x_1958_ = lean_box(0);
v_isShared_1959_ = v_isSharedCheck_1963_;
goto v_resetjp_1957_;
}
v_resetjp_1957_:
{
lean_object* v___x_1961_; 
if (v_isShared_1959_ == 0)
{
v___x_1961_ = v___x_1958_;
goto v_reusejp_1960_;
}
else
{
lean_object* v_reuseFailAlloc_1962_; 
v_reuseFailAlloc_1962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1962_, 0, v_a_1956_);
v___x_1961_ = v_reuseFailAlloc_1962_;
goto v_reusejp_1960_;
}
v_reusejp_1960_:
{
return v___x_1961_;
}
}
}
}
else
{
lean_del_object(v___x_1346_);
return v___x_1509_;
}
}
else
{
lean_object* v_a_1964_; lean_object* v___x_1966_; uint8_t v_isShared_1967_; uint8_t v_isSharedCheck_1971_; 
lean_dec(v_a_1500_);
lean_dec(v___x_1491_);
lean_del_object(v___x_1346_);
v_a_1964_ = lean_ctor_get(v___x_1502_, 0);
v_isSharedCheck_1971_ = !lean_is_exclusive(v___x_1502_);
if (v_isSharedCheck_1971_ == 0)
{
v___x_1966_ = v___x_1502_;
v_isShared_1967_ = v_isSharedCheck_1971_;
goto v_resetjp_1965_;
}
else
{
lean_inc(v_a_1964_);
lean_dec(v___x_1502_);
v___x_1966_ = lean_box(0);
v_isShared_1967_ = v_isSharedCheck_1971_;
goto v_resetjp_1965_;
}
v_resetjp_1965_:
{
lean_object* v___x_1969_; 
if (v_isShared_1967_ == 0)
{
v___x_1969_ = v___x_1966_;
goto v_reusejp_1968_;
}
else
{
lean_object* v_reuseFailAlloc_1970_; 
v_reuseFailAlloc_1970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1970_, 0, v_a_1964_);
v___x_1969_ = v_reuseFailAlloc_1970_;
goto v_reusejp_1968_;
}
v_reusejp_1968_:
{
return v___x_1969_;
}
}
}
}
else
{
lean_object* v_a_1972_; lean_object* v___x_1974_; uint8_t v_isShared_1975_; uint8_t v_isSharedCheck_1979_; 
lean_dec(v___x_1491_);
lean_del_object(v___x_1346_);
v_a_1972_ = lean_ctor_get(v___x_1499_, 0);
v_isSharedCheck_1979_ = !lean_is_exclusive(v___x_1499_);
if (v_isSharedCheck_1979_ == 0)
{
v___x_1974_ = v___x_1499_;
v_isShared_1975_ = v_isSharedCheck_1979_;
goto v_resetjp_1973_;
}
else
{
lean_inc(v_a_1972_);
lean_dec(v___x_1499_);
v___x_1974_ = lean_box(0);
v_isShared_1975_ = v_isSharedCheck_1979_;
goto v_resetjp_1973_;
}
v_resetjp_1973_:
{
lean_object* v___x_1977_; 
if (v_isShared_1975_ == 0)
{
v___x_1977_ = v___x_1974_;
goto v_reusejp_1976_;
}
else
{
lean_object* v_reuseFailAlloc_1978_; 
v_reuseFailAlloc_1978_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1978_, 0, v_a_1972_);
v___x_1977_ = v_reuseFailAlloc_1978_;
goto v_reusejp_1976_;
}
v_reusejp_1976_:
{
return v___x_1977_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___lam__2___boxed(lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_){
_start:
{
lean_object* v_res_1998_; 
v_res_1998_ = lp_mathlib_exists__delab___lam__2(v___y_1991_, v___y_1992_, v___y_1993_, v___y_1994_, v___y_1995_, v___y_1996_);
lean_dec(v___y_1996_);
lean_dec_ref(v___y_1995_);
lean_dec(v___y_1994_);
lean_dec_ref(v___y_1993_);
lean_dec(v___y_1992_);
lean_dec_ref(v___y_1991_);
return v_res_1998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab(lean_object* v_a_2000_, lean_object* v_a_2001_, lean_object* v_a_2002_, lean_object* v_a_2003_, lean_object* v_a_2004_, lean_object* v_a_2005_){
_start:
{
lean_object* v___f_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; 
v___f_2007_ = ((lean_object*)(lp_mathlib_exists__delab___closed__0));
v___x_2008_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___closed__2));
v___x_2009_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_2008_, v___f_2007_, v_a_2000_, v_a_2001_, v_a_2002_, v_a_2003_, v_a_2004_, v_a_2005_);
return v___x_2009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_exists__delab___boxed(lean_object* v_a_2010_, lean_object* v_a_2011_, lean_object* v_a_2012_, lean_object* v_a_2013_, lean_object* v_a_2014_, lean_object* v_a_2015_, lean_object* v_a_2016_){
_start:
{
lean_object* v_res_2017_; 
v_res_2017_ = lp_mathlib_exists__delab(v_a_2010_, v_a_2011_, v_a_2012_, v_a_2013_, v_a_2014_, v_a_2015_);
lean_dec(v_a_2015_);
lean_dec_ref(v_a_2014_);
lean_dec(v_a_2013_);
lean_dec_ref(v_a_2012_);
lean_dec(v_a_2011_);
lean_dec_ref(v_a_2010_);
return v_res_2017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4(lean_object* v_00_u03b1_2018_, lean_object* v_child_2019_, lean_object* v_childIdx_2020_, lean_object* v_x_2021_, lean_object* v___y_2022_, lean_object* v___y_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_){
_start:
{
lean_object* v___x_2029_; 
v___x_2029_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___redArg(v_child_2019_, v_childIdx_2020_, v_x_2021_, v___y_2022_, v___y_2023_, v___y_2024_, v___y_2025_, v___y_2026_, v___y_2027_);
return v___x_2029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4___boxed(lean_object* v_00_u03b1_2030_, lean_object* v_child_2031_, lean_object* v_childIdx_2032_, lean_object* v_x_2033_, lean_object* v___y_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_, lean_object* v___y_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_){
_start:
{
lean_object* v_res_2041_; 
v_res_2041_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4_spec__4(v_00_u03b1_2030_, v_child_2031_, v_childIdx_2032_, v_x_2033_, v___y_2034_, v___y_2035_, v___y_2036_, v___y_2037_, v___y_2038_, v___y_2039_);
lean_dec(v___y_2039_);
lean_dec_ref(v___y_2038_);
lean_dec(v___y_2037_);
lean_dec_ref(v___y_2036_);
lean_dec(v___y_2035_);
lean_dec_ref(v___y_2034_);
return v_res_2041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4(lean_object* v_00_u03b1_2042_, lean_object* v_x_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_){
_start:
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___redArg(v_x_2043_, v___y_2044_, v___y_2045_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4___boxed(lean_object* v_00_u03b1_2052_, lean_object* v_x_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_){
_start:
{
lean_object* v_res_2061_; 
v_res_2061_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingDomain___at___00exists__delab_spec__4(v_00_u03b1_2052_, v_x_2053_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_, v___y_2059_);
lean_dec(v___y_2059_);
lean_dec_ref(v___y_2058_);
lean_dec(v___y_2057_);
lean_dec_ref(v___y_2056_);
lean_dec(v___y_2055_);
lean_dec_ref(v___y_2054_);
return v_res_2061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5(lean_object* v_00_u03b1_2062_, lean_object* v_x_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_){
_start:
{
lean_object* v___x_2071_; 
v___x_2071_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(v_x_2063_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_, v___y_2068_, v___y_2069_);
return v___x_2071_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___boxed(lean_object* v_00_u03b1_2072_, lean_object* v_x_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_, lean_object* v___y_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_){
_start:
{
lean_object* v_res_2081_; 
v_res_2081_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5(v_00_u03b1_2072_, v_x_2073_, v___y_2074_, v___y_2075_, v___y_2076_, v___y_2077_, v___y_2078_, v___y_2079_);
lean_dec(v___y_2079_);
lean_dec_ref(v___y_2078_);
lean_dec(v___y_2077_);
lean_dec_ref(v___y_2076_);
lean_dec(v___y_2075_);
lean_dec_ref(v___y_2074_);
return v_res_2081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg(lean_object* v___y_2082_){
_start:
{
lean_object* v_subExpr_2084_; lean_object* v_pos_2085_; lean_object* v___x_2086_; 
v_subExpr_2084_ = lean_ctor_get(v___y_2082_, 3);
v_pos_2085_ = lean_ctor_get(v_subExpr_2084_, 1);
lean_inc(v_pos_2085_);
v___x_2086_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2086_, 0, v_pos_2085_);
return v___x_2086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg___boxed(lean_object* v___y_2087_, lean_object* v___y_2088_){
_start:
{
lean_object* v_res_2089_; 
v_res_2089_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg(v___y_2087_);
lean_dec_ref(v___y_2087_);
return v_res_2089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___redArg(lean_object* v_argIdx_2090_, lean_object* v_x_2091_, lean_object* v___y_2092_, lean_object* v___y_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_){
_start:
{
lean_object* v___x_2099_; lean_object* v_a_2100_; lean_object* v___x_2101_; lean_object* v_a_2102_; lean_object* v_optionsPerPos_2103_; lean_object* v_currNamespace_2104_; lean_object* v_openDecls_2105_; uint8_t v_inPattern_2106_; lean_object* v_depth_2107_; lean_object* v_lctxInitIndices_2108_; lean_object* v_nargs_2109_; lean_object* v___x_2110_; lean_object* v_dummy_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v_args_2115_; lean_object* v___x_2116_; lean_object* v_newPos_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; 
v___x_2099_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_2092_);
v_a_2100_ = lean_ctor_get(v___x_2099_, 0);
lean_inc(v_a_2100_);
lean_dec_ref(v___x_2099_);
v___x_2101_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg(v___y_2092_);
v_a_2102_ = lean_ctor_get(v___x_2101_, 0);
lean_inc(v_a_2102_);
lean_dec_ref(v___x_2101_);
v_optionsPerPos_2103_ = lean_ctor_get(v___y_2092_, 0);
v_currNamespace_2104_ = lean_ctor_get(v___y_2092_, 1);
v_openDecls_2105_ = lean_ctor_get(v___y_2092_, 2);
v_inPattern_2106_ = lean_ctor_get_uint8(v___y_2092_, sizeof(void*)*6);
v_depth_2107_ = lean_ctor_get(v___y_2092_, 4);
v_lctxInitIndices_2108_ = lean_ctor_get(v___y_2092_, 5);
v_nargs_2109_ = l_Lean_Expr_getAppNumArgs(v_a_2100_);
v___x_2110_ = l_Lean_instInhabitedExpr;
v_dummy_2111_ = lean_obj_once(&lp_mathlib_exists__delab___lam__2___closed__0, &lp_mathlib_exists__delab___lam__2___closed__0_once, _init_lp_mathlib_exists__delab___lam__2___closed__0);
lean_inc(v_nargs_2109_);
v___x_2112_ = lean_mk_array(v_nargs_2109_, v_dummy_2111_);
v___x_2113_ = lean_unsigned_to_nat(1u);
v___x_2114_ = lean_nat_sub(v_nargs_2109_, v___x_2113_);
lean_dec(v_nargs_2109_);
v_args_2115_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_2100_, v___x_2112_, v___x_2114_);
v___x_2116_ = lean_array_get_size(v_args_2115_);
v_newPos_2117_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_2116_, v_argIdx_2090_, v_a_2102_);
lean_dec(v_a_2102_);
v___x_2118_ = lean_array_get(v___x_2110_, v_args_2115_, v_argIdx_2090_);
lean_dec_ref(v_args_2115_);
v___x_2119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2119_, 0, v___x_2118_);
lean_ctor_set(v___x_2119_, 1, v_newPos_2117_);
lean_inc(v_lctxInitIndices_2108_);
lean_inc(v_depth_2107_);
lean_inc(v_openDecls_2105_);
lean_inc(v_currNamespace_2104_);
lean_inc(v_optionsPerPos_2103_);
v___x_2120_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_2120_, 0, v_optionsPerPos_2103_);
lean_ctor_set(v___x_2120_, 1, v_currNamespace_2104_);
lean_ctor_set(v___x_2120_, 2, v_openDecls_2105_);
lean_ctor_set(v___x_2120_, 3, v___x_2119_);
lean_ctor_set(v___x_2120_, 4, v_depth_2107_);
lean_ctor_set(v___x_2120_, 5, v_lctxInitIndices_2108_);
lean_ctor_set_uint8(v___x_2120_, sizeof(void*)*6, v_inPattern_2106_);
lean_inc(v___y_2097_);
lean_inc_ref(v___y_2096_);
lean_inc(v___y_2095_);
lean_inc_ref(v___y_2094_);
lean_inc(v___y_2093_);
v___x_2121_ = lean_apply_7(v_x_2091_, v___x_2120_, v___y_2093_, v___y_2094_, v___y_2095_, v___y_2096_, v___y_2097_, lean_box(0));
return v___x_2121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___redArg___boxed(lean_object* v_argIdx_2122_, lean_object* v_x_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_){
_start:
{
lean_object* v_res_2131_; 
v_res_2131_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___redArg(v_argIdx_2122_, v_x_2123_, v___y_2124_, v___y_2125_, v___y_2126_, v___y_2127_, v___y_2128_, v___y_2129_);
lean_dec(v___y_2129_);
lean_dec_ref(v___y_2128_);
lean_dec(v___y_2127_);
lean_dec_ref(v___y_2126_);
lean_dec(v___y_2125_);
lean_dec_ref(v___y_2124_);
lean_dec(v_argIdx_2122_);
return v_res_2131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0(lean_object* v_00_u03b1_2132_, lean_object* v_argIdx_2133_, lean_object* v_x_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_, lean_object* v___y_2140_){
_start:
{
lean_object* v___x_2142_; 
v___x_2142_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___redArg(v_argIdx_2133_, v_x_2134_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_, v___y_2139_, v___y_2140_);
return v___x_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0___boxed(lean_object* v_00_u03b1_2143_, lean_object* v_argIdx_2144_, lean_object* v_x_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_){
_start:
{
lean_object* v_res_2153_; 
v_res_2153_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0(v_00_u03b1_2143_, v_argIdx_2144_, v_x_2145_, v___y_2146_, v___y_2147_, v___y_2148_, v___y_2149_, v___y_2150_, v___y_2151_);
lean_dec(v___y_2151_);
lean_dec_ref(v___y_2150_);
lean_dec(v___y_2149_);
lean_dec_ref(v___y_2148_);
lean_dec(v___y_2147_);
lean_dec_ref(v___y_2146_);
lean_dec(v_argIdx_2144_);
return v_res_2153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn___lam__0(lean_object* v___y_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_){
_start:
{
lean_object* v___x_2193_; lean_object* v_a_2194_; lean_object* v_dummy_2195_; lean_object* v_nargs_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; uint8_t v___x_2202_; 
v___x_2193_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00exists__delab_spec__0___redArg(v___y_2165_);
v_a_2194_ = lean_ctor_get(v___x_2193_, 0);
lean_inc(v_a_2194_);
lean_dec_ref(v___x_2193_);
v_dummy_2195_ = lean_obj_once(&lp_mathlib_exists__delab___lam__2___closed__0, &lp_mathlib_exists__delab___lam__2___closed__0_once, _init_lp_mathlib_exists__delab___lam__2___closed__0);
v_nargs_2196_ = l_Lean_Expr_getAppNumArgs(v_a_2194_);
lean_inc(v_nargs_2196_);
v___x_2197_ = lean_mk_array(v_nargs_2196_, v_dummy_2195_);
v___x_2198_ = lean_unsigned_to_nat(1u);
v___x_2199_ = lean_nat_sub(v_nargs_2196_, v___x_2198_);
lean_dec(v_nargs_2196_);
v___x_2200_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_2194_, v___x_2197_, v___x_2199_);
v___x_2201_ = lean_array_get_size(v___x_2200_);
v___x_2202_ = lean_nat_dec_eq(v___x_2201_, v___x_2198_);
if (v___x_2202_ == 0)
{
lean_object* v___x_2203_; 
lean_dec_ref(v___x_2200_);
v___x_2203_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_2203_;
}
else
{
lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; uint8_t v___x_2208_; 
v___x_2204_ = lean_unsigned_to_nat(0u);
v___x_2205_ = lean_array_fget(v___x_2200_, v___x_2204_);
lean_dec_ref(v___x_2200_);
v___x_2206_ = ((lean_object*)(lp_mathlib_delabNotIn___lam__0___closed__4));
v___x_2207_ = lean_unsigned_to_nat(5u);
v___x_2208_ = l_Lean_Expr_isAppOfArity(v___x_2205_, v___x_2206_, v___x_2207_);
lean_dec(v___x_2205_);
if (v___x_2208_ == 0)
{
lean_object* v___x_2209_; 
v___x_2209_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_2209_) == 0)
{
lean_dec_ref_known(v___x_2209_, 1);
goto v___jp_2172_;
}
else
{
lean_object* v_a_2210_; lean_object* v___x_2212_; uint8_t v_isShared_2213_; uint8_t v_isSharedCheck_2217_; 
v_a_2210_ = lean_ctor_get(v___x_2209_, 0);
v_isSharedCheck_2217_ = !lean_is_exclusive(v___x_2209_);
if (v_isSharedCheck_2217_ == 0)
{
v___x_2212_ = v___x_2209_;
v_isShared_2213_ = v_isSharedCheck_2217_;
goto v_resetjp_2211_;
}
else
{
lean_inc(v_a_2210_);
lean_dec(v___x_2209_);
v___x_2212_ = lean_box(0);
v_isShared_2213_ = v_isSharedCheck_2217_;
goto v_resetjp_2211_;
}
v_resetjp_2211_:
{
lean_object* v___x_2215_; 
if (v_isShared_2213_ == 0)
{
v___x_2215_ = v___x_2212_;
goto v_reusejp_2214_;
}
else
{
lean_object* v_reuseFailAlloc_2216_; 
v_reuseFailAlloc_2216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2216_, 0, v_a_2210_);
v___x_2215_ = v_reuseFailAlloc_2216_;
goto v_reusejp_2214_;
}
v_reusejp_2214_:
{
return v___x_2215_;
}
}
}
}
else
{
goto v___jp_2172_;
}
}
v___jp_2172_:
{
lean_object* v___x_2173_; lean_object* v___x_2174_; 
v___x_2173_ = ((lean_object*)(lp_mathlib_delabNotIn___lam__0___closed__0));
v___x_2174_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(v___x_2173_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_);
if (lean_obj_tag(v___x_2174_) == 0)
{
lean_object* v_a_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; 
v_a_2175_ = lean_ctor_get(v___x_2174_, 0);
lean_inc(v_a_2175_);
lean_dec_ref_known(v___x_2174_, 1);
v___x_2176_ = ((lean_object*)(lp_mathlib_delabNotIn___lam__0___closed__1));
v___x_2177_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00exists__delab_spec__5___redArg(v___x_2176_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_, v___y_2169_, v___y_2170_);
if (lean_obj_tag(v___x_2177_) == 0)
{
lean_object* v_a_2178_; lean_object* v___x_2180_; uint8_t v_isShared_2181_; uint8_t v_isSharedCheck_2192_; 
v_a_2178_ = lean_ctor_get(v___x_2177_, 0);
v_isSharedCheck_2192_ = !lean_is_exclusive(v___x_2177_);
if (v_isSharedCheck_2192_ == 0)
{
v___x_2180_ = v___x_2177_;
v_isShared_2181_ = v_isSharedCheck_2192_;
goto v_resetjp_2179_;
}
else
{
lean_inc(v_a_2178_);
lean_dec(v___x_2177_);
v___x_2180_ = lean_box(0);
v_isShared_2181_ = v_isSharedCheck_2192_;
goto v_resetjp_2179_;
}
v_resetjp_2179_:
{
lean_object* v_ref_2182_; uint8_t v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2190_; 
v_ref_2182_ = lean_ctor_get(v___y_2169_, 5);
v___x_2183_ = 0;
v___x_2184_ = l_Lean_SourceInfo_fromRef(v_ref_2182_, v___x_2183_);
v___x_2185_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__16));
v___x_2186_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___lam__0___closed__43));
lean_inc(v___x_2184_);
v___x_2187_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2187_, 0, v___x_2184_);
lean_ctor_set(v___x_2187_, 1, v___x_2186_);
v___x_2188_ = l_Lean_Syntax_node3(v___x_2184_, v___x_2185_, v_a_2178_, v___x_2187_, v_a_2175_);
if (v_isShared_2181_ == 0)
{
lean_ctor_set(v___x_2180_, 0, v___x_2188_);
v___x_2190_ = v___x_2180_;
goto v_reusejp_2189_;
}
else
{
lean_object* v_reuseFailAlloc_2191_; 
v_reuseFailAlloc_2191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2191_, 0, v___x_2188_);
v___x_2190_ = v_reuseFailAlloc_2191_;
goto v_reusejp_2189_;
}
v_reusejp_2189_:
{
return v___x_2190_;
}
}
}
else
{
lean_dec(v_a_2175_);
return v___x_2177_;
}
}
else
{
return v___x_2174_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn___lam__0___boxed(lean_object* v___y_2218_, lean_object* v___y_2219_, lean_object* v___y_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_){
_start:
{
lean_object* v_res_2225_; 
v_res_2225_ = lp_mathlib_delabNotIn___lam__0(v___y_2218_, v___y_2219_, v___y_2220_, v___y_2221_, v___y_2222_, v___y_2223_);
lean_dec(v___y_2223_);
lean_dec_ref(v___y_2222_);
lean_dec(v___y_2221_);
lean_dec_ref(v___y_2220_);
lean_dec(v___y_2219_);
lean_dec_ref(v___y_2218_);
return v_res_2225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn(lean_object* v_a_2227_, lean_object* v_a_2228_, lean_object* v_a_2229_, lean_object* v_a_2230_, lean_object* v_a_2231_, lean_object* v_a_2232_){
_start:
{
lean_object* v___f_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; 
v___f_2234_ = ((lean_object*)(lp_mathlib_delabNotIn___closed__0));
v___x_2235_ = ((lean_object*)(lp_mathlib_PiNotation_delabPi___closed__2));
v___x_2236_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_2235_, v___f_2234_, v_a_2227_, v_a_2228_, v_a_2229_, v_a_2230_, v_a_2231_, v_a_2232_);
return v___x_2236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_delabNotIn___boxed(lean_object* v_a_2237_, lean_object* v_a_2238_, lean_object* v_a_2239_, lean_object* v_a_2240_, lean_object* v_a_2241_, lean_object* v_a_2242_, lean_object* v_a_2243_){
_start:
{
lean_object* v_res_2244_; 
v_res_2244_ = lp_mathlib_delabNotIn(v_a_2237_, v_a_2238_, v_a_2239_, v_a_2240_, v_a_2241_, v_a_2242_);
lean_dec(v_a_2242_);
lean_dec_ref(v_a_2241_);
lean_dec(v_a_2240_);
lean_dec_ref(v_a_2239_);
lean_dec(v_a_2238_);
lean_dec_ref(v_a_2237_);
return v_res_2244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0(lean_object* v___y_2245_, lean_object* v___y_2246_, lean_object* v___y_2247_, lean_object* v___y_2248_, lean_object* v___y_2249_, lean_object* v___y_2250_){
_start:
{
lean_object* v___x_2252_; 
v___x_2252_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___redArg(v___y_2245_);
return v___x_2252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0___boxed(lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_){
_start:
{
lean_object* v_res_2260_; 
v_res_2260_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00delabNotIn_spec__0_spec__0(v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_, v___y_2257_, v___y_2258_);
lean_dec(v___y_2258_);
lean_dec_ref(v___y_2257_);
lean_dec(v___y_2256_);
lean_dec_ref(v___y_2255_);
lean_dec(v___y_2254_);
lean_dec_ref(v___y_2253_);
return v_res_2260_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_PPOptions(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Util_PPOptions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_PiNotation_piNotation = _init_lp_mathlib_PiNotation_piNotation();
lean_mark_persistent(lp_mathlib_PiNotation_piNotation);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_PPOptions(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_Delaborators(uint8_t builtin) {
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
res = initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_PPOptions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_Delaborators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_Delaborators(builtin);
}
#ifdef __cplusplus
}
#endif
