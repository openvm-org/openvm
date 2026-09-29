// Lean compiler output
// Module: Mathlib.Data.Fintype.Defs
// Imports: public import Init public meta import Init public import Mathlib.Basic.Finite.Defs public import Mathlib.Data.Finset.Filter public import Mathlib.Order.Lex
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
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
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
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
uint8_t l_Lean_Expr_binderInfo(lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
uint8_t l_List_nodupDecidable___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* lean_array_fget(lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPFunBinderTypes___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isLambda(lean_object*);
lean_object* lp_mathlib_Multiset_pmap___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "setBuilder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__2_value),LEAN_SCALAR_PTR_LITERAL(55, 252, 174, 2, 80, 49, 173, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__6_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__9_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__13_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∉_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__15_value),LEAN_SCALAR_PTR_LITERAL(147, 253, 164, 249, 200, 108, 121, 70)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≠_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__17_value),LEAN_SCALAR_PTR_LITERAL(39, 40, 245, 52, 138, 78, 140, 19)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__21_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Finset.filter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__23_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "filter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__25_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__26_value),LEAN_SCALAR_PTR_LITERAL(88, 243, 224, 152, 142, 113, 169, 220)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__28_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__30_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__32_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__34_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__37_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__41_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__42_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "SubExpr"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__44_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__43_value),LEAN_SCALAR_PTR_LITERAL(170, 131, 175, 90, 105, 49, 153, 209)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__44_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "PrettyPrinter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__46_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Delaborator"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__47_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__46_value),LEAN_SCALAR_PTR_LITERAL(120, 167, 117, 148, 131, 202, 42, 4)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__47_value),LEAN_SCALAR_PTR_LITERAL(79, 126, 247, 124, 241, 28, 11, 244)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__43_value),LEAN_SCALAR_PTR_LITERAL(231, 152, 1, 212, 81, 225, 23, 202)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__48_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__46_value),LEAN_SCALAR_PTR_LITERAL(120, 167, 117, 148, 131, 202, 42, 4)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__47_value),LEAN_SCALAR_PTR_LITERAL(79, 126, 247, 124, 241, 28, 11, 244)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__50_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__52_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__54_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__54_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__56_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__56_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__57_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__58_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__59_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__61_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__61_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__58_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__62_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__63_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__63_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__58_value),LEAN_SCALAR_PTR_LITERAL(228, 185, 96, 51, 222, 54, 124, 240)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__63_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__64_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__65_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__65_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__66_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__67_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__67_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__68_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__68_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__70_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__70_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__71_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__71_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__72_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__69_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__73_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__74_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__66_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__74_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__75_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__64_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__75_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__76_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__62_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__76_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__77_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__60_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__77_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__42_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__78_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__57_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__79_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__80_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__55_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__80_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__81_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__53_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__81_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__82_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__51_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__82_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__83 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__83_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__49_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__83_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__84_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__45_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__84_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__42_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__85_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__89 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__89_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__89_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↦"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_ᶜ"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__94 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__94_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__94_value),LEAN_SCALAR_PTR_LITERAL(128, 3, 137, 103, 191, 193, 176, 89)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__95_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "singleton"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__96_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__97_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__97;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__96_value),LEAN_SCALAR_PTR_LITERAL(208, 33, 246, 107, 223, 5, 156, 82)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__98_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Singleton"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__99 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__99_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__99_value),LEAN_SCALAR_PTR_LITERAL(190, 73, 36, 155, 228, 35, 161, 122)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__96_value),LEAN_SCALAR_PTR_LITERAL(185, 48, 115, 60, 21, 14, 217, 215)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__101_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__101_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__102_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ᶜ"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__103_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__25_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__104 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__104_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__105 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__105_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__19_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__20_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__105_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__107 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__107_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Finset.univ"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__108 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__108_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "univ"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__110 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__110_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__25_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__110_value),LEAN_SCALAR_PTR_LITERAL(177, 108, 234, 69, 25, 31, 35, 26)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__112 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__112_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__112_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__113 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__113_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Compl"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "compl"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(86, 104, 100, 165, 159, 188, 212, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(162, 20, 47, 39, 134, 212, 205, 41)}};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred∈_"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__8_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(150, 164, 254, 63, 76, 57, 126, 92)}};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∈"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∉"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__11_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__12_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___boxed, .m_arity = 10, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__12_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≠"};
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__14_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPFunBinderTypes___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidablePiFintype___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidablePiFintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidablePiFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidablePiFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidablePiFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidablePiFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableForallFintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableForallFintype___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableForallFintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableForallFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableExistsFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableExistsFintype___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableExistsFintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableExistsFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_instDecidableLEForall___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instDecidableLEForall___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_instDecidableLEForall___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instDecidableLEForall___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_instDecidableLEForall(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instDecidableLEForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableMemRangeFintype___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableMemRangeFintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableMemRangeFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableMemRangeFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableMemRangeFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableMemRangeFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEquivFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEquivFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEquivFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEquivFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEmbeddingFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableInjectiveFintype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableInjectiveFintype___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableInjectiveFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableInjectiveFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fintype_subtype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fintype_subtype___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fintype_subtype___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fintype_subtype___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofFinset(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Bool_fintype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Bool_fintype___closed__0 = (const lean_object*)&lp_mathlib_Bool_fintype___closed__0_value;
static const lean_ctor_object lp_mathlib_Bool_fintype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Bool_fintype___closed__0_value)}};
static const lean_object* lp_mathlib_Bool_fintype___closed__1 = (const lean_object*)&lp_mathlib_Bool_fintype___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Bool_fintype = (const lean_object*)&lp_mathlib_Bool_fintype___closed__1_value;
static const lean_ctor_object lp_mathlib_Ordering_fintype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ordering_fintype___closed__0 = (const lean_object*)&lp_mathlib_Ordering_fintype___closed__0_value;
static const lean_ctor_object lp_mathlib_Ordering_fintype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_Ordering_fintype___closed__0_value)}};
static const lean_object* lp_mathlib_Ordering_fintype___closed__1 = (const lean_object*)&lp_mathlib_Ordering_fintype___closed__1_value;
static const lean_ctor_object lp_mathlib_Ordering_fintype___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Ordering_fintype___closed__1_value)}};
static const lean_object* lp_mathlib_Ordering_fintype___closed__2 = (const lean_object*)&lp_mathlib_Ordering_fintype___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Ordering_fintype = (const lean_object*)&lp_mathlib_Ordering_fintype___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg(lean_object* v_f_1_, lean_object* v_s_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
v___x_4_ = lp_mathlib_Multiset_map___redArg(v_f_1_, v_s_2_);
v___x_5_ = l_List_nodupDecidable___redArg(v_inst_3_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg___boxed(lean_object* v_f_6_, lean_object* v_s_7_, lean_object* v_inst_8_){
_start:
{
uint8_t v_res_9_; lean_object* v_r_10_; 
v_res_9_ = lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg(v_f_6_, v_s_7_, v_inst_8_);
v_r_10_ = lean_box(v_res_9_);
return v_r_10_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_f_13_, lean_object* v_s_14_, lean_object* v_inst_15_){
_start:
{
uint8_t v___x_16_; 
v___x_16_ = lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg(v_f_13_, v_s_14_, v_inst_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___boxed(lean_object* v_00_u03b1_17_, lean_object* v_00_u03b2_18_, lean_object* v_f_19_, lean_object* v_s_20_, lean_object* v_inst_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq(v_00_u03b1_17_, v_00_u03b2_18_, v_f_19_, v_s_20_, v_inst_21_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0(lean_object* v_inst_24_, lean_object* v_t_x27_25_, lean_object* v_a_26_){
_start:
{
uint8_t v___x_27_; 
v___x_27_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_24_, v_a_26_, v_t_x27_25_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0___boxed(lean_object* v_inst_28_, lean_object* v_t_x27_29_, lean_object* v_a_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0(v_inst_28_, v_t_x27_29_, v_a_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg(lean_object* v_f_33_, lean_object* v_s_34_, lean_object* v_t_x27_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___f_37_; uint8_t v___x_38_; uint8_t v___x_39_; uint8_t v___x_40_; 
lean_inc(v_t_x27_35_);
lean_inc_ref_n(v_inst_36_, 2);
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_37_, 0, v_inst_36_);
lean_closure_set(v___f_37_, 1, v_t_x27_35_);
lean_inc_n(v_s_34_, 2);
lean_inc_n(v_f_33_, 2);
v___x_38_ = lp_mathlib_List_instDecidableSurjOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_33_, v_s_34_, v_t_x27_35_, v_inst_36_);
v___x_39_ = lp_mathlib_List_instDecidableInjOnCoeFinsetOfDecidableEq___redArg(v_f_33_, v_s_34_, v_inst_36_);
v___x_40_ = lp_mathlib_List_instDecidableMapsToCoeFinsetOfDecidablePredMemSet___aux__1___redArg(v_f_33_, v_s_34_, v___f_37_);
if (v___x_40_ == 0)
{
return v___x_40_;
}
else
{
if (v___x_39_ == 0)
{
return v___x_39_;
}
else
{
return v___x_38_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg___boxed(lean_object* v_f_41_, lean_object* v_s_42_, lean_object* v_t_x27_43_, lean_object* v_inst_44_){
_start:
{
uint8_t v_res_45_; lean_object* v_r_46_; 
v_res_45_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_41_, v_s_42_, v_t_x27_43_, v_inst_44_);
v_r_46_ = lean_box(v_res_45_);
return v_r_46_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1(lean_object* v_00_u03b1_47_, lean_object* v_00_u03b2_48_, lean_object* v_f_49_, lean_object* v_s_50_, lean_object* v_t_x27_51_, lean_object* v_inst_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_49_, v_s_50_, v_t_x27_51_, v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___boxed(lean_object* v_00_u03b1_54_, lean_object* v_00_u03b2_55_, lean_object* v_f_56_, lean_object* v_s_57_, lean_object* v_t_x27_58_, lean_object* v_inst_59_){
_start:
{
uint8_t v_res_60_; lean_object* v_r_61_; 
v_res_60_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1(v_00_u03b1_54_, v_00_u03b2_55_, v_f_56_, v_s_57_, v_t_x27_58_, v_inst_59_);
v_r_61_ = lean_box(v_res_60_);
return v_r_61_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___redArg(lean_object* v_f_62_, lean_object* v_s_63_, lean_object* v_t_x27_64_, lean_object* v_inst_65_){
_start:
{
uint8_t v___x_66_; 
v___x_66_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_62_, v_s_63_, v_t_x27_64_, v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___redArg___boxed(lean_object* v_f_67_, lean_object* v_s_68_, lean_object* v_t_x27_69_, lean_object* v_inst_70_){
_start:
{
uint8_t v_res_71_; lean_object* v_r_72_; 
v_res_71_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___redArg(v_f_67_, v_s_68_, v_t_x27_69_, v_inst_70_);
v_r_72_ = lean_box(v_res_71_);
return v_r_72_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq(lean_object* v_00_u03b1_73_, lean_object* v_00_u03b2_74_, lean_object* v_f_75_, lean_object* v_s_76_, lean_object* v_t_x27_77_, lean_object* v_inst_78_){
_start:
{
uint8_t v___x_79_; 
v___x_79_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___aux__1___redArg(v_f_75_, v_s_76_, v_t_x27_77_, v_inst_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq___boxed(lean_object* v_00_u03b1_80_, lean_object* v_00_u03b2_81_, lean_object* v_f_82_, lean_object* v_s_83_, lean_object* v_t_x27_84_, lean_object* v_inst_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib_List_instDecidableBijOnCoeFinsetOfDecidableEq(v_00_u03b1_80_, v_00_u03b2_81_, v_f_82_, v_s_83_, v_t_x27_84_, v_inst_85_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ___redArg(lean_object* v_inst_88_){
_start:
{
lean_inc(v_inst_88_);
return v_inst_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ___redArg___boxed(lean_object* v_inst_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Finset_univ___redArg(v_inst_89_);
lean_dec(v_inst_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_){
_start:
{
lean_inc(v_inst_92_);
return v_inst_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_univ___boxed(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Finset_univ(v_00_u03b1_93_, v_inst_94_);
lean_dec(v_inst_94_);
return v_res_95_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_96_ = lean_box(0);
v___x_97_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_98_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v___x_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg(){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___closed__0);
v___x_101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg___boxed(lean_object* v___y_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0(lean_object* v_00_u03b1_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___boxed(lean_object* v_00_u03b1_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0(v_00_u03b1_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_);
lean_dec(v___y_119_);
lean_dec_ref(v___y_118_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
return v_res_121_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__23));
v___x_165_ = l_String_toRawSubstring_x27(v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_197_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__39));
v___x_198_ = l_String_toRawSubstring_x27(v___x_197_);
return v___x_198_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91(void){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = l_Array_mkArray0(lean_box(0));
return v___x_326_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__97(void){
_start:
{
lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_333_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__96));
v___x_334_ = l_String_toRawSubstring_x27(v___x_333_);
return v___x_334_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; 
v___x_358_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__108));
v___x_359_ = l_String_toRawSubstring_x27(v___x_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf(lean_object* v_x_370_, lean_object* v_x_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_){
_start:
{
lean_object* v___x_379_; uint8_t v___x_380_; 
v___x_379_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3));
lean_inc(v_x_370_);
v___x_380_ = l_Lean_Syntax_isOfKind(v_x_370_, v___x_379_);
if (v___x_380_ == 0)
{
lean_object* v___x_381_; 
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v___x_381_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_381_;
}
else
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; uint8_t v___x_385_; 
v___x_382_ = lean_unsigned_to_nat(1u);
v___x_383_ = l_Lean_Syntax_getArg(v_x_370_, v___x_382_);
v___x_384_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7));
lean_inc(v___x_383_);
v___x_385_ = l_Lean_Syntax_isOfKind(v___x_383_, v___x_384_);
if (v___x_385_ == 0)
{
lean_object* v___x_386_; 
lean_dec(v___x_383_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v___x_386_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_386_;
}
else
{
lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; uint8_t v___x_390_; 
v___x_387_ = lean_unsigned_to_nat(0u);
v___x_388_ = l_Lean_Syntax_getArg(v___x_383_, v___x_387_);
v___x_389_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10));
lean_inc(v___x_388_);
v___x_390_ = l_Lean_Syntax_isOfKind(v___x_388_, v___x_389_);
if (v___x_390_ == 0)
{
lean_object* v___x_391_; 
lean_dec(v___x_388_);
lean_dec(v___x_383_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v___x_391_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_391_;
}
else
{
lean_object* v___x_392_; lean_object* v___x_393_; uint8_t v___x_394_; 
v___x_392_ = l_Lean_Syntax_getArg(v___x_388_, v___x_387_);
lean_dec(v___x_388_);
v___x_393_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__12));
lean_inc(v___x_392_);
v___x_394_ = l_Lean_Syntax_isOfKind(v___x_392_, v___x_393_);
if (v___x_394_ == 0)
{
lean_object* v___x_395_; 
lean_dec(v___x_392_);
lean_dec(v___x_383_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v___x_395_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_395_;
}
else
{
lean_object* v___x_396_; uint8_t v___x_397_; 
v___x_396_ = l_Lean_Syntax_getArg(v___x_383_, v___x_382_);
lean_dec(v___x_383_);
lean_inc(v___x_396_);
v___x_397_ = l_Lean_Syntax_matchesNull(v___x_396_, v___x_387_);
if (v___x_397_ == 0)
{
uint8_t v___x_398_; 
lean_inc(v___x_396_);
v___x_398_ = l_Lean_Syntax_matchesNull(v___x_396_, v___x_382_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; 
lean_dec(v___x_396_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v___x_399_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_399_;
}
else
{
lean_object* v___x_400_; lean_object* v___x_401_; uint8_t v___x_402_; 
v___x_400_ = l_Lean_Syntax_getArg(v___x_396_, v___x_387_);
lean_dec(v___x_396_);
v___x_401_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__14));
lean_inc(v___x_400_);
v___x_402_ = l_Lean_Syntax_isOfKind(v___x_400_, v___x_401_);
if (v___x_402_ == 0)
{
lean_object* v___x_403_; uint8_t v___x_404_; 
v___x_403_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16));
lean_inc(v___x_400_);
v___x_404_ = l_Lean_Syntax_isOfKind(v___x_400_, v___x_403_);
if (v___x_404_ == 0)
{
lean_object* v___x_405_; uint8_t v___x_406_; 
v___x_405_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18));
lean_inc(v___x_400_);
v___x_406_ = l_Lean_Syntax_isOfKind(v___x_400_, v___x_405_);
if (v___x_406_ == 0)
{
lean_object* v___x_407_; 
lean_dec(v___x_400_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v___x_407_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_407_;
}
else
{
lean_object* v___x_408_; 
lean_inc(v_x_371_);
v___x_408_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_408_) == 0)
{
lean_object* v_a_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___y_414_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___y_418_; lean_object* v___y_419_; uint8_t v___x_472_; 
v_a_409_ = lean_ctor_get(v___x_408_, 0);
lean_inc(v_a_409_);
lean_dec_ref_known(v___x_408_, 1);
v___x_410_ = l_Lean_Syntax_getArg(v___x_400_, v___x_382_);
lean_dec(v___x_400_);
v___x_411_ = lean_unsigned_to_nat(3u);
v___x_412_ = l_Lean_Syntax_getArg(v_x_370_, v___x_411_);
lean_dec(v_x_370_);
v___x_472_ = lean_unbox(v_a_409_);
lean_dec(v_a_409_);
if (v___x_472_ == 0)
{
lean_object* v___x_473_; lean_object* v_a_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_481_; 
lean_dec(v___x_412_);
lean_dec(v___x_410_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
v___x_473_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
v_a_474_ = lean_ctor_get(v___x_473_, 0);
v_isSharedCheck_481_ = !lean_is_exclusive(v___x_473_);
if (v_isSharedCheck_481_ == 0)
{
v___x_476_ = v___x_473_;
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_a_474_);
lean_dec(v___x_473_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_481_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
lean_object* v___x_479_; 
if (v_isShared_477_ == 0)
{
v___x_479_ = v___x_476_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v_a_474_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
else
{
v___y_414_ = v_a_372_;
v___y_415_ = v_a_373_;
v___y_416_ = v_a_374_;
v___y_417_ = v_a_375_;
v___y_418_ = v_a_376_;
v___y_419_ = v_a_377_;
goto v___jp_413_;
}
v___jp_413_:
{
lean_object* v_ref_420_; lean_object* v_quotContext_421_; lean_object* v_currMacroScope_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; 
v_ref_420_ = lean_ctor_get(v___y_418_, 5);
v_quotContext_421_ = lean_ctor_get(v___y_418_, 10);
v_currMacroScope_422_ = lean_ctor_get(v___y_418_, 11);
v___x_423_ = l_Lean_SourceInfo_fromRef(v_ref_420_, v___x_404_);
v___x_424_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22));
v___x_425_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24);
v___x_426_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27));
lean_inc_n(v_currMacroScope_422_, 3);
lean_inc_n(v_quotContext_421_, 3);
v___x_427_ = l_Lean_addMacroScope(v_quotContext_421_, v___x_426_, v_currMacroScope_422_);
v___x_428_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29));
lean_inc_n(v___x_423_, 20);
v___x_429_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_429_, 0, v___x_423_);
lean_ctor_set(v___x_429_, 1, v___x_425_);
lean_ctor_set(v___x_429_, 2, v___x_427_);
lean_ctor_set(v___x_429_, 3, v___x_428_);
v___x_430_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_431_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33));
v___x_432_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35));
v___x_433_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36));
v___x_434_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_434_, 0, v___x_423_);
lean_ctor_set(v___x_434_, 1, v___x_433_);
v___x_435_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38));
v___x_436_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40);
v___x_437_ = lean_box(0);
v___x_438_ = l_Lean_addMacroScope(v_quotContext_421_, v___x_437_, v_currMacroScope_422_);
v___x_439_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86));
v___x_440_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_440_, 0, v___x_423_);
lean_ctor_set(v___x_440_, 1, v___x_436_);
lean_ctor_set(v___x_440_, 2, v___x_438_);
lean_ctor_set(v___x_440_, 3, v___x_439_);
v___x_441_ = l_Lean_Syntax_node1(v___x_423_, v___x_435_, v___x_440_);
v___x_442_ = l_Lean_Syntax_node2(v___x_423_, v___x_432_, v___x_434_, v___x_441_);
v___x_443_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87));
v___x_444_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88));
v___x_445_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_445_, 0, v___x_423_);
lean_ctor_set(v___x_445_, 1, v___x_443_);
v___x_446_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90));
v___x_447_ = l_Lean_Syntax_node1(v___x_423_, v___x_430_, v___x_392_);
v___x_448_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91);
v___x_449_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_449_, 0, v___x_423_);
lean_ctor_set(v___x_449_, 1, v___x_430_);
lean_ctor_set(v___x_449_, 2, v___x_448_);
v___x_450_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92));
v___x_451_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_451_, 0, v___x_423_);
lean_ctor_set(v___x_451_, 1, v___x_450_);
v___x_452_ = l_Lean_Syntax_node4(v___x_423_, v___x_446_, v___x_447_, v___x_449_, v___x_451_, v___x_412_);
v___x_453_ = l_Lean_Syntax_node2(v___x_423_, v___x_444_, v___x_445_, v___x_452_);
v___x_454_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93));
v___x_455_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_455_, 0, v___x_423_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
lean_inc_ref(v___x_455_);
lean_inc(v___x_442_);
v___x_456_ = l_Lean_Syntax_node3(v___x_423_, v___x_431_, v___x_442_, v___x_453_, v___x_455_);
v___x_457_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__95));
v___x_458_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__97, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__97_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__97);
v___x_459_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__98));
v___x_460_ = l_Lean_addMacroScope(v_quotContext_421_, v___x_459_, v_currMacroScope_422_);
v___x_461_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__102));
v___x_462_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_462_, 0, v___x_423_);
lean_ctor_set(v___x_462_, 1, v___x_458_);
lean_ctor_set(v___x_462_, 2, v___x_460_);
lean_ctor_set(v___x_462_, 3, v___x_461_);
v___x_463_ = l_Lean_Syntax_node1(v___x_423_, v___x_430_, v___x_410_);
v___x_464_ = l_Lean_Syntax_node2(v___x_423_, v___x_424_, v___x_462_, v___x_463_);
v___x_465_ = l_Lean_Syntax_node3(v___x_423_, v___x_431_, v___x_442_, v___x_464_, v___x_455_);
v___x_466_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__103));
v___x_467_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_467_, 0, v___x_423_);
lean_ctor_set(v___x_467_, 1, v___x_466_);
v___x_468_ = l_Lean_Syntax_node2(v___x_423_, v___x_457_, v___x_465_, v___x_467_);
v___x_469_ = l_Lean_Syntax_node2(v___x_423_, v___x_430_, v___x_456_, v___x_468_);
v___x_470_ = l_Lean_Syntax_node2(v___x_423_, v___x_424_, v___x_429_, v___x_469_);
v___x_471_ = l_Lean_Elab_Term_elabTerm(v___x_470_, v_x_371_, v___x_394_, v___x_394_, v___y_414_, v___y_415_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
return v___x_471_;
}
}
else
{
lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_489_; 
lean_dec(v___x_400_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v_a_482_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_489_ == 0)
{
v___x_484_ = v___x_408_;
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v___x_408_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_487_; 
if (v_isShared_485_ == 0)
{
v___x_487_ = v___x_484_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_a_482_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
}
}
else
{
lean_object* v___x_490_; 
lean_inc(v_x_371_);
v___x_490_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_490_) == 0)
{
lean_object* v_a_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___y_496_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_499_; lean_object* v___y_500_; lean_object* v___y_501_; lean_object* v___y_547_; lean_object* v___y_548_; lean_object* v___y_549_; lean_object* v___y_550_; lean_object* v___y_551_; lean_object* v___y_552_; lean_object* v_a_563_; lean_object* v___y_572_; uint8_t v___y_573_; lean_object* v___y_576_; uint8_t v___x_581_; 
v_a_491_ = lean_ctor_get(v___x_490_, 0);
lean_inc(v_a_491_);
lean_dec_ref_known(v___x_490_, 1);
v___x_492_ = l_Lean_Syntax_getArg(v___x_400_, v___x_382_);
lean_dec(v___x_400_);
v___x_493_ = lean_unsigned_to_nat(3u);
v___x_494_ = l_Lean_Syntax_getArg(v_x_370_, v___x_493_);
lean_dec(v_x_370_);
v___x_581_ = lean_unbox(v_a_491_);
lean_dec(v_a_491_);
if (v___x_581_ == 0)
{
lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_582_ = lean_box(0);
lean_inc(v___x_492_);
v___x_583_ = l_Lean_Elab_Term_elabTerm(v___x_492_, v___x_582_, v___x_394_, v___x_394_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v_a_584_; lean_object* v___x_585_; 
v_a_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_584_);
lean_dec_ref_known(v___x_583_, 1);
lean_inc(v_a_377_);
lean_inc_ref(v_a_376_);
lean_inc(v_a_375_);
lean_inc_ref(v_a_374_);
v___x_585_ = lean_infer_type(v_a_584_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_585_) == 0)
{
lean_object* v_a_586_; lean_object* v___x_587_; 
v_a_586_ = lean_ctor_get(v___x_585_, 0);
lean_inc(v_a_586_);
lean_dec_ref_known(v___x_585_, 1);
v___x_587_ = l_Lean_Meta_whnfR(v_a_586_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
v___y_576_ = v___x_587_;
goto v___jp_575_;
}
else
{
v___y_576_ = v___x_585_;
goto v___jp_575_;
}
}
else
{
v___y_576_ = v___x_583_;
goto v___jp_575_;
}
}
else
{
v___y_496_ = v_a_372_;
v___y_497_ = v_a_373_;
v___y_498_ = v_a_374_;
v___y_499_ = v_a_375_;
v___y_500_ = v_a_376_;
v___y_501_ = v_a_377_;
goto v___jp_495_;
}
v___jp_495_:
{
lean_object* v_ref_502_; lean_object* v_quotContext_503_; lean_object* v_currMacroScope_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v_ref_502_ = lean_ctor_get(v___y_500_, 5);
v_quotContext_503_ = lean_ctor_get(v___y_500_, 10);
v_currMacroScope_504_ = lean_ctor_get(v___y_500_, 11);
v___x_505_ = l_Lean_SourceInfo_fromRef(v_ref_502_, v___x_402_);
v___x_506_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22));
v___x_507_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24);
v___x_508_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27));
lean_inc_n(v_currMacroScope_504_, 2);
lean_inc_n(v_quotContext_503_, 2);
v___x_509_ = l_Lean_addMacroScope(v_quotContext_503_, v___x_508_, v_currMacroScope_504_);
v___x_510_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29));
lean_inc_n(v___x_505_, 16);
v___x_511_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_511_, 0, v___x_505_);
lean_ctor_set(v___x_511_, 1, v___x_507_);
lean_ctor_set(v___x_511_, 2, v___x_509_);
lean_ctor_set(v___x_511_, 3, v___x_510_);
v___x_512_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_513_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33));
v___x_514_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35));
v___x_515_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36));
v___x_516_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_516_, 0, v___x_505_);
lean_ctor_set(v___x_516_, 1, v___x_515_);
v___x_517_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38));
v___x_518_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40);
v___x_519_ = lean_box(0);
v___x_520_ = l_Lean_addMacroScope(v_quotContext_503_, v___x_519_, v_currMacroScope_504_);
v___x_521_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86));
v___x_522_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_522_, 0, v___x_505_);
lean_ctor_set(v___x_522_, 1, v___x_518_);
lean_ctor_set(v___x_522_, 2, v___x_520_);
lean_ctor_set(v___x_522_, 3, v___x_521_);
v___x_523_ = l_Lean_Syntax_node1(v___x_505_, v___x_517_, v___x_522_);
v___x_524_ = l_Lean_Syntax_node2(v___x_505_, v___x_514_, v___x_516_, v___x_523_);
v___x_525_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87));
v___x_526_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88));
v___x_527_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_527_, 0, v___x_505_);
lean_ctor_set(v___x_527_, 1, v___x_525_);
v___x_528_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90));
v___x_529_ = l_Lean_Syntax_node1(v___x_505_, v___x_512_, v___x_392_);
v___x_530_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91);
v___x_531_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_531_, 0, v___x_505_);
lean_ctor_set(v___x_531_, 1, v___x_512_);
lean_ctor_set(v___x_531_, 2, v___x_530_);
v___x_532_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92));
v___x_533_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_533_, 0, v___x_505_);
lean_ctor_set(v___x_533_, 1, v___x_532_);
v___x_534_ = l_Lean_Syntax_node4(v___x_505_, v___x_528_, v___x_529_, v___x_531_, v___x_533_, v___x_494_);
v___x_535_ = l_Lean_Syntax_node2(v___x_505_, v___x_526_, v___x_527_, v___x_534_);
v___x_536_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93));
v___x_537_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_505_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = l_Lean_Syntax_node3(v___x_505_, v___x_513_, v___x_524_, v___x_535_, v___x_537_);
v___x_539_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__95));
v___x_540_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__103));
v___x_541_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_541_, 0, v___x_505_);
lean_ctor_set(v___x_541_, 1, v___x_540_);
v___x_542_ = l_Lean_Syntax_node2(v___x_505_, v___x_539_, v___x_492_, v___x_541_);
v___x_543_ = l_Lean_Syntax_node2(v___x_505_, v___x_512_, v___x_538_, v___x_542_);
v___x_544_ = l_Lean_Syntax_node2(v___x_505_, v___x_506_, v___x_511_, v___x_543_);
v___x_545_ = l_Lean_Elab_Term_elabTerm(v___x_544_, v_x_371_, v___x_394_, v___x_394_, v___y_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_, v___y_501_);
return v___x_545_;
}
v___jp_546_:
{
lean_object* v___x_553_; lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_561_; 
v___x_553_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
v_a_554_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_561_ == 0)
{
v___x_556_ = v___x_553_;
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_553_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_561_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___x_559_; 
if (v_isShared_557_ == 0)
{
v___x_559_ = v___x_556_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v_a_554_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
}
v___jp_562_:
{
lean_object* v___x_564_; 
v___x_564_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_563_, v_a_375_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v_a_565_; lean_object* v___x_566_; uint8_t v___x_567_; 
v_a_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_a_565_);
lean_dec_ref_known(v___x_564_, 1);
v___x_566_ = l_Lean_Expr_cleanupAnnotations(v_a_565_);
v___x_567_ = l_Lean_Expr_isApp(v___x_566_);
if (v___x_567_ == 0)
{
lean_dec_ref(v___x_566_);
lean_dec(v___x_494_);
lean_dec(v___x_492_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
v___y_547_ = v_a_372_;
v___y_548_ = v_a_373_;
v___y_549_ = v_a_374_;
v___y_550_ = v_a_375_;
v___y_551_ = v_a_376_;
v___y_552_ = v_a_377_;
goto v___jp_546_;
}
else
{
lean_object* v___x_568_; lean_object* v___x_569_; uint8_t v___x_570_; 
v___x_568_ = l_Lean_Expr_appFnCleanup___redArg(v___x_566_);
v___x_569_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__104));
v___x_570_ = l_Lean_Expr_isConstOf(v___x_568_, v___x_569_);
lean_dec_ref(v___x_568_);
if (v___x_570_ == 0)
{
lean_dec(v___x_494_);
lean_dec(v___x_492_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
v___y_547_ = v_a_372_;
v___y_548_ = v_a_373_;
v___y_549_ = v_a_374_;
v___y_550_ = v_a_375_;
v___y_551_ = v_a_376_;
v___y_552_ = v_a_377_;
goto v___jp_546_;
}
else
{
v___y_496_ = v_a_372_;
v___y_497_ = v_a_373_;
v___y_498_ = v_a_374_;
v___y_499_ = v_a_375_;
v___y_500_ = v_a_376_;
v___y_501_ = v_a_377_;
goto v___jp_495_;
}
}
}
else
{
lean_dec(v___x_494_);
lean_dec(v___x_492_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
return v___x_564_;
}
}
v___jp_571_:
{
if (v___y_573_ == 0)
{
lean_object* v___x_574_; 
lean_dec_ref(v___y_572_);
v___x_574_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
return v___x_574_;
}
else
{
return v___y_572_;
}
}
v___jp_575_:
{
if (lean_obj_tag(v___y_576_) == 0)
{
lean_object* v_a_577_; 
v_a_577_ = lean_ctor_get(v___y_576_, 0);
lean_inc(v_a_577_);
lean_dec_ref_known(v___y_576_, 1);
v_a_563_ = v_a_577_;
goto v___jp_562_;
}
else
{
lean_object* v_a_578_; uint8_t v___x_579_; 
lean_dec(v___x_494_);
lean_dec(v___x_492_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
v_a_578_ = lean_ctor_get(v___y_576_, 0);
v___x_579_ = l_Lean_Exception_isInterrupt(v_a_578_);
if (v___x_579_ == 0)
{
uint8_t v___x_580_; 
lean_inc(v_a_578_);
v___x_580_ = l_Lean_Exception_isRuntime(v_a_578_);
v___y_572_ = v___y_576_;
v___y_573_ = v___x_580_;
goto v___jp_571_;
}
else
{
v___y_572_ = v___y_576_;
v___y_573_ = v___x_579_;
goto v___jp_571_;
}
}
}
}
else
{
lean_object* v_a_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_595_; 
lean_dec(v___x_400_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v_a_588_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_595_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_595_ == 0)
{
v___x_590_ = v___x_490_;
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_a_588_);
lean_dec(v___x_490_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_595_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___x_593_; 
if (v_isShared_591_ == 0)
{
v___x_593_ = v___x_590_;
goto v_reusejp_592_;
}
else
{
lean_object* v_reuseFailAlloc_594_; 
v_reuseFailAlloc_594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_594_, 0, v_a_588_);
v___x_593_ = v_reuseFailAlloc_594_;
goto v_reusejp_592_;
}
v_reusejp_592_:
{
return v___x_593_;
}
}
}
}
}
else
{
lean_object* v___x_596_; 
lean_inc(v_x_371_);
v___x_596_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_596_) == 0)
{
lean_object* v_a_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___y_602_; lean_object* v___y_603_; lean_object* v___y_604_; lean_object* v___y_605_; lean_object* v___y_606_; lean_object* v___y_607_; uint8_t v___x_656_; 
v_a_597_ = lean_ctor_get(v___x_596_, 0);
lean_inc(v_a_597_);
lean_dec_ref_known(v___x_596_, 1);
v___x_598_ = l_Lean_Syntax_getArg(v___x_400_, v___x_382_);
lean_dec(v___x_400_);
v___x_599_ = lean_unsigned_to_nat(3u);
v___x_600_ = l_Lean_Syntax_getArg(v_x_370_, v___x_599_);
lean_dec(v_x_370_);
v___x_656_ = lean_unbox(v_a_597_);
lean_dec(v_a_597_);
if (v___x_656_ == 0)
{
lean_object* v___x_657_; lean_object* v_a_658_; lean_object* v___x_660_; uint8_t v_isShared_661_; uint8_t v_isSharedCheck_665_; 
lean_dec(v___x_600_);
lean_dec(v___x_598_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
v___x_657_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
v_a_658_ = lean_ctor_get(v___x_657_, 0);
v_isSharedCheck_665_ = !lean_is_exclusive(v___x_657_);
if (v_isSharedCheck_665_ == 0)
{
v___x_660_ = v___x_657_;
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
else
{
lean_inc(v_a_658_);
lean_dec(v___x_657_);
v___x_660_ = lean_box(0);
v_isShared_661_ = v_isSharedCheck_665_;
goto v_resetjp_659_;
}
v_resetjp_659_:
{
lean_object* v___x_663_; 
if (v_isShared_661_ == 0)
{
v___x_663_ = v___x_660_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_664_; 
v_reuseFailAlloc_664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_664_, 0, v_a_658_);
v___x_663_ = v_reuseFailAlloc_664_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
return v___x_663_;
}
}
}
else
{
v___y_602_ = v_a_372_;
v___y_603_ = v_a_373_;
v___y_604_ = v_a_374_;
v___y_605_ = v_a_375_;
v___y_606_ = v_a_376_;
v___y_607_ = v_a_377_;
goto v___jp_601_;
}
v___jp_601_:
{
lean_object* v_ref_608_; lean_object* v_quotContext_609_; lean_object* v_currMacroScope_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v_ref_608_ = lean_ctor_get(v___y_606_, 5);
v_quotContext_609_ = lean_ctor_get(v___y_606_, 10);
v_currMacroScope_610_ = lean_ctor_get(v___y_606_, 11);
v___x_611_ = l_Lean_SourceInfo_fromRef(v_ref_608_, v___x_397_);
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22));
v___x_613_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24);
v___x_614_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27));
lean_inc_n(v_currMacroScope_610_, 3);
lean_inc_n(v_quotContext_609_, 3);
v___x_615_ = l_Lean_addMacroScope(v_quotContext_609_, v___x_614_, v_currMacroScope_610_);
v___x_616_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29));
lean_inc_n(v___x_611_, 17);
v___x_617_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_617_, 0, v___x_611_);
lean_ctor_set(v___x_617_, 1, v___x_613_);
lean_ctor_set(v___x_617_, 2, v___x_615_);
lean_ctor_set(v___x_617_, 3, v___x_616_);
v___x_618_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_619_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33));
v___x_620_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35));
v___x_621_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36));
v___x_622_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_611_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38));
v___x_624_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40);
v___x_625_ = lean_box(0);
v___x_626_ = l_Lean_addMacroScope(v_quotContext_609_, v___x_625_, v_currMacroScope_610_);
v___x_627_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86));
v___x_628_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_628_, 0, v___x_611_);
lean_ctor_set(v___x_628_, 1, v___x_624_);
lean_ctor_set(v___x_628_, 2, v___x_626_);
lean_ctor_set(v___x_628_, 3, v___x_627_);
v___x_629_ = l_Lean_Syntax_node1(v___x_611_, v___x_623_, v___x_628_);
v___x_630_ = l_Lean_Syntax_node2(v___x_611_, v___x_620_, v___x_622_, v___x_629_);
v___x_631_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87));
v___x_632_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88));
v___x_633_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_633_, 0, v___x_611_);
lean_ctor_set(v___x_633_, 1, v___x_631_);
v___x_634_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90));
v___x_635_ = l_Lean_Syntax_node1(v___x_611_, v___x_618_, v___x_392_);
v___x_636_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__106));
v___x_637_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__107));
v___x_638_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_611_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
v___x_639_ = l_Lean_Syntax_node2(v___x_611_, v___x_636_, v___x_638_, v___x_598_);
v___x_640_ = l_Lean_Syntax_node1(v___x_611_, v___x_618_, v___x_639_);
v___x_641_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92));
v___x_642_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_642_, 0, v___x_611_);
lean_ctor_set(v___x_642_, 1, v___x_641_);
v___x_643_ = l_Lean_Syntax_node4(v___x_611_, v___x_634_, v___x_635_, v___x_640_, v___x_642_, v___x_600_);
v___x_644_ = l_Lean_Syntax_node2(v___x_611_, v___x_632_, v___x_633_, v___x_643_);
v___x_645_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93));
v___x_646_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_646_, 0, v___x_611_);
lean_ctor_set(v___x_646_, 1, v___x_645_);
v___x_647_ = l_Lean_Syntax_node3(v___x_611_, v___x_619_, v___x_630_, v___x_644_, v___x_646_);
v___x_648_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109);
v___x_649_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111));
v___x_650_ = l_Lean_addMacroScope(v_quotContext_609_, v___x_649_, v_currMacroScope_610_);
v___x_651_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__113));
v___x_652_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_652_, 0, v___x_611_);
lean_ctor_set(v___x_652_, 1, v___x_648_);
lean_ctor_set(v___x_652_, 2, v___x_650_);
lean_ctor_set(v___x_652_, 3, v___x_651_);
v___x_653_ = l_Lean_Syntax_node2(v___x_611_, v___x_618_, v___x_647_, v___x_652_);
v___x_654_ = l_Lean_Syntax_node2(v___x_611_, v___x_612_, v___x_617_, v___x_653_);
v___x_655_ = l_Lean_Elab_Term_elabTerm(v___x_654_, v_x_371_, v___x_394_, v___x_394_, v___y_602_, v___y_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_);
return v___x_655_;
}
}
else
{
lean_object* v_a_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_673_; 
lean_dec(v___x_400_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v_a_666_ = lean_ctor_get(v___x_596_, 0);
v_isSharedCheck_673_ = !lean_is_exclusive(v___x_596_);
if (v_isSharedCheck_673_ == 0)
{
v___x_668_ = v___x_596_;
v_isShared_669_ = v_isSharedCheck_673_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_a_666_);
lean_dec(v___x_596_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_673_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___x_671_; 
if (v_isShared_669_ == 0)
{
v___x_671_ = v___x_668_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v_a_666_);
v___x_671_ = v_reuseFailAlloc_672_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
return v___x_671_;
}
}
}
}
}
}
else
{
lean_object* v___x_674_; 
lean_dec(v___x_396_);
lean_inc(v_x_371_);
v___x_674_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_371_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_, v_a_377_);
if (lean_obj_tag(v___x_674_) == 0)
{
lean_object* v_a_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___y_679_; lean_object* v___y_680_; lean_object* v___y_681_; lean_object* v___y_682_; lean_object* v___y_683_; lean_object* v___y_684_; uint8_t v___x_731_; 
v_a_675_ = lean_ctor_get(v___x_674_, 0);
lean_inc(v_a_675_);
lean_dec_ref_known(v___x_674_, 1);
v___x_676_ = lean_unsigned_to_nat(3u);
v___x_677_ = l_Lean_Syntax_getArg(v_x_370_, v___x_676_);
lean_dec(v_x_370_);
v___x_731_ = lean_unbox(v_a_675_);
lean_dec(v_a_675_);
if (v___x_731_ == 0)
{
lean_object* v___x_732_; lean_object* v_a_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_740_; 
lean_dec(v___x_677_);
lean_dec(v___x_392_);
lean_dec(v_x_371_);
v___x_732_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderSetOf_spec__0___redArg();
v_a_733_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_740_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_740_ == 0)
{
v___x_735_ = v___x_732_;
v_isShared_736_ = v_isSharedCheck_740_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_a_733_);
lean_dec(v___x_732_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_740_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v___x_738_; 
if (v_isShared_736_ == 0)
{
v___x_738_ = v___x_735_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v_a_733_);
v___x_738_ = v_reuseFailAlloc_739_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
return v___x_738_;
}
}
}
else
{
v___y_679_ = v_a_372_;
v___y_680_ = v_a_373_;
v___y_681_ = v_a_374_;
v___y_682_ = v_a_375_;
v___y_683_ = v_a_376_;
v___y_684_ = v_a_377_;
goto v___jp_678_;
}
v___jp_678_:
{
lean_object* v_ref_685_; lean_object* v_quotContext_686_; lean_object* v_currMacroScope_687_; uint8_t v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; 
v_ref_685_ = lean_ctor_get(v___y_683_, 5);
v_quotContext_686_ = lean_ctor_get(v___y_683_, 10);
v_currMacroScope_687_ = lean_ctor_get(v___y_683_, 11);
v___x_688_ = 0;
v___x_689_ = l_Lean_SourceInfo_fromRef(v_ref_685_, v___x_688_);
v___x_690_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__22));
v___x_691_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__24);
v___x_692_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__27));
lean_inc_n(v_currMacroScope_687_, 3);
lean_inc_n(v_quotContext_686_, 3);
v___x_693_ = l_Lean_addMacroScope(v_quotContext_686_, v___x_692_, v_currMacroScope_687_);
v___x_694_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__29));
lean_inc_n(v___x_689_, 15);
v___x_695_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_695_, 0, v___x_689_);
lean_ctor_set(v___x_695_, 1, v___x_691_);
lean_ctor_set(v___x_695_, 2, v___x_693_);
lean_ctor_set(v___x_695_, 3, v___x_694_);
v___x_696_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_697_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__33));
v___x_698_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__35));
v___x_699_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__36));
v___x_700_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_700_, 0, v___x_689_);
lean_ctor_set(v___x_700_, 1, v___x_699_);
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__38));
v___x_702_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__40);
v___x_703_ = lean_box(0);
v___x_704_ = l_Lean_addMacroScope(v_quotContext_686_, v___x_703_, v_currMacroScope_687_);
v___x_705_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__86));
v___x_706_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_706_, 0, v___x_689_);
lean_ctor_set(v___x_706_, 1, v___x_702_);
lean_ctor_set(v___x_706_, 2, v___x_704_);
lean_ctor_set(v___x_706_, 3, v___x_705_);
v___x_707_ = l_Lean_Syntax_node1(v___x_689_, v___x_701_, v___x_706_);
v___x_708_ = l_Lean_Syntax_node2(v___x_689_, v___x_698_, v___x_700_, v___x_707_);
v___x_709_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__87));
v___x_710_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__88));
v___x_711_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_711_, 0, v___x_689_);
lean_ctor_set(v___x_711_, 1, v___x_709_);
v___x_712_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__90));
v___x_713_ = l_Lean_Syntax_node1(v___x_689_, v___x_696_, v___x_392_);
v___x_714_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91);
v___x_715_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_715_, 0, v___x_689_);
lean_ctor_set(v___x_715_, 1, v___x_696_);
lean_ctor_set(v___x_715_, 2, v___x_714_);
v___x_716_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__92));
v___x_717_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_717_, 0, v___x_689_);
lean_ctor_set(v___x_717_, 1, v___x_716_);
v___x_718_ = l_Lean_Syntax_node4(v___x_689_, v___x_712_, v___x_713_, v___x_715_, v___x_717_, v___x_677_);
v___x_719_ = l_Lean_Syntax_node2(v___x_689_, v___x_710_, v___x_711_, v___x_718_);
v___x_720_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__93));
v___x_721_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_721_, 0, v___x_689_);
lean_ctor_set(v___x_721_, 1, v___x_720_);
v___x_722_ = l_Lean_Syntax_node3(v___x_689_, v___x_697_, v___x_708_, v___x_719_, v___x_721_);
v___x_723_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__109);
v___x_724_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111));
v___x_725_ = l_Lean_addMacroScope(v_quotContext_686_, v___x_724_, v_currMacroScope_687_);
v___x_726_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__113));
v___x_727_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_727_, 0, v___x_689_);
lean_ctor_set(v___x_727_, 1, v___x_723_);
lean_ctor_set(v___x_727_, 2, v___x_725_);
lean_ctor_set(v___x_727_, 3, v___x_726_);
v___x_728_ = l_Lean_Syntax_node2(v___x_689_, v___x_696_, v___x_722_, v___x_727_);
v___x_729_ = l_Lean_Syntax_node2(v___x_689_, v___x_690_, v___x_695_, v___x_728_);
v___x_730_ = l_Lean_Elab_Term_elabTerm(v___x_729_, v_x_371_, v___x_394_, v___x_394_, v___y_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
return v___x_730_;
}
}
else
{
lean_object* v_a_741_; lean_object* v___x_743_; uint8_t v_isShared_744_; uint8_t v_isSharedCheck_748_; 
lean_dec(v___x_392_);
lean_dec(v_x_371_);
lean_dec(v_x_370_);
v_a_741_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_748_ == 0)
{
v___x_743_ = v___x_674_;
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
else
{
lean_inc(v_a_741_);
lean_dec(v___x_674_);
v___x_743_ = lean_box(0);
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
v_resetjp_742_:
{
lean_object* v___x_746_; 
if (v_isShared_744_ == 0)
{
v___x_746_ = v___x_743_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_a_741_);
v___x_746_ = v_reuseFailAlloc_747_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
return v___x_746_;
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___boxed(lean_object* v_x_749_, lean_object* v_x_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v_a_757_){
_start:
{
lean_object* v_res_758_; 
v_res_758_ = lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf(v_x_749_, v_x_750_, v_a_751_, v_a_752_, v_a_753_, v_a_754_, v_a_755_, v_a_756_);
lean_dec(v_a_756_);
lean_dec_ref(v_a_755_);
lean_dec(v_a_754_);
lean_dec_ref(v_a_753_);
lean_dec(v_a_752_);
lean_dec_ref(v_a_751_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(lean_object* v___y_759_){
_start:
{
lean_object* v_subExpr_761_; lean_object* v_expr_762_; lean_object* v___x_763_; 
v_subExpr_761_ = lean_ctor_get(v___y_759_, 3);
v_expr_762_ = lean_ctor_get(v_subExpr_761_, 0);
lean_inc_ref(v_expr_762_);
v___x_763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_763_, 0, v_expr_762_);
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg___boxed(lean_object* v___y_764_, lean_object* v___y_765_){
_start:
{
lean_object* v_res_766_; 
v_res_766_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(v___y_764_);
lean_dec_ref(v___y_764_);
return v_res_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0(lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_){
_start:
{
lean_object* v___x_774_; 
v___x_774_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(v___y_767_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___boxed(lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_){
_start:
{
lean_object* v_res_782_; 
v_res_782_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0(v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_);
lean_dec(v___y_780_);
lean_dec_ref(v___y_779_);
lean_dec(v___y_778_);
lean_dec_ref(v___y_777_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
return v_res_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__0(lean_object* v_x_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_){
_start:
{
lean_object* v___x_791_; 
v___x_791_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_791_, 0, v_x_783_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__0___boxed(lean_object* v_x_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_){
_start:
{
lean_object* v_res_800_; 
v_res_800_ = lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__0(v_x_792_, v___y_793_, v___y_794_, v___y_795_, v___y_796_, v___y_797_, v___y_798_);
lean_dec(v___y_798_);
lean_dec_ref(v___y_797_);
lean_dec(v___y_796_);
lean_dec_ref(v___y_795_);
lean_dec(v___y_794_);
lean_dec_ref(v___y_793_);
return v_res_800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg(lean_object* v___y_801_){
_start:
{
lean_object* v_subExpr_803_; lean_object* v_pos_804_; lean_object* v___x_805_; 
v_subExpr_803_ = lean_ctor_get(v___y_801_, 3);
v_pos_804_ = lean_ctor_get(v_subExpr_803_, 1);
lean_inc(v_pos_804_);
v___x_805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_805_, 0, v_pos_804_);
return v___x_805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg___boxed(lean_object* v___y_806_, lean_object* v___y_807_){
_start:
{
lean_object* v_res_808_; 
v_res_808_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg(v___y_806_);
lean_dec_ref(v___y_806_);
return v_res_808_;
}
}
static lean_object* _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_809_; lean_object* v_dummy_810_; 
v___x_809_ = lean_box(0);
v_dummy_810_ = l_Lean_Expr_sort___override(v___x_809_);
return v_dummy_810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(lean_object* v_argIdx_811_, lean_object* v_x_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_){
_start:
{
lean_object* v___x_820_; lean_object* v_a_821_; lean_object* v___x_822_; lean_object* v_a_823_; lean_object* v_optionsPerPos_824_; lean_object* v_currNamespace_825_; lean_object* v_openDecls_826_; uint8_t v_inPattern_827_; lean_object* v_depth_828_; lean_object* v_lctxInitIndices_829_; lean_object* v_nargs_830_; lean_object* v___x_831_; lean_object* v_dummy_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v_args_836_; lean_object* v___x_837_; lean_object* v_newPos_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; 
v___x_820_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(v___y_813_);
v_a_821_ = lean_ctor_get(v___x_820_, 0);
lean_inc(v_a_821_);
lean_dec_ref(v___x_820_);
v___x_822_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg(v___y_813_);
v_a_823_ = lean_ctor_get(v___x_822_, 0);
lean_inc(v_a_823_);
lean_dec_ref(v___x_822_);
v_optionsPerPos_824_ = lean_ctor_get(v___y_813_, 0);
v_currNamespace_825_ = lean_ctor_get(v___y_813_, 1);
v_openDecls_826_ = lean_ctor_get(v___y_813_, 2);
v_inPattern_827_ = lean_ctor_get_uint8(v___y_813_, sizeof(void*)*6);
v_depth_828_ = lean_ctor_get(v___y_813_, 4);
v_lctxInitIndices_829_ = lean_ctor_get(v___y_813_, 5);
v_nargs_830_ = l_Lean_Expr_getAppNumArgs(v_a_821_);
v___x_831_ = l_Lean_instInhabitedExpr;
v_dummy_832_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0);
lean_inc(v_nargs_830_);
v___x_833_ = lean_mk_array(v_nargs_830_, v_dummy_832_);
v___x_834_ = lean_unsigned_to_nat(1u);
v___x_835_ = lean_nat_sub(v_nargs_830_, v___x_834_);
lean_dec(v_nargs_830_);
v_args_836_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_821_, v___x_833_, v___x_835_);
v___x_837_ = lean_array_get_size(v_args_836_);
v_newPos_838_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_837_, v_argIdx_811_, v_a_823_);
lean_dec(v_a_823_);
v___x_839_ = lean_array_get(v___x_831_, v_args_836_, v_argIdx_811_);
lean_dec_ref(v_args_836_);
v___x_840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_840_, 0, v___x_839_);
lean_ctor_set(v___x_840_, 1, v_newPos_838_);
lean_inc(v_lctxInitIndices_829_);
lean_inc(v_depth_828_);
lean_inc(v_openDecls_826_);
lean_inc(v_currNamespace_825_);
lean_inc(v_optionsPerPos_824_);
v___x_841_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_841_, 0, v_optionsPerPos_824_);
lean_ctor_set(v___x_841_, 1, v_currNamespace_825_);
lean_ctor_set(v___x_841_, 2, v_openDecls_826_);
lean_ctor_set(v___x_841_, 3, v___x_840_);
lean_ctor_set(v___x_841_, 4, v_depth_828_);
lean_ctor_set(v___x_841_, 5, v_lctxInitIndices_829_);
lean_ctor_set_uint8(v___x_841_, sizeof(void*)*6, v_inPattern_827_);
lean_inc(v___y_818_);
lean_inc_ref(v___y_817_);
lean_inc(v___y_816_);
lean_inc_ref(v___y_815_);
lean_inc(v___y_814_);
v___x_842_ = lean_apply_7(v_x_812_, v___x_841_, v___y_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_, lean_box(0));
return v___x_842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___boxed(lean_object* v_argIdx_843_, lean_object* v_x_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v_argIdx_843_, v_x_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
lean_dec(v_argIdx_843_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__1(lean_object* v_x_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_){
_start:
{
lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_861_ = lean_box(0);
v___x_862_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_862_, 0, v___x_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__1___boxed(lean_object* v_x_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_){
_start:
{
lean_object* v_res_871_; 
v_res_871_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__1(v_x_863_, v___y_864_, v___y_865_, v___y_866_, v___y_867_, v___y_868_, v___y_869_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
lean_dec(v___y_867_);
lean_dec_ref(v___y_866_);
lean_dec(v___y_865_);
lean_dec_ref(v___y_864_);
lean_dec_ref(v_x_863_);
return v_res_871_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg(lean_object* v_child_872_, lean_object* v_childIdx_873_, lean_object* v_x_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_){
_start:
{
lean_object* v_subExpr_882_; lean_object* v_optionsPerPos_883_; lean_object* v_currNamespace_884_; lean_object* v_openDecls_885_; uint8_t v_inPattern_886_; lean_object* v_depth_887_; lean_object* v_lctxInitIndices_888_; lean_object* v_pos_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; 
v_subExpr_882_ = lean_ctor_get(v___y_875_, 3);
v_optionsPerPos_883_ = lean_ctor_get(v___y_875_, 0);
v_currNamespace_884_ = lean_ctor_get(v___y_875_, 1);
v_openDecls_885_ = lean_ctor_get(v___y_875_, 2);
v_inPattern_886_ = lean_ctor_get_uint8(v___y_875_, sizeof(void*)*6);
v_depth_887_ = lean_ctor_get(v___y_875_, 4);
v_lctxInitIndices_888_ = lean_ctor_get(v___y_875_, 5);
v_pos_889_ = lean_ctor_get(v_subExpr_882_, 1);
v___x_890_ = l_Lean_SubExpr_Pos_push(v_pos_889_, v_childIdx_873_);
v___x_891_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_891_, 0, v_child_872_);
lean_ctor_set(v___x_891_, 1, v___x_890_);
lean_inc(v_lctxInitIndices_888_);
lean_inc(v_depth_887_);
lean_inc(v_openDecls_885_);
lean_inc(v_currNamespace_884_);
lean_inc(v_optionsPerPos_883_);
v___x_892_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_892_, 0, v_optionsPerPos_883_);
lean_ctor_set(v___x_892_, 1, v_currNamespace_884_);
lean_ctor_set(v___x_892_, 2, v_openDecls_885_);
lean_ctor_set(v___x_892_, 3, v___x_891_);
lean_ctor_set(v___x_892_, 4, v_depth_887_);
lean_ctor_set(v___x_892_, 5, v_lctxInitIndices_888_);
lean_ctor_set_uint8(v___x_892_, sizeof(void*)*6, v_inPattern_886_);
lean_inc(v___y_880_);
lean_inc_ref(v___y_879_);
lean_inc(v___y_878_);
lean_inc_ref(v___y_877_);
lean_inc(v___y_876_);
v___x_893_ = lean_apply_7(v_x_874_, v___x_892_, v___y_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_, lean_box(0));
return v___x_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_child_894_, lean_object* v_childIdx_895_, lean_object* v_x_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
lean_object* v_res_904_; 
v_res_904_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg(v_child_894_, v_childIdx_895_, v_x_896_, v___y_897_, v___y_898_, v___y_899_, v___y_900_, v___y_901_, v___y_902_);
lean_dec(v___y_902_);
lean_dec_ref(v___y_901_);
lean_dec(v___y_900_);
lean_dec_ref(v___y_899_);
lean_dec(v___y_898_);
lean_dec_ref(v___y_897_);
return v_res_904_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___lam__0(lean_object* v_v_905_, lean_object* v_a_906_, lean_object* v_x_907_, lean_object* v_fvar_908_, lean_object* v___y_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
lean_object* v___x_916_; 
lean_inc(v___y_914_);
lean_inc_ref(v___y_913_);
lean_inc(v___y_912_);
lean_inc_ref(v___y_911_);
lean_inc(v___y_910_);
lean_inc_ref(v___y_909_);
lean_inc_ref(v_fvar_908_);
v___x_916_ = lean_apply_8(v_v_905_, v_fvar_908_, v___y_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_, lean_box(0));
if (lean_obj_tag(v___x_916_) == 0)
{
lean_object* v_a_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v_a_917_ = lean_ctor_get(v___x_916_, 0);
lean_inc(v_a_917_);
lean_dec_ref_known(v___x_916_, 1);
v___x_918_ = l_Lean_Expr_bindingBody_x21(v_a_906_);
v___x_919_ = lean_expr_instantiate1(v___x_918_, v_fvar_908_);
lean_dec_ref(v_fvar_908_);
lean_dec_ref(v___x_918_);
v___x_920_ = lean_unsigned_to_nat(1u);
v___x_921_ = lean_apply_1(v_x_907_, v_a_917_);
v___x_922_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg(v___x_919_, v___x_920_, v___x_921_, v___y_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, v___y_914_);
return v___x_922_;
}
else
{
lean_object* v_a_923_; lean_object* v___x_925_; uint8_t v_isShared_926_; uint8_t v_isSharedCheck_930_; 
lean_dec_ref(v_fvar_908_);
lean_dec_ref(v_x_907_);
v_a_923_ = lean_ctor_get(v___x_916_, 0);
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_916_);
if (v_isSharedCheck_930_ == 0)
{
v___x_925_ = v___x_916_;
v_isShared_926_ = v_isSharedCheck_930_;
goto v_resetjp_924_;
}
else
{
lean_inc(v_a_923_);
lean_dec(v___x_916_);
v___x_925_ = lean_box(0);
v_isShared_926_ = v_isSharedCheck_930_;
goto v_resetjp_924_;
}
v_resetjp_924_:
{
lean_object* v___x_928_; 
if (v_isShared_926_ == 0)
{
v___x_928_ = v___x_925_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v_a_923_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v_v_931_, lean_object* v_a_932_, lean_object* v_x_933_, lean_object* v_fvar_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_){
_start:
{
lean_object* v_res_942_; 
v_res_942_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___lam__0(v_v_931_, v_a_932_, v_x_933_, v_fvar_934_, v___y_935_, v___y_936_, v___y_937_, v___y_938_, v___y_939_, v___y_940_);
lean_dec(v___y_940_);
lean_dec_ref(v___y_939_);
lean_dec(v___y_938_);
lean_dec_ref(v___y_937_);
lean_dec(v___y_936_);
lean_dec_ref(v___y_935_);
lean_dec_ref(v_a_932_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___lam__0(lean_object* v_k_943_, lean_object* v___y_944_, lean_object* v___y_945_, lean_object* v_b_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_){
_start:
{
lean_object* v___x_952_; 
lean_inc(v___y_950_);
lean_inc_ref(v___y_949_);
lean_inc(v___y_948_);
lean_inc_ref(v___y_947_);
lean_inc(v___y_945_);
lean_inc_ref(v___y_944_);
v___x_952_ = lean_apply_8(v_k_943_, v_b_946_, v___y_944_, v___y_945_, v___y_947_, v___y_948_, v___y_949_, v___y_950_, lean_box(0));
return v___x_952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___lam__0___boxed(lean_object* v_k_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v_b_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_){
_start:
{
lean_object* v_res_962_; 
v_res_962_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___lam__0(v_k_953_, v___y_954_, v___y_955_, v_b_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_955_);
lean_dec_ref(v___y_954_);
return v_res_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg(lean_object* v_name_963_, uint8_t v_bi_964_, lean_object* v_type_965_, lean_object* v_k_966_, uint8_t v_kind_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_){
_start:
{
lean_object* v___f_975_; lean_object* v___x_976_; 
lean_inc(v___y_969_);
lean_inc_ref(v___y_968_);
v___f_975_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_975_, 0, v_k_966_);
lean_closure_set(v___f_975_, 1, v___y_968_);
lean_closure_set(v___f_975_, 2, v___y_969_);
v___x_976_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_963_, v_bi_964_, v_type_965_, v___f_975_, v_kind_967_, v___y_970_, v___y_971_, v___y_972_, v___y_973_);
if (lean_obj_tag(v___x_976_) == 0)
{
return v___x_976_;
}
else
{
lean_object* v_a_977_; lean_object* v___x_979_; uint8_t v_isShared_980_; uint8_t v_isSharedCheck_984_; 
v_a_977_ = lean_ctor_get(v___x_976_, 0);
v_isSharedCheck_984_ = !lean_is_exclusive(v___x_976_);
if (v_isSharedCheck_984_ == 0)
{
v___x_979_ = v___x_976_;
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
else
{
lean_inc(v_a_977_);
lean_dec(v___x_976_);
v___x_979_ = lean_box(0);
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
v_resetjp_978_:
{
lean_object* v___x_982_; 
if (v_isShared_980_ == 0)
{
v___x_982_ = v___x_979_;
goto v_reusejp_981_;
}
else
{
lean_object* v_reuseFailAlloc_983_; 
v_reuseFailAlloc_983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_983_, 0, v_a_977_);
v___x_982_ = v_reuseFailAlloc_983_;
goto v_reusejp_981_;
}
v_reusejp_981_:
{
return v___x_982_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg___boxed(lean_object* v_name_985_, lean_object* v_bi_986_, lean_object* v_type_987_, lean_object* v_k_988_, lean_object* v_kind_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
uint8_t v_bi_boxed_997_; uint8_t v_kind_boxed_998_; lean_object* v_res_999_; 
v_bi_boxed_997_ = lean_unbox(v_bi_986_);
v_kind_boxed_998_ = lean_unbox(v_kind_989_);
v_res_999_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg(v_name_985_, v_bi_boxed_997_, v_type_987_, v_k_988_, v_kind_boxed_998_, v___y_990_, v___y_991_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
lean_dec(v___y_993_);
lean_dec_ref(v___y_992_);
lean_dec(v___y_991_);
lean_dec_ref(v___y_990_);
return v_res_999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg(lean_object* v_n_1000_, lean_object* v_v_1001_, lean_object* v_x_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_){
_start:
{
lean_object* v___x_1010_; lean_object* v_a_1011_; lean_object* v___f_1012_; uint8_t v___x_1013_; lean_object* v___x_1014_; uint8_t v___x_1015_; lean_object* v___x_1016_; 
v___x_1010_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(v___y_1003_);
v_a_1011_ = lean_ctor_get(v___x_1010_, 0);
lean_inc_n(v_a_1011_, 2);
lean_dec_ref(v___x_1010_);
v___f_1012_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___lam__0___boxed), 11, 3);
lean_closure_set(v___f_1012_, 0, v_v_1001_);
lean_closure_set(v___f_1012_, 1, v_a_1011_);
lean_closure_set(v___f_1012_, 2, v_x_1002_);
v___x_1013_ = l_Lean_Expr_binderInfo(v_a_1011_);
v___x_1014_ = l_Lean_Expr_bindingDomain_x21(v_a_1011_);
lean_dec(v_a_1011_);
v___x_1015_ = 0;
v___x_1016_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg(v_n_1000_, v___x_1013_, v___x_1014_, v___f_1012_, v___x_1015_, v___y_1003_, v___y_1004_, v___y_1005_, v___y_1006_, v___y_1007_, v___y_1008_);
return v___x_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg___boxed(lean_object* v_n_1017_, lean_object* v_v_1018_, lean_object* v_x_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_){
_start:
{
lean_object* v_res_1027_; 
v_res_1027_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg(v_n_1017_, v_v_1018_, v_x_1019_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_, v___y_1024_, v___y_1025_);
lean_dec(v___y_1025_);
lean_dec_ref(v___y_1024_);
lean_dec(v___y_1023_);
lean_dec_ref(v___y_1022_);
lean_dec(v___y_1021_);
lean_dec_ref(v___y_1020_);
return v_res_1027_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__0(lean_object* v_x_1028_, lean_object* v_x_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v___x_1037_; 
lean_inc(v___y_1035_);
lean_inc_ref(v___y_1034_);
lean_inc(v___y_1033_);
lean_inc_ref(v___y_1032_);
lean_inc(v___y_1031_);
lean_inc_ref(v___y_1030_);
v___x_1037_ = lean_apply_7(v_x_1028_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_, lean_box(0));
return v___x_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__0___boxed(lean_object* v_x_1038_, lean_object* v_x_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v_res_1047_; 
v_res_1047_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__0(v_x_1038_, v_x_1039_, v___y_1040_, v___y_1041_, v___y_1042_, v___y_1043_, v___y_1044_, v___y_1045_);
lean_dec(v___y_1045_);
lean_dec_ref(v___y_1044_);
lean_dec(v___y_1043_);
lean_dec_ref(v___y_1042_);
lean_dec(v___y_1041_);
lean_dec_ref(v___y_1040_);
return v_res_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg(lean_object* v_n_1049_, lean_object* v_x_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_){
_start:
{
lean_object* v___f_1058_; lean_object* v___f_1059_; lean_object* v___x_1060_; 
v___f_1058_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1058_, 0, v_x_1050_);
v___f_1059_ = ((lean_object*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___closed__0));
v___x_1060_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg(v_n_1049_, v___f_1059_, v___f_1058_, v___y_1051_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_);
return v___x_1060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg___boxed(lean_object* v_n_1061_, lean_object* v_x_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_){
_start:
{
lean_object* v_res_1070_; 
v_res_1070_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg(v_n_1061_, v_x_1062_, v___y_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
lean_dec(v___y_1068_);
lean_dec_ref(v___y_1067_);
lean_dec(v___y_1066_);
lean_dec_ref(v___y_1065_);
lean_dec(v___y_1064_);
lean_dec_ref(v___y_1063_);
return v_res_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2(lean_object* v_00_u03b1_1071_, lean_object* v_n_1072_, lean_object* v_x_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_){
_start:
{
lean_object* v___x_1081_; 
v___x_1081_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___redArg(v_n_1072_, v_x_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_, v___y_1079_);
return v___x_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___boxed(lean_object* v_00_u03b1_1082_, lean_object* v_n_1083_, lean_object* v_x_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2(v_00_u03b1_1082_, v_n_1083_, v_x_1084_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
lean_dec(v___y_1090_);
lean_dec_ref(v___y_1089_);
lean_dec(v___y_1088_);
lean_dec_ref(v___y_1087_);
lean_dec(v___y_1086_);
lean_dec_ref(v___y_1085_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1(lean_object* v_00_u03b1_1093_, lean_object* v_argIdx_1094_, lean_object* v_x_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_){
_start:
{
lean_object* v___x_1103_; 
v___x_1103_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v_argIdx_1094_, v_x_1095_, v___y_1096_, v___y_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_);
return v___x_1103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___boxed(lean_object* v_00_u03b1_1104_, lean_object* v_argIdx_1105_, lean_object* v_x_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
lean_object* v_res_1114_; 
v_res_1114_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1(v_00_u03b1_1104_, v_argIdx_1105_, v_x_1106_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_, v___y_1111_, v___y_1112_);
lean_dec(v___y_1112_);
lean_dec_ref(v___y_1111_);
lean_dec(v___y_1110_);
lean_dec_ref(v___y_1109_);
lean_dec(v___y_1108_);
lean_dec_ref(v___y_1107_);
lean_dec(v_argIdx_1105_);
return v_res_1114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1(lean_object* v___f_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_){
_start:
{
lean_object* v___x_1149_; lean_object* v_a_1150_; lean_object* v_dummy_1151_; lean_object* v_nargs_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; uint8_t v___x_1159_; 
v___x_1149_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Mathlib_Meta_delabFinsetFilter_spec__0___redArg(v___y_1142_);
v_a_1150_ = lean_ctor_get(v___x_1149_, 0);
lean_inc(v_a_1150_);
lean_dec_ref(v___x_1149_);
v_dummy_1151_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg___closed__0);
v_nargs_1152_ = l_Lean_Expr_getAppNumArgs(v_a_1150_);
lean_inc(v_nargs_1152_);
v___x_1153_ = lean_mk_array(v_nargs_1152_, v_dummy_1151_);
v___x_1154_ = lean_unsigned_to_nat(1u);
v___x_1155_ = lean_nat_sub(v_nargs_1152_, v___x_1154_);
lean_dec(v_nargs_1152_);
v___x_1156_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_1150_, v___x_1153_, v___x_1155_);
v___x_1157_ = lean_array_get_size(v___x_1156_);
v___x_1158_ = lean_unsigned_to_nat(4u);
v___x_1159_ = lean_nat_dec_eq(v___x_1157_, v___x_1158_);
if (v___x_1159_ == 0)
{
lean_object* v___x_1160_; 
lean_dec_ref(v___x_1156_);
lean_dec_ref(v___f_1141_);
v___x_1160_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_1160_;
}
else
{
lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; uint8_t v___x_1347_; 
v___x_1161_ = lean_array_fget(v___x_1156_, v___x_1154_);
v___x_1162_ = lean_unsigned_to_nat(3u);
v___x_1163_ = lean_array_fget(v___x_1156_, v___x_1162_);
lean_dec_ref(v___x_1156_);
v___x_1347_ = l_Lean_Expr_isLambda(v___x_1161_);
lean_dec(v___x_1161_);
if (v___x_1347_ == 0)
{
lean_object* v___x_1348_; 
v___x_1348_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_1348_) == 0)
{
lean_dec_ref_known(v___x_1348_, 1);
goto v___jp_1164_;
}
else
{
lean_object* v_a_1349_; lean_object* v___x_1351_; uint8_t v_isShared_1352_; uint8_t v_isSharedCheck_1356_; 
lean_dec(v___x_1163_);
lean_dec_ref(v___f_1141_);
v_a_1349_ = lean_ctor_get(v___x_1348_, 0);
v_isSharedCheck_1356_ = !lean_is_exclusive(v___x_1348_);
if (v_isSharedCheck_1356_ == 0)
{
v___x_1351_ = v___x_1348_;
v_isShared_1352_ = v_isSharedCheck_1356_;
goto v_resetjp_1350_;
}
else
{
lean_inc(v_a_1349_);
lean_dec(v___x_1348_);
v___x_1351_ = lean_box(0);
v_isShared_1352_ = v_isSharedCheck_1356_;
goto v_resetjp_1350_;
}
v_resetjp_1350_:
{
lean_object* v___x_1354_; 
if (v_isShared_1352_ == 0)
{
v___x_1354_ = v___x_1351_;
goto v_reusejp_1353_;
}
else
{
lean_object* v_reuseFailAlloc_1355_; 
v_reuseFailAlloc_1355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1355_, 0, v_a_1349_);
v___x_1354_ = v_reuseFailAlloc_1355_;
goto v_reusejp_1353_;
}
v_reusejp_1353_:
{
return v___x_1354_;
}
}
}
}
else
{
goto v___jp_1164_;
}
v___jp_1164_:
{
uint8_t v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; 
v___x_1165_ = 0;
v___x_1166_ = l_Lean_NameSet_empty;
v___x_1167_ = lean_box(v___x_1165_);
v___x_1168_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_Delaborator_withBindingBodyUnusedName___boxed), 11, 4);
lean_closure_set(v___x_1168_, 0, lean_box(0));
lean_closure_set(v___x_1168_, 1, v___f_1141_);
lean_closure_set(v___x_1168_, 2, v___x_1167_);
lean_closure_set(v___x_1168_, 3, v___x_1166_);
v___x_1169_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v___x_1154_, v___x_1168_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1169_) == 0)
{
lean_object* v_a_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v_a_1170_ = lean_ctor_get(v___x_1169_, 0);
lean_inc(v_a_1170_);
lean_dec_ref_known(v___x_1169_, 1);
v___x_1171_ = l_Lean_TSyntax_getId(v_a_1170_);
v___x_1172_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__0));
v___x_1173_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2___boxed), 10, 3);
lean_closure_set(v___x_1173_, 0, lean_box(0));
lean_closure_set(v___x_1173_, 1, v___x_1171_);
lean_closure_set(v___x_1173_, 2, v___x_1172_);
v___x_1174_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v___x_1154_, v___x_1173_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1174_) == 0)
{
lean_object* v_a_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; uint8_t v___x_1178_; 
v_a_1175_ = lean_ctor_get(v___x_1174_, 0);
lean_inc(v_a_1175_);
lean_dec_ref_known(v___x_1174_, 1);
v___x_1176_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__111));
v___x_1177_ = lean_unsigned_to_nat(2u);
v___x_1178_ = l_Lean_Expr_isAppOfArity(v___x_1163_, v___x_1176_, v___x_1177_);
if (v___x_1178_ == 0)
{
lean_object* v___x_1179_; uint8_t v___x_1180_; 
v___x_1179_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__3));
v___x_1180_ = l_Lean_Expr_isAppOfArity(v___x_1163_, v___x_1179_, v___x_1162_);
if (v___x_1180_ == 0)
{
lean_object* v___x_1181_; 
lean_dec(v___x_1163_);
v___x_1181_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v___x_1162_, v___x_1172_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1181_) == 0)
{
lean_object* v_a_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1209_; 
v_a_1182_ = lean_ctor_get(v___x_1181_, 0);
v_isSharedCheck_1209_ = !lean_is_exclusive(v___x_1181_);
if (v_isSharedCheck_1209_ == 0)
{
v___x_1184_ = v___x_1181_;
v_isShared_1185_ = v_isSharedCheck_1209_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_a_1182_);
lean_dec(v___x_1181_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1209_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v_ref_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1207_; 
v_ref_1186_ = lean_ctor_get(v___y_1146_, 5);
v___x_1187_ = l_Lean_SourceInfo_fromRef(v_ref_1186_, v___x_1180_);
v___x_1188_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3));
v___x_1189_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4));
lean_inc_n(v___x_1187_, 8);
v___x_1190_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1187_);
lean_ctor_set(v___x_1190_, 1, v___x_1189_);
v___x_1191_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7));
v___x_1192_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10));
v___x_1193_ = l_Lean_Syntax_node1(v___x_1187_, v___x_1192_, v_a_1170_);
v___x_1194_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_1195_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__6));
v___x_1196_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__7));
v___x_1197_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1187_);
lean_ctor_set(v___x_1197_, 1, v___x_1196_);
v___x_1198_ = l_Lean_Syntax_node2(v___x_1187_, v___x_1195_, v___x_1197_, v_a_1182_);
v___x_1199_ = l_Lean_Syntax_node1(v___x_1187_, v___x_1194_, v___x_1198_);
v___x_1200_ = l_Lean_Syntax_node2(v___x_1187_, v___x_1191_, v___x_1193_, v___x_1199_);
v___x_1201_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8));
v___x_1202_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1202_, 0, v___x_1187_);
lean_ctor_set(v___x_1202_, 1, v___x_1201_);
v___x_1203_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9));
v___x_1204_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1204_, 0, v___x_1187_);
lean_ctor_set(v___x_1204_, 1, v___x_1203_);
v___x_1205_ = l_Lean_Syntax_node5(v___x_1187_, v___x_1188_, v___x_1190_, v___x_1200_, v___x_1202_, v_a_1175_, v___x_1204_);
if (v_isShared_1185_ == 0)
{
lean_ctor_set(v___x_1184_, 0, v___x_1205_);
v___x_1207_ = v___x_1184_;
goto v_reusejp_1206_;
}
else
{
lean_object* v_reuseFailAlloc_1208_; 
v_reuseFailAlloc_1208_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1208_, 0, v___x_1205_);
v___x_1207_ = v_reuseFailAlloc_1208_;
goto v_reusejp_1206_;
}
v_reusejp_1206_:
{
return v___x_1207_;
}
}
}
else
{
lean_dec(v_a_1175_);
lean_dec(v_a_1170_);
return v___x_1181_;
}
}
else
{
lean_object* v_nargs_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; uint8_t v___x_1215_; 
v_nargs_1210_ = l_Lean_Expr_getAppNumArgs(v___x_1163_);
lean_inc(v_nargs_1210_);
v___x_1211_ = lean_mk_array(v_nargs_1210_, v_dummy_1151_);
v___x_1212_ = lean_nat_sub(v_nargs_1210_, v___x_1154_);
lean_dec(v_nargs_1210_);
v___x_1213_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v___x_1163_, v___x_1211_, v___x_1212_);
v___x_1214_ = lean_array_get_size(v___x_1213_);
v___x_1215_ = lean_nat_dec_eq(v___x_1214_, v___x_1162_);
if (v___x_1215_ == 0)
{
lean_object* v___x_1216_; 
lean_dec_ref(v___x_1213_);
lean_dec(v_a_1175_);
lean_dec(v_a_1170_);
v___x_1216_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_1216_;
}
else
{
lean_object* v___x_1217_; lean_object* v___x_1218_; uint8_t v___x_1219_; 
v___x_1217_ = lean_array_fget(v___x_1213_, v___x_1177_);
lean_dec_ref(v___x_1213_);
v___x_1218_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__100));
v___x_1219_ = l_Lean_Expr_isAppOfArity(v___x_1217_, v___x_1218_, v___x_1158_);
lean_dec(v___x_1217_);
if (v___x_1219_ == 0)
{
lean_object* v___x_1220_; lean_object* v___x_1221_; 
v___x_1220_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__10));
v___x_1221_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v___x_1162_, v___x_1220_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1221_) == 0)
{
lean_object* v_a_1222_; lean_object* v___x_1224_; uint8_t v_isShared_1225_; uint8_t v_isSharedCheck_1249_; 
v_a_1222_ = lean_ctor_get(v___x_1221_, 0);
v_isSharedCheck_1249_ = !lean_is_exclusive(v___x_1221_);
if (v_isSharedCheck_1249_ == 0)
{
v___x_1224_ = v___x_1221_;
v_isShared_1225_ = v_isSharedCheck_1249_;
goto v_resetjp_1223_;
}
else
{
lean_inc(v_a_1222_);
lean_dec(v___x_1221_);
v___x_1224_ = lean_box(0);
v_isShared_1225_ = v_isSharedCheck_1249_;
goto v_resetjp_1223_;
}
v_resetjp_1223_:
{
lean_object* v_ref_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1247_; 
v_ref_1226_ = lean_ctor_get(v___y_1146_, 5);
v___x_1227_ = l_Lean_SourceInfo_fromRef(v_ref_1226_, v___x_1219_);
v___x_1228_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3));
v___x_1229_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4));
lean_inc_n(v___x_1227_, 8);
v___x_1230_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1230_, 0, v___x_1227_);
lean_ctor_set(v___x_1230_, 1, v___x_1229_);
v___x_1231_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7));
v___x_1232_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10));
v___x_1233_ = l_Lean_Syntax_node1(v___x_1227_, v___x_1232_, v_a_1170_);
v___x_1234_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_1235_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__16));
v___x_1236_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__11));
v___x_1237_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1237_, 0, v___x_1227_);
lean_ctor_set(v___x_1237_, 1, v___x_1236_);
v___x_1238_ = l_Lean_Syntax_node2(v___x_1227_, v___x_1235_, v___x_1237_, v_a_1222_);
v___x_1239_ = l_Lean_Syntax_node1(v___x_1227_, v___x_1234_, v___x_1238_);
v___x_1240_ = l_Lean_Syntax_node2(v___x_1227_, v___x_1231_, v___x_1233_, v___x_1239_);
v___x_1241_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8));
v___x_1242_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1242_, 0, v___x_1227_);
lean_ctor_set(v___x_1242_, 1, v___x_1241_);
v___x_1243_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9));
v___x_1244_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1227_);
lean_ctor_set(v___x_1244_, 1, v___x_1243_);
v___x_1245_ = l_Lean_Syntax_node5(v___x_1227_, v___x_1228_, v___x_1230_, v___x_1240_, v___x_1242_, v_a_1175_, v___x_1244_);
if (v_isShared_1225_ == 0)
{
lean_ctor_set(v___x_1224_, 0, v___x_1245_);
v___x_1247_ = v___x_1224_;
goto v_reusejp_1246_;
}
else
{
lean_object* v_reuseFailAlloc_1248_; 
v_reuseFailAlloc_1248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1248_, 0, v___x_1245_);
v___x_1247_ = v_reuseFailAlloc_1248_;
goto v_reusejp_1246_;
}
v_reusejp_1246_:
{
return v___x_1247_;
}
}
}
else
{
lean_dec(v_a_1175_);
lean_dec(v_a_1170_);
return v___x_1221_;
}
}
else
{
lean_object* v___x_1250_; lean_object* v___x_1251_; 
v___x_1250_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__13));
v___x_1251_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v___x_1162_, v___x_1250_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1251_) == 0)
{
lean_object* v_a_1252_; lean_object* v___x_1254_; uint8_t v_isShared_1255_; uint8_t v_isSharedCheck_1279_; 
v_a_1252_ = lean_ctor_get(v___x_1251_, 0);
v_isSharedCheck_1279_ = !lean_is_exclusive(v___x_1251_);
if (v_isSharedCheck_1279_ == 0)
{
v___x_1254_ = v___x_1251_;
v_isShared_1255_ = v_isSharedCheck_1279_;
goto v_resetjp_1253_;
}
else
{
lean_inc(v_a_1252_);
lean_dec(v___x_1251_);
v___x_1254_ = lean_box(0);
v_isShared_1255_ = v_isSharedCheck_1279_;
goto v_resetjp_1253_;
}
v_resetjp_1253_:
{
lean_object* v_ref_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1277_; 
v_ref_1256_ = lean_ctor_get(v___y_1146_, 5);
v___x_1257_ = l_Lean_SourceInfo_fromRef(v_ref_1256_, v___x_1178_);
v___x_1258_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3));
v___x_1259_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4));
lean_inc_n(v___x_1257_, 8);
v___x_1260_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1260_, 0, v___x_1257_);
lean_ctor_set(v___x_1260_, 1, v___x_1259_);
v___x_1261_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7));
v___x_1262_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10));
v___x_1263_ = l_Lean_Syntax_node1(v___x_1257_, v___x_1262_, v_a_1170_);
v___x_1264_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_1265_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__18));
v___x_1266_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__14));
v___x_1267_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1267_, 0, v___x_1257_);
lean_ctor_set(v___x_1267_, 1, v___x_1266_);
v___x_1268_ = l_Lean_Syntax_node2(v___x_1257_, v___x_1265_, v___x_1267_, v_a_1252_);
v___x_1269_ = l_Lean_Syntax_node1(v___x_1257_, v___x_1264_, v___x_1268_);
v___x_1270_ = l_Lean_Syntax_node2(v___x_1257_, v___x_1261_, v___x_1263_, v___x_1269_);
v___x_1271_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8));
v___x_1272_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1257_);
lean_ctor_set(v___x_1272_, 1, v___x_1271_);
v___x_1273_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9));
v___x_1274_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1257_);
lean_ctor_set(v___x_1274_, 1, v___x_1273_);
v___x_1275_ = l_Lean_Syntax_node5(v___x_1257_, v___x_1258_, v___x_1260_, v___x_1270_, v___x_1272_, v_a_1175_, v___x_1274_);
if (v_isShared_1255_ == 0)
{
lean_ctor_set(v___x_1254_, 0, v___x_1275_);
v___x_1277_ = v___x_1254_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1278_; 
v_reuseFailAlloc_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1278_, 0, v___x_1275_);
v___x_1277_ = v_reuseFailAlloc_1278_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
return v___x_1277_;
}
}
}
else
{
lean_dec(v_a_1175_);
lean_dec(v_a_1170_);
return v___x_1251_;
}
}
}
}
}
else
{
lean_object* v___x_1280_; lean_object* v___x_1281_; 
lean_dec(v___x_1163_);
v___x_1280_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__15));
v___x_1281_ = l_Lean_PrettyPrinter_Delaborator_getPPOption___redArg(v___x_1280_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1281_) == 0)
{
lean_object* v_a_1282_; lean_object* v___x_1284_; uint8_t v_isShared_1285_; uint8_t v_isSharedCheck_1338_; 
v_a_1282_ = lean_ctor_get(v___x_1281_, 0);
v_isSharedCheck_1338_ = !lean_is_exclusive(v___x_1281_);
if (v_isSharedCheck_1338_ == 0)
{
v___x_1284_ = v___x_1281_;
v_isShared_1285_ = v_isSharedCheck_1338_;
goto v_resetjp_1283_;
}
else
{
lean_inc(v_a_1282_);
lean_dec(v___x_1281_);
v___x_1284_ = lean_box(0);
v_isShared_1285_ = v_isSharedCheck_1338_;
goto v_resetjp_1283_;
}
v_resetjp_1283_:
{
uint8_t v___x_1286_; 
v___x_1286_ = lean_unbox(v_a_1282_);
if (v___x_1286_ == 0)
{
lean_object* v_ref_1287_; uint8_t v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1306_; 
v_ref_1287_ = lean_ctor_get(v___y_1146_, 5);
v___x_1288_ = lean_unbox(v_a_1282_);
lean_dec(v_a_1282_);
v___x_1289_ = l_Lean_SourceInfo_fromRef(v_ref_1287_, v___x_1288_);
v___x_1290_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3));
v___x_1291_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4));
lean_inc_n(v___x_1289_, 6);
v___x_1292_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1289_);
lean_ctor_set(v___x_1292_, 1, v___x_1291_);
v___x_1293_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7));
v___x_1294_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10));
v___x_1295_ = l_Lean_Syntax_node1(v___x_1289_, v___x_1294_, v_a_1170_);
v___x_1296_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_1297_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__91);
v___x_1298_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1289_);
lean_ctor_set(v___x_1298_, 1, v___x_1296_);
lean_ctor_set(v___x_1298_, 2, v___x_1297_);
v___x_1299_ = l_Lean_Syntax_node2(v___x_1289_, v___x_1293_, v___x_1295_, v___x_1298_);
v___x_1300_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8));
v___x_1301_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1301_, 0, v___x_1289_);
lean_ctor_set(v___x_1301_, 1, v___x_1300_);
v___x_1302_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9));
v___x_1303_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1303_, 0, v___x_1289_);
lean_ctor_set(v___x_1303_, 1, v___x_1302_);
v___x_1304_ = l_Lean_Syntax_node5(v___x_1289_, v___x_1290_, v___x_1292_, v___x_1299_, v___x_1301_, v_a_1175_, v___x_1303_);
if (v_isShared_1285_ == 0)
{
lean_ctor_set(v___x_1284_, 0, v___x_1304_);
v___x_1306_ = v___x_1284_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1307_; 
v_reuseFailAlloc_1307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1307_, 0, v___x_1304_);
v___x_1306_ = v_reuseFailAlloc_1307_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
return v___x_1306_;
}
}
else
{
lean_object* v___x_1308_; lean_object* v___x_1309_; 
lean_del_object(v___x_1284_);
lean_dec(v_a_1282_);
v___x_1308_ = lean_unsigned_to_nat(0u);
v___x_1309_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1___redArg(v___x_1308_, v___x_1172_, v___y_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_, v___y_1147_);
if (lean_obj_tag(v___x_1309_) == 0)
{
lean_object* v_a_1310_; lean_object* v___x_1312_; uint8_t v_isShared_1313_; uint8_t v_isSharedCheck_1337_; 
v_a_1310_ = lean_ctor_get(v___x_1309_, 0);
v_isSharedCheck_1337_ = !lean_is_exclusive(v___x_1309_);
if (v_isSharedCheck_1337_ == 0)
{
v___x_1312_ = v___x_1309_;
v_isShared_1313_ = v_isSharedCheck_1337_;
goto v_resetjp_1311_;
}
else
{
lean_inc(v_a_1310_);
lean_dec(v___x_1309_);
v___x_1312_ = lean_box(0);
v_isShared_1313_ = v_isSharedCheck_1337_;
goto v_resetjp_1311_;
}
v_resetjp_1311_:
{
lean_object* v_ref_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1335_; 
v_ref_1314_ = lean_ctor_get(v___y_1146_, 5);
v___x_1315_ = l_Lean_SourceInfo_fromRef(v_ref_1314_, v___x_1165_);
v___x_1316_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__3));
v___x_1317_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__4));
lean_inc_n(v___x_1315_, 8);
v___x_1318_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1315_);
lean_ctor_set(v___x_1318_, 1, v___x_1317_);
v___x_1319_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__7));
v___x_1320_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__10));
v___x_1321_ = l_Lean_Syntax_node1(v___x_1315_, v___x_1320_, v_a_1170_);
v___x_1322_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__31));
v___x_1323_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__14));
v___x_1324_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderSetOf___closed__107));
v___x_1325_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1325_, 0, v___x_1315_);
lean_ctor_set(v___x_1325_, 1, v___x_1324_);
v___x_1326_ = l_Lean_Syntax_node2(v___x_1315_, v___x_1323_, v___x_1325_, v_a_1310_);
v___x_1327_ = l_Lean_Syntax_node1(v___x_1315_, v___x_1322_, v___x_1326_);
v___x_1328_ = l_Lean_Syntax_node2(v___x_1315_, v___x_1319_, v___x_1321_, v___x_1327_);
v___x_1329_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__8));
v___x_1330_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1315_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
v___x_1331_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___closed__9));
v___x_1332_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1332_, 0, v___x_1315_);
lean_ctor_set(v___x_1332_, 1, v___x_1331_);
v___x_1333_ = l_Lean_Syntax_node5(v___x_1315_, v___x_1316_, v___x_1318_, v___x_1328_, v___x_1330_, v_a_1175_, v___x_1332_);
if (v_isShared_1313_ == 0)
{
lean_ctor_set(v___x_1312_, 0, v___x_1333_);
v___x_1335_ = v___x_1312_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v___x_1333_);
v___x_1335_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
return v___x_1335_;
}
}
}
else
{
lean_dec(v_a_1175_);
lean_dec(v_a_1170_);
return v___x_1309_;
}
}
}
}
else
{
lean_object* v_a_1339_; lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1346_; 
lean_dec(v_a_1175_);
lean_dec(v_a_1170_);
v_a_1339_ = lean_ctor_get(v___x_1281_, 0);
v_isSharedCheck_1346_ = !lean_is_exclusive(v___x_1281_);
if (v_isSharedCheck_1346_ == 0)
{
v___x_1341_ = v___x_1281_;
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
else
{
lean_inc(v_a_1339_);
lean_dec(v___x_1281_);
v___x_1341_ = lean_box(0);
v_isShared_1342_ = v_isSharedCheck_1346_;
goto v_resetjp_1340_;
}
v_resetjp_1340_:
{
lean_object* v___x_1344_; 
if (v_isShared_1342_ == 0)
{
v___x_1344_ = v___x_1341_;
goto v_reusejp_1343_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v_a_1339_);
v___x_1344_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1343_;
}
v_reusejp_1343_:
{
return v___x_1344_;
}
}
}
}
}
else
{
lean_dec(v_a_1170_);
lean_dec(v___x_1163_);
return v___x_1174_;
}
}
else
{
lean_dec(v___x_1163_);
return v___x_1169_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1___boxed(lean_object* v___f_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_){
_start:
{
lean_object* v_res_1365_; 
v_res_1365_ = lp_mathlib_Mathlib_Meta_delabFinsetFilter___lam__1(v___f_1357_, v___y_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_);
lean_dec(v___y_1363_);
lean_dec_ref(v___y_1362_);
lean_dec(v___y_1361_);
lean_dec_ref(v___y_1360_);
lean_dec(v___y_1359_);
lean_dec_ref(v___y_1358_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter(lean_object* v_a_1370_, lean_object* v_a_1371_, lean_object* v_a_1372_, lean_object* v_a_1373_, lean_object* v_a_1374_, lean_object* v_a_1375_){
_start:
{
lean_object* v___f_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; 
v___f_1377_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__1));
v___x_1378_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_delabFinsetFilter___closed__2));
v___x_1379_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_1378_, v___f_1377_, v_a_1370_, v_a_1371_, v_a_1372_, v_a_1373_, v_a_1374_, v_a_1375_);
return v___x_1379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_delabFinsetFilter___boxed(lean_object* v_a_1380_, lean_object* v_a_1381_, lean_object* v_a_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_){
_start:
{
lean_object* v_res_1387_; 
v_res_1387_ = lp_mathlib_Mathlib_Meta_delabFinsetFilter(v_a_1380_, v_a_1381_, v_a_1382_, v_a_1383_, v_a_1384_, v_a_1385_);
lean_dec(v_a_1385_);
lean_dec_ref(v_a_1384_);
lean_dec(v_a_1383_);
lean_dec_ref(v_a_1382_);
lean_dec(v_a_1381_);
lean_dec_ref(v_a_1380_);
return v_res_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1(lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_){
_start:
{
lean_object* v___x_1395_; 
v___x_1395_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___redArg(v___y_1388_);
return v___x_1395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1___boxed(lean_object* v___y_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_){
_start:
{
lean_object* v_res_1403_; 
v_res_1403_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00Mathlib_Meta_delabFinsetFilter_spec__1_spec__1(v___y_1396_, v___y_1397_, v___y_1398_, v___y_1399_, v___y_1400_, v___y_1401_);
lean_dec(v___y_1401_);
lean_dec_ref(v___y_1400_);
lean_dec(v___y_1399_);
lean_dec_ref(v___y_1398_);
lean_dec(v___y_1397_);
lean_dec_ref(v___y_1396_);
return v_res_1403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4(lean_object* v_00_u03b1_1404_, lean_object* v_child_1405_, lean_object* v_childIdx_1406_, lean_object* v_x_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_){
_start:
{
lean_object* v___x_1415_; 
v___x_1415_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___redArg(v_child_1405_, v_childIdx_1406_, v_x_1407_, v___y_1408_, v___y_1409_, v___y_1410_, v___y_1411_, v___y_1412_, v___y_1413_);
return v___x_1415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4___boxed(lean_object* v_00_u03b1_1416_, lean_object* v_child_1417_, lean_object* v_childIdx_1418_, lean_object* v_x_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_){
_start:
{
lean_object* v_res_1427_; 
v_res_1427_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__4(v_00_u03b1_1416_, v_child_1417_, v_childIdx_1418_, v_x_1419_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_, v___y_1424_, v___y_1425_);
lean_dec(v___y_1425_);
lean_dec_ref(v___y_1424_);
lean_dec(v___y_1423_);
lean_dec_ref(v___y_1422_);
lean_dec(v___y_1421_);
lean_dec_ref(v___y_1420_);
return v_res_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5(lean_object* v_00_u03b1_1428_, lean_object* v_name_1429_, uint8_t v_bi_1430_, lean_object* v_type_1431_, lean_object* v_k_1432_, uint8_t v_kind_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_){
_start:
{
lean_object* v___x_1441_; 
v___x_1441_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___redArg(v_name_1429_, v_bi_1430_, v_type_1431_, v_k_1432_, v_kind_1433_, v___y_1434_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_);
return v___x_1441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5___boxed(lean_object* v_00_u03b1_1442_, lean_object* v_name_1443_, lean_object* v_bi_1444_, lean_object* v_type_1445_, lean_object* v_k_1446_, lean_object* v_kind_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_){
_start:
{
uint8_t v_bi_boxed_1455_; uint8_t v_kind_boxed_1456_; lean_object* v_res_1457_; 
v_bi_boxed_1455_ = lean_unbox(v_bi_1444_);
v_kind_boxed_1456_ = lean_unbox(v_kind_1447_);
v_res_1457_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3_spec__5(v_00_u03b1_1442_, v_name_1443_, v_bi_boxed_1455_, v_type_1445_, v_k_1446_, v_kind_boxed_1456_, v___y_1448_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_, v___y_1453_);
lean_dec(v___y_1453_);
lean_dec_ref(v___y_1452_);
lean_dec(v___y_1451_);
lean_dec_ref(v___y_1450_);
lean_dec(v___y_1449_);
lean_dec_ref(v___y_1448_);
return v_res_1457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3(lean_object* v_00_u03b1_1458_, lean_object* v_00_u03b2_1459_, lean_object* v_n_1460_, lean_object* v_v_1461_, lean_object* v_x_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_){
_start:
{
lean_object* v___x_1470_; 
v___x_1470_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___redArg(v_n_1460_, v_v_1461_, v_x_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_);
return v___x_1470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1471_, lean_object* v_00_u03b2_1472_, lean_object* v_n_1473_, lean_object* v_v_1474_, lean_object* v_x_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_){
_start:
{
lean_object* v_res_1483_; 
v_res_1483_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00Mathlib_Meta_delabFinsetFilter_spec__2_spec__3(v_00_u03b1_1471_, v_00_u03b2_1472_, v_n_1473_, v_v_1474_, v_x_1475_, v___y_1476_, v___y_1477_, v___y_1478_, v___y_1479_, v___y_1480_, v___y_1481_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
lean_dec(v___y_1479_);
lean_dec_ref(v___y_1478_);
lean_dec(v___y_1477_);
lean_dec_ref(v___y_1476_);
return v_res_1483_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidablePiFintype___redArg___lam__0(lean_object* v_f_1484_, lean_object* v_g_1485_, lean_object* v_inst_1486_, lean_object* v_a_1487_, lean_object* v_h_1488_){
_start:
{
lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; uint8_t v___x_1492_; 
lean_inc_n(v_a_1487_, 2);
v___x_1489_ = lean_apply_1(v_f_1484_, v_a_1487_);
v___x_1490_ = lean_apply_1(v_g_1485_, v_a_1487_);
v___x_1491_ = lean_apply_3(v_inst_1486_, v_a_1487_, v___x_1489_, v___x_1490_);
v___x_1492_ = lean_unbox(v___x_1491_);
return v___x_1492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidablePiFintype___redArg___lam__0___boxed(lean_object* v_f_1493_, lean_object* v_g_1494_, lean_object* v_inst_1495_, lean_object* v_a_1496_, lean_object* v_h_1497_){
_start:
{
uint8_t v_res_1498_; lean_object* v_r_1499_; 
v_res_1498_ = lp_mathlib_Fintype_decidablePiFintype___redArg___lam__0(v_f_1493_, v_g_1494_, v_inst_1495_, v_a_1496_, v_h_1497_);
v_r_1499_ = lean_box(v_res_1498_);
return v_r_1499_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidablePiFintype___redArg(lean_object* v_inst_1500_, lean_object* v_inst_1501_, lean_object* v_f_1502_, lean_object* v_g_1503_){
_start:
{
lean_object* v___f_1504_; uint8_t v___x_1505_; 
v___f_1504_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidablePiFintype___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_1504_, 0, v_f_1502_);
lean_closure_set(v___f_1504_, 1, v_g_1503_);
lean_closure_set(v___f_1504_, 2, v_inst_1500_);
v___x_1505_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_inst_1501_, v___f_1504_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidablePiFintype___redArg___boxed(lean_object* v_inst_1506_, lean_object* v_inst_1507_, lean_object* v_f_1508_, lean_object* v_g_1509_){
_start:
{
uint8_t v_res_1510_; lean_object* v_r_1511_; 
v_res_1510_ = lp_mathlib_Fintype_decidablePiFintype___redArg(v_inst_1506_, v_inst_1507_, v_f_1508_, v_g_1509_);
v_r_1511_ = lean_box(v_res_1510_);
return v_r_1511_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidablePiFintype(lean_object* v_00_u03b1_1512_, lean_object* v_00_u03b2_1513_, lean_object* v_inst_1514_, lean_object* v_inst_1515_, lean_object* v_f_1516_, lean_object* v_g_1517_){
_start:
{
uint8_t v___x_1518_; 
v___x_1518_ = lp_mathlib_Fintype_decidablePiFintype___redArg(v_inst_1514_, v_inst_1515_, v_f_1516_, v_g_1517_);
return v___x_1518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidablePiFintype___boxed(lean_object* v_00_u03b1_1519_, lean_object* v_00_u03b2_1520_, lean_object* v_inst_1521_, lean_object* v_inst_1522_, lean_object* v_f_1523_, lean_object* v_g_1524_){
_start:
{
uint8_t v_res_1525_; lean_object* v_r_1526_; 
v_res_1525_ = lp_mathlib_Fintype_decidablePiFintype(v_00_u03b1_1519_, v_00_u03b2_1520_, v_inst_1521_, v_inst_1522_, v_f_1523_, v_g_1524_);
v_r_1526_ = lean_box(v_res_1525_);
return v_r_1526_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg___lam__0(lean_object* v_inst_1527_, lean_object* v_a_1528_, lean_object* v_h_1529_){
_start:
{
lean_object* v___x_1530_; uint8_t v___x_1531_; 
v___x_1530_ = lean_apply_1(v_inst_1527_, v_a_1528_);
v___x_1531_ = lean_unbox(v___x_1530_);
return v___x_1531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableForallFintype___redArg___lam__0___boxed(lean_object* v_inst_1532_, lean_object* v_a_1533_, lean_object* v_h_1534_){
_start:
{
uint8_t v_res_1535_; lean_object* v_r_1536_; 
v_res_1535_ = lp_mathlib_Fintype_decidableForallFintype___redArg___lam__0(v_inst_1532_, v_a_1533_, v_h_1534_);
v_r_1536_ = lean_box(v_res_1535_);
return v_r_1536_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object* v_inst_1537_, lean_object* v_inst_1538_){
_start:
{
lean_object* v___f_1539_; uint8_t v___x_1540_; 
v___f_1539_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableForallFintype___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1539_, 0, v_inst_1537_);
v___x_1540_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_inst_1538_, v___f_1539_);
return v___x_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableForallFintype___redArg___boxed(lean_object* v_inst_1541_, lean_object* v_inst_1542_){
_start:
{
uint8_t v_res_1543_; lean_object* v_r_1544_; 
v_res_1543_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v_inst_1541_, v_inst_1542_);
v_r_1544_ = lean_box(v_res_1543_);
return v_r_1544_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableForallFintype(lean_object* v_00_u03b1_1545_, lean_object* v_p_1546_, lean_object* v_inst_1547_, lean_object* v_inst_1548_){
_start:
{
uint8_t v___x_1549_; 
v___x_1549_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v_inst_1547_, v_inst_1548_);
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableForallFintype___boxed(lean_object* v_00_u03b1_1550_, lean_object* v_p_1551_, lean_object* v_inst_1552_, lean_object* v_inst_1553_){
_start:
{
uint8_t v_res_1554_; lean_object* v_r_1555_; 
v_res_1554_ = lp_mathlib_Fintype_decidableForallFintype(v_00_u03b1_1550_, v_p_1551_, v_inst_1552_, v_inst_1553_);
v_r_1555_ = lean_box(v_res_1554_);
return v_r_1555_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableExistsFintype___redArg(lean_object* v_inst_1556_, lean_object* v_inst_1557_){
_start:
{
uint8_t v___x_1558_; 
v___x_1558_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_1557_, v_inst_1556_);
return v___x_1558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableExistsFintype___redArg___boxed(lean_object* v_inst_1559_, lean_object* v_inst_1560_){
_start:
{
uint8_t v_res_1561_; lean_object* v_r_1562_; 
v_res_1561_ = lp_mathlib_Fintype_decidableExistsFintype___redArg(v_inst_1559_, v_inst_1560_);
v_r_1562_ = lean_box(v_res_1561_);
return v_r_1562_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableExistsFintype(lean_object* v_00_u03b1_1563_, lean_object* v_p_1564_, lean_object* v_inst_1565_, lean_object* v_inst_1566_){
_start:
{
uint8_t v___x_1567_; 
v___x_1567_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_1566_, v_inst_1565_);
return v___x_1567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableExistsFintype___boxed(lean_object* v_00_u03b1_1568_, lean_object* v_p_1569_, lean_object* v_inst_1570_, lean_object* v_inst_1571_){
_start:
{
uint8_t v_res_1572_; lean_object* v_r_1573_; 
v_res_1572_ = lp_mathlib_Fintype_decidableExistsFintype(v_00_u03b1_1568_, v_p_1569_, v_inst_1570_, v_inst_1571_);
v_r_1573_ = lean_box(v_res_1572_);
return v_r_1573_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_instDecidableLEForall___redArg___lam__0(lean_object* v_x_1574_, lean_object* v_x_1575_, lean_object* v_inst_1576_, lean_object* v_a_1577_){
_start:
{
lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; uint8_t v___x_1581_; 
lean_inc_n(v_a_1577_, 2);
v___x_1578_ = lean_apply_1(v_x_1574_, v_a_1577_);
v___x_1579_ = lean_apply_1(v_x_1575_, v_a_1577_);
v___x_1580_ = lean_apply_3(v_inst_1576_, v_a_1577_, v___x_1578_, v___x_1579_);
v___x_1581_ = lean_unbox(v___x_1580_);
return v___x_1581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instDecidableLEForall___redArg___lam__0___boxed(lean_object* v_x_1582_, lean_object* v_x_1583_, lean_object* v_inst_1584_, lean_object* v_a_1585_){
_start:
{
uint8_t v_res_1586_; lean_object* v_r_1587_; 
v_res_1586_ = lp_mathlib_Fintype_instDecidableLEForall___redArg___lam__0(v_x_1582_, v_x_1583_, v_inst_1584_, v_a_1585_);
v_r_1587_ = lean_box(v_res_1586_);
return v_r_1587_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_instDecidableLEForall___redArg(lean_object* v_inst_1588_, lean_object* v_inst_1589_, lean_object* v_x_1590_, lean_object* v_x_1591_){
_start:
{
lean_object* v___f_1592_; uint8_t v___x_1593_; 
v___f_1592_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_instDecidableLEForall___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_1592_, 0, v_x_1590_);
lean_closure_set(v___f_1592_, 1, v_x_1591_);
lean_closure_set(v___f_1592_, 2, v_inst_1588_);
v___x_1593_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_1592_, v_inst_1589_);
return v___x_1593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instDecidableLEForall___redArg___boxed(lean_object* v_inst_1594_, lean_object* v_inst_1595_, lean_object* v_x_1596_, lean_object* v_x_1597_){
_start:
{
uint8_t v_res_1598_; lean_object* v_r_1599_; 
v_res_1598_ = lp_mathlib_Fintype_instDecidableLEForall___redArg(v_inst_1594_, v_inst_1595_, v_x_1596_, v_x_1597_);
v_r_1599_ = lean_box(v_res_1598_);
return v_r_1599_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_instDecidableLEForall(lean_object* v_00_u03b1_1600_, lean_object* v_00_u03b2_1601_, lean_object* v_inst_1602_, lean_object* v_inst_1603_, lean_object* v_inst_1604_, lean_object* v_x_1605_, lean_object* v_x_1606_){
_start:
{
uint8_t v___x_1607_; 
v___x_1607_ = lp_mathlib_Fintype_instDecidableLEForall___redArg(v_inst_1603_, v_inst_1604_, v_x_1605_, v_x_1606_);
return v___x_1607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_instDecidableLEForall___boxed(lean_object* v_00_u03b1_1608_, lean_object* v_00_u03b2_1609_, lean_object* v_inst_1610_, lean_object* v_inst_1611_, lean_object* v_inst_1612_, lean_object* v_x_1613_, lean_object* v_x_1614_){
_start:
{
uint8_t v_res_1615_; lean_object* v_r_1616_; 
v_res_1615_ = lp_mathlib_Fintype_instDecidableLEForall(v_00_u03b1_1608_, v_00_u03b2_1609_, v_inst_1610_, v_inst_1611_, v_inst_1612_, v_x_1613_, v_x_1614_);
lean_dec_ref(v_inst_1610_);
v_r_1616_ = lean_box(v_res_1615_);
return v_r_1616_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableMemRangeFintype___redArg___lam__0(lean_object* v_f_1617_, lean_object* v_inst_1618_, lean_object* v_x_1619_, lean_object* v_a_1620_){
_start:
{
lean_object* v___x_1621_; lean_object* v___x_1622_; uint8_t v___x_1623_; 
v___x_1621_ = lean_apply_1(v_f_1617_, v_a_1620_);
v___x_1622_ = lean_apply_2(v_inst_1618_, v___x_1621_, v_x_1619_);
v___x_1623_ = lean_unbox(v___x_1622_);
return v___x_1623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableMemRangeFintype___redArg___lam__0___boxed(lean_object* v_f_1624_, lean_object* v_inst_1625_, lean_object* v_x_1626_, lean_object* v_a_1627_){
_start:
{
uint8_t v_res_1628_; lean_object* v_r_1629_; 
v_res_1628_ = lp_mathlib_Fintype_decidableMemRangeFintype___redArg___lam__0(v_f_1624_, v_inst_1625_, v_x_1626_, v_a_1627_);
v_r_1629_ = lean_box(v_res_1628_);
return v_r_1629_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableMemRangeFintype___redArg(lean_object* v_inst_1630_, lean_object* v_inst_1631_, lean_object* v_f_1632_, lean_object* v_x_1633_){
_start:
{
lean_object* v___f_1634_; uint8_t v___x_1635_; 
v___f_1634_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableMemRangeFintype___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_1634_, 0, v_f_1632_);
lean_closure_set(v___f_1634_, 1, v_inst_1631_);
lean_closure_set(v___f_1634_, 2, v_x_1633_);
v___x_1635_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_1630_, v___f_1634_);
return v___x_1635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableMemRangeFintype___redArg___boxed(lean_object* v_inst_1636_, lean_object* v_inst_1637_, lean_object* v_f_1638_, lean_object* v_x_1639_){
_start:
{
uint8_t v_res_1640_; lean_object* v_r_1641_; 
v_res_1640_ = lp_mathlib_Fintype_decidableMemRangeFintype___redArg(v_inst_1636_, v_inst_1637_, v_f_1638_, v_x_1639_);
v_r_1641_ = lean_box(v_res_1640_);
return v_r_1641_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableMemRangeFintype(lean_object* v_00_u03b1_1642_, lean_object* v_00_u03b2_1643_, lean_object* v_inst_1644_, lean_object* v_inst_1645_, lean_object* v_f_1646_, lean_object* v_x_1647_){
_start:
{
uint8_t v___x_1648_; 
v___x_1648_ = lp_mathlib_Fintype_decidableMemRangeFintype___redArg(v_inst_1644_, v_inst_1645_, v_f_1646_, v_x_1647_);
return v___x_1648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableMemRangeFintype___boxed(lean_object* v_00_u03b1_1649_, lean_object* v_00_u03b2_1650_, lean_object* v_inst_1651_, lean_object* v_inst_1652_, lean_object* v_f_1653_, lean_object* v_x_1654_){
_start:
{
uint8_t v_res_1655_; lean_object* v_r_1656_; 
v_res_1655_ = lp_mathlib_Fintype_decidableMemRangeFintype(v_00_u03b1_1649_, v_00_u03b2_1650_, v_inst_1651_, v_inst_1652_, v_f_1653_, v_x_1654_);
v_r_1656_ = lean_box(v_res_1655_);
return v_r_1656_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__0(lean_object* v_inst_1657_, uint8_t v___x_1658_, lean_object* v_inst_1659_, lean_object* v_a_1660_, lean_object* v_a_1661_){
_start:
{
lean_object* v___x_1662_; uint8_t v___x_1663_; 
lean_inc(v_a_1661_);
v___x_1662_ = lean_apply_1(v_inst_1657_, v_a_1661_);
v___x_1663_ = lean_unbox(v___x_1662_);
if (v___x_1663_ == 0)
{
lean_dec(v_a_1661_);
lean_dec(v_a_1660_);
lean_dec_ref(v_inst_1659_);
return v___x_1658_;
}
else
{
lean_object* v___x_1664_; uint8_t v___x_1665_; 
v___x_1664_ = lean_apply_2(v_inst_1659_, v_a_1660_, v_a_1661_);
v___x_1665_ = lean_unbox(v___x_1664_);
return v___x_1665_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__0___boxed(lean_object* v_inst_1666_, lean_object* v___x_1667_, lean_object* v_inst_1668_, lean_object* v_a_1669_, lean_object* v_a_1670_){
_start:
{
uint8_t v___x_49__boxed_1671_; uint8_t v_res_1672_; lean_object* v_r_1673_; 
v___x_49__boxed_1671_ = lean_unbox(v___x_1667_);
v_res_1672_ = lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__0(v_inst_1666_, v___x_49__boxed_1671_, v_inst_1668_, v_a_1669_, v_a_1670_);
v_r_1673_ = lean_box(v_res_1672_);
return v_r_1673_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__1(lean_object* v_inst_1674_, lean_object* v_inst_1675_, lean_object* v_inst_1676_, lean_object* v_a_1677_){
_start:
{
lean_object* v___x_1678_; uint8_t v___x_1679_; 
lean_inc_ref(v_inst_1674_);
lean_inc(v_a_1677_);
v___x_1678_ = lean_apply_1(v_inst_1674_, v_a_1677_);
v___x_1679_ = lean_unbox(v___x_1678_);
if (v___x_1679_ == 0)
{
uint8_t v___x_1680_; 
lean_dec(v_a_1677_);
lean_dec(v_inst_1676_);
lean_dec_ref(v_inst_1675_);
lean_dec_ref(v_inst_1674_);
v___x_1680_ = 1;
return v___x_1680_;
}
else
{
lean_object* v___f_1681_; uint8_t v___x_1682_; 
v___f_1681_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_1681_, 0, v_inst_1674_);
lean_closure_set(v___f_1681_, 1, v___x_1678_);
lean_closure_set(v___f_1681_, 2, v_inst_1675_);
lean_closure_set(v___f_1681_, 3, v_a_1677_);
v___x_1682_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_1681_, v_inst_1676_);
return v___x_1682_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__1___boxed(lean_object* v_inst_1683_, lean_object* v_inst_1684_, lean_object* v_inst_1685_, lean_object* v_a_1686_){
_start:
{
uint8_t v_res_1687_; lean_object* v_r_1688_; 
v_res_1687_ = lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__1(v_inst_1683_, v_inst_1684_, v_inst_1685_, v_a_1686_);
v_r_1688_ = lean_box(v_res_1687_);
return v_r_1688_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton___redArg(lean_object* v_inst_1689_, lean_object* v_inst_1690_, lean_object* v_inst_1691_){
_start:
{
lean_object* v___f_1692_; uint8_t v___x_1693_; 
lean_inc(v_inst_1689_);
v___f_1692_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableSubsingleton___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_1692_, 0, v_inst_1691_);
lean_closure_set(v___f_1692_, 1, v_inst_1690_);
lean_closure_set(v___f_1692_, 2, v_inst_1689_);
v___x_1693_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_1692_, v_inst_1689_);
return v___x_1693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___redArg___boxed(lean_object* v_inst_1694_, lean_object* v_inst_1695_, lean_object* v_inst_1696_){
_start:
{
uint8_t v_res_1697_; lean_object* v_r_1698_; 
v_res_1697_ = lp_mathlib_Fintype_decidableSubsingleton___redArg(v_inst_1694_, v_inst_1695_, v_inst_1696_);
v_r_1698_ = lean_box(v_res_1697_);
return v_r_1698_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSubsingleton(lean_object* v_00_u03b1_1699_, lean_object* v_inst_1700_, lean_object* v_inst_1701_, lean_object* v_s_1702_, lean_object* v_inst_1703_){
_start:
{
uint8_t v___x_1704_; 
v___x_1704_ = lp_mathlib_Fintype_decidableSubsingleton___redArg(v_inst_1700_, v_inst_1701_, v_inst_1703_);
return v___x_1704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSubsingleton___boxed(lean_object* v_00_u03b1_1705_, lean_object* v_inst_1706_, lean_object* v_inst_1707_, lean_object* v_s_1708_, lean_object* v_inst_1709_){
_start:
{
uint8_t v_res_1710_; lean_object* v_r_1711_; 
v_res_1710_ = lp_mathlib_Fintype_decidableSubsingleton(v_00_u03b1_1705_, v_inst_1706_, v_inst_1707_, v_s_1708_, v_inst_1709_);
v_r_1711_ = lean_box(v_res_1710_);
return v_r_1711_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0(lean_object* v_inst_1712_, lean_object* v_a_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_){
_start:
{
lean_object* v___x_1716_; uint8_t v___x_1717_; 
v___x_1716_ = lean_apply_2(v_inst_1712_, v___y_1714_, v___y_1715_);
v___x_1717_ = lean_unbox(v___x_1716_);
return v___x_1717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0___boxed(lean_object* v_inst_1718_, lean_object* v_a_1719_, lean_object* v___y_1720_, lean_object* v___y_1721_){
_start:
{
uint8_t v_res_1722_; lean_object* v_r_1723_; 
v_res_1722_ = lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0(v_inst_1718_, v_a_1719_, v___y_1720_, v___y_1721_);
lean_dec(v_a_1719_);
v_r_1723_ = lean_box(v_res_1722_);
return v_r_1723_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEquivFintype___redArg(lean_object* v_inst_1724_, lean_object* v_inst_1725_, lean_object* v_a_1726_, lean_object* v_b_1727_){
_start:
{
lean_object* v_toFun_1728_; lean_object* v_toFun_1729_; lean_object* v___f_1730_; uint8_t v___x_1731_; 
v_toFun_1728_ = lean_ctor_get(v_a_1726_, 0);
lean_inc(v_toFun_1728_);
lean_dec_ref(v_a_1726_);
v_toFun_1729_ = lean_ctor_get(v_b_1727_, 0);
lean_inc(v_toFun_1729_);
lean_dec_ref(v_b_1727_);
v___f_1730_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1730_, 0, v_inst_1724_);
v___x_1731_ = lp_mathlib_Fintype_decidablePiFintype___redArg(v___f_1730_, v_inst_1725_, v_toFun_1728_, v_toFun_1729_);
return v___x_1731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEquivFintype___redArg___boxed(lean_object* v_inst_1732_, lean_object* v_inst_1733_, lean_object* v_a_1734_, lean_object* v_b_1735_){
_start:
{
uint8_t v_res_1736_; lean_object* v_r_1737_; 
v_res_1736_ = lp_mathlib_Fintype_decidableEqEquivFintype___redArg(v_inst_1732_, v_inst_1733_, v_a_1734_, v_b_1735_);
v_r_1737_ = lean_box(v_res_1736_);
return v_r_1737_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEquivFintype(lean_object* v_00_u03b1_1738_, lean_object* v_00_u03b2_1739_, lean_object* v_inst_1740_, lean_object* v_inst_1741_, lean_object* v_a_1742_, lean_object* v_b_1743_){
_start:
{
uint8_t v___x_1744_; 
v___x_1744_ = lp_mathlib_Fintype_decidableEqEquivFintype___redArg(v_inst_1740_, v_inst_1741_, v_a_1742_, v_b_1743_);
return v___x_1744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEquivFintype___boxed(lean_object* v_00_u03b1_1745_, lean_object* v_00_u03b2_1746_, lean_object* v_inst_1747_, lean_object* v_inst_1748_, lean_object* v_a_1749_, lean_object* v_b_1750_){
_start:
{
uint8_t v_res_1751_; lean_object* v_r_1752_; 
v_res_1751_ = lp_mathlib_Fintype_decidableEqEquivFintype(v_00_u03b1_1745_, v_00_u03b2_1746_, v_inst_1747_, v_inst_1748_, v_a_1749_, v_b_1750_);
v_r_1752_ = lean_box(v_res_1751_);
return v_r_1752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___lam__1(lean_object* v_a_1753_, lean_object* v___y_1754_){
_start:
{
lean_object* v___x_1755_; 
v___x_1755_ = lean_apply_1(v_a_1753_, v___y_1754_);
return v___x_1755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___lam__0(lean_object* v_b_1756_, lean_object* v___y_1757_){
_start:
{
lean_object* v___x_1758_; 
v___x_1758_ = lean_apply_1(v_b_1756_, v___y_1757_);
return v___x_1758_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg(lean_object* v_inst_1759_, lean_object* v_inst_1760_, lean_object* v_a_1761_, lean_object* v_b_1762_){
_start:
{
lean_object* v___f_1763_; lean_object* v___f_1764_; lean_object* v___f_1765_; uint8_t v___x_1766_; 
v___f_1763_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableEqEquivFintype___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_1763_, 0, v_inst_1759_);
v___f_1764_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1764_, 0, v_a_1761_);
v___f_1765_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1765_, 0, v_b_1762_);
v___x_1766_ = lp_mathlib_Fintype_decidablePiFintype___redArg(v___f_1763_, v_inst_1760_, v___f_1764_, v___f_1765_);
return v___x_1766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg___boxed(lean_object* v_inst_1767_, lean_object* v_inst_1768_, lean_object* v_a_1769_, lean_object* v_b_1770_){
_start:
{
uint8_t v_res_1771_; lean_object* v_r_1772_; 
v_res_1771_ = lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg(v_inst_1767_, v_inst_1768_, v_a_1769_, v_b_1770_);
v_r_1772_ = lean_box(v_res_1771_);
return v_r_1772_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableEqEmbeddingFintype(lean_object* v_00_u03b1_1773_, lean_object* v_00_u03b2_1774_, lean_object* v_inst_1775_, lean_object* v_inst_1776_, lean_object* v_a_1777_, lean_object* v_b_1778_){
_start:
{
uint8_t v___x_1779_; 
v___x_1779_ = lp_mathlib_Fintype_decidableEqEmbeddingFintype___redArg(v_inst_1775_, v_inst_1776_, v_a_1777_, v_b_1778_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableEqEmbeddingFintype___boxed(lean_object* v_00_u03b1_1780_, lean_object* v_00_u03b2_1781_, lean_object* v_inst_1782_, lean_object* v_inst_1783_, lean_object* v_a_1784_, lean_object* v_b_1785_){
_start:
{
uint8_t v_res_1786_; lean_object* v_r_1787_; 
v_res_1786_ = lp_mathlib_Fintype_decidableEqEmbeddingFintype(v_00_u03b1_1780_, v_00_u03b2_1781_, v_inst_1782_, v_inst_1783_, v_a_1784_, v_b_1785_);
v_r_1787_ = lean_box(v_res_1786_);
return v_r_1787_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableInjectiveFintype___redArg(lean_object* v_inst_1788_, lean_object* v_inst_1789_, lean_object* v_f_1790_){
_start:
{
lean_object* v___x_1791_; uint8_t v___x_1792_; 
v___x_1791_ = lp_mathlib_Multiset_map___redArg(v_f_1790_, v_inst_1789_);
v___x_1792_ = l_List_nodupDecidable___redArg(v_inst_1788_, v___x_1791_);
return v___x_1792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableInjectiveFintype___redArg___boxed(lean_object* v_inst_1793_, lean_object* v_inst_1794_, lean_object* v_f_1795_){
_start:
{
uint8_t v_res_1796_; lean_object* v_r_1797_; 
v_res_1796_ = lp_mathlib_Fintype_decidableInjectiveFintype___redArg(v_inst_1793_, v_inst_1794_, v_f_1795_);
v_r_1797_ = lean_box(v_res_1796_);
return v_r_1797_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableInjectiveFintype(lean_object* v_00_u03b1_1798_, lean_object* v_00_u03b2_1799_, lean_object* v_inst_1800_, lean_object* v_inst_1801_, lean_object* v_f_1802_){
_start:
{
uint8_t v___x_1803_; 
v___x_1803_ = lp_mathlib_Fintype_decidableInjectiveFintype___redArg(v_inst_1800_, v_inst_1801_, v_f_1802_);
return v___x_1803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableInjectiveFintype___boxed(lean_object* v_00_u03b1_1804_, lean_object* v_00_u03b2_1805_, lean_object* v_inst_1806_, lean_object* v_inst_1807_, lean_object* v_f_1808_){
_start:
{
uint8_t v_res_1809_; lean_object* v_r_1810_; 
v_res_1809_ = lp_mathlib_Fintype_decidableInjectiveFintype(v_00_u03b1_1804_, v_00_u03b2_1805_, v_inst_1806_, v_inst_1807_, v_f_1808_);
v_r_1810_ = lean_box(v_res_1809_);
return v_r_1810_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__0(lean_object* v_x_1811_, lean_object* v_inst_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_){
_start:
{
lean_object* v___x_1815_; lean_object* v___x_1816_; uint8_t v___x_1817_; 
v___x_1815_ = lean_apply_1(v_x_1811_, v_a_1814_);
v___x_1816_ = lean_apply_2(v_inst_1812_, v___x_1815_, v_a_1813_);
v___x_1817_ = lean_unbox(v___x_1816_);
return v___x_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__0___boxed(lean_object* v_x_1818_, lean_object* v_inst_1819_, lean_object* v_a_1820_, lean_object* v_a_1821_){
_start:
{
uint8_t v_res_1822_; lean_object* v_r_1823_; 
v_res_1822_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__0(v_x_1818_, v_inst_1819_, v_a_1820_, v_a_1821_);
v_r_1823_ = lean_box(v_res_1822_);
return v_r_1823_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__1(lean_object* v_x_1824_, lean_object* v_inst_1825_, lean_object* v_inst_1826_, lean_object* v_a_1827_){
_start:
{
lean_object* v___f_1828_; uint8_t v___x_1829_; 
v___f_1828_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_1828_, 0, v_x_1824_);
lean_closure_set(v___f_1828_, 1, v_inst_1825_);
lean_closure_set(v___f_1828_, 2, v_a_1827_);
v___x_1829_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_1826_, v___f_1828_);
return v___x_1829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__1___boxed(lean_object* v_x_1830_, lean_object* v_inst_1831_, lean_object* v_inst_1832_, lean_object* v_a_1833_){
_start:
{
uint8_t v_res_1834_; lean_object* v_r_1835_; 
v_res_1834_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__1(v_x_1830_, v_inst_1831_, v_inst_1832_, v_a_1833_);
v_r_1835_ = lean_box(v_res_1834_);
return v_r_1835_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(lean_object* v_inst_1836_, lean_object* v_inst_1837_, lean_object* v_inst_1838_, lean_object* v_x_1839_){
_start:
{
lean_object* v___f_1840_; uint8_t v___x_1841_; 
v___f_1840_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_1840_, 0, v_x_1839_);
lean_closure_set(v___f_1840_, 1, v_inst_1836_);
lean_closure_set(v___f_1840_, 2, v_inst_1837_);
v___x_1841_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_1840_, v_inst_1838_);
return v___x_1841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg___boxed(lean_object* v_inst_1842_, lean_object* v_inst_1843_, lean_object* v_inst_1844_, lean_object* v_x_1845_){
_start:
{
uint8_t v_res_1846_; lean_object* v_r_1847_; 
v_res_1846_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(v_inst_1842_, v_inst_1843_, v_inst_1844_, v_x_1845_);
v_r_1847_ = lean_box(v_res_1846_);
return v_r_1847_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1(lean_object* v_00_u03b1_1848_, lean_object* v_00_u03b2_1849_, lean_object* v_inst_1850_, lean_object* v_inst_1851_, lean_object* v_inst_1852_, lean_object* v_x_1853_){
_start:
{
uint8_t v___x_1854_; 
v___x_1854_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(v_inst_1850_, v_inst_1851_, v_inst_1852_, v_x_1853_);
return v___x_1854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___boxed(lean_object* v_00_u03b1_1855_, lean_object* v_00_u03b2_1856_, lean_object* v_inst_1857_, lean_object* v_inst_1858_, lean_object* v_inst_1859_, lean_object* v_x_1860_){
_start:
{
uint8_t v_res_1861_; lean_object* v_r_1862_; 
v_res_1861_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1(v_00_u03b1_1855_, v_00_u03b2_1856_, v_inst_1857_, v_inst_1858_, v_inst_1859_, v_x_1860_);
v_r_1862_ = lean_box(v_res_1861_);
return v_r_1862_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype___redArg(lean_object* v_inst_1863_, lean_object* v_inst_1864_, lean_object* v_inst_1865_, lean_object* v_x_1866_){
_start:
{
uint8_t v___x_1867_; 
v___x_1867_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(v_inst_1863_, v_inst_1864_, v_inst_1865_, v_x_1866_);
return v___x_1867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___redArg___boxed(lean_object* v_inst_1868_, lean_object* v_inst_1869_, lean_object* v_inst_1870_, lean_object* v_x_1871_){
_start:
{
uint8_t v_res_1872_; lean_object* v_r_1873_; 
v_res_1872_ = lp_mathlib_Fintype_decidableSurjectiveFintype___redArg(v_inst_1868_, v_inst_1869_, v_inst_1870_, v_x_1871_);
v_r_1873_ = lean_box(v_res_1872_);
return v_r_1873_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableSurjectiveFintype(lean_object* v_00_u03b1_1874_, lean_object* v_00_u03b2_1875_, lean_object* v_inst_1876_, lean_object* v_inst_1877_, lean_object* v_inst_1878_, lean_object* v_x_1879_){
_start:
{
uint8_t v___x_1880_; 
v___x_1880_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(v_inst_1876_, v_inst_1877_, v_inst_1878_, v_x_1879_);
return v___x_1880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableSurjectiveFintype___boxed(lean_object* v_00_u03b1_1881_, lean_object* v_00_u03b2_1882_, lean_object* v_inst_1883_, lean_object* v_inst_1884_, lean_object* v_inst_1885_, lean_object* v_x_1886_){
_start:
{
uint8_t v_res_1887_; lean_object* v_r_1888_; 
v_res_1887_ = lp_mathlib_Fintype_decidableSurjectiveFintype(v_00_u03b1_1881_, v_00_u03b2_1882_, v_inst_1883_, v_inst_1884_, v_inst_1885_, v_x_1886_);
v_r_1888_ = lean_box(v_res_1887_);
return v_r_1888_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg(lean_object* v_inst_1889_, lean_object* v_inst_1890_, lean_object* v_inst_1891_, lean_object* v_x_1892_){
_start:
{
uint8_t v___x_1893_; uint8_t v___x_1894_; 
lean_inc(v_x_1892_);
lean_inc(v_inst_1890_);
lean_inc_ref(v_inst_1889_);
v___x_1893_ = lp_mathlib_Fintype_decidableSurjectiveFintype___aux__1___redArg(v_inst_1889_, v_inst_1890_, v_inst_1891_, v_x_1892_);
v___x_1894_ = lp_mathlib_Fintype_decidableInjectiveFintype___redArg(v_inst_1889_, v_inst_1890_, v_x_1892_);
if (v___x_1894_ == 0)
{
return v___x_1894_;
}
else
{
return v___x_1893_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg___boxed(lean_object* v_inst_1895_, lean_object* v_inst_1896_, lean_object* v_inst_1897_, lean_object* v_x_1898_){
_start:
{
uint8_t v_res_1899_; lean_object* v_r_1900_; 
v_res_1899_ = lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg(v_inst_1895_, v_inst_1896_, v_inst_1897_, v_x_1898_);
v_r_1900_ = lean_box(v_res_1899_);
return v_r_1900_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype___aux__1(lean_object* v_00_u03b1_1901_, lean_object* v_00_u03b2_1902_, lean_object* v_inst_1903_, lean_object* v_inst_1904_, lean_object* v_inst_1905_, lean_object* v_x_1906_){
_start:
{
uint8_t v___x_1907_; 
v___x_1907_ = lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg(v_inst_1903_, v_inst_1904_, v_inst_1905_, v_x_1906_);
return v___x_1907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___boxed(lean_object* v_00_u03b1_1908_, lean_object* v_00_u03b2_1909_, lean_object* v_inst_1910_, lean_object* v_inst_1911_, lean_object* v_inst_1912_, lean_object* v_x_1913_){
_start:
{
uint8_t v_res_1914_; lean_object* v_r_1915_; 
v_res_1914_ = lp_mathlib_Fintype_decidableBijectiveFintype___aux__1(v_00_u03b1_1908_, v_00_u03b2_1909_, v_inst_1910_, v_inst_1911_, v_inst_1912_, v_x_1913_);
v_r_1915_ = lean_box(v_res_1914_);
return v_r_1915_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype___redArg(lean_object* v_inst_1916_, lean_object* v_inst_1917_, lean_object* v_inst_1918_, lean_object* v_x_1919_){
_start:
{
uint8_t v___x_1920_; 
v___x_1920_ = lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg(v_inst_1916_, v_inst_1917_, v_inst_1918_, v_x_1919_);
return v___x_1920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___redArg___boxed(lean_object* v_inst_1921_, lean_object* v_inst_1922_, lean_object* v_inst_1923_, lean_object* v_x_1924_){
_start:
{
uint8_t v_res_1925_; lean_object* v_r_1926_; 
v_res_1925_ = lp_mathlib_Fintype_decidableBijectiveFintype___redArg(v_inst_1921_, v_inst_1922_, v_inst_1923_, v_x_1924_);
v_r_1926_ = lean_box(v_res_1925_);
return v_r_1926_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableBijectiveFintype(lean_object* v_00_u03b1_1927_, lean_object* v_00_u03b2_1928_, lean_object* v_inst_1929_, lean_object* v_inst_1930_, lean_object* v_inst_1931_, lean_object* v_x_1932_){
_start:
{
uint8_t v___x_1933_; 
v___x_1933_ = lp_mathlib_Fintype_decidableBijectiveFintype___aux__1___redArg(v_inst_1929_, v_inst_1930_, v_inst_1931_, v_x_1932_);
return v___x_1933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableBijectiveFintype___boxed(lean_object* v_00_u03b1_1934_, lean_object* v_00_u03b2_1935_, lean_object* v_inst_1936_, lean_object* v_inst_1937_, lean_object* v_inst_1938_, lean_object* v_x_1939_){
_start:
{
uint8_t v_res_1940_; lean_object* v_r_1941_; 
v_res_1940_ = lp_mathlib_Fintype_decidableBijectiveFintype(v_00_u03b1_1934_, v_00_u03b2_1935_, v_inst_1936_, v_inst_1937_, v_inst_1938_, v_x_1939_);
v_r_1941_ = lean_box(v_res_1940_);
return v_r_1941_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___lam__0(lean_object* v_f_1942_, lean_object* v_g_1943_, lean_object* v_inst_1944_, lean_object* v_a_1945_){
_start:
{
lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; uint8_t v___x_1949_; 
lean_inc(v_a_1945_);
v___x_1946_ = lean_apply_1(v_f_1942_, v_a_1945_);
v___x_1947_ = lean_apply_1(v_g_1943_, v___x_1946_);
v___x_1948_ = lean_apply_2(v_inst_1944_, v___x_1947_, v_a_1945_);
v___x_1949_ = lean_unbox(v___x_1948_);
return v___x_1949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___lam__0___boxed(lean_object* v_f_1950_, lean_object* v_g_1951_, lean_object* v_inst_1952_, lean_object* v_a_1953_){
_start:
{
uint8_t v_res_1954_; lean_object* v_r_1955_; 
v_res_1954_ = lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___lam__0(v_f_1950_, v_g_1951_, v_inst_1952_, v_a_1953_);
v_r_1955_ = lean_box(v_res_1954_);
return v_r_1955_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg(lean_object* v_inst_1956_, lean_object* v_inst_1957_, lean_object* v_f_1958_, lean_object* v_g_1959_){
_start:
{
lean_object* v___f_1960_; uint8_t v___x_1961_; 
v___f_1960_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_1960_, 0, v_f_1958_);
lean_closure_set(v___f_1960_, 1, v_g_1959_);
lean_closure_set(v___f_1960_, 2, v_inst_1956_);
v___x_1961_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_1960_, v_inst_1957_);
return v___x_1961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg___boxed(lean_object* v_inst_1962_, lean_object* v_inst_1963_, lean_object* v_f_1964_, lean_object* v_g_1965_){
_start:
{
uint8_t v_res_1966_; lean_object* v_r_1967_; 
v_res_1966_ = lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg(v_inst_1962_, v_inst_1963_, v_f_1964_, v_g_1965_);
v_r_1967_ = lean_box(v_res_1966_);
return v_r_1967_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___aux__1(lean_object* v_00_u03b1_1968_, lean_object* v_00_u03b2_1969_, lean_object* v_inst_1970_, lean_object* v_inst_1971_, lean_object* v_f_1972_, lean_object* v_g_1973_){
_start:
{
uint8_t v___x_1974_; 
v___x_1974_ = lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg(v_inst_1970_, v_inst_1971_, v_f_1972_, v_g_1973_);
return v___x_1974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___boxed(lean_object* v_00_u03b1_1975_, lean_object* v_00_u03b2_1976_, lean_object* v_inst_1977_, lean_object* v_inst_1978_, lean_object* v_f_1979_, lean_object* v_g_1980_){
_start:
{
uint8_t v_res_1981_; lean_object* v_r_1982_; 
v_res_1981_ = lp_mathlib_Fintype_decidableRightInverseFintype___aux__1(v_00_u03b1_1975_, v_00_u03b2_1976_, v_inst_1977_, v_inst_1978_, v_f_1979_, v_g_1980_);
v_r_1982_ = lean_box(v_res_1981_);
return v_r_1982_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype___redArg(lean_object* v_inst_1983_, lean_object* v_inst_1984_, lean_object* v_f_1985_, lean_object* v_g_1986_){
_start:
{
uint8_t v___x_1987_; 
v___x_1987_ = lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg(v_inst_1983_, v_inst_1984_, v_f_1985_, v_g_1986_);
return v___x_1987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___redArg___boxed(lean_object* v_inst_1988_, lean_object* v_inst_1989_, lean_object* v_f_1990_, lean_object* v_g_1991_){
_start:
{
uint8_t v_res_1992_; lean_object* v_r_1993_; 
v_res_1992_ = lp_mathlib_Fintype_decidableRightInverseFintype___redArg(v_inst_1988_, v_inst_1989_, v_f_1990_, v_g_1991_);
v_r_1993_ = lean_box(v_res_1992_);
return v_r_1993_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableRightInverseFintype(lean_object* v_00_u03b1_1994_, lean_object* v_00_u03b2_1995_, lean_object* v_inst_1996_, lean_object* v_inst_1997_, lean_object* v_f_1998_, lean_object* v_g_1999_){
_start:
{
uint8_t v___x_2000_; 
v___x_2000_ = lp_mathlib_Fintype_decidableRightInverseFintype___aux__1___redArg(v_inst_1996_, v_inst_1997_, v_f_1998_, v_g_1999_);
return v___x_2000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableRightInverseFintype___boxed(lean_object* v_00_u03b1_2001_, lean_object* v_00_u03b2_2002_, lean_object* v_inst_2003_, lean_object* v_inst_2004_, lean_object* v_f_2005_, lean_object* v_g_2006_){
_start:
{
uint8_t v_res_2007_; lean_object* v_r_2008_; 
v_res_2007_ = lp_mathlib_Fintype_decidableRightInverseFintype(v_00_u03b1_2001_, v_00_u03b2_2002_, v_inst_2003_, v_inst_2004_, v_f_2005_, v_g_2006_);
v_r_2008_ = lean_box(v_res_2007_);
return v_r_2008_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___lam__0(lean_object* v_g_2009_, lean_object* v_f_2010_, lean_object* v_inst_2011_, lean_object* v_a_2012_){
_start:
{
lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; uint8_t v___x_2016_; 
lean_inc(v_a_2012_);
v___x_2013_ = lean_apply_1(v_g_2009_, v_a_2012_);
v___x_2014_ = lean_apply_1(v_f_2010_, v___x_2013_);
v___x_2015_ = lean_apply_2(v_inst_2011_, v___x_2014_, v_a_2012_);
v___x_2016_ = lean_unbox(v___x_2015_);
return v___x_2016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___lam__0___boxed(lean_object* v_g_2017_, lean_object* v_f_2018_, lean_object* v_inst_2019_, lean_object* v_a_2020_){
_start:
{
uint8_t v_res_2021_; lean_object* v_r_2022_; 
v_res_2021_ = lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___lam__0(v_g_2017_, v_f_2018_, v_inst_2019_, v_a_2020_);
v_r_2022_ = lean_box(v_res_2021_);
return v_r_2022_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg(lean_object* v_inst_2023_, lean_object* v_inst_2024_, lean_object* v_f_2025_, lean_object* v_g_2026_){
_start:
{
lean_object* v___f_2027_; uint8_t v___x_2028_; 
v___f_2027_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_2027_, 0, v_g_2026_);
lean_closure_set(v___f_2027_, 1, v_f_2025_);
lean_closure_set(v___f_2027_, 2, v_inst_2023_);
v___x_2028_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_2027_, v_inst_2024_);
return v___x_2028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg___boxed(lean_object* v_inst_2029_, lean_object* v_inst_2030_, lean_object* v_f_2031_, lean_object* v_g_2032_){
_start:
{
uint8_t v_res_2033_; lean_object* v_r_2034_; 
v_res_2033_ = lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg(v_inst_2029_, v_inst_2030_, v_f_2031_, v_g_2032_);
v_r_2034_ = lean_box(v_res_2033_);
return v_r_2034_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1(lean_object* v_00_u03b1_2035_, lean_object* v_00_u03b2_2036_, lean_object* v_inst_2037_, lean_object* v_inst_2038_, lean_object* v_f_2039_, lean_object* v_g_2040_){
_start:
{
uint8_t v___x_2041_; 
v___x_2041_ = lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg(v_inst_2037_, v_inst_2038_, v_f_2039_, v_g_2040_);
return v___x_2041_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___boxed(lean_object* v_00_u03b1_2042_, lean_object* v_00_u03b2_2043_, lean_object* v_inst_2044_, lean_object* v_inst_2045_, lean_object* v_f_2046_, lean_object* v_g_2047_){
_start:
{
uint8_t v_res_2048_; lean_object* v_r_2049_; 
v_res_2048_ = lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1(v_00_u03b1_2042_, v_00_u03b2_2043_, v_inst_2044_, v_inst_2045_, v_f_2046_, v_g_2047_);
v_r_2049_ = lean_box(v_res_2048_);
return v_r_2049_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype___redArg(lean_object* v_inst_2050_, lean_object* v_inst_2051_, lean_object* v_f_2052_, lean_object* v_g_2053_){
_start:
{
uint8_t v___x_2054_; 
v___x_2054_ = lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg(v_inst_2050_, v_inst_2051_, v_f_2052_, v_g_2053_);
return v___x_2054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___redArg___boxed(lean_object* v_inst_2055_, lean_object* v_inst_2056_, lean_object* v_f_2057_, lean_object* v_g_2058_){
_start:
{
uint8_t v_res_2059_; lean_object* v_r_2060_; 
v_res_2059_ = lp_mathlib_Fintype_decidableLeftInverseFintype___redArg(v_inst_2055_, v_inst_2056_, v_f_2057_, v_g_2058_);
v_r_2060_ = lean_box(v_res_2059_);
return v_r_2060_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_decidableLeftInverseFintype(lean_object* v_00_u03b1_2061_, lean_object* v_00_u03b2_2062_, lean_object* v_inst_2063_, lean_object* v_inst_2064_, lean_object* v_f_2065_, lean_object* v_g_2066_){
_start:
{
uint8_t v___x_2067_; 
v___x_2067_ = lp_mathlib_Fintype_decidableLeftInverseFintype___aux__1___redArg(v_inst_2063_, v_inst_2064_, v_f_2065_, v_g_2066_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_decidableLeftInverseFintype___boxed(lean_object* v_00_u03b1_2068_, lean_object* v_00_u03b2_2069_, lean_object* v_inst_2070_, lean_object* v_inst_2071_, lean_object* v_f_2072_, lean_object* v_g_2073_){
_start:
{
uint8_t v_res_2074_; lean_object* v_r_2075_; 
v_res_2074_ = lp_mathlib_Fintype_decidableLeftInverseFintype(v_00_u03b1_2068_, v_00_u03b2_2069_, v_inst_2070_, v_inst_2071_, v_f_2072_, v_g_2073_);
v_r_2075_ = lean_box(v_res_2074_);
return v_r_2075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype___redArg___lam__0(lean_object* v_val_2076_, lean_object* v_property_2077_){
_start:
{
lean_inc(v_val_2076_);
return v_val_2076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype___redArg___lam__0___boxed(lean_object* v_val_2078_, lean_object* v_property_2079_){
_start:
{
lean_object* v_res_2080_; 
v_res_2080_ = lp_mathlib_Fintype_subtype___redArg___lam__0(v_val_2078_, v_property_2079_);
lean_dec(v_val_2078_);
return v_res_2080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object* v_s_2082_){
_start:
{
lean_object* v___f_2083_; lean_object* v___x_2084_; 
v___f_2083_ = ((lean_object*)(lp_mathlib_Fintype_subtype___redArg___closed__0));
v___x_2084_ = lp_mathlib_Multiset_pmap___redArg(v___f_2083_, v_s_2082_);
return v___x_2084_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_subtype(lean_object* v_00_u03b1_2085_, lean_object* v_p_2086_, lean_object* v_s_2087_, lean_object* v_H_2088_){
_start:
{
lean_object* v___x_2089_; 
v___x_2089_ = lp_mathlib_Fintype_subtype___redArg(v_s_2087_);
return v___x_2089_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofFinset___redArg(lean_object* v_s_2090_){
_start:
{
lean_object* v___x_2091_; 
v___x_2091_ = lp_mathlib_Fintype_subtype___redArg(v_s_2090_);
return v___x_2091_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_ofFinset(lean_object* v_00_u03b1_2092_, lean_object* v_p_2093_, lean_object* v_s_2094_, lean_object* v_H_2095_){
_start:
{
lean_object* v___x_2096_; 
v___x_2096_ = lp_mathlib_Fintype_subtype___redArg(v_s_2094_);
return v___x_2096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype___redArg(lean_object* v_inst_2119_){
_start:
{
lean_inc(v_inst_2119_);
return v_inst_2119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype___redArg___boxed(lean_object* v_inst_2120_){
_start:
{
lean_object* v_res_2121_; 
v_res_2121_ = lp_mathlib_OrderDual_fintype___redArg(v_inst_2120_);
lean_dec(v_inst_2120_);
return v_res_2121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype(lean_object* v_00_u03b1_2122_, lean_object* v_inst_2123_){
_start:
{
lean_inc(v_inst_2123_);
return v_inst_2123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_fintype___boxed(lean_object* v_00_u03b1_2124_, lean_object* v_inst_2125_){
_start:
{
lean_object* v_res_2126_; 
v_res_2126_ = lp_mathlib_OrderDual_fintype(v_00_u03b1_2124_, v_inst_2125_);
lean_dec(v_inst_2125_);
return v_res_2126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype___redArg(lean_object* v_inst_2127_){
_start:
{
lean_inc(v_inst_2127_);
return v_inst_2127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype___redArg___boxed(lean_object* v_inst_2128_){
_start:
{
lean_object* v_res_2129_; 
v_res_2129_ = lp_mathlib_Lex_fintype___redArg(v_inst_2128_);
lean_dec(v_inst_2128_);
return v_res_2129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype(lean_object* v_00_u03b1_2130_, lean_object* v_inst_2131_){
_start:
{
lean_inc(v_inst_2131_);
return v_inst_2131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_fintype___boxed(lean_object* v_00_u03b1_2132_, lean_object* v_inst_2133_){
_start:
{
lean_object* v_res_2134_; 
v_res_2134_ = lp_mathlib_Lex_fintype(v_00_u03b1_2132_, v_inst_2133_);
lean_dec(v_inst_2133_);
return v_res_2134_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
