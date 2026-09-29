// Lean compiler output
// Module: Mathlib.Order.Filter.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Insert public import Mathlib.Order.SetNotation public import Mathlib.Order.BooleanAlgebra.Set public import Mathlib.Order.Bounds.Defs
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_runTermElab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Term_Quotation_precheck(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_getBinders(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchScoped(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Array_reverse___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_focus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instMembership(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_copy(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_comk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_principal(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_term_U0001d4df___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Filter"};
static const lean_object* lp_mathlib_Filter_term_U0001d4df___closed__0 = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value;
static const lean_string_object lp_mathlib_Filter_term_U0001d4df___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 5, .m_data = "term𝓟"};
static const lean_object* lp_mathlib_Filter_term_U0001d4df___closed__1 = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__1_value;
static const lean_ctor_object lp_mathlib_Filter_term_U0001d4df___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_term_U0001d4df___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__1_value),LEAN_SCALAR_PTR_LITERAL(124, 131, 227, 231, 186, 0, 101, 22)}};
static const lean_object* lp_mathlib_Filter_term_U0001d4df___closed__2 = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__2_value;
static const lean_string_object lp_mathlib_Filter_term_U0001d4df___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 1, .m_data = "𝓟"};
static const lean_object* lp_mathlib_Filter_term_U0001d4df___closed__3 = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter_term_U0001d4df___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__3_value)}};
static const lean_object* lp_mathlib_Filter_term_U0001d4df___closed__4 = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__4_value;
static const lean_ctor_object lp_mathlib_Filter_term_U0001d4df___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__4_value)}};
static const lean_object* lp_mathlib_Filter_term_U0001d4df___closed__5 = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_term_U0001d4df = (const lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__5_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Filter.principal"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__1;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "principal"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__2 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(209, 235, 28, 198, 3, 158, 96, 185)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__4 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__5 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__1 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instPure___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instPure___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instPure___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instPure___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instPure___closed__0 = (const lean_object*)&lp_mathlib_Filter_instPure___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_instPure = (const lean_object*)&lp_mathlib_Filter_instPure___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_join(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Filter_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_Filter_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSupSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Filter_instSupSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instSupSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instSupSet___closed__0 = (const lean_object*)&lp_mathlib_Filter_instSupSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSupSet(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_sInf(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_sInf, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Filter_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Filter_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInfSet(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTop(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInf___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instInf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instInf___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instInf___closed__0 = (const lean_object*)&lp_mathlib_Filter_instInf___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInf(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSDiff(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instHNot___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Filter_instHNot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instHNot___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instHNot___closed__0 = (const lean_object*)&lp_mathlib_Filter_instHNot___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instHNot(lean_object*);
static const lean_string_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 12, .m_data = "term∀ᶠ_In_,_"};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 30, 144, 56, 152, 223, 236, 188)}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value;
static const lean_string_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∀ᶠ"};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__4_value)}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__5_value;
static lean_once_cell_t lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__6;
static const lean_string_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__7 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__7_value)}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__8_value;
static lean_once_cell_t lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__9;
static const lean_string_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__10 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12_value;
static lean_once_cell_t lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__13;
static const lean_string_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__14 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__14_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__14_value)}};
static const lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__15 = (const lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__15_value;
static lean_once_cell_t lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__16;
static lean_once_cell_t lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__17;
static lean_once_cell_t lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Filter_term_u2200_u1da0__In___x2c__;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Notation3"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termExpand_binders%(_=>_)_,_"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 176, 22, 214, 10, 13, 147, 22)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(120, 7, 237, 26, 3, 243, 131, 214)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "expand_binders%"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__4_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__5_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "p"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__6 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(34, 153, 146, 175, 179, 220, 230, 134)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__9 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__9_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__10 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__10_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__11 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__11_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__12 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__12_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__13 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Filter.Eventually"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__15 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__15_value;
static lean_once_cell_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__16;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Eventually"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__17 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__17_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(170, 201, 114, 242, 122, 67, 34, 48)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__19 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__19_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__20 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__20_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__21 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__21_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__23 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__23_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__24 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__24_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "∀ᶠ "};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__3_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__4_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__5_value;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extBinders"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__6_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__6_value),LEAN_SCALAR_PTR_LITERAL(142, 202, 111, 171, 129, 134, 17, 161)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7_value;
static lean_once_cell_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "extBinderCollection"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__9_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__9_value),LEAN_SCALAR_PTR_LITERAL(144, 58, 22, 199, 215, 82, 42, 232)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__11_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__3_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__3_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__4_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__5_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__4_value),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__5_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__6 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 12, .m_data = "term∃ᶠ_In_,_"};
static const lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(127, 196, 164, 19, 70, 139, 2, 153)}};
static const lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∃ᶠ"};
static const lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__2_value)}};
static const lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__4;
static lean_once_cell_t lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__5;
static lean_once_cell_t lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__6;
static lean_once_cell_t lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__7;
static lean_once_cell_t lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__8;
static lean_once_cell_t lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Filter_term_u2203_u1da0__In___x2c__;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Filter.Frequently"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__1;
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Frequently"};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(46, 70, 186, 141, 48, 174, 129, 118)}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__4 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__5 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "∃ᶠ "};
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__0 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__1___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__0_value),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__1 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__2 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__4_value),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__3 = (const lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_eventuallyEqStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "eventuallyEqStx"};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__0 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(115, 243, 238, 200, 228, 59, 85, 141)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__1 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__1_value;
static const lean_string_object lp_mathlib_Filter_eventuallyEqStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " =ᶠ["};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__2 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__2_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__3 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__3_value),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__4 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__4_value;
static const lean_string_object lp_mathlib_Filter_eventuallyEqStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__5 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__5_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__5_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__6 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__4_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__6_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__7 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__7_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__11_value),((lean_object*)(((size_t)(50) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__8 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__8_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__7_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__8_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__9 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__9_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyEqStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(51) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__9_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyEqStx___closed__10 = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_eventuallyEqStx = (const lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__10_value;
static const lean_string_object lp_mathlib_Filter_eventuallyLEStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "eventuallyLEStx"};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__0 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 119, 129, 124, 155, 229, 70, 96)}};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__1 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__1_value;
static const lean_string_object lp_mathlib_Filter_eventuallyLEStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ≤ᶠ["};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__2 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__2_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__3 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__3_value),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__4 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__4_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__6_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__5 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__5_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__5_value),((lean_object*)&lp_mathlib_Filter_eventuallyEqStx___closed__8_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__6 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Filter_eventuallyLEStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(51) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__6_value)}};
static const lean_object* lp_mathlib_Filter_eventuallyLEStx___closed__7 = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_eventuallyLEStx = (const lean_object*)&lp_mathlib_Filter_eventuallyLEStx___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_elabEventuallyRelSides___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Filter_elabEventuallyRelSides___closed__0 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyRelSides___closed__0_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyRelSides___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_elabEventuallyRelSides___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib_Filter_elabEventuallyRelSides___closed__1 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyRelSides___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_elabEventuallyEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Filter.EventuallyEq"};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__0 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__0_value;
static lean_once_cell_t lp_mathlib_Filter_elabEventuallyEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__1;
static const lean_string_object lp_mathlib_Filter_elabEventuallyEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "EventuallyEq"};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__2 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__2_value),LEAN_SCALAR_PTR_LITERAL(3, 137, 11, 84, 20, 233, 4, 80)}};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__3 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__4 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__4_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__5 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__5_value;
static const lean_string_object lp_mathlib_Filter_elabEventuallyEq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "Filter.EventuallyEqSet"};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__6 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__6_value;
static lean_once_cell_t lp_mathlib_Filter_elabEventuallyEq___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__7;
static const lean_string_object lp_mathlib_Filter_elabEventuallyEq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "EventuallyEqSet"};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__8 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__8_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__8_value),LEAN_SCALAR_PTR_LITERAL(103, 6, 138, 83, 246, 9, 18, 87)}};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__9 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__9_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__10 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__10_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyEq___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyEq___closed__11 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyEq___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_elabEventuallyLE___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Filter.EventuallyLE"};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__0 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__0_value;
static lean_once_cell_t lp_mathlib_Filter_elabEventuallyLE___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__1;
static const lean_string_object lp_mathlib_Filter_elabEventuallyLE___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "EventuallyLE"};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__2 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__2_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__2_value),LEAN_SCALAR_PTR_LITERAL(201, 46, 243, 165, 122, 59, 63, 210)}};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__3 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__3_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__4 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__4_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__5 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__5_value;
static const lean_string_object lp_mathlib_Filter_elabEventuallyLE___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Filter.EventuallySubset"};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__6 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__6_value;
static lean_once_cell_t lp_mathlib_Filter_elabEventuallyLE___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__7;
static const lean_string_object lp_mathlib_Filter_elabEventuallyLE___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "EventuallySubset"};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__8 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__8_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__8_value),LEAN_SCALAR_PTR_LITERAL(121, 219, 68, 65, 245, 199, 176, 102)}};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__9 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__9_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__10 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__10_value;
static const lean_ctor_object lp_mathlib_Filter_elabEventuallyLE___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Filter_elabEventuallyLE___closed__11 = (const lean_object*)&lp_mathlib_Filter_elabEventuallyLE___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_unexpandEventuallyEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = "=ᶠ["};
static const lean_object* lp_mathlib_Filter_unexpandEventuallyEq___closed__0 = (const lean_object*)&lp_mathlib_Filter_unexpandEventuallyEq___closed__0_value;
static const lean_string_object lp_mathlib_Filter_unexpandEventuallyEq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Filter_unexpandEventuallyEq___closed__1 = (const lean_object*)&lp_mathlib_Filter_unexpandEventuallyEq___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEq(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEq___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEqSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEqSet___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Filter_unexpandEventuallyLE___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 3, .m_data = "≤ᶠ["};
static const lean_object* lp_mathlib_Filter_unexpandEventuallyLE___closed__0 = (const lean_object*)&lp_mathlib_Filter_unexpandEventuallyLE___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyLE(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyLE___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallySubset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallySubset___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_comap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_coprod(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSProd___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instSProd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instSProd___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instSProd___closed__0 = (const lean_object*)&lp_mathlib_Filter_instSProd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_pi_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_pi_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_pi(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_pi___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_bind(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_seq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_curry(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instBind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_bind___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instBind___closed__0 = (const lean_object*)&lp_mathlib_Filter_instBind___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_instBind = (const lean_object*)&lp_mathlib_Filter_instBind___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instFunctor___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instFunctor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instFunctor___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instFunctor___closed__0 = (const lean_object*)&lp_mathlib_Filter_instFunctor___closed__0_value;
static const lean_closure_object lp_mathlib_Filter_instFunctor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_map___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instFunctor___closed__1 = (const lean_object*)&lp_mathlib_Filter_instFunctor___closed__1_value;
static const lean_ctor_object lp_mathlib_Filter_instFunctor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Filter_instFunctor___closed__1_value),((lean_object*)&lp_mathlib_Filter_instFunctor___closed__0_value)}};
static const lean_object* lp_mathlib_Filter_instFunctor___closed__2 = (const lean_object*)&lp_mathlib_Filter_instFunctor___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_instFunctor = (const lean_object*)&lp_mathlib_Filter_instFunctor___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_lift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_lift_x27(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "filterUpwards"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__0_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__1_value),LEAN_SCALAR_PTR_LITERAL(208, 52, 97, 161, 239, 253, 121, 232)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "filter_upwards"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12_value),((lean_object*)&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__24_value),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Filter_unexpandEventuallyEq___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__17_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__19_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__22_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__21_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__11_value),((lean_object*)(((size_t)(1024) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__25_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__29_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__32_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__33_value),((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__34_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__31_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_filterUpwards___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__37_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_filterUpwards = (const lean_object*)&lp_mathlib_Mathlib_Tactic_filterUpwards___closed__37_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "univ_mem'"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 250, 104, 86, 128, 54, 69, 241)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mp_mem"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_term_U0001d4df___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(79, 3, 110, 86, 190, 23, 90, 75)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Filter.mp_mem"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__3;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "dsimp"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "negConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zeta"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__11;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(56, 247, 87, 81, 188, 35, 250, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Set.mem_ofPred_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "mem_ofPred_eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Filter_elabEventuallyRelSides___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__18_value),LEAN_SCALAR_PTR_LITERAL(108, 115, 205, 134, 57, 52, 7, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__1_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___boxed__const__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instMembership(lean_object* v_00_u03b1_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_copy(lean_object* v_00_u03b1_3_, lean_object* v_f_4_, lean_object* v_S_5_, lean_object* v_hmem_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_comk(lean_object* v_00_u03b1_8_, lean_object* v_p_9_, lean_object* v_he_10_, lean_object* v_hmono_11_, lean_object* v_hunion_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_box(0);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_principal(lean_object* v_00_u03b1_14_, lean_object* v_s_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_box(0);
return v___x_16_;
}
}
static lean_object* _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__1(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_31_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__0));
v___x_32_ = l_String_toRawSubstring_x27(v___x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1(lean_object* v_x_43_, lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; uint8_t v___x_47_; 
v___x_46_ = ((lean_object*)(lp_mathlib_Filter_term_U0001d4df___closed__2));
v___x_47_ = l_Lean_Syntax_isOfKind(v_x_43_, v___x_46_);
if (v___x_47_ == 0)
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = lean_box(1);
v___x_49_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_49_, 0, v___x_48_);
lean_ctor_set(v___x_49_, 1, v_a_45_);
return v___x_49_;
}
else
{
lean_object* v_quotContext_50_; lean_object* v_currMacroScope_51_; lean_object* v_ref_52_; uint8_t v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v_quotContext_50_ = lean_ctor_get(v_a_44_, 1);
v_currMacroScope_51_ = lean_ctor_get(v_a_44_, 2);
v_ref_52_ = lean_ctor_get(v_a_44_, 5);
v___x_53_ = 0;
v___x_54_ = l_Lean_SourceInfo_fromRef(v_ref_52_, v___x_53_);
v___x_55_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__1, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__1_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__1);
v___x_56_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__3));
lean_inc(v_currMacroScope_51_);
lean_inc(v_quotContext_50_);
v___x_57_ = l_Lean_addMacroScope(v_quotContext_50_, v___x_56_, v_currMacroScope_51_);
v___x_58_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___closed__5));
v___x_59_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_59_, 0, v___x_54_);
lean_ctor_set(v___x_59_, 1, v___x_55_);
lean_ctor_set(v___x_59_, 2, v___x_57_);
lean_ctor_set(v___x_59_, 3, v___x_58_);
v___x_60_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v_a_45_);
return v___x_60_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1___boxed(lean_object* v_x_61_, lean_object* v_a_62_, lean_object* v_a_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_U0001d4df__1(v_x_61_, v_a_62_, v_a_63_);
lean_dec_ref(v_a_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v___x_71_; uint8_t v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___closed__1));
lean_inc(v_x_68_);
v___x_72_ = l_Lean_Syntax_isOfKind(v_x_68_, v___x_71_);
if (v___x_72_ == 0)
{
lean_object* v___x_73_; lean_object* v___x_74_; 
lean_dec(v_x_68_);
v___x_73_ = lean_box(0);
v___x_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_a_70_);
return v___x_74_;
}
else
{
lean_object* v_ref_75_; uint8_t v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v_ref_75_ = l_Lean_replaceRef(v_x_68_, v_a_69_);
lean_dec(v_x_68_);
v___x_76_ = 0;
v___x_77_ = l_Lean_SourceInfo_fromRef(v_ref_75_, v___x_76_);
lean_dec(v_ref_75_);
v___x_78_ = ((lean_object*)(lp_mathlib_Filter_term_U0001d4df___closed__2));
v___x_79_ = ((lean_object*)(lp_mathlib_Filter_term_U0001d4df___closed__3));
lean_inc(v___x_77_);
v___x_80_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_77_);
lean_ctor_set(v___x_80_, 1, v___x_79_);
v___x_81_ = l_Lean_Syntax_node1(v___x_77_, v___x_78_, v___x_80_);
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v_a_70_);
return v___x_82_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1___boxed(lean_object* v_x_83_, lean_object* v_a_84_, lean_object* v_a_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______unexpand__Filter__principal__1(v_x_83_, v_a_84_, v_a_85_);
lean_dec(v_a_84_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instPure___lam__0(lean_object* v_00_u03b1_87_, lean_object* v_x_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_box(0);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instPure___lam__0___boxed(lean_object* v_00_u03b1_90_, lean_object* v_x_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Filter_instPure___lam__0(v_00_u03b1_90_, v_x_91_);
lean_dec(v_x_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_join(lean_object* v_00_u03b1_95_, lean_object* v_f_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_box(0);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instPartialOrder(lean_object* v_00_u03b1_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = ((lean_object*)(lp_mathlib_Filter_instPartialOrder___closed__0));
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSupSet___lam__0(lean_object* v_S_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_box(0);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSupSet(lean_object* v_00_u03b1_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = ((lean_object*)(lp_mathlib_Filter_instSupSet___closed__0));
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_sInf(lean_object* v_00_u03b1_108_, lean_object* v_s_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_box(0);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInfSet(lean_object* v_00_u03b1_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = ((lean_object*)(lp_mathlib_Filter_instInfSet___closed__0));
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instTop(lean_object* v_00_u03b1_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_box(0);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instBot(lean_object* v_00_u03b1_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lean_box(0);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInf___lam__0(lean_object* v_f_118_, lean_object* v_g_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_box(0);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instInf(lean_object* v_00_u03b1_122_){
_start:
{
lean_object* v___f_123_; 
v___f_123_ = ((lean_object*)(lp_mathlib_Filter_instInf___closed__0));
return v___f_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSup(lean_object* v_00_u03b1_124_){
_start:
{
lean_object* v___f_125_; 
v___f_125_ = ((lean_object*)(lp_mathlib_Filter_instInf___closed__0));
return v___f_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSDiff(lean_object* v_00_u03b1_126_){
_start:
{
lean_object* v___f_127_; 
v___f_127_ = ((lean_object*)(lp_mathlib_Filter_instInf___closed__0));
return v___f_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instHNot___lam__0(lean_object* v_f_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_box(0);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instHNot(lean_object* v_00_u03b1_131_){
_start:
{
lean_object* v___f_132_; 
v___f_132_ = ((lean_object*)(lp_mathlib_Filter_instHNot___closed__0));
return v___f_132_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_143_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_144_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__5));
v___x_145_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_146_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_146_, 0, v___x_145_);
lean_ctor_set(v___x_146_, 1, v___x_144_);
lean_ctor_set(v___x_146_, 2, v___x_143_);
return v___x_146_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__9(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_150_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__8));
v___x_151_ = lean_obj_once(&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__6, &lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__6_once, _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__6);
v___x_152_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_153_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v___x_151_);
lean_ctor_set(v___x_153_, 2, v___x_150_);
return v___x_153_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__13(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_160_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12));
v___x_161_ = lean_obj_once(&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__9, &lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__9_once, _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__9);
v___x_162_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_163_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v___x_161_);
lean_ctor_set(v___x_163_, 2, v___x_160_);
return v___x_163_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__16(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_167_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__15));
v___x_168_ = lean_obj_once(&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__13, &lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__13_once, _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__13);
v___x_169_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_170_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v___x_168_);
lean_ctor_set(v___x_170_, 2, v___x_167_);
return v___x_170_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__17(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; 
v___x_171_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12));
v___x_172_ = lean_obj_once(&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__16, &lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__16_once, _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__16);
v___x_173_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_174_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v___x_172_);
lean_ctor_set(v___x_174_, 2, v___x_171_);
return v___x_174_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__18(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_175_ = lean_obj_once(&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__17, &lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__17_once, _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__17);
v___x_176_ = lean_unsigned_to_nat(1022u);
v___x_177_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1));
v___x_178_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_176_);
lean_ctor_set(v___x_178_, 2, v___x_175_);
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c__(void){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_obj_once(&lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__18, &lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__18_once, _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__18);
return v___x_179_;
}
}
static lean_object* _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__6));
v___x_191_ = l_String_toRawSubstring_x27(v___x_190_);
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__16(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__15));
v___x_206_ = l_String_toRawSubstring_x27(v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1(lean_object* v_x_222_, lean_object* v_a_223_, lean_object* v_a_224_){
_start:
{
lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_225_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1));
lean_inc(v_x_222_);
v___x_226_ = l_Lean_Syntax_isOfKind(v_x_222_, v___x_225_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; lean_object* v___x_228_; 
lean_dec(v_x_222_);
v___x_227_ = lean_box(1);
v___x_228_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_228_, 0, v___x_227_);
lean_ctor_set(v___x_228_, 1, v_a_224_);
return v___x_228_;
}
else
{
lean_object* v_quotContext_229_; lean_object* v_currMacroScope_230_; lean_object* v_ref_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; uint8_t v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v_quotContext_229_ = lean_ctor_get(v_a_223_, 1);
v_currMacroScope_230_ = lean_ctor_get(v_a_223_, 2);
v_ref_231_ = lean_ctor_get(v_a_223_, 5);
v___x_232_ = lean_unsigned_to_nat(1u);
v___x_233_ = l_Lean_Syntax_getArg(v_x_222_, v___x_232_);
v___x_234_ = lean_unsigned_to_nat(3u);
v___x_235_ = l_Lean_Syntax_getArg(v_x_222_, v___x_234_);
v___x_236_ = lean_unsigned_to_nat(5u);
v___x_237_ = l_Lean_Syntax_getArg(v_x_222_, v___x_236_);
lean_dec(v_x_222_);
v___x_238_ = 0;
v___x_239_ = l_Lean_SourceInfo_fromRef(v_ref_231_, v___x_238_);
v___x_240_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3));
v___x_241_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__4));
lean_inc_n(v___x_239_, 9);
v___x_242_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_239_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__5));
v___x_244_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_239_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7);
v___x_246_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_230_, 2);
lean_inc_n(v_quotContext_229_, 2);
v___x_247_ = l_Lean_addMacroScope(v_quotContext_229_, v___x_246_, v_currMacroScope_230_);
v___x_248_ = lean_box(0);
v___x_249_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_249_, 0, v___x_239_);
lean_ctor_set(v___x_249_, 1, v___x_245_);
lean_ctor_set(v___x_249_, 2, v___x_247_);
lean_ctor_set(v___x_249_, 3, v___x_248_);
v___x_250_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__9));
v___x_251_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_239_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v___x_252_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_253_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__16, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__16_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__16);
v___x_254_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18));
v___x_255_ = l_Lean_addMacroScope(v_quotContext_229_, v___x_254_, v_currMacroScope_230_);
v___x_256_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__20));
v___x_257_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_257_, 0, v___x_239_);
lean_ctor_set(v___x_257_, 1, v___x_253_);
lean_ctor_set(v___x_257_, 2, v___x_255_);
lean_ctor_set(v___x_257_, 3, v___x_256_);
v___x_258_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
lean_inc_ref(v___x_249_);
v___x_259_ = l_Lean_Syntax_node2(v___x_239_, v___x_258_, v___x_249_, v___x_235_);
v___x_260_ = l_Lean_Syntax_node2(v___x_239_, v___x_252_, v___x_257_, v___x_259_);
v___x_261_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__23));
v___x_262_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_262_, 0, v___x_239_);
lean_ctor_set(v___x_262_, 1, v___x_261_);
v___x_263_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__24));
v___x_264_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_239_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = lean_unsigned_to_nat(9u);
v___x_266_ = lean_mk_empty_array_with_capacity(v___x_265_);
v___x_267_ = lean_array_push(v___x_266_, v___x_242_);
v___x_268_ = lean_array_push(v___x_267_, v___x_244_);
v___x_269_ = lean_array_push(v___x_268_, v___x_249_);
v___x_270_ = lean_array_push(v___x_269_, v___x_251_);
v___x_271_ = lean_array_push(v___x_270_, v___x_260_);
v___x_272_ = lean_array_push(v___x_271_, v___x_262_);
v___x_273_ = lean_array_push(v___x_272_, v___x_233_);
v___x_274_ = lean_array_push(v___x_273_, v___x_264_);
v___x_275_ = lean_array_push(v___x_274_, v___x_237_);
v___x_276_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_276_, 0, v___x_239_);
lean_ctor_set(v___x_276_, 1, v___x_240_);
lean_ctor_set(v___x_276_, 2, v___x_275_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set(v___x_277_, 1, v_a_224_);
return v___x_277_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___boxed(lean_object* v_x_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1(v_x_278_, v_a_279_, v_a_280_);
lean_dec_ref(v_a_279_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg(lean_object* v___y_282_){
_start:
{
lean_object* v_subExpr_284_; lean_object* v_expr_285_; lean_object* v___x_286_; 
v_subExpr_284_ = lean_ctor_get(v___y_282_, 3);
v_expr_285_ = lean_ctor_get(v_subExpr_284_, 0);
lean_inc_ref(v_expr_285_);
v___x_286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_286_, 0, v_expr_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg___boxed(lean_object* v___y_287_, lean_object* v___y_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg(v___y_287_);
lean_dec_ref(v___y_287_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0(lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg(v___y_290_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___boxed(lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0(v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
return v_res_305_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__0(lean_object* v_x_306_){
_start:
{
lean_object* v___x_307_; uint8_t v___x_308_; 
v___x_307_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__18));
v___x_308_ = l_Lean_Expr_isConstOf(v_x_306_, v___x_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__0___boxed(lean_object* v_x_309_){
_start:
{
uint8_t v_res_310_; lean_object* v_r_311_; 
v_res_310_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__0(v_x_309_);
lean_dec_ref(v_x_309_);
v_r_311_ = lean_box(v_res_310_);
return v_r_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__1(lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_320_, 0, v___y_312_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__1___boxed(lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__1(v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2(uint8_t v___x_331_, lean_object* v___x_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_ref_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v_ref_342_ = lean_ctor_get(v___y_339_, 5);
v___x_343_ = l_Lean_SourceInfo_fromRef(v_ref_342_, v___x_331_);
v___x_344_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__1));
v___x_345_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_343_, 3);
v___x_346_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_343_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__7));
v___x_348_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_343_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
v___x_349_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__14));
v___x_350_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_343_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = l_Lean_Syntax_node6(v___x_343_, v___x_344_, v___x_346_, v___x_332_, v___x_348_, v_a_333_, v___x_350_, v_a_334_);
v___x_352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___boxed(lean_object* v___x_353_, lean_object* v___x_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
uint8_t v___x_7736__boxed_364_; lean_object* v_res_365_; 
v___x_7736__boxed_364_ = lean_unbox(v___x_353_);
v_res_365_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2(v___x_7736__boxed_364_, v___x_354_, v_a_355_, v_a_356_, v___y_357_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
lean_dec(v___y_362_);
lean_dec_ref(v___y_361_);
lean_dec(v___y_360_);
lean_dec_ref(v___y_359_);
lean_dec(v___y_358_);
lean_dec_ref(v___y_357_);
return v_res_365_;
}
}
static lean_object* _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8(void){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = l_Array_mkArray0(lean_box(0));
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3(lean_object* v___f_389_, lean_object* v___f_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v___x_398_; lean_object* v_a_399_; lean_object* v___x_401_; uint8_t v_isShared_402_; uint8_t v_isSharedCheck_449_; 
v___x_398_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg(v___y_391_);
v_a_399_ = lean_ctor_get(v___x_398_, 0);
v_isSharedCheck_449_ = !lean_is_exclusive(v___x_398_);
if (v_isSharedCheck_449_ == 0)
{
v___x_401_ = v___x_398_;
v_isShared_402_ = v_isSharedCheck_449_;
goto v_resetjp_400_;
}
else
{
lean_inc(v_a_399_);
lean_dec(v___x_398_);
v___x_401_ = lean_box(0);
v_isShared_402_ = v_isSharedCheck_449_;
goto v_resetjp_400_;
}
v_resetjp_400_:
{
lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___y_406_; lean_object* v___x_438_; lean_object* v___x_439_; 
v___x_403_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__1));
v___x_404_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__3));
v___x_438_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_439_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_403_, v___x_438_, v___y_391_, v___y_393_);
if (lean_obj_tag(v___x_439_) == 0)
{
lean_object* v_a_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v_a_440_ = lean_ctor_get(v___x_439_, 0);
lean_inc(v_a_440_);
lean_dec_ref_known(v___x_439_, 1);
v___x_441_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8));
v___x_442_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_442_, 0, v___f_389_);
v___x_443_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_443_, 0, v___x_442_);
lean_closure_set(v___x_443_, 1, v___f_390_);
v___x_444_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__11));
v___x_445_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_445_, 0, v___x_443_);
lean_closure_set(v___x_445_, 1, v___x_444_);
v___x_446_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__12));
v___x_447_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_447_, 0, v___x_445_);
lean_closure_set(v___x_447_, 1, v___x_446_);
v___x_448_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_403_, v___x_441_, v___x_447_, v_a_440_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
v___y_406_ = v___x_448_;
goto v___jp_405_;
}
else
{
lean_dec_ref(v___f_390_);
lean_dec_ref(v___f_389_);
v___y_406_ = v___x_439_;
goto v___jp_405_;
}
v___jp_405_:
{
if (lean_obj_tag(v___y_406_) == 0)
{
lean_object* v_a_407_; lean_object* v_ref_408_; lean_object* v___x_410_; 
v_a_407_ = lean_ctor_get(v___y_406_, 0);
lean_inc(v_a_407_);
lean_dec_ref_known(v___y_406_, 1);
v_ref_408_ = lean_ctor_get(v___y_395_, 5);
if (v_isShared_402_ == 0)
{
lean_ctor_set_tag(v___x_401_, 1);
v___x_410_ = v___x_401_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v_a_399_);
v___x_410_ = v_reuseFailAlloc_429_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
lean_object* v___x_411_; 
lean_inc_ref(v___x_410_);
v___x_411_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_407_, v___x_404_, v___x_410_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_411_) == 0)
{
lean_object* v_a_412_; lean_object* v___x_413_; 
v_a_412_ = lean_ctor_get(v___x_411_, 0);
lean_inc(v_a_412_);
lean_dec_ref_known(v___x_411_, 1);
v___x_413_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_407_, v___x_403_, v___x_410_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_413_) == 0)
{
lean_object* v_a_414_; uint8_t v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___f_427_; lean_object* v___x_428_; 
v_a_414_ = lean_ctor_get(v___x_413_, 0);
lean_inc(v_a_414_);
lean_dec_ref_known(v___x_413_, 1);
v___x_415_ = 0;
v___x_416_ = l_Lean_SourceInfo_fromRef(v_ref_408_, v___x_415_);
v___x_417_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7));
v___x_418_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_419_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8);
v___x_420_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_407_);
lean_dec(v_a_407_);
v___x_421_ = l_Array_append___redArg(v___x_419_, v___x_420_);
lean_dec_ref(v___x_420_);
lean_inc_n(v___x_416_, 2);
v___x_422_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_422_, 0, v___x_416_);
lean_ctor_set(v___x_422_, 1, v___x_418_);
lean_ctor_set(v___x_422_, 2, v___x_421_);
v___x_423_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10));
v___x_424_ = l_Lean_Syntax_node1(v___x_416_, v___x_423_, v___x_422_);
v___x_425_ = l_Lean_Syntax_node1(v___x_416_, v___x_417_, v___x_424_);
v___x_426_ = lean_box(v___x_415_);
v___f_427_ = lean_alloc_closure((void*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__2___boxed), 11, 4);
lean_closure_set(v___f_427_, 0, v___x_426_);
lean_closure_set(v___f_427_, 1, v___x_425_);
lean_closure_set(v___f_427_, 2, v_a_412_);
lean_closure_set(v___f_427_, 3, v_a_414_);
v___x_428_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_427_, v___y_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
return v___x_428_;
}
else
{
lean_dec(v_a_412_);
lean_dec(v_a_407_);
return v___x_413_;
}
}
else
{
lean_dec_ref(v___x_410_);
lean_dec(v_a_407_);
return v___x_411_;
}
}
}
else
{
lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
lean_del_object(v___x_401_);
lean_dec(v_a_399_);
v_a_430_ = lean_ctor_get(v___y_406_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___y_406_);
if (v_isSharedCheck_437_ == 0)
{
v___x_432_ = v___y_406_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___y_406_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_433_ == 0)
{
v___x_435_ = v___x_432_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_a_430_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___boxed(lean_object* v___f_450_, lean_object* v___f_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3(v___f_450_, v___f_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_);
lean_dec(v___y_457_);
lean_dec_ref(v___y_456_);
lean_dec(v___y_455_);
lean_dec_ref(v___y_454_);
lean_dec(v___y_453_);
lean_dec_ref(v___y_452_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1(lean_object* v_a_473_, lean_object* v_a_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_, lean_object* v_a_478_){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_480_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__3));
v___x_481_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__6));
v___x_482_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_480_, v___x_481_, v_a_473_, v_a_474_, v_a_475_, v_a_476_, v_a_477_, v_a_478_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___boxed(lean_object* v_a_483_, lean_object* v_a_484_, lean_object* v_a_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1(v_a_483_, v_a_484_, v_a_485_, v_a_486_, v_a_487_, v_a_488_);
lean_dec(v_a_488_);
lean_dec_ref(v_a_487_);
lean_dec(v_a_486_);
lean_dec_ref(v_a_485_);
lean_dec(v_a_484_);
lean_dec_ref(v_a_483_);
return v_res_490_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_498_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_499_ = ((lean_object*)(lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__3));
v___x_500_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_501_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_501_, 0, v___x_500_);
lean_ctor_set(v___x_501_, 1, v___x_499_);
lean_ctor_set(v___x_501_, 2, v___x_498_);
return v___x_501_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__5(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
v___x_502_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__8));
v___x_503_ = lean_obj_once(&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__4, &lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__4_once, _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__4);
v___x_504_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_505_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_505_, 0, v___x_504_);
lean_ctor_set(v___x_505_, 1, v___x_503_);
lean_ctor_set(v___x_505_, 2, v___x_502_);
return v___x_505_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
v___x_506_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12));
v___x_507_ = lean_obj_once(&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__5, &lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__5_once, _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__5);
v___x_508_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_509_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
lean_ctor_set(v___x_509_, 1, v___x_507_);
lean_ctor_set(v___x_509_, 2, v___x_506_);
return v___x_509_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_510_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__15));
v___x_511_ = lean_obj_once(&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__6, &lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__6_once, _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__6);
v___x_512_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_513_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v___x_511_);
lean_ctor_set(v___x_513_, 2, v___x_510_);
return v___x_513_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__8(void){
_start:
{
lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_514_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__12));
v___x_515_ = lean_obj_once(&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__7, &lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__7_once, _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__7);
v___x_516_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__3));
v___x_517_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_517_, 0, v___x_516_);
lean_ctor_set(v___x_517_, 1, v___x_515_);
lean_ctor_set(v___x_517_, 2, v___x_514_);
return v___x_517_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__9(void){
_start:
{
lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_518_ = lean_obj_once(&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__8, &lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__8_once, _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__8);
v___x_519_ = lean_unsigned_to_nat(1022u);
v___x_520_ = ((lean_object*)(lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1));
v___x_521_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_521_, 0, v___x_520_);
lean_ctor_set(v___x_521_, 1, v___x_519_);
lean_ctor_set(v___x_521_, 2, v___x_518_);
return v___x_521_;
}
}
static lean_object* _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c__(void){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lean_obj_once(&lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__9, &lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__9_once, _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__9);
return v___x_522_;
}
}
static lean_object* _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__1(void){
_start:
{
lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_524_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__0));
v___x_525_ = l_String_toRawSubstring_x27(v___x_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1(lean_object* v_x_536_, lean_object* v_a_537_, lean_object* v_a_538_){
_start:
{
lean_object* v___x_539_; uint8_t v___x_540_; 
v___x_539_ = ((lean_object*)(lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1));
lean_inc(v_x_536_);
v___x_540_ = l_Lean_Syntax_isOfKind(v_x_536_, v___x_539_);
if (v___x_540_ == 0)
{
lean_object* v___x_541_; lean_object* v___x_542_; 
lean_dec(v_x_536_);
v___x_541_ = lean_box(1);
v___x_542_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_542_, 0, v___x_541_);
lean_ctor_set(v___x_542_, 1, v_a_538_);
return v___x_542_;
}
else
{
lean_object* v_quotContext_543_; lean_object* v_currMacroScope_544_; lean_object* v_ref_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; uint8_t v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v_quotContext_543_ = lean_ctor_get(v_a_537_, 1);
v_currMacroScope_544_ = lean_ctor_get(v_a_537_, 2);
v_ref_545_ = lean_ctor_get(v_a_537_, 5);
v___x_546_ = lean_unsigned_to_nat(1u);
v___x_547_ = l_Lean_Syntax_getArg(v_x_536_, v___x_546_);
v___x_548_ = lean_unsigned_to_nat(3u);
v___x_549_ = l_Lean_Syntax_getArg(v_x_536_, v___x_548_);
v___x_550_ = lean_unsigned_to_nat(5u);
v___x_551_ = l_Lean_Syntax_getArg(v_x_536_, v___x_550_);
lean_dec(v_x_536_);
v___x_552_ = 0;
v___x_553_ = l_Lean_SourceInfo_fromRef(v_ref_545_, v___x_552_);
v___x_554_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__3));
v___x_555_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__4));
lean_inc_n(v___x_553_, 9);
v___x_556_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_556_, 0, v___x_553_);
lean_ctor_set(v___x_556_, 1, v___x_555_);
v___x_557_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__5));
v___x_558_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_553_);
lean_ctor_set(v___x_558_, 1, v___x_557_);
v___x_559_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__7);
v___x_560_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_544_, 2);
lean_inc_n(v_quotContext_543_, 2);
v___x_561_ = l_Lean_addMacroScope(v_quotContext_543_, v___x_560_, v_currMacroScope_544_);
v___x_562_ = lean_box(0);
v___x_563_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_563_, 0, v___x_553_);
lean_ctor_set(v___x_563_, 1, v___x_559_);
lean_ctor_set(v___x_563_, 2, v___x_561_);
lean_ctor_set(v___x_563_, 3, v___x_562_);
v___x_564_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__9));
v___x_565_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_565_, 0, v___x_553_);
lean_ctor_set(v___x_565_, 1, v___x_564_);
v___x_566_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_567_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__1, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__1_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__1);
v___x_568_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3));
v___x_569_ = l_Lean_addMacroScope(v_quotContext_543_, v___x_568_, v_currMacroScope_544_);
v___x_570_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__5));
v___x_571_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_571_, 0, v___x_553_);
lean_ctor_set(v___x_571_, 1, v___x_567_);
lean_ctor_set(v___x_571_, 2, v___x_569_);
lean_ctor_set(v___x_571_, 3, v___x_570_);
v___x_572_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
lean_inc_ref(v___x_563_);
v___x_573_ = l_Lean_Syntax_node2(v___x_553_, v___x_572_, v___x_563_, v___x_549_);
v___x_574_ = l_Lean_Syntax_node2(v___x_553_, v___x_566_, v___x_571_, v___x_573_);
v___x_575_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__23));
v___x_576_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_576_, 0, v___x_553_);
lean_ctor_set(v___x_576_, 1, v___x_575_);
v___x_577_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__24));
v___x_578_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_578_, 0, v___x_553_);
lean_ctor_set(v___x_578_, 1, v___x_577_);
v___x_579_ = lean_unsigned_to_nat(9u);
v___x_580_ = lean_mk_empty_array_with_capacity(v___x_579_);
v___x_581_ = lean_array_push(v___x_580_, v___x_556_);
v___x_582_ = lean_array_push(v___x_581_, v___x_558_);
v___x_583_ = lean_array_push(v___x_582_, v___x_563_);
v___x_584_ = lean_array_push(v___x_583_, v___x_565_);
v___x_585_ = lean_array_push(v___x_584_, v___x_574_);
v___x_586_ = lean_array_push(v___x_585_, v___x_576_);
v___x_587_ = lean_array_push(v___x_586_, v___x_547_);
v___x_588_ = lean_array_push(v___x_587_, v___x_578_);
v___x_589_ = lean_array_push(v___x_588_, v___x_551_);
v___x_590_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_590_, 0, v___x_553_);
lean_ctor_set(v___x_590_, 1, v___x_554_);
lean_ctor_set(v___x_590_, 2, v___x_589_);
v___x_591_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_591_, 0, v___x_590_);
lean_ctor_set(v___x_591_, 1, v_a_538_);
return v___x_591_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___boxed(lean_object* v_x_592_, lean_object* v_a_593_, lean_object* v_a_594_){
_start:
{
lean_object* v_res_595_; 
v_res_595_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1(v_x_592_, v_a_593_, v_a_594_);
lean_dec_ref(v_a_593_);
return v_res_595_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__0(lean_object* v_x_596_){
_start:
{
lean_object* v___x_597_; uint8_t v___x_598_; 
v___x_597_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2203_u1da0__In___x2c____1___closed__3));
v___x_598_ = l_Lean_Expr_isConstOf(v_x_596_, v___x_597_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__0___boxed(lean_object* v_x_599_){
_start:
{
uint8_t v_res_600_; lean_object* v_r_601_; 
v_res_600_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__0(v_x_599_);
lean_dec_ref(v_x_599_);
v_r_601_ = lean_box(v_res_600_);
return v_r_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2(uint8_t v___x_603_, lean_object* v___x_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
lean_object* v_ref_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; 
v_ref_614_ = lean_ctor_get(v___y_611_, 5);
v___x_615_ = l_Lean_SourceInfo_fromRef(v_ref_614_, v___x_603_);
v___x_616_ = ((lean_object*)(lp_mathlib_Filter_term_u2203_u1da0__In___x2c___00__closed__1));
v___x_617_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_615_, 3);
v___x_618_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_618_, 0, v___x_615_);
lean_ctor_set(v___x_618_, 1, v___x_617_);
v___x_619_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__7));
v___x_620_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_615_);
lean_ctor_set(v___x_620_, 1, v___x_619_);
v___x_621_ = ((lean_object*)(lp_mathlib_Filter_term_u2200_u1da0__In___x2c___00__closed__14));
v___x_622_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_615_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = l_Lean_Syntax_node6(v___x_615_, v___x_616_, v___x_618_, v___x_604_, v___x_620_, v_a_605_, v___x_622_, v_a_606_);
v___x_624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_624_, 0, v___x_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___boxed(lean_object* v___x_625_, lean_object* v___x_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_){
_start:
{
uint8_t v___x_7333__boxed_636_; lean_object* v_res_637_; 
v___x_7333__boxed_636_ = lean_unbox(v___x_625_);
v_res_637_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2(v___x_7333__boxed_636_, v___x_626_, v_a_627_, v_a_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_, v___y_633_, v___y_634_);
lean_dec(v___y_634_);
lean_dec_ref(v___y_633_);
lean_dec(v___y_632_);
lean_dec_ref(v___y_631_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__1(lean_object* v___f_638_, lean_object* v___f_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
lean_object* v___x_647_; lean_object* v_a_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_698_; 
v___x_647_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1_spec__0___redArg(v___y_640_);
v_a_648_ = lean_ctor_get(v___x_647_, 0);
v_isSharedCheck_698_ = !lean_is_exclusive(v___x_647_);
if (v_isSharedCheck_698_ == 0)
{
v___x_650_ = v___x_647_;
v_isShared_651_ = v_isSharedCheck_698_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_a_648_);
lean_dec(v___x_647_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_698_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___y_655_; lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_652_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__1));
v___x_653_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__3));
v___x_687_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_688_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_652_, v___x_687_, v___y_640_, v___y_642_);
if (lean_obj_tag(v___x_688_) == 0)
{
lean_object* v_a_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
v_a_689_ = lean_ctor_get(v___x_688_, 0);
lean_inc(v_a_689_);
lean_dec_ref_known(v___x_688_, 1);
v___x_690_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__8));
v___x_691_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_691_, 0, v___f_638_);
v___x_692_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_692_, 0, v___x_691_);
lean_closure_set(v___x_692_, 1, v___f_639_);
v___x_693_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__11));
v___x_694_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_694_, 0, v___x_692_);
lean_closure_set(v___x_694_, 1, v___x_693_);
v___x_695_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__12));
v___x_696_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_696_, 0, v___x_694_);
lean_closure_set(v___x_696_, 1, v___x_695_);
v___x_697_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_652_, v___x_690_, v___x_696_, v_a_689_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
v___y_655_ = v___x_697_;
goto v___jp_654_;
}
else
{
lean_dec_ref(v___f_639_);
lean_dec_ref(v___f_638_);
v___y_655_ = v___x_688_;
goto v___jp_654_;
}
v___jp_654_:
{
if (lean_obj_tag(v___y_655_) == 0)
{
lean_object* v_a_656_; lean_object* v_ref_657_; lean_object* v___x_659_; 
v_a_656_ = lean_ctor_get(v___y_655_, 0);
lean_inc(v_a_656_);
lean_dec_ref_known(v___y_655_, 1);
v_ref_657_ = lean_ctor_get(v___y_644_, 5);
if (v_isShared_651_ == 0)
{
lean_ctor_set_tag(v___x_650_, 1);
v___x_659_ = v___x_650_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_678_; 
v_reuseFailAlloc_678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_678_, 0, v_a_648_);
v___x_659_ = v_reuseFailAlloc_678_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
lean_object* v___x_660_; 
lean_inc_ref(v___x_659_);
v___x_660_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_656_, v___x_653_, v___x_659_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
if (lean_obj_tag(v___x_660_) == 0)
{
lean_object* v_a_661_; lean_object* v___x_662_; 
v_a_661_ = lean_ctor_get(v___x_660_, 0);
lean_inc(v_a_661_);
lean_dec_ref_known(v___x_660_, 1);
v___x_662_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_656_, v___x_652_, v___x_659_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
if (lean_obj_tag(v___x_662_) == 0)
{
lean_object* v_a_663_; uint8_t v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___f_676_; lean_object* v___x_677_; 
v_a_663_ = lean_ctor_get(v___x_662_, 0);
lean_inc(v_a_663_);
lean_dec_ref_known(v___x_662_, 1);
v___x_664_ = 0;
v___x_665_ = l_Lean_SourceInfo_fromRef(v_ref_657_, v___x_664_);
v___x_666_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__7));
v___x_667_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_668_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8);
v___x_669_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_656_);
lean_dec(v_a_656_);
v___x_670_ = l_Array_append___redArg(v___x_668_, v___x_669_);
lean_dec_ref(v___x_669_);
lean_inc_n(v___x_665_, 2);
v___x_671_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_671_, 0, v___x_665_);
lean_ctor_set(v___x_671_, 1, v___x_667_);
lean_ctor_set(v___x_671_, 2, v___x_670_);
v___x_672_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__10));
v___x_673_ = l_Lean_Syntax_node1(v___x_665_, v___x_672_, v___x_671_);
v___x_674_ = l_Lean_Syntax_node1(v___x_665_, v___x_666_, v___x_673_);
v___x_675_ = lean_box(v___x_664_);
v___f_676_ = lean_alloc_closure((void*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__2___boxed), 11, 4);
lean_closure_set(v___f_676_, 0, v___x_675_);
lean_closure_set(v___f_676_, 1, v___x_674_);
lean_closure_set(v___f_676_, 2, v_a_661_);
lean_closure_set(v___f_676_, 3, v_a_663_);
v___x_677_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_676_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_);
return v___x_677_;
}
else
{
lean_dec(v_a_661_);
lean_dec(v_a_656_);
return v___x_662_;
}
}
else
{
lean_dec_ref(v___x_659_);
lean_dec(v_a_656_);
return v___x_660_;
}
}
}
else
{
lean_object* v_a_679_; lean_object* v___x_681_; uint8_t v_isShared_682_; uint8_t v_isSharedCheck_686_; 
lean_del_object(v___x_650_);
lean_dec(v_a_648_);
v_a_679_ = lean_ctor_get(v___y_655_, 0);
v_isSharedCheck_686_ = !lean_is_exclusive(v___y_655_);
if (v_isSharedCheck_686_ == 0)
{
v___x_681_ = v___y_655_;
v_isShared_682_ = v_isSharedCheck_686_;
goto v_resetjp_680_;
}
else
{
lean_inc(v_a_679_);
lean_dec(v___y_655_);
v___x_681_ = lean_box(0);
v_isShared_682_ = v_isSharedCheck_686_;
goto v_resetjp_680_;
}
v_resetjp_680_:
{
lean_object* v___x_684_; 
if (v_isShared_682_ == 0)
{
v___x_684_ = v___x_681_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_685_; 
v_reuseFailAlloc_685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_685_, 0, v_a_679_);
v___x_684_ = v_reuseFailAlloc_685_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
return v___x_684_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__1___boxed(lean_object* v___f_699_, lean_object* v___f_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
lean_object* v_res_708_; 
v_res_708_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___lam__1(v___f_699_, v___f_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_);
lean_dec(v___y_706_);
lean_dec_ref(v___y_705_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
lean_dec(v___y_702_);
lean_dec_ref(v___y_701_);
return v_res_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1(lean_object* v_a_719_, lean_object* v_a_720_, lean_object* v_a_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_){
_start:
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_726_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___closed__3));
v___x_727_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___closed__3));
v___x_728_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_726_, v___x_727_, v_a_719_, v_a_720_, v_a_721_, v_a_722_, v_a_723_, v_a_724_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1___boxed(lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_, lean_object* v_a_734_, lean_object* v_a_735_){
_start:
{
lean_object* v_res_736_; 
v_res_736_ = lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2203_u1da0__In___x2c____1(v_a_729_, v_a_730_, v_a_731_, v_a_732_, v_a_733_, v_a_734_);
lean_dec(v_a_734_);
lean_dec_ref(v_a_733_);
lean_dec(v_a_732_);
lean_dec_ref(v_a_731_);
lean_dec(v_a_730_);
lean_dec_ref(v_a_729_);
return v_res_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg(lean_object* v_e_793_, lean_object* v___y_794_){
_start:
{
uint8_t v___x_796_; 
v___x_796_ = l_Lean_Expr_hasMVar(v_e_793_);
if (v___x_796_ == 0)
{
lean_object* v___x_797_; 
v___x_797_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_797_, 0, v_e_793_);
return v___x_797_;
}
else
{
lean_object* v___x_798_; lean_object* v_mctx_799_; lean_object* v___x_800_; lean_object* v_fst_801_; lean_object* v_snd_802_; lean_object* v___x_803_; lean_object* v_cache_804_; lean_object* v_zetaDeltaFVarIds_805_; lean_object* v_postponed_806_; lean_object* v_diag_807_; lean_object* v___x_809_; uint8_t v_isShared_810_; uint8_t v_isSharedCheck_816_; 
v___x_798_ = lean_st_ref_get(v___y_794_);
v_mctx_799_ = lean_ctor_get(v___x_798_, 0);
lean_inc_ref(v_mctx_799_);
lean_dec(v___x_798_);
v___x_800_ = l_Lean_instantiateMVarsCore(v_mctx_799_, v_e_793_);
v_fst_801_ = lean_ctor_get(v___x_800_, 0);
lean_inc(v_fst_801_);
v_snd_802_ = lean_ctor_get(v___x_800_, 1);
lean_inc(v_snd_802_);
lean_dec_ref(v___x_800_);
v___x_803_ = lean_st_ref_take(v___y_794_);
v_cache_804_ = lean_ctor_get(v___x_803_, 1);
v_zetaDeltaFVarIds_805_ = lean_ctor_get(v___x_803_, 2);
v_postponed_806_ = lean_ctor_get(v___x_803_, 3);
v_diag_807_ = lean_ctor_get(v___x_803_, 4);
v_isSharedCheck_816_ = !lean_is_exclusive(v___x_803_);
if (v_isSharedCheck_816_ == 0)
{
lean_object* v_unused_817_; 
v_unused_817_ = lean_ctor_get(v___x_803_, 0);
lean_dec(v_unused_817_);
v___x_809_ = v___x_803_;
v_isShared_810_ = v_isSharedCheck_816_;
goto v_resetjp_808_;
}
else
{
lean_inc(v_diag_807_);
lean_inc(v_postponed_806_);
lean_inc(v_zetaDeltaFVarIds_805_);
lean_inc(v_cache_804_);
lean_dec(v___x_803_);
v___x_809_ = lean_box(0);
v_isShared_810_ = v_isSharedCheck_816_;
goto v_resetjp_808_;
}
v_resetjp_808_:
{
lean_object* v___x_812_; 
if (v_isShared_810_ == 0)
{
lean_ctor_set(v___x_809_, 0, v_snd_802_);
v___x_812_ = v___x_809_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_815_; 
v_reuseFailAlloc_815_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_815_, 0, v_snd_802_);
lean_ctor_set(v_reuseFailAlloc_815_, 1, v_cache_804_);
lean_ctor_set(v_reuseFailAlloc_815_, 2, v_zetaDeltaFVarIds_805_);
lean_ctor_set(v_reuseFailAlloc_815_, 3, v_postponed_806_);
lean_ctor_set(v_reuseFailAlloc_815_, 4, v_diag_807_);
v___x_812_ = v_reuseFailAlloc_815_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
lean_object* v___x_813_; lean_object* v___x_814_; 
v___x_813_ = lean_st_ref_set(v___y_794_, v___x_812_);
v___x_814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_814_, 0, v_fst_801_);
return v___x_814_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg___boxed(lean_object* v_e_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
lean_object* v_res_821_; 
v_res_821_ = lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg(v_e_818_, v___y_819_);
lean_dec(v___y_819_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0(lean_object* v_e_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_){
_start:
{
lean_object* v___x_830_; 
v___x_830_ = lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg(v_e_822_, v___y_826_);
return v___x_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___boxed(lean_object* v_e_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0(v_e_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_, v___y_837_);
lean_dec(v___y_837_);
lean_dec_ref(v___y_836_);
lean_dec(v___y_835_);
lean_dec_ref(v___y_834_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides___lam__0(lean_object* v_e_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_){
_start:
{
lean_object* v___x_848_; uint8_t v___x_849_; lean_object* v___x_850_; 
v___x_848_ = lean_box(0);
v___x_849_ = 1;
v___x_850_ = l_Lean_Elab_Term_elabTerm(v_e_840_, v___x_848_, v___x_849_, v___x_849_, v___y_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
if (lean_obj_tag(v___x_850_) == 0)
{
lean_object* v_a_851_; lean_object* v___x_852_; 
v_a_851_ = lean_ctor_get(v___x_850_, 0);
lean_inc_n(v_a_851_, 2);
lean_dec_ref_known(v___x_850_, 1);
lean_inc(v___y_846_);
lean_inc_ref(v___y_845_);
lean_inc(v___y_844_);
lean_inc_ref(v___y_843_);
v___x_852_ = lean_infer_type(v_a_851_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
if (lean_obj_tag(v___x_852_) == 0)
{
lean_object* v_a_853_; lean_object* v___x_854_; lean_object* v_a_855_; lean_object* v___x_856_; 
v_a_853_ = lean_ctor_get(v___x_852_, 0);
lean_inc(v_a_853_);
lean_dec_ref_known(v___x_852_, 1);
v___x_854_ = lp_mathlib_Lean_instantiateMVars___at___00Filter_elabEventuallyRelSides_spec__0___redArg(v_a_853_, v___y_844_);
v_a_855_ = lean_ctor_get(v___x_854_, 0);
lean_inc(v_a_855_);
lean_dec_ref(v___x_854_);
v___x_856_ = l_Lean_Meta_whnfR(v_a_855_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
if (lean_obj_tag(v___x_856_) == 0)
{
lean_object* v_a_857_; lean_object* v___x_858_; 
v_a_857_ = lean_ctor_get(v___x_856_, 0);
lean_inc(v_a_857_);
lean_dec_ref_known(v___x_856_, 1);
v___x_858_ = l_Lean_Elab_Term_exprToSyntax(v_a_851_, v___y_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_);
if (lean_obj_tag(v___x_858_) == 0)
{
lean_object* v_a_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_867_; 
v_a_859_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_867_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_867_ == 0)
{
v___x_861_ = v___x_858_;
v_isShared_862_ = v_isSharedCheck_867_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_a_859_);
lean_dec(v___x_858_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_867_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v___x_863_; lean_object* v___x_865_; 
v___x_863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_863_, 0, v_a_857_);
lean_ctor_set(v___x_863_, 1, v_a_859_);
if (v_isShared_862_ == 0)
{
lean_ctor_set(v___x_861_, 0, v___x_863_);
v___x_865_ = v___x_861_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_866_; 
v_reuseFailAlloc_866_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_866_, 0, v___x_863_);
v___x_865_ = v_reuseFailAlloc_866_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
return v___x_865_;
}
}
}
else
{
lean_object* v_a_868_; lean_object* v___x_870_; uint8_t v_isShared_871_; uint8_t v_isSharedCheck_875_; 
lean_dec(v_a_857_);
v_a_868_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_875_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_875_ == 0)
{
v___x_870_ = v___x_858_;
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
else
{
lean_inc(v_a_868_);
lean_dec(v___x_858_);
v___x_870_ = lean_box(0);
v_isShared_871_ = v_isSharedCheck_875_;
goto v_resetjp_869_;
}
v_resetjp_869_:
{
lean_object* v___x_873_; 
if (v_isShared_871_ == 0)
{
v___x_873_ = v___x_870_;
goto v_reusejp_872_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v_a_868_);
v___x_873_ = v_reuseFailAlloc_874_;
goto v_reusejp_872_;
}
v_reusejp_872_:
{
return v___x_873_;
}
}
}
}
else
{
lean_object* v_a_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_883_; 
lean_dec(v_a_851_);
v_a_876_ = lean_ctor_get(v___x_856_, 0);
v_isSharedCheck_883_ = !lean_is_exclusive(v___x_856_);
if (v_isSharedCheck_883_ == 0)
{
v___x_878_ = v___x_856_;
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_a_876_);
lean_dec(v___x_856_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_883_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_881_; 
if (v_isShared_879_ == 0)
{
v___x_881_ = v___x_878_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_a_876_);
v___x_881_ = v_reuseFailAlloc_882_;
goto v_reusejp_880_;
}
v_reusejp_880_:
{
return v___x_881_;
}
}
}
}
else
{
lean_object* v_a_884_; lean_object* v___x_886_; uint8_t v_isShared_887_; uint8_t v_isSharedCheck_891_; 
lean_dec(v_a_851_);
v_a_884_ = lean_ctor_get(v___x_852_, 0);
v_isSharedCheck_891_ = !lean_is_exclusive(v___x_852_);
if (v_isSharedCheck_891_ == 0)
{
v___x_886_ = v___x_852_;
v_isShared_887_ = v_isSharedCheck_891_;
goto v_resetjp_885_;
}
else
{
lean_inc(v_a_884_);
lean_dec(v___x_852_);
v___x_886_ = lean_box(0);
v_isShared_887_ = v_isSharedCheck_891_;
goto v_resetjp_885_;
}
v_resetjp_885_:
{
lean_object* v___x_889_; 
if (v_isShared_887_ == 0)
{
v___x_889_ = v___x_886_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_890_; 
v_reuseFailAlloc_890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_890_, 0, v_a_884_);
v___x_889_ = v_reuseFailAlloc_890_;
goto v_reusejp_888_;
}
v_reusejp_888_:
{
return v___x_889_;
}
}
}
}
else
{
lean_object* v_a_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_899_; 
v_a_892_ = lean_ctor_get(v___x_850_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_850_);
if (v_isSharedCheck_899_ == 0)
{
v___x_894_ = v___x_850_;
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_a_892_);
lean_dec(v___x_850_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_897_; 
if (v_isShared_895_ == 0)
{
v___x_897_ = v___x_894_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v_a_892_);
v___x_897_ = v_reuseFailAlloc_898_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
return v___x_897_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides___lam__0___boxed(lean_object* v_e_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_){
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_mathlib_Filter_elabEventuallyRelSides___lam__0(v_e_900_, v___y_901_, v___y_902_, v___y_903_, v___y_904_, v___y_905_, v___y_906_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
lean_dec(v___y_904_);
lean_dec_ref(v___y_903_);
lean_dec(v___y_902_);
lean_dec_ref(v___y_901_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides(lean_object* v_x_912_, lean_object* v_y_913_, lean_object* v_a_914_, lean_object* v_a_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_a_918_, lean_object* v_a_919_){
_start:
{
lean_object* v___x_921_; 
v___x_921_ = lp_mathlib_Filter_elabEventuallyRelSides___lam__0(v_x_912_, v_a_914_, v_a_915_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_921_) == 0)
{
lean_object* v_a_922_; lean_object* v___x_924_; uint8_t v_isShared_925_; uint8_t v_isSharedCheck_978_; 
v_a_922_ = lean_ctor_get(v___x_921_, 0);
v_isSharedCheck_978_ = !lean_is_exclusive(v___x_921_);
if (v_isSharedCheck_978_ == 0)
{
v___x_924_ = v___x_921_;
v_isShared_925_ = v_isSharedCheck_978_;
goto v_resetjp_923_;
}
else
{
lean_inc(v_a_922_);
lean_dec(v___x_921_);
v___x_924_ = lean_box(0);
v_isShared_925_ = v_isSharedCheck_978_;
goto v_resetjp_923_;
}
v_resetjp_923_:
{
lean_object* v_fst_926_; lean_object* v_snd_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_977_; 
v_fst_926_ = lean_ctor_get(v_a_922_, 0);
v_snd_927_ = lean_ctor_get(v_a_922_, 1);
v_isSharedCheck_977_ = !lean_is_exclusive(v_a_922_);
if (v_isSharedCheck_977_ == 0)
{
v___x_929_ = v_a_922_;
v_isShared_930_ = v_isSharedCheck_977_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_snd_927_);
lean_inc(v_fst_926_);
lean_dec(v_a_922_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_977_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_931_; uint8_t v___x_932_; 
v___x_931_ = l_Lean_Expr_getAppFn(v_fst_926_);
v___x_932_ = l_Lean_Expr_isMVar(v___x_931_);
lean_dec_ref(v___x_931_);
if (v___x_932_ == 0)
{
lean_object* v___x_933_; lean_object* v___x_934_; uint8_t v___x_935_; lean_object* v___x_936_; lean_object* v___x_938_; 
v___x_933_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyRelSides___closed__1));
v___x_934_ = lean_unsigned_to_nat(1u);
v___x_935_ = l_Lean_Expr_isAppOfArity(v_fst_926_, v___x_933_, v___x_934_);
lean_dec(v_fst_926_);
v___x_936_ = lean_box(v___x_935_);
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 1, v___x_936_);
lean_ctor_set(v___x_929_, 0, v_y_913_);
v___x_938_ = v___x_929_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_y_913_);
lean_ctor_set(v_reuseFailAlloc_943_, 1, v___x_936_);
v___x_938_ = v_reuseFailAlloc_943_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
lean_object* v___x_939_; lean_object* v___x_941_; 
v___x_939_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_939_, 0, v_snd_927_);
lean_ctor_set(v___x_939_, 1, v___x_938_);
if (v_isShared_925_ == 0)
{
lean_ctor_set(v___x_924_, 0, v___x_939_);
v___x_941_ = v___x_924_;
goto v_reusejp_940_;
}
else
{
lean_object* v_reuseFailAlloc_942_; 
v_reuseFailAlloc_942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_942_, 0, v___x_939_);
v___x_941_ = v_reuseFailAlloc_942_;
goto v_reusejp_940_;
}
v_reusejp_940_:
{
return v___x_941_;
}
}
}
else
{
lean_object* v___x_944_; 
lean_dec(v_fst_926_);
lean_del_object(v___x_924_);
v___x_944_ = lp_mathlib_Filter_elabEventuallyRelSides___lam__0(v_y_913_, v_a_914_, v_a_915_, v_a_916_, v_a_917_, v_a_918_, v_a_919_);
if (lean_obj_tag(v___x_944_) == 0)
{
lean_object* v_a_945_; lean_object* v___x_947_; uint8_t v_isShared_948_; uint8_t v_isSharedCheck_968_; 
v_a_945_ = lean_ctor_get(v___x_944_, 0);
v_isSharedCheck_968_ = !lean_is_exclusive(v___x_944_);
if (v_isSharedCheck_968_ == 0)
{
v___x_947_ = v___x_944_;
v_isShared_948_ = v_isSharedCheck_968_;
goto v_resetjp_946_;
}
else
{
lean_inc(v_a_945_);
lean_dec(v___x_944_);
v___x_947_ = lean_box(0);
v_isShared_948_ = v_isSharedCheck_968_;
goto v_resetjp_946_;
}
v_resetjp_946_:
{
lean_object* v_fst_949_; lean_object* v_snd_950_; lean_object* v___x_952_; uint8_t v_isShared_953_; uint8_t v_isSharedCheck_967_; 
v_fst_949_ = lean_ctor_get(v_a_945_, 0);
v_snd_950_ = lean_ctor_get(v_a_945_, 1);
v_isSharedCheck_967_ = !lean_is_exclusive(v_a_945_);
if (v_isSharedCheck_967_ == 0)
{
v___x_952_ = v_a_945_;
v_isShared_953_ = v_isSharedCheck_967_;
goto v_resetjp_951_;
}
else
{
lean_inc(v_snd_950_);
lean_inc(v_fst_949_);
lean_dec(v_a_945_);
v___x_952_ = lean_box(0);
v_isShared_953_ = v_isSharedCheck_967_;
goto v_resetjp_951_;
}
v_resetjp_951_:
{
lean_object* v___x_954_; lean_object* v___x_955_; uint8_t v___x_956_; lean_object* v___x_957_; lean_object* v___x_959_; 
v___x_954_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyRelSides___closed__1));
v___x_955_ = lean_unsigned_to_nat(1u);
v___x_956_ = l_Lean_Expr_isAppOfArity(v_fst_949_, v___x_954_, v___x_955_);
lean_dec(v_fst_949_);
v___x_957_ = lean_box(v___x_956_);
if (v_isShared_953_ == 0)
{
lean_ctor_set(v___x_952_, 1, v___x_957_);
lean_ctor_set(v___x_952_, 0, v_snd_950_);
v___x_959_ = v___x_952_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_snd_950_);
lean_ctor_set(v_reuseFailAlloc_966_, 1, v___x_957_);
v___x_959_ = v_reuseFailAlloc_966_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
lean_object* v___x_961_; 
if (v_isShared_930_ == 0)
{
lean_ctor_set(v___x_929_, 1, v___x_959_);
lean_ctor_set(v___x_929_, 0, v_snd_927_);
v___x_961_ = v___x_929_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_965_; 
v_reuseFailAlloc_965_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_965_, 0, v_snd_927_);
lean_ctor_set(v_reuseFailAlloc_965_, 1, v___x_959_);
v___x_961_ = v_reuseFailAlloc_965_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
lean_object* v___x_963_; 
if (v_isShared_948_ == 0)
{
lean_ctor_set(v___x_947_, 0, v___x_961_);
v___x_963_ = v___x_947_;
goto v_reusejp_962_;
}
else
{
lean_object* v_reuseFailAlloc_964_; 
v_reuseFailAlloc_964_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_964_, 0, v___x_961_);
v___x_963_ = v_reuseFailAlloc_964_;
goto v_reusejp_962_;
}
v_reusejp_962_:
{
return v___x_963_;
}
}
}
}
}
}
else
{
lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_976_; 
lean_del_object(v___x_929_);
lean_dec(v_snd_927_);
v_a_969_ = lean_ctor_get(v___x_944_, 0);
v_isSharedCheck_976_ = !lean_is_exclusive(v___x_944_);
if (v_isSharedCheck_976_ == 0)
{
v___x_971_ = v___x_944_;
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_944_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_976_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_974_; 
if (v_isShared_972_ == 0)
{
v___x_974_ = v___x_971_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_975_; 
v_reuseFailAlloc_975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_975_, 0, v_a_969_);
v___x_974_ = v_reuseFailAlloc_975_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
return v___x_974_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_986_; 
lean_dec(v_y_913_);
v_a_979_ = lean_ctor_get(v___x_921_, 0);
v_isSharedCheck_986_ = !lean_is_exclusive(v___x_921_);
if (v_isSharedCheck_986_ == 0)
{
v___x_981_ = v___x_921_;
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_a_979_);
lean_dec(v___x_921_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v___x_984_; 
if (v_isShared_982_ == 0)
{
v___x_984_ = v___x_981_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_985_; 
v_reuseFailAlloc_985_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_985_, 0, v_a_979_);
v___x_984_ = v_reuseFailAlloc_985_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
return v___x_984_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyRelSides___boxed(lean_object* v_x_987_, lean_object* v_y_988_, lean_object* v_a_989_, lean_object* v_a_990_, lean_object* v_a_991_, lean_object* v_a_992_, lean_object* v_a_993_, lean_object* v_a_994_, lean_object* v_a_995_){
_start:
{
lean_object* v_res_996_; 
v_res_996_ = lp_mathlib_Filter_elabEventuallyRelSides(v_x_987_, v_y_988_, v_a_989_, v_a_990_, v_a_991_, v_a_992_, v_a_993_, v_a_994_);
lean_dec(v_a_994_);
lean_dec_ref(v_a_993_);
lean_dec(v_a_992_);
lean_dec_ref(v_a_991_);
lean_dec(v_a_990_);
lean_dec_ref(v_a_989_);
return v_res_996_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; 
v___x_997_ = lean_box(0);
v___x_998_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_999_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_999_, 0, v___x_998_);
lean_ctor_set(v___x_999_, 1, v___x_997_);
return v___x_999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg(){
_start:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_1001_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0);
v___x_1002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1001_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___boxed(lean_object* v___y_1003_){
_start:
{
lean_object* v_res_1004_; 
v_res_1004_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg();
return v_res_1004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0(lean_object* v_00_u03b1_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_){
_start:
{
lean_object* v___x_1013_; 
v___x_1013_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg();
return v___x_1013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___boxed(lean_object* v_00_u03b1_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_){
_start:
{
lean_object* v_res_1022_; 
v_res_1022_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0(v_00_u03b1_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_);
lean_dec(v___y_1020_);
lean_dec_ref(v___y_1019_);
lean_dec(v___y_1018_);
lean_dec_ref(v___y_1017_);
lean_dec(v___y_1016_);
lean_dec_ref(v___y_1015_);
return v_res_1022_;
}
}
static lean_object* _init_lp_mathlib_Filter_elabEventuallyEq___closed__1(void){
_start:
{
lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1024_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyEq___closed__0));
v___x_1025_ = l_String_toRawSubstring_x27(v___x_1024_);
return v___x_1025_;
}
}
static lean_object* _init_lp_mathlib_Filter_elabEventuallyEq___closed__7(void){
_start:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1037_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyEq___closed__6));
v___x_1038_ = l_String_toRawSubstring_x27(v___x_1037_);
return v___x_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyEq(lean_object* v_stx_1049_, lean_object* v_expectedType_x3f_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_, lean_object* v_a_1055_, lean_object* v_a_1056_){
_start:
{
lean_object* v___x_1058_; uint8_t v___x_1059_; 
v___x_1058_ = ((lean_object*)(lp_mathlib_Filter_eventuallyEqStx___closed__1));
lean_inc(v_stx_1049_);
v___x_1059_ = l_Lean_Syntax_isOfKind(v_stx_1049_, v___x_1058_);
if (v___x_1059_ == 0)
{
lean_object* v___x_1060_; 
lean_dec(v_expectedType_x3f_1050_);
lean_dec(v_stx_1049_);
v___x_1060_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg();
return v___x_1060_;
}
else
{
lean_object* v___x_1061_; lean_object* v_x_1062_; lean_object* v___x_1063_; lean_object* v_y_1064_; lean_object* v___x_1065_; 
v___x_1061_ = lean_unsigned_to_nat(0u);
v_x_1062_ = l_Lean_Syntax_getArg(v_stx_1049_, v___x_1061_);
v___x_1063_ = lean_unsigned_to_nat(4u);
v_y_1064_ = l_Lean_Syntax_getArg(v_stx_1049_, v___x_1063_);
v___x_1065_ = lp_mathlib_Filter_elabEventuallyRelSides(v_x_1062_, v_y_1064_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_);
if (lean_obj_tag(v___x_1065_) == 0)
{
lean_object* v_a_1066_; lean_object* v_snd_1067_; lean_object* v_fst_1068_; lean_object* v_fst_1069_; lean_object* v_snd_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; uint8_t v___x_1073_; 
v_a_1066_ = lean_ctor_get(v___x_1065_, 0);
lean_inc(v_a_1066_);
lean_dec_ref_known(v___x_1065_, 1);
v_snd_1067_ = lean_ctor_get(v_a_1066_, 1);
lean_inc(v_snd_1067_);
v_fst_1068_ = lean_ctor_get(v_a_1066_, 0);
lean_inc(v_fst_1068_);
lean_dec(v_a_1066_);
v_fst_1069_ = lean_ctor_get(v_snd_1067_, 0);
lean_inc(v_fst_1069_);
v_snd_1070_ = lean_ctor_get(v_snd_1067_, 1);
lean_inc(v_snd_1070_);
lean_dec(v_snd_1067_);
v___x_1071_ = lean_unsigned_to_nat(2u);
v___x_1072_ = l_Lean_Syntax_getArg(v_stx_1049_, v___x_1071_);
lean_dec(v_stx_1049_);
v___x_1073_ = lean_unbox(v_snd_1070_);
if (v___x_1073_ == 0)
{
lean_object* v_ref_1074_; lean_object* v_quotContext_1075_; lean_object* v_currMacroScope_1076_; uint8_t v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; 
v_ref_1074_ = lean_ctor_get(v_a_1055_, 5);
v_quotContext_1075_ = lean_ctor_get(v_a_1055_, 10);
v_currMacroScope_1076_ = lean_ctor_get(v_a_1055_, 11);
v___x_1077_ = lean_unbox(v_snd_1070_);
lean_dec(v_snd_1070_);
v___x_1078_ = l_Lean_SourceInfo_fromRef(v_ref_1074_, v___x_1077_);
v___x_1079_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_1080_ = lean_obj_once(&lp_mathlib_Filter_elabEventuallyEq___closed__1, &lp_mathlib_Filter_elabEventuallyEq___closed__1_once, _init_lp_mathlib_Filter_elabEventuallyEq___closed__1);
v___x_1081_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyEq___closed__3));
lean_inc(v_currMacroScope_1076_);
lean_inc(v_quotContext_1075_);
v___x_1082_ = l_Lean_addMacroScope(v_quotContext_1075_, v___x_1081_, v_currMacroScope_1076_);
v___x_1083_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyEq___closed__5));
lean_inc_n(v___x_1078_, 2);
v___x_1084_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1084_, 0, v___x_1078_);
lean_ctor_set(v___x_1084_, 1, v___x_1080_);
lean_ctor_set(v___x_1084_, 2, v___x_1082_);
lean_ctor_set(v___x_1084_, 3, v___x_1083_);
v___x_1085_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_1086_ = l_Lean_Syntax_node3(v___x_1078_, v___x_1085_, v___x_1072_, v_fst_1068_, v_fst_1069_);
v___x_1087_ = l_Lean_Syntax_node2(v___x_1078_, v___x_1079_, v___x_1084_, v___x_1086_);
v___x_1088_ = l_Lean_Elab_Term_elabTerm(v___x_1087_, v_expectedType_x3f_1050_, v___x_1059_, v___x_1059_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_);
return v___x_1088_;
}
else
{
lean_object* v_ref_1089_; lean_object* v_quotContext_1090_; lean_object* v_currMacroScope_1091_; uint8_t v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; 
lean_dec(v_snd_1070_);
v_ref_1089_ = lean_ctor_get(v_a_1055_, 5);
v_quotContext_1090_ = lean_ctor_get(v_a_1055_, 10);
v_currMacroScope_1091_ = lean_ctor_get(v_a_1055_, 11);
v___x_1092_ = 0;
v___x_1093_ = l_Lean_SourceInfo_fromRef(v_ref_1089_, v___x_1092_);
v___x_1094_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_1095_ = lean_obj_once(&lp_mathlib_Filter_elabEventuallyEq___closed__7, &lp_mathlib_Filter_elabEventuallyEq___closed__7_once, _init_lp_mathlib_Filter_elabEventuallyEq___closed__7);
v___x_1096_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyEq___closed__9));
lean_inc(v_currMacroScope_1091_);
lean_inc(v_quotContext_1090_);
v___x_1097_ = l_Lean_addMacroScope(v_quotContext_1090_, v___x_1096_, v_currMacroScope_1091_);
v___x_1098_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyEq___closed__11));
lean_inc_n(v___x_1093_, 2);
v___x_1099_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1093_);
lean_ctor_set(v___x_1099_, 1, v___x_1095_);
lean_ctor_set(v___x_1099_, 2, v___x_1097_);
lean_ctor_set(v___x_1099_, 3, v___x_1098_);
v___x_1100_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_1101_ = l_Lean_Syntax_node3(v___x_1093_, v___x_1100_, v___x_1072_, v_fst_1068_, v_fst_1069_);
v___x_1102_ = l_Lean_Syntax_node2(v___x_1093_, v___x_1094_, v___x_1099_, v___x_1101_);
v___x_1103_ = l_Lean_Elab_Term_elabTerm(v___x_1102_, v_expectedType_x3f_1050_, v___x_1059_, v___x_1059_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_, v_a_1055_, v_a_1056_);
return v___x_1103_;
}
}
else
{
lean_object* v_a_1104_; lean_object* v___x_1106_; uint8_t v_isShared_1107_; uint8_t v_isSharedCheck_1111_; 
lean_dec(v_expectedType_x3f_1050_);
lean_dec(v_stx_1049_);
v_a_1104_ = lean_ctor_get(v___x_1065_, 0);
v_isSharedCheck_1111_ = !lean_is_exclusive(v___x_1065_);
if (v_isSharedCheck_1111_ == 0)
{
v___x_1106_ = v___x_1065_;
v_isShared_1107_ = v_isSharedCheck_1111_;
goto v_resetjp_1105_;
}
else
{
lean_inc(v_a_1104_);
lean_dec(v___x_1065_);
v___x_1106_ = lean_box(0);
v_isShared_1107_ = v_isSharedCheck_1111_;
goto v_resetjp_1105_;
}
v_resetjp_1105_:
{
lean_object* v___x_1109_; 
if (v_isShared_1107_ == 0)
{
v___x_1109_ = v___x_1106_;
goto v_reusejp_1108_;
}
else
{
lean_object* v_reuseFailAlloc_1110_; 
v_reuseFailAlloc_1110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1110_, 0, v_a_1104_);
v___x_1109_ = v_reuseFailAlloc_1110_;
goto v_reusejp_1108_;
}
v_reusejp_1108_:
{
return v___x_1109_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyEq___boxed(lean_object* v_stx_1112_, lean_object* v_expectedType_x3f_1113_, lean_object* v_a_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_, lean_object* v_a_1117_, lean_object* v_a_1118_, lean_object* v_a_1119_, lean_object* v_a_1120_){
_start:
{
lean_object* v_res_1121_; 
v_res_1121_ = lp_mathlib_Filter_elabEventuallyEq(v_stx_1112_, v_expectedType_x3f_1113_, v_a_1114_, v_a_1115_, v_a_1116_, v_a_1117_, v_a_1118_, v_a_1119_);
lean_dec(v_a_1119_);
lean_dec_ref(v_a_1118_);
lean_dec(v_a_1117_);
lean_dec_ref(v_a_1116_);
lean_dec(v_a_1115_);
lean_dec_ref(v_a_1114_);
return v_res_1121_;
}
}
static lean_object* _init_lp_mathlib_Filter_elabEventuallyLE___closed__1(void){
_start:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; 
v___x_1123_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyLE___closed__0));
v___x_1124_ = l_String_toRawSubstring_x27(v___x_1123_);
return v___x_1124_;
}
}
static lean_object* _init_lp_mathlib_Filter_elabEventuallyLE___closed__7(void){
_start:
{
lean_object* v___x_1136_; lean_object* v___x_1137_; 
v___x_1136_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyLE___closed__6));
v___x_1137_ = l_String_toRawSubstring_x27(v___x_1136_);
return v___x_1137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyLE(lean_object* v_stx_1148_, lean_object* v_expectedType_x3f_1149_, lean_object* v_a_1150_, lean_object* v_a_1151_, lean_object* v_a_1152_, lean_object* v_a_1153_, lean_object* v_a_1154_, lean_object* v_a_1155_){
_start:
{
lean_object* v___x_1157_; uint8_t v___x_1158_; 
v___x_1157_ = ((lean_object*)(lp_mathlib_Filter_eventuallyLEStx___closed__1));
lean_inc(v_stx_1148_);
v___x_1158_ = l_Lean_Syntax_isOfKind(v_stx_1148_, v___x_1157_);
if (v___x_1158_ == 0)
{
lean_object* v___x_1159_; 
lean_dec(v_expectedType_x3f_1149_);
lean_dec(v_stx_1148_);
v___x_1159_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg();
return v___x_1159_;
}
else
{
lean_object* v___x_1160_; lean_object* v_x_1161_; lean_object* v___x_1162_; lean_object* v_y_1163_; lean_object* v___x_1164_; 
v___x_1160_ = lean_unsigned_to_nat(0u);
v_x_1161_ = l_Lean_Syntax_getArg(v_stx_1148_, v___x_1160_);
v___x_1162_ = lean_unsigned_to_nat(4u);
v_y_1163_ = l_Lean_Syntax_getArg(v_stx_1148_, v___x_1162_);
v___x_1164_ = lp_mathlib_Filter_elabEventuallyRelSides(v_x_1161_, v_y_1163_, v_a_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_, v_a_1155_);
if (lean_obj_tag(v___x_1164_) == 0)
{
lean_object* v_a_1165_; lean_object* v_snd_1166_; lean_object* v_fst_1167_; lean_object* v_fst_1168_; lean_object* v_snd_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; uint8_t v___x_1172_; 
v_a_1165_ = lean_ctor_get(v___x_1164_, 0);
lean_inc(v_a_1165_);
lean_dec_ref_known(v___x_1164_, 1);
v_snd_1166_ = lean_ctor_get(v_a_1165_, 1);
lean_inc(v_snd_1166_);
v_fst_1167_ = lean_ctor_get(v_a_1165_, 0);
lean_inc(v_fst_1167_);
lean_dec(v_a_1165_);
v_fst_1168_ = lean_ctor_get(v_snd_1166_, 0);
lean_inc(v_fst_1168_);
v_snd_1169_ = lean_ctor_get(v_snd_1166_, 1);
lean_inc(v_snd_1169_);
lean_dec(v_snd_1166_);
v___x_1170_ = lean_unsigned_to_nat(2u);
v___x_1171_ = l_Lean_Syntax_getArg(v_stx_1148_, v___x_1170_);
lean_dec(v_stx_1148_);
v___x_1172_ = lean_unbox(v_snd_1169_);
if (v___x_1172_ == 0)
{
lean_object* v_ref_1173_; lean_object* v_quotContext_1174_; lean_object* v_currMacroScope_1175_; uint8_t v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; 
v_ref_1173_ = lean_ctor_get(v_a_1154_, 5);
v_quotContext_1174_ = lean_ctor_get(v_a_1154_, 10);
v_currMacroScope_1175_ = lean_ctor_get(v_a_1154_, 11);
v___x_1176_ = lean_unbox(v_snd_1169_);
lean_dec(v_snd_1169_);
v___x_1177_ = l_Lean_SourceInfo_fromRef(v_ref_1173_, v___x_1176_);
v___x_1178_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_1179_ = lean_obj_once(&lp_mathlib_Filter_elabEventuallyLE___closed__1, &lp_mathlib_Filter_elabEventuallyLE___closed__1_once, _init_lp_mathlib_Filter_elabEventuallyLE___closed__1);
v___x_1180_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyLE___closed__3));
lean_inc(v_currMacroScope_1175_);
lean_inc(v_quotContext_1174_);
v___x_1181_ = l_Lean_addMacroScope(v_quotContext_1174_, v___x_1180_, v_currMacroScope_1175_);
v___x_1182_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyLE___closed__5));
lean_inc_n(v___x_1177_, 2);
v___x_1183_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1177_);
lean_ctor_set(v___x_1183_, 1, v___x_1179_);
lean_ctor_set(v___x_1183_, 2, v___x_1181_);
lean_ctor_set(v___x_1183_, 3, v___x_1182_);
v___x_1184_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_1185_ = l_Lean_Syntax_node3(v___x_1177_, v___x_1184_, v___x_1171_, v_fst_1167_, v_fst_1168_);
v___x_1186_ = l_Lean_Syntax_node2(v___x_1177_, v___x_1178_, v___x_1183_, v___x_1185_);
v___x_1187_ = l_Lean_Elab_Term_elabTerm(v___x_1186_, v_expectedType_x3f_1149_, v___x_1158_, v___x_1158_, v_a_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_, v_a_1155_);
return v___x_1187_;
}
else
{
lean_object* v_ref_1188_; lean_object* v_quotContext_1189_; lean_object* v_currMacroScope_1190_; uint8_t v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
lean_dec(v_snd_1169_);
v_ref_1188_ = lean_ctor_get(v_a_1154_, 5);
v_quotContext_1189_ = lean_ctor_get(v_a_1154_, 10);
v_currMacroScope_1190_ = lean_ctor_get(v_a_1154_, 11);
v___x_1191_ = 0;
v___x_1192_ = l_Lean_SourceInfo_fromRef(v_ref_1188_, v___x_1191_);
v___x_1193_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_1194_ = lean_obj_once(&lp_mathlib_Filter_elabEventuallyLE___closed__7, &lp_mathlib_Filter_elabEventuallyLE___closed__7_once, _init_lp_mathlib_Filter_elabEventuallyLE___closed__7);
v___x_1195_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyLE___closed__9));
lean_inc(v_currMacroScope_1190_);
lean_inc(v_quotContext_1189_);
v___x_1196_ = l_Lean_addMacroScope(v_quotContext_1189_, v___x_1195_, v_currMacroScope_1190_);
v___x_1197_ = ((lean_object*)(lp_mathlib_Filter_elabEventuallyLE___closed__11));
lean_inc_n(v___x_1192_, 2);
v___x_1198_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1198_, 0, v___x_1192_);
lean_ctor_set(v___x_1198_, 1, v___x_1194_);
lean_ctor_set(v___x_1198_, 2, v___x_1196_);
lean_ctor_set(v___x_1198_, 3, v___x_1197_);
v___x_1199_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_1200_ = l_Lean_Syntax_node3(v___x_1192_, v___x_1199_, v___x_1171_, v_fst_1167_, v_fst_1168_);
v___x_1201_ = l_Lean_Syntax_node2(v___x_1192_, v___x_1193_, v___x_1198_, v___x_1200_);
v___x_1202_ = l_Lean_Elab_Term_elabTerm(v___x_1201_, v_expectedType_x3f_1149_, v___x_1158_, v___x_1158_, v_a_1150_, v_a_1151_, v_a_1152_, v_a_1153_, v_a_1154_, v_a_1155_);
return v___x_1202_;
}
}
else
{
lean_object* v_a_1203_; lean_object* v___x_1205_; uint8_t v_isShared_1206_; uint8_t v_isSharedCheck_1210_; 
lean_dec(v_expectedType_x3f_1149_);
lean_dec(v_stx_1148_);
v_a_1203_ = lean_ctor_get(v___x_1164_, 0);
v_isSharedCheck_1210_ = !lean_is_exclusive(v___x_1164_);
if (v_isSharedCheck_1210_ == 0)
{
v___x_1205_ = v___x_1164_;
v_isShared_1206_ = v_isSharedCheck_1210_;
goto v_resetjp_1204_;
}
else
{
lean_inc(v_a_1203_);
lean_dec(v___x_1164_);
v___x_1205_ = lean_box(0);
v_isShared_1206_ = v_isSharedCheck_1210_;
goto v_resetjp_1204_;
}
v_resetjp_1204_:
{
lean_object* v___x_1208_; 
if (v_isShared_1206_ == 0)
{
v___x_1208_ = v___x_1205_;
goto v_reusejp_1207_;
}
else
{
lean_object* v_reuseFailAlloc_1209_; 
v_reuseFailAlloc_1209_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1209_, 0, v_a_1203_);
v___x_1208_ = v_reuseFailAlloc_1209_;
goto v_reusejp_1207_;
}
v_reusejp_1207_:
{
return v___x_1208_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_elabEventuallyLE___boxed(lean_object* v_stx_1211_, lean_object* v_expectedType_x3f_1212_, lean_object* v_a_1213_, lean_object* v_a_1214_, lean_object* v_a_1215_, lean_object* v_a_1216_, lean_object* v_a_1217_, lean_object* v_a_1218_, lean_object* v_a_1219_){
_start:
{
lean_object* v_res_1220_; 
v_res_1220_ = lp_mathlib_Filter_elabEventuallyLE(v_stx_1211_, v_expectedType_x3f_1212_, v_a_1213_, v_a_1214_, v_a_1215_, v_a_1216_, v_a_1217_, v_a_1218_);
lean_dec(v_a_1218_);
lean_dec_ref(v_a_1217_);
lean_dec(v_a_1216_);
lean_dec_ref(v_a_1215_);
lean_dec(v_a_1214_);
lean_dec_ref(v_a_1213_);
return v_res_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg(){
_start:
{
lean_object* v___x_1222_; lean_object* v___x_1223_; 
v___x_1222_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0);
v___x_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1223_, 0, v___x_1222_);
return v___x_1223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg___boxed(lean_object* v___y_1224_){
_start:
{
lean_object* v_res_1225_; 
v_res_1225_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg();
return v_res_1225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0(lean_object* v_00_u03b1_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_, lean_object* v___y_1230_, lean_object* v___y_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_){
_start:
{
lean_object* v___x_1235_; 
v___x_1235_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg();
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___boxed(lean_object* v_00_u03b1_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0(v_00_u03b1_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_, v___y_1241_, v___y_1242_, v___y_1243_);
lean_dec(v___y_1243_);
lean_dec_ref(v___y_1242_);
lean_dec(v___y_1241_);
lean_dec_ref(v___y_1240_);
lean_dec(v___y_1239_);
lean_dec_ref(v___y_1238_);
lean_dec(v___y_1237_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyEq(lean_object* v_x_1246_, lean_object* v_a_1247_, lean_object* v_a_1248_, lean_object* v_a_1249_, lean_object* v_a_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_){
_start:
{
lean_object* v___x_1255_; uint8_t v___x_1256_; 
v___x_1255_ = ((lean_object*)(lp_mathlib_Filter_eventuallyEqStx___closed__1));
lean_inc(v_x_1246_);
v___x_1256_ = l_Lean_Syntax_isOfKind(v_x_1246_, v___x_1255_);
if (v___x_1256_ == 0)
{
lean_object* v___x_1257_; 
lean_dec(v_x_1246_);
v___x_1257_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg();
return v___x_1257_;
}
else
{
lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; 
v___x_1258_ = lean_unsigned_to_nat(0u);
v___x_1259_ = l_Lean_Syntax_getArg(v_x_1246_, v___x_1258_);
v___x_1260_ = l_Lean_Elab_Term_Quotation_precheck(v___x_1259_, v_a_1247_, v_a_1248_, v_a_1249_, v_a_1250_, v_a_1251_, v_a_1252_, v_a_1253_);
if (lean_obj_tag(v___x_1260_) == 0)
{
lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
lean_dec_ref_known(v___x_1260_, 1);
v___x_1261_ = lean_unsigned_to_nat(2u);
v___x_1262_ = l_Lean_Syntax_getArg(v_x_1246_, v___x_1261_);
v___x_1263_ = l_Lean_Elab_Term_Quotation_precheck(v___x_1262_, v_a_1247_, v_a_1248_, v_a_1249_, v_a_1250_, v_a_1251_, v_a_1252_, v_a_1253_);
if (lean_obj_tag(v___x_1263_) == 0)
{
lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; 
lean_dec_ref_known(v___x_1263_, 1);
v___x_1264_ = lean_unsigned_to_nat(4u);
v___x_1265_ = l_Lean_Syntax_getArg(v_x_1246_, v___x_1264_);
lean_dec(v_x_1246_);
v___x_1266_ = l_Lean_Elab_Term_Quotation_precheck(v___x_1265_, v_a_1247_, v_a_1248_, v_a_1249_, v_a_1250_, v_a_1251_, v_a_1252_, v_a_1253_);
return v___x_1266_;
}
else
{
lean_dec(v_x_1246_);
return v___x_1263_;
}
}
else
{
lean_dec(v_x_1246_);
return v___x_1260_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyEq___boxed(lean_object* v_x_1267_, lean_object* v_a_1268_, lean_object* v_a_1269_, lean_object* v_a_1270_, lean_object* v_a_1271_, lean_object* v_a_1272_, lean_object* v_a_1273_, lean_object* v_a_1274_, lean_object* v_a_1275_){
_start:
{
lean_object* v_res_1276_; 
v_res_1276_ = lp_mathlib_Filter_precheckEventuallyEq(v_x_1267_, v_a_1268_, v_a_1269_, v_a_1270_, v_a_1271_, v_a_1272_, v_a_1273_, v_a_1274_);
lean_dec(v_a_1274_);
lean_dec_ref(v_a_1273_);
lean_dec(v_a_1272_);
lean_dec_ref(v_a_1271_);
lean_dec(v_a_1270_);
lean_dec_ref(v_a_1269_);
lean_dec(v_a_1268_);
return v_res_1276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyLE(lean_object* v_x_1277_, lean_object* v_a_1278_, lean_object* v_a_1279_, lean_object* v_a_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_, lean_object* v_a_1283_, lean_object* v_a_1284_){
_start:
{
lean_object* v___x_1286_; uint8_t v___x_1287_; 
v___x_1286_ = ((lean_object*)(lp_mathlib_Filter_eventuallyLEStx___closed__1));
lean_inc(v_x_1277_);
v___x_1287_ = l_Lean_Syntax_isOfKind(v_x_1277_, v___x_1286_);
if (v___x_1287_ == 0)
{
lean_object* v___x_1288_; 
lean_dec(v_x_1277_);
v___x_1288_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_precheckEventuallyEq_spec__0___redArg();
return v___x_1288_;
}
else
{
lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; 
v___x_1289_ = lean_unsigned_to_nat(0u);
v___x_1290_ = l_Lean_Syntax_getArg(v_x_1277_, v___x_1289_);
v___x_1291_ = l_Lean_Elab_Term_Quotation_precheck(v___x_1290_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
if (lean_obj_tag(v___x_1291_) == 0)
{
lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; 
lean_dec_ref_known(v___x_1291_, 1);
v___x_1292_ = lean_unsigned_to_nat(2u);
v___x_1293_ = l_Lean_Syntax_getArg(v_x_1277_, v___x_1292_);
v___x_1294_ = l_Lean_Elab_Term_Quotation_precheck(v___x_1293_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
if (lean_obj_tag(v___x_1294_) == 0)
{
lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; 
lean_dec_ref_known(v___x_1294_, 1);
v___x_1295_ = lean_unsigned_to_nat(4u);
v___x_1296_ = l_Lean_Syntax_getArg(v_x_1277_, v___x_1295_);
lean_dec(v_x_1277_);
v___x_1297_ = l_Lean_Elab_Term_Quotation_precheck(v___x_1296_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_, v_a_1283_, v_a_1284_);
return v___x_1297_;
}
else
{
lean_dec(v_x_1277_);
return v___x_1294_;
}
}
else
{
lean_dec(v_x_1277_);
return v___x_1291_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_precheckEventuallyLE___boxed(lean_object* v_x_1298_, lean_object* v_a_1299_, lean_object* v_a_1300_, lean_object* v_a_1301_, lean_object* v_a_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_, lean_object* v_a_1305_, lean_object* v_a_1306_){
_start:
{
lean_object* v_res_1307_; 
v_res_1307_ = lp_mathlib_Filter_precheckEventuallyLE(v_x_1298_, v_a_1299_, v_a_1300_, v_a_1301_, v_a_1302_, v_a_1303_, v_a_1304_, v_a_1305_);
lean_dec(v_a_1305_);
lean_dec_ref(v_a_1304_);
lean_dec(v_a_1303_);
lean_dec_ref(v_a_1302_);
lean_dec(v_a_1301_);
lean_dec_ref(v_a_1300_);
lean_dec(v_a_1299_);
return v_res_1307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEq(lean_object* v_x_1310_, lean_object* v_a_1311_, lean_object* v_a_1312_){
_start:
{
lean_object* v___x_1313_; uint8_t v___x_1314_; 
v___x_1313_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
lean_inc(v_x_1310_);
v___x_1314_ = l_Lean_Syntax_isOfKind(v_x_1310_, v___x_1313_);
if (v___x_1314_ == 0)
{
lean_object* v___x_1315_; lean_object* v___x_1316_; 
lean_dec(v_x_1310_);
v___x_1315_ = lean_box(0);
v___x_1316_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1316_, 0, v___x_1315_);
lean_ctor_set(v___x_1316_, 1, v_a_1312_);
return v___x_1316_;
}
else
{
lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; uint8_t v___x_1320_; 
v___x_1317_ = lean_unsigned_to_nat(1u);
v___x_1318_ = l_Lean_Syntax_getArg(v_x_1310_, v___x_1317_);
lean_dec(v_x_1310_);
v___x_1319_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_1318_);
v___x_1320_ = l_Lean_Syntax_matchesNull(v___x_1318_, v___x_1319_);
if (v___x_1320_ == 0)
{
lean_object* v___x_1321_; lean_object* v___x_1322_; 
lean_dec(v___x_1318_);
v___x_1321_ = lean_box(0);
v___x_1322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1322_, 0, v___x_1321_);
lean_ctor_set(v___x_1322_, 1, v_a_1312_);
return v___x_1322_;
}
else
{
lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; uint8_t v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; 
v___x_1323_ = lean_unsigned_to_nat(0u);
v___x_1324_ = l_Lean_Syntax_getArg(v___x_1318_, v___x_1323_);
v___x_1325_ = l_Lean_Syntax_getArg(v___x_1318_, v___x_1317_);
v___x_1326_ = lean_unsigned_to_nat(2u);
v___x_1327_ = l_Lean_Syntax_getArg(v___x_1318_, v___x_1326_);
lean_dec(v___x_1318_);
v___x_1328_ = 0;
v___x_1329_ = l_Lean_SourceInfo_fromRef(v_a_1311_, v___x_1328_);
v___x_1330_ = ((lean_object*)(lp_mathlib_Filter_eventuallyEqStx___closed__1));
v___x_1331_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__0));
lean_inc_n(v___x_1329_, 2);
v___x_1332_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1332_, 0, v___x_1329_);
lean_ctor_set(v___x_1332_, 1, v___x_1331_);
v___x_1333_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__1));
v___x_1334_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1334_, 0, v___x_1329_);
lean_ctor_set(v___x_1334_, 1, v___x_1333_);
v___x_1335_ = l_Lean_Syntax_node5(v___x_1329_, v___x_1330_, v___x_1325_, v___x_1332_, v___x_1324_, v___x_1334_, v___x_1327_);
v___x_1336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1336_, 0, v___x_1335_);
lean_ctor_set(v___x_1336_, 1, v_a_1312_);
return v___x_1336_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEq___boxed(lean_object* v_x_1337_, lean_object* v_a_1338_, lean_object* v_a_1339_){
_start:
{
lean_object* v_res_1340_; 
v_res_1340_ = lp_mathlib_Filter_unexpandEventuallyEq(v_x_1337_, v_a_1338_, v_a_1339_);
lean_dec(v_a_1338_);
return v_res_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEqSet(lean_object* v_x_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_){
_start:
{
lean_object* v___x_1344_; uint8_t v___x_1345_; 
v___x_1344_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
lean_inc(v_x_1341_);
v___x_1345_ = l_Lean_Syntax_isOfKind(v_x_1341_, v___x_1344_);
if (v___x_1345_ == 0)
{
lean_object* v___x_1346_; lean_object* v___x_1347_; 
lean_dec(v_x_1341_);
v___x_1346_ = lean_box(0);
v___x_1347_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1347_, 0, v___x_1346_);
lean_ctor_set(v___x_1347_, 1, v_a_1343_);
return v___x_1347_;
}
else
{
lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; uint8_t v___x_1351_; 
v___x_1348_ = lean_unsigned_to_nat(1u);
v___x_1349_ = l_Lean_Syntax_getArg(v_x_1341_, v___x_1348_);
lean_dec(v_x_1341_);
v___x_1350_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_1349_);
v___x_1351_ = l_Lean_Syntax_matchesNull(v___x_1349_, v___x_1350_);
if (v___x_1351_ == 0)
{
lean_object* v___x_1352_; lean_object* v___x_1353_; 
lean_dec(v___x_1349_);
v___x_1352_ = lean_box(0);
v___x_1353_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1352_);
lean_ctor_set(v___x_1353_, 1, v_a_1343_);
return v___x_1353_;
}
else
{
lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; uint8_t v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; 
v___x_1354_ = lean_unsigned_to_nat(0u);
v___x_1355_ = l_Lean_Syntax_getArg(v___x_1349_, v___x_1354_);
v___x_1356_ = l_Lean_Syntax_getArg(v___x_1349_, v___x_1348_);
v___x_1357_ = lean_unsigned_to_nat(2u);
v___x_1358_ = l_Lean_Syntax_getArg(v___x_1349_, v___x_1357_);
lean_dec(v___x_1349_);
v___x_1359_ = 0;
v___x_1360_ = l_Lean_SourceInfo_fromRef(v_a_1342_, v___x_1359_);
v___x_1361_ = ((lean_object*)(lp_mathlib_Filter_eventuallyEqStx___closed__1));
v___x_1362_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__0));
lean_inc_n(v___x_1360_, 2);
v___x_1363_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1363_, 0, v___x_1360_);
lean_ctor_set(v___x_1363_, 1, v___x_1362_);
v___x_1364_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__1));
v___x_1365_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1365_, 0, v___x_1360_);
lean_ctor_set(v___x_1365_, 1, v___x_1364_);
v___x_1366_ = l_Lean_Syntax_node5(v___x_1360_, v___x_1361_, v___x_1356_, v___x_1363_, v___x_1355_, v___x_1365_, v___x_1358_);
v___x_1367_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1367_, 0, v___x_1366_);
lean_ctor_set(v___x_1367_, 1, v_a_1343_);
return v___x_1367_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyEqSet___boxed(lean_object* v_x_1368_, lean_object* v_a_1369_, lean_object* v_a_1370_){
_start:
{
lean_object* v_res_1371_; 
v_res_1371_ = lp_mathlib_Filter_unexpandEventuallyEqSet(v_x_1368_, v_a_1369_, v_a_1370_);
lean_dec(v_a_1369_);
return v_res_1371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyLE(lean_object* v_x_1373_, lean_object* v_a_1374_, lean_object* v_a_1375_){
_start:
{
lean_object* v___x_1376_; uint8_t v___x_1377_; 
v___x_1376_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
lean_inc(v_x_1373_);
v___x_1377_ = l_Lean_Syntax_isOfKind(v_x_1373_, v___x_1376_);
if (v___x_1377_ == 0)
{
lean_object* v___x_1378_; lean_object* v___x_1379_; 
lean_dec(v_x_1373_);
v___x_1378_ = lean_box(0);
v___x_1379_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1379_, 0, v___x_1378_);
lean_ctor_set(v___x_1379_, 1, v_a_1375_);
return v___x_1379_;
}
else
{
lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; uint8_t v___x_1383_; 
v___x_1380_ = lean_unsigned_to_nat(1u);
v___x_1381_ = l_Lean_Syntax_getArg(v_x_1373_, v___x_1380_);
lean_dec(v_x_1373_);
v___x_1382_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_1381_);
v___x_1383_ = l_Lean_Syntax_matchesNull(v___x_1381_, v___x_1382_);
if (v___x_1383_ == 0)
{
lean_object* v___x_1384_; lean_object* v___x_1385_; 
lean_dec(v___x_1381_);
v___x_1384_ = lean_box(0);
v___x_1385_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1385_, 0, v___x_1384_);
lean_ctor_set(v___x_1385_, 1, v_a_1375_);
return v___x_1385_;
}
else
{
lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; uint8_t v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; 
v___x_1386_ = lean_unsigned_to_nat(0u);
v___x_1387_ = l_Lean_Syntax_getArg(v___x_1381_, v___x_1386_);
v___x_1388_ = l_Lean_Syntax_getArg(v___x_1381_, v___x_1380_);
v___x_1389_ = lean_unsigned_to_nat(2u);
v___x_1390_ = l_Lean_Syntax_getArg(v___x_1381_, v___x_1389_);
lean_dec(v___x_1381_);
v___x_1391_ = 0;
v___x_1392_ = l_Lean_SourceInfo_fromRef(v_a_1374_, v___x_1391_);
v___x_1393_ = ((lean_object*)(lp_mathlib_Filter_eventuallyLEStx___closed__1));
v___x_1394_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyLE___closed__0));
lean_inc_n(v___x_1392_, 2);
v___x_1395_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1395_, 0, v___x_1392_);
lean_ctor_set(v___x_1395_, 1, v___x_1394_);
v___x_1396_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__1));
v___x_1397_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1397_, 0, v___x_1392_);
lean_ctor_set(v___x_1397_, 1, v___x_1396_);
v___x_1398_ = l_Lean_Syntax_node5(v___x_1392_, v___x_1393_, v___x_1388_, v___x_1395_, v___x_1387_, v___x_1397_, v___x_1390_);
v___x_1399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1399_, 0, v___x_1398_);
lean_ctor_set(v___x_1399_, 1, v_a_1375_);
return v___x_1399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallyLE___boxed(lean_object* v_x_1400_, lean_object* v_a_1401_, lean_object* v_a_1402_){
_start:
{
lean_object* v_res_1403_; 
v_res_1403_ = lp_mathlib_Filter_unexpandEventuallyLE(v_x_1400_, v_a_1401_, v_a_1402_);
lean_dec(v_a_1401_);
return v_res_1403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallySubset(lean_object* v_x_1404_, lean_object* v_a_1405_, lean_object* v_a_1406_){
_start:
{
lean_object* v___x_1407_; uint8_t v___x_1408_; 
v___x_1407_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
lean_inc(v_x_1404_);
v___x_1408_ = l_Lean_Syntax_isOfKind(v_x_1404_, v___x_1407_);
if (v___x_1408_ == 0)
{
lean_object* v___x_1409_; lean_object* v___x_1410_; 
lean_dec(v_x_1404_);
v___x_1409_ = lean_box(0);
v___x_1410_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1410_, 0, v___x_1409_);
lean_ctor_set(v___x_1410_, 1, v_a_1406_);
return v___x_1410_;
}
else
{
lean_object* v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1413_; uint8_t v___x_1414_; 
v___x_1411_ = lean_unsigned_to_nat(1u);
v___x_1412_ = l_Lean_Syntax_getArg(v_x_1404_, v___x_1411_);
lean_dec(v_x_1404_);
v___x_1413_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_1412_);
v___x_1414_ = l_Lean_Syntax_matchesNull(v___x_1412_, v___x_1413_);
if (v___x_1414_ == 0)
{
lean_object* v___x_1415_; lean_object* v___x_1416_; 
lean_dec(v___x_1412_);
v___x_1415_ = lean_box(0);
v___x_1416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1416_, 0, v___x_1415_);
lean_ctor_set(v___x_1416_, 1, v_a_1406_);
return v___x_1416_;
}
else
{
lean_object* v___x_1417_; lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; uint8_t v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; 
v___x_1417_ = lean_unsigned_to_nat(0u);
v___x_1418_ = l_Lean_Syntax_getArg(v___x_1412_, v___x_1417_);
v___x_1419_ = l_Lean_Syntax_getArg(v___x_1412_, v___x_1411_);
v___x_1420_ = lean_unsigned_to_nat(2u);
v___x_1421_ = l_Lean_Syntax_getArg(v___x_1412_, v___x_1420_);
lean_dec(v___x_1412_);
v___x_1422_ = 0;
v___x_1423_ = l_Lean_SourceInfo_fromRef(v_a_1405_, v___x_1422_);
v___x_1424_ = ((lean_object*)(lp_mathlib_Filter_eventuallyLEStx___closed__1));
v___x_1425_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyLE___closed__0));
lean_inc_n(v___x_1423_, 2);
v___x_1426_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1426_, 0, v___x_1423_);
lean_ctor_set(v___x_1426_, 1, v___x_1425_);
v___x_1427_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__1));
v___x_1428_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1428_, 0, v___x_1423_);
lean_ctor_set(v___x_1428_, 1, v___x_1427_);
v___x_1429_ = l_Lean_Syntax_node5(v___x_1423_, v___x_1424_, v___x_1419_, v___x_1426_, v___x_1418_, v___x_1428_, v___x_1421_);
v___x_1430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1429_);
lean_ctor_set(v___x_1430_, 1, v_a_1406_);
return v___x_1430_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_unexpandEventuallySubset___boxed(lean_object* v_x_1431_, lean_object* v_a_1432_, lean_object* v_a_1433_){
_start:
{
lean_object* v_res_1434_; 
v_res_1434_ = lp_mathlib_Filter_unexpandEventuallySubset(v_x_1431_, v_a_1432_, v_a_1433_);
lean_dec(v_a_1432_);
return v_res_1434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_map(lean_object* v_00_u03b1_1435_, lean_object* v_00_u03b2_1436_, lean_object* v_m_1437_, lean_object* v_f_1438_){
_start:
{
lean_object* v___x_1439_; 
v___x_1439_ = lean_box(0);
return v___x_1439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_map___boxed(lean_object* v_00_u03b1_1440_, lean_object* v_00_u03b2_1441_, lean_object* v_m_1442_, lean_object* v_f_1443_){
_start:
{
lean_object* v_res_1444_; 
v_res_1444_ = lp_mathlib_Filter_map(v_00_u03b1_1440_, v_00_u03b2_1441_, v_m_1442_, v_f_1443_);
lean_dec(v_m_1442_);
return v_res_1444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_comap(lean_object* v_00_u03b1_1445_, lean_object* v_00_u03b2_1446_, lean_object* v_m_1447_, lean_object* v_f_1448_){
_start:
{
lean_object* v___x_1449_; 
v___x_1449_ = lean_box(0);
return v___x_1449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_comap___boxed(lean_object* v_00_u03b1_1450_, lean_object* v_00_u03b2_1451_, lean_object* v_m_1452_, lean_object* v_f_1453_){
_start:
{
lean_object* v_res_1454_; 
v_res_1454_ = lp_mathlib_Filter_comap(v_00_u03b1_1450_, v_00_u03b2_1451_, v_m_1452_, v_f_1453_);
lean_dec(v_m_1452_);
return v_res_1454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_coprod(lean_object* v_00_u03b1_1455_, lean_object* v_00_u03b2_1456_, lean_object* v_f_1457_, lean_object* v_g_1458_){
_start:
{
lean_object* v___x_1459_; 
v___x_1459_ = lean_box(0);
return v___x_1459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSProd___lam__0(lean_object* v_f_1460_, lean_object* v_g_1461_){
_start:
{
lean_object* v___x_1462_; 
v___x_1462_ = lean_box(0);
return v___x_1462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instSProd(lean_object* v_00_u03b1_1464_, lean_object* v_00_u03b2_1465_){
_start:
{
lean_object* v___f_1466_; 
v___f_1466_ = ((lean_object*)(lp_mathlib_Filter_instSProd___closed__0));
return v___f_1466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_pi_spec__0(lean_object* v_00_u03b9_1467_, lean_object* v_00_u03b1_1468_, lean_object* v_00_u03b9_1469_, lean_object* v_s_1470_){
_start:
{
lean_object* v___x_1471_; 
v___x_1471_ = lean_box(0);
return v___x_1471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_pi_spec__0___boxed(lean_object* v_00_u03b9_1472_, lean_object* v_00_u03b1_1473_, lean_object* v_00_u03b9_1474_, lean_object* v_s_1475_){
_start:
{
lean_object* v_res_1476_; 
v_res_1476_ = lp_mathlib_iInf___at___00Filter_pi_spec__0(v_00_u03b9_1472_, v_00_u03b1_1473_, v_00_u03b9_1474_, v_s_1475_);
lean_dec_ref(v_s_1475_);
return v_res_1476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_pi(lean_object* v_00_u03b9_1477_, lean_object* v_00_u03b1_1478_, lean_object* v_f_1479_){
_start:
{
lean_object* v___x_1480_; 
v___x_1480_ = lean_box(0);
return v___x_1480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_pi___boxed(lean_object* v_00_u03b9_1481_, lean_object* v_00_u03b1_1482_, lean_object* v_f_1483_){
_start:
{
lean_object* v_res_1484_; 
v_res_1484_ = lp_mathlib_Filter_pi(v_00_u03b9_1481_, v_00_u03b1_1482_, v_f_1483_);
lean_dec_ref(v_f_1483_);
return v_res_1484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_bind(lean_object* v_00_u03b1_1485_, lean_object* v_00_u03b2_1486_, lean_object* v_f_1487_, lean_object* v_m_1488_){
_start:
{
lean_object* v___x_1489_; 
v___x_1489_ = lean_box(0);
return v___x_1489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_bind___boxed(lean_object* v_00_u03b1_1490_, lean_object* v_00_u03b2_1491_, lean_object* v_f_1492_, lean_object* v_m_1493_){
_start:
{
lean_object* v_res_1494_; 
v_res_1494_ = lp_mathlib_Filter_bind(v_00_u03b1_1490_, v_00_u03b2_1491_, v_f_1492_, v_m_1493_);
lean_dec_ref(v_m_1493_);
return v_res_1494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_seq(lean_object* v_00_u03b1_1495_, lean_object* v_00_u03b2_1496_, lean_object* v_f_1497_, lean_object* v_g_1498_){
_start:
{
lean_object* v___x_1499_; 
v___x_1499_ = lean_box(0);
return v___x_1499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_curry(lean_object* v_00_u03b1_1500_, lean_object* v_00_u03b2_1501_, lean_object* v_f_1502_, lean_object* v_g_1503_){
_start:
{
lean_object* v___x_1504_; 
v___x_1504_ = lean_box(0);
return v___x_1504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instFunctor___lam__0(lean_object* v_00_u03b1_1507_, lean_object* v_00_u03b2_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_){
_start:
{
lean_object* v___x_1511_; 
v___x_1511_ = lean_box(0);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instFunctor___lam__0___boxed(lean_object* v_00_u03b1_1512_, lean_object* v_00_u03b2_1513_, lean_object* v___y_1514_, lean_object* v___y_1515_){
_start:
{
lean_object* v_res_1516_; 
v_res_1516_ = lp_mathlib_Filter_instFunctor___lam__0(v_00_u03b1_1512_, v_00_u03b2_1513_, v___y_1514_, v___y_1515_);
lean_dec(v___y_1514_);
return v_res_1516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__0(lean_object* v_00_u03b2_1523_, lean_object* v_00_u03b9_1524_, lean_object* v_s_1525_){
_start:
{
lean_object* v___x_1526_; 
v___x_1526_ = lean_box(0);
return v___x_1526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__0___boxed(lean_object* v_00_u03b2_1527_, lean_object* v_00_u03b9_1528_, lean_object* v_s_1529_){
_start:
{
lean_object* v_res_1530_; 
v_res_1530_ = lp_mathlib_iInf___at___00Filter_lift_spec__0(v_00_u03b2_1527_, v_00_u03b9_1528_, v_s_1529_);
lean_dec_ref(v_s_1529_);
return v_res_1530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__1(lean_object* v_00_u03b2_1531_, lean_object* v_00_u03b9_1532_, lean_object* v_s_1533_){
_start:
{
lean_object* v___x_1534_; 
v___x_1534_ = lean_box(0);
return v___x_1534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Filter_lift_spec__1___boxed(lean_object* v_00_u03b2_1535_, lean_object* v_00_u03b9_1536_, lean_object* v_s_1537_){
_start:
{
lean_object* v_res_1538_; 
v_res_1538_ = lp_mathlib_iInf___at___00Filter_lift_spec__1(v_00_u03b2_1535_, v_00_u03b9_1536_, v_s_1537_);
lean_dec_ref(v_s_1537_);
return v_res_1538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_lift(lean_object* v_00_u03b1_1539_, lean_object* v_00_u03b2_1540_, lean_object* v_f_1541_, lean_object* v_g_1542_){
_start:
{
lean_object* v___x_1543_; 
v___x_1543_ = lean_box(0);
return v___x_1543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_lift___boxed(lean_object* v_00_u03b1_1544_, lean_object* v_00_u03b2_1545_, lean_object* v_f_1546_, lean_object* v_g_1547_){
_start:
{
lean_object* v_res_1548_; 
v_res_1548_ = lp_mathlib_Filter_lift(v_00_u03b1_1544_, v_00_u03b2_1545_, v_f_1546_, v_g_1547_);
lean_dec_ref(v_g_1547_);
return v_res_1548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_lift_x27(lean_object* v_00_u03b1_1549_, lean_object* v_00_u03b2_1550_, lean_object* v_f_1551_, lean_object* v_h_1552_){
_start:
{
lean_object* v___x_1553_; 
v___x_1553_ = lean_box(0);
return v___x_1553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1653_; lean_object* v___x_1654_; 
v___x_1653_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Filter_elabEventuallyEq_spec__0___redArg___closed__0);
v___x_1654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1654_, 0, v___x_1653_);
return v___x_1654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg___boxed(lean_object* v___y_1655_){
_start:
{
lean_object* v_res_1656_; 
v_res_1656_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v_res_1656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0(lean_object* v_00_u03b1_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_, lean_object* v___y_1664_, lean_object* v___y_1665_){
_start:
{
lean_object* v___x_1667_; 
v___x_1667_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v___x_1667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___boxed(lean_object* v_00_u03b1_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_, lean_object* v___y_1674_, lean_object* v___y_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_){
_start:
{
lean_object* v_res_1678_; 
v_res_1678_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0(v_00_u03b1_1668_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_, v___y_1674_, v___y_1675_, v___y_1676_);
lean_dec(v___y_1676_);
lean_dec_ref(v___y_1675_);
lean_dec(v___y_1674_);
lean_dec_ref(v___y_1673_);
lean_dec(v___y_1672_);
lean_dec_ref(v___y_1671_);
lean_dec(v___y_1670_);
lean_dec_ref(v___y_1669_);
return v_res_1678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___lam__0(lean_object* v_x_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_, lean_object* v___y_1685_, lean_object* v___y_1686_, lean_object* v___y_1687_){
_start:
{
lean_object* v___x_1689_; 
lean_inc(v___y_1683_);
lean_inc_ref(v___y_1682_);
lean_inc(v___y_1681_);
lean_inc_ref(v___y_1680_);
v___x_1689_ = lean_apply_9(v_x_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_, v___y_1684_, v___y_1685_, v___y_1686_, v___y_1687_, lean_box(0));
return v___x_1689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___lam__0___boxed(lean_object* v_x_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_, lean_object* v___y_1696_, lean_object* v___y_1697_, lean_object* v___y_1698_, lean_object* v___y_1699_){
_start:
{
lean_object* v_res_1700_; 
v_res_1700_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___lam__0(v_x_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_, v___y_1695_, v___y_1696_, v___y_1697_, v___y_1698_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
lean_dec(v___y_1692_);
lean_dec_ref(v___y_1691_);
return v_res_1700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg(lean_object* v_mvarId_1701_, lean_object* v_x_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_){
_start:
{
lean_object* v___f_1712_; lean_object* v___x_1713_; 
lean_inc(v___y_1706_);
lean_inc_ref(v___y_1705_);
lean_inc(v___y_1704_);
lean_inc_ref(v___y_1703_);
v___f_1712_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1712_, 0, v_x_1702_);
lean_closure_set(v___f_1712_, 1, v___y_1703_);
lean_closure_set(v___f_1712_, 2, v___y_1704_);
lean_closure_set(v___f_1712_, 3, v___y_1705_);
lean_closure_set(v___f_1712_, 4, v___y_1706_);
v___x_1713_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1701_, v___f_1712_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_);
if (lean_obj_tag(v___x_1713_) == 0)
{
return v___x_1713_;
}
else
{
lean_object* v_a_1714_; lean_object* v___x_1716_; uint8_t v_isShared_1717_; uint8_t v_isSharedCheck_1721_; 
v_a_1714_ = lean_ctor_get(v___x_1713_, 0);
v_isSharedCheck_1721_ = !lean_is_exclusive(v___x_1713_);
if (v_isSharedCheck_1721_ == 0)
{
v___x_1716_ = v___x_1713_;
v_isShared_1717_ = v_isSharedCheck_1721_;
goto v_resetjp_1715_;
}
else
{
lean_inc(v_a_1714_);
lean_dec(v___x_1713_);
v___x_1716_ = lean_box(0);
v_isShared_1717_ = v_isSharedCheck_1721_;
goto v_resetjp_1715_;
}
v_resetjp_1715_:
{
lean_object* v___x_1719_; 
if (v_isShared_1717_ == 0)
{
v___x_1719_ = v___x_1716_;
goto v_reusejp_1718_;
}
else
{
lean_object* v_reuseFailAlloc_1720_; 
v_reuseFailAlloc_1720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1720_, 0, v_a_1714_);
v___x_1719_ = v_reuseFailAlloc_1720_;
goto v_reusejp_1718_;
}
v_reusejp_1718_:
{
return v___x_1719_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg___boxed(lean_object* v_mvarId_1722_, lean_object* v_x_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_){
_start:
{
lean_object* v_res_1733_; 
v_res_1733_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg(v_mvarId_1722_, v_x_1723_, v___y_1724_, v___y_1725_, v___y_1726_, v___y_1727_, v___y_1728_, v___y_1729_, v___y_1730_, v___y_1731_);
lean_dec(v___y_1731_);
lean_dec_ref(v___y_1730_);
lean_dec(v___y_1729_);
lean_dec_ref(v___y_1728_);
lean_dec(v___y_1727_);
lean_dec_ref(v___y_1726_);
lean_dec(v___y_1725_);
lean_dec_ref(v___y_1724_);
return v_res_1733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2(lean_object* v_00_u03b1_1734_, lean_object* v_mvarId_1735_, lean_object* v_x_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_){
_start:
{
lean_object* v___x_1746_; 
v___x_1746_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg(v_mvarId_1735_, v_x_1736_, v___y_1737_, v___y_1738_, v___y_1739_, v___y_1740_, v___y_1741_, v___y_1742_, v___y_1743_, v___y_1744_);
return v___x_1746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___boxed(lean_object* v_00_u03b1_1747_, lean_object* v_mvarId_1748_, lean_object* v_x_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_, lean_object* v___y_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_){
_start:
{
lean_object* v_res_1759_; 
v_res_1759_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2(v_00_u03b1_1747_, v_mvarId_1748_, v_x_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_, v___y_1755_, v___y_1756_, v___y_1757_);
lean_dec(v___y_1757_);
lean_dec_ref(v___y_1756_);
lean_dec(v___y_1755_);
lean_dec_ref(v___y_1754_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec_ref(v___y_1750_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__0(lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_){
_start:
{
lean_object* v_ref_1769_; uint8_t v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; 
v_ref_1769_ = lean_ctor_get(v___y_1766_, 5);
v___x_1770_ = 0;
v___x_1771_ = l_Lean_SourceInfo_fromRef(v_ref_1769_, v___x_1770_);
v___x_1772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1772_, 0, v___x_1771_);
return v___x_1772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__0___boxed(lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_){
_start:
{
lean_object* v_res_1782_; 
v_res_1782_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__0(v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_, v___y_1777_, v___y_1778_, v___y_1779_, v___y_1780_);
lean_dec(v___y_1780_);
lean_dec_ref(v___y_1779_);
lean_dec(v___y_1778_);
lean_dec_ref(v___y_1777_);
lean_dec(v___y_1776_);
lean_dec_ref(v___y_1775_);
lean_dec(v___y_1774_);
lean_dec_ref(v___y_1773_);
return v_res_1782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1(lean_object* v_config_1787_, lean_object* v___y_1788_, lean_object* v___y_1789_, lean_object* v___y_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_){
_start:
{
lean_object* v___x_1797_; 
v___x_1797_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1789_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_);
if (lean_obj_tag(v___x_1797_) == 0)
{
lean_object* v_a_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; 
v_a_1798_ = lean_ctor_get(v___x_1797_, 0);
lean_inc(v_a_1798_);
lean_dec_ref_known(v___x_1797_, 1);
v___x_1799_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___closed__1));
v___x_1800_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v___x_1799_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_);
if (lean_obj_tag(v___x_1800_) == 0)
{
lean_object* v_a_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; 
v_a_1801_ = lean_ctor_get(v___x_1800_, 0);
lean_inc(v_a_1801_);
lean_dec_ref_known(v___x_1800_, 1);
v___x_1802_ = lean_box(0);
v___x_1803_ = l_Lean_MVarId_apply(v_a_1798_, v_a_1801_, v_config_1787_, v___x_1802_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_);
if (lean_obj_tag(v___x_1803_) == 0)
{
lean_object* v_a_1804_; lean_object* v___x_1805_; 
v_a_1804_ = lean_ctor_get(v___x_1803_, 0);
lean_inc(v_a_1804_);
lean_dec_ref_known(v___x_1803_, 1);
v___x_1805_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_1804_, v___y_1789_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_);
if (lean_obj_tag(v___x_1805_) == 0)
{
lean_object* v___x_1807_; uint8_t v_isShared_1808_; uint8_t v_isSharedCheck_1813_; 
v_isSharedCheck_1813_ = !lean_is_exclusive(v___x_1805_);
if (v_isSharedCheck_1813_ == 0)
{
lean_object* v_unused_1814_; 
v_unused_1814_ = lean_ctor_get(v___x_1805_, 0);
lean_dec(v_unused_1814_);
v___x_1807_ = v___x_1805_;
v_isShared_1808_ = v_isSharedCheck_1813_;
goto v_resetjp_1806_;
}
else
{
lean_dec(v___x_1805_);
v___x_1807_ = lean_box(0);
v_isShared_1808_ = v_isSharedCheck_1813_;
goto v_resetjp_1806_;
}
v_resetjp_1806_:
{
lean_object* v___x_1809_; lean_object* v___x_1811_; 
v___x_1809_ = lean_box(0);
if (v_isShared_1808_ == 0)
{
lean_ctor_set(v___x_1807_, 0, v___x_1809_);
v___x_1811_ = v___x_1807_;
goto v_reusejp_1810_;
}
else
{
lean_object* v_reuseFailAlloc_1812_; 
v_reuseFailAlloc_1812_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1812_, 0, v___x_1809_);
v___x_1811_ = v_reuseFailAlloc_1812_;
goto v_reusejp_1810_;
}
v_reusejp_1810_:
{
return v___x_1811_;
}
}
}
else
{
return v___x_1805_;
}
}
else
{
lean_object* v_a_1815_; lean_object* v___x_1817_; uint8_t v_isShared_1818_; uint8_t v_isSharedCheck_1822_; 
v_a_1815_ = lean_ctor_get(v___x_1803_, 0);
v_isSharedCheck_1822_ = !lean_is_exclusive(v___x_1803_);
if (v_isSharedCheck_1822_ == 0)
{
v___x_1817_ = v___x_1803_;
v_isShared_1818_ = v_isSharedCheck_1822_;
goto v_resetjp_1816_;
}
else
{
lean_inc(v_a_1815_);
lean_dec(v___x_1803_);
v___x_1817_ = lean_box(0);
v_isShared_1818_ = v_isSharedCheck_1822_;
goto v_resetjp_1816_;
}
v_resetjp_1816_:
{
lean_object* v___x_1820_; 
if (v_isShared_1818_ == 0)
{
v___x_1820_ = v___x_1817_;
goto v_reusejp_1819_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v_a_1815_);
v___x_1820_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1819_;
}
v_reusejp_1819_:
{
return v___x_1820_;
}
}
}
}
else
{
lean_object* v_a_1823_; lean_object* v___x_1825_; uint8_t v_isShared_1826_; uint8_t v_isSharedCheck_1830_; 
lean_dec(v_a_1798_);
lean_dec_ref(v_config_1787_);
v_a_1823_ = lean_ctor_get(v___x_1800_, 0);
v_isSharedCheck_1830_ = !lean_is_exclusive(v___x_1800_);
if (v_isSharedCheck_1830_ == 0)
{
v___x_1825_ = v___x_1800_;
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
else
{
lean_inc(v_a_1823_);
lean_dec(v___x_1800_);
v___x_1825_ = lean_box(0);
v_isShared_1826_ = v_isSharedCheck_1830_;
goto v_resetjp_1824_;
}
v_resetjp_1824_:
{
lean_object* v___x_1828_; 
if (v_isShared_1826_ == 0)
{
v___x_1828_ = v___x_1825_;
goto v_reusejp_1827_;
}
else
{
lean_object* v_reuseFailAlloc_1829_; 
v_reuseFailAlloc_1829_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1829_, 0, v_a_1823_);
v___x_1828_ = v_reuseFailAlloc_1829_;
goto v_reusejp_1827_;
}
v_reusejp_1827_:
{
return v___x_1828_;
}
}
}
}
else
{
lean_object* v_a_1831_; lean_object* v___x_1833_; uint8_t v_isShared_1834_; uint8_t v_isSharedCheck_1838_; 
lean_dec_ref(v_config_1787_);
v_a_1831_ = lean_ctor_get(v___x_1797_, 0);
v_isSharedCheck_1838_ = !lean_is_exclusive(v___x_1797_);
if (v_isSharedCheck_1838_ == 0)
{
v___x_1833_ = v___x_1797_;
v_isShared_1834_ = v_isSharedCheck_1838_;
goto v_resetjp_1832_;
}
else
{
lean_inc(v_a_1831_);
lean_dec(v___x_1797_);
v___x_1833_ = lean_box(0);
v_isShared_1834_ = v_isSharedCheck_1838_;
goto v_resetjp_1832_;
}
v_resetjp_1832_:
{
lean_object* v___x_1836_; 
if (v_isShared_1834_ == 0)
{
v___x_1836_ = v___x_1833_;
goto v_reusejp_1835_;
}
else
{
lean_object* v_reuseFailAlloc_1837_; 
v_reuseFailAlloc_1837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1837_, 0, v_a_1831_);
v___x_1836_ = v_reuseFailAlloc_1837_;
goto v_reusejp_1835_;
}
v_reusejp_1835_:
{
return v___x_1836_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___boxed(lean_object* v_config_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_, lean_object* v___y_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_){
_start:
{
lean_object* v_res_1849_; 
v_res_1849_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1(v_config_1839_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_, v___y_1844_, v___y_1845_, v___y_1846_, v___y_1847_);
lean_dec(v___y_1847_);
lean_dec_ref(v___y_1846_);
lean_dec(v___y_1845_);
lean_dec_ref(v___y_1844_);
lean_dec(v___y_1843_);
lean_dec_ref(v___y_1842_);
lean_dec(v___y_1841_);
lean_dec_ref(v___y_1840_);
return v_res_1849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8_spec__9___redArg(lean_object* v_x_1850_, lean_object* v_x_1851_, lean_object* v_x_1852_, lean_object* v_x_1853_){
_start:
{
lean_object* v_ks_1854_; lean_object* v_vs_1855_; lean_object* v___x_1857_; uint8_t v_isShared_1858_; uint8_t v_isSharedCheck_1879_; 
v_ks_1854_ = lean_ctor_get(v_x_1850_, 0);
v_vs_1855_ = lean_ctor_get(v_x_1850_, 1);
v_isSharedCheck_1879_ = !lean_is_exclusive(v_x_1850_);
if (v_isSharedCheck_1879_ == 0)
{
v___x_1857_ = v_x_1850_;
v_isShared_1858_ = v_isSharedCheck_1879_;
goto v_resetjp_1856_;
}
else
{
lean_inc(v_vs_1855_);
lean_inc(v_ks_1854_);
lean_dec(v_x_1850_);
v___x_1857_ = lean_box(0);
v_isShared_1858_ = v_isSharedCheck_1879_;
goto v_resetjp_1856_;
}
v_resetjp_1856_:
{
lean_object* v___x_1859_; uint8_t v___x_1860_; 
v___x_1859_ = lean_array_get_size(v_ks_1854_);
v___x_1860_ = lean_nat_dec_lt(v_x_1851_, v___x_1859_);
if (v___x_1860_ == 0)
{
lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1864_; 
lean_dec(v_x_1851_);
v___x_1861_ = lean_array_push(v_ks_1854_, v_x_1852_);
v___x_1862_ = lean_array_push(v_vs_1855_, v_x_1853_);
if (v_isShared_1858_ == 0)
{
lean_ctor_set(v___x_1857_, 1, v___x_1862_);
lean_ctor_set(v___x_1857_, 0, v___x_1861_);
v___x_1864_ = v___x_1857_;
goto v_reusejp_1863_;
}
else
{
lean_object* v_reuseFailAlloc_1865_; 
v_reuseFailAlloc_1865_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1865_, 0, v___x_1861_);
lean_ctor_set(v_reuseFailAlloc_1865_, 1, v___x_1862_);
v___x_1864_ = v_reuseFailAlloc_1865_;
goto v_reusejp_1863_;
}
v_reusejp_1863_:
{
return v___x_1864_;
}
}
else
{
lean_object* v_k_x27_1866_; uint8_t v___x_1867_; 
v_k_x27_1866_ = lean_array_fget_borrowed(v_ks_1854_, v_x_1851_);
v___x_1867_ = l_Lean_instBEqMVarId_beq(v_x_1852_, v_k_x27_1866_);
if (v___x_1867_ == 0)
{
lean_object* v___x_1869_; 
if (v_isShared_1858_ == 0)
{
v___x_1869_ = v___x_1857_;
goto v_reusejp_1868_;
}
else
{
lean_object* v_reuseFailAlloc_1873_; 
v_reuseFailAlloc_1873_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1873_, 0, v_ks_1854_);
lean_ctor_set(v_reuseFailAlloc_1873_, 1, v_vs_1855_);
v___x_1869_ = v_reuseFailAlloc_1873_;
goto v_reusejp_1868_;
}
v_reusejp_1868_:
{
lean_object* v___x_1870_; lean_object* v___x_1871_; 
v___x_1870_ = lean_unsigned_to_nat(1u);
v___x_1871_ = lean_nat_add(v_x_1851_, v___x_1870_);
lean_dec(v_x_1851_);
v_x_1850_ = v___x_1869_;
v_x_1851_ = v___x_1871_;
goto _start;
}
}
else
{
lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1877_; 
v___x_1874_ = lean_array_fset(v_ks_1854_, v_x_1851_, v_x_1852_);
v___x_1875_ = lean_array_fset(v_vs_1855_, v_x_1851_, v_x_1853_);
lean_dec(v_x_1851_);
if (v_isShared_1858_ == 0)
{
lean_ctor_set(v___x_1857_, 1, v___x_1875_);
lean_ctor_set(v___x_1857_, 0, v___x_1874_);
v___x_1877_ = v___x_1857_;
goto v_reusejp_1876_;
}
else
{
lean_object* v_reuseFailAlloc_1878_; 
v_reuseFailAlloc_1878_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1878_, 0, v___x_1874_);
lean_ctor_set(v_reuseFailAlloc_1878_, 1, v___x_1875_);
v___x_1877_ = v_reuseFailAlloc_1878_;
goto v_reusejp_1876_;
}
v_reusejp_1876_:
{
return v___x_1877_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8___redArg(lean_object* v_n_1880_, lean_object* v_k_1881_, lean_object* v_v_1882_){
_start:
{
lean_object* v___x_1883_; lean_object* v___x_1884_; 
v___x_1883_ = lean_unsigned_to_nat(0u);
v___x_1884_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8_spec__9___redArg(v_n_1880_, v___x_1883_, v_k_1881_, v_v_1882_);
return v___x_1884_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_1885_; 
v___x_1885_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(lean_object* v_x_1886_, size_t v_x_1887_, size_t v_x_1888_, lean_object* v_x_1889_, lean_object* v_x_1890_){
_start:
{
if (lean_obj_tag(v_x_1886_) == 0)
{
lean_object* v_es_1891_; size_t v___x_1892_; size_t v___x_1893_; lean_object* v_j_1894_; lean_object* v___x_1895_; uint8_t v___x_1896_; 
v_es_1891_ = lean_ctor_get(v_x_1886_, 0);
v___x_1892_ = ((size_t)31ULL);
v___x_1893_ = lean_usize_land(v_x_1887_, v___x_1892_);
v_j_1894_ = lean_usize_to_nat(v___x_1893_);
v___x_1895_ = lean_array_get_size(v_es_1891_);
v___x_1896_ = lean_nat_dec_lt(v_j_1894_, v___x_1895_);
if (v___x_1896_ == 0)
{
lean_dec(v_j_1894_);
lean_dec(v_x_1890_);
lean_dec(v_x_1889_);
return v_x_1886_;
}
else
{
lean_object* v___x_1898_; uint8_t v_isShared_1899_; uint8_t v_isSharedCheck_1935_; 
lean_inc_ref(v_es_1891_);
v_isSharedCheck_1935_ = !lean_is_exclusive(v_x_1886_);
if (v_isSharedCheck_1935_ == 0)
{
lean_object* v_unused_1936_; 
v_unused_1936_ = lean_ctor_get(v_x_1886_, 0);
lean_dec(v_unused_1936_);
v___x_1898_ = v_x_1886_;
v_isShared_1899_ = v_isSharedCheck_1935_;
goto v_resetjp_1897_;
}
else
{
lean_dec(v_x_1886_);
v___x_1898_ = lean_box(0);
v_isShared_1899_ = v_isSharedCheck_1935_;
goto v_resetjp_1897_;
}
v_resetjp_1897_:
{
lean_object* v_v_1900_; lean_object* v___x_1901_; lean_object* v_xs_x27_1902_; lean_object* v___y_1904_; 
v_v_1900_ = lean_array_fget(v_es_1891_, v_j_1894_);
v___x_1901_ = lean_box(0);
v_xs_x27_1902_ = lean_array_fset(v_es_1891_, v_j_1894_, v___x_1901_);
switch(lean_obj_tag(v_v_1900_))
{
case 0:
{
lean_object* v_key_1909_; lean_object* v_val_1910_; lean_object* v___x_1912_; uint8_t v_isShared_1913_; uint8_t v_isSharedCheck_1920_; 
v_key_1909_ = lean_ctor_get(v_v_1900_, 0);
v_val_1910_ = lean_ctor_get(v_v_1900_, 1);
v_isSharedCheck_1920_ = !lean_is_exclusive(v_v_1900_);
if (v_isSharedCheck_1920_ == 0)
{
v___x_1912_ = v_v_1900_;
v_isShared_1913_ = v_isSharedCheck_1920_;
goto v_resetjp_1911_;
}
else
{
lean_inc(v_val_1910_);
lean_inc(v_key_1909_);
lean_dec(v_v_1900_);
v___x_1912_ = lean_box(0);
v_isShared_1913_ = v_isSharedCheck_1920_;
goto v_resetjp_1911_;
}
v_resetjp_1911_:
{
uint8_t v___x_1914_; 
v___x_1914_ = l_Lean_instBEqMVarId_beq(v_x_1889_, v_key_1909_);
if (v___x_1914_ == 0)
{
lean_object* v___x_1915_; lean_object* v___x_1916_; 
lean_del_object(v___x_1912_);
v___x_1915_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1909_, v_val_1910_, v_x_1889_, v_x_1890_);
v___x_1916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1916_, 0, v___x_1915_);
v___y_1904_ = v___x_1916_;
goto v___jp_1903_;
}
else
{
lean_object* v___x_1918_; 
lean_dec(v_val_1910_);
lean_dec(v_key_1909_);
if (v_isShared_1913_ == 0)
{
lean_ctor_set(v___x_1912_, 1, v_x_1890_);
lean_ctor_set(v___x_1912_, 0, v_x_1889_);
v___x_1918_ = v___x_1912_;
goto v_reusejp_1917_;
}
else
{
lean_object* v_reuseFailAlloc_1919_; 
v_reuseFailAlloc_1919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1919_, 0, v_x_1889_);
lean_ctor_set(v_reuseFailAlloc_1919_, 1, v_x_1890_);
v___x_1918_ = v_reuseFailAlloc_1919_;
goto v_reusejp_1917_;
}
v_reusejp_1917_:
{
v___y_1904_ = v___x_1918_;
goto v___jp_1903_;
}
}
}
}
case 1:
{
lean_object* v_node_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_1933_; 
v_node_1921_ = lean_ctor_get(v_v_1900_, 0);
v_isSharedCheck_1933_ = !lean_is_exclusive(v_v_1900_);
if (v_isSharedCheck_1933_ == 0)
{
v___x_1923_ = v_v_1900_;
v_isShared_1924_ = v_isSharedCheck_1933_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_node_1921_);
lean_dec(v_v_1900_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_1933_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
size_t v___x_1925_; size_t v___x_1926_; size_t v___x_1927_; size_t v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1931_; 
v___x_1925_ = ((size_t)5ULL);
v___x_1926_ = lean_usize_shift_right(v_x_1887_, v___x_1925_);
v___x_1927_ = ((size_t)1ULL);
v___x_1928_ = lean_usize_add(v_x_1888_, v___x_1927_);
v___x_1929_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(v_node_1921_, v___x_1926_, v___x_1928_, v_x_1889_, v_x_1890_);
if (v_isShared_1924_ == 0)
{
lean_ctor_set(v___x_1923_, 0, v___x_1929_);
v___x_1931_ = v___x_1923_;
goto v_reusejp_1930_;
}
else
{
lean_object* v_reuseFailAlloc_1932_; 
v_reuseFailAlloc_1932_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1932_, 0, v___x_1929_);
v___x_1931_ = v_reuseFailAlloc_1932_;
goto v_reusejp_1930_;
}
v_reusejp_1930_:
{
v___y_1904_ = v___x_1931_;
goto v___jp_1903_;
}
}
}
default: 
{
lean_object* v___x_1934_; 
v___x_1934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1934_, 0, v_x_1889_);
lean_ctor_set(v___x_1934_, 1, v_x_1890_);
v___y_1904_ = v___x_1934_;
goto v___jp_1903_;
}
}
v___jp_1903_:
{
lean_object* v___x_1905_; lean_object* v___x_1907_; 
v___x_1905_ = lean_array_fset(v_xs_x27_1902_, v_j_1894_, v___y_1904_);
lean_dec(v_j_1894_);
if (v_isShared_1899_ == 0)
{
lean_ctor_set(v___x_1898_, 0, v___x_1905_);
v___x_1907_ = v___x_1898_;
goto v_reusejp_1906_;
}
else
{
lean_object* v_reuseFailAlloc_1908_; 
v_reuseFailAlloc_1908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1908_, 0, v___x_1905_);
v___x_1907_ = v_reuseFailAlloc_1908_;
goto v_reusejp_1906_;
}
v_reusejp_1906_:
{
return v___x_1907_;
}
}
}
}
}
else
{
lean_object* v_ks_1937_; lean_object* v_vs_1938_; lean_object* v___x_1940_; uint8_t v_isShared_1941_; uint8_t v_isSharedCheck_1958_; 
v_ks_1937_ = lean_ctor_get(v_x_1886_, 0);
v_vs_1938_ = lean_ctor_get(v_x_1886_, 1);
v_isSharedCheck_1958_ = !lean_is_exclusive(v_x_1886_);
if (v_isSharedCheck_1958_ == 0)
{
v___x_1940_ = v_x_1886_;
v_isShared_1941_ = v_isSharedCheck_1958_;
goto v_resetjp_1939_;
}
else
{
lean_inc(v_vs_1938_);
lean_inc(v_ks_1937_);
lean_dec(v_x_1886_);
v___x_1940_ = lean_box(0);
v_isShared_1941_ = v_isSharedCheck_1958_;
goto v_resetjp_1939_;
}
v_resetjp_1939_:
{
lean_object* v___x_1943_; 
if (v_isShared_1941_ == 0)
{
v___x_1943_ = v___x_1940_;
goto v_reusejp_1942_;
}
else
{
lean_object* v_reuseFailAlloc_1957_; 
v_reuseFailAlloc_1957_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1957_, 0, v_ks_1937_);
lean_ctor_set(v_reuseFailAlloc_1957_, 1, v_vs_1938_);
v___x_1943_ = v_reuseFailAlloc_1957_;
goto v_reusejp_1942_;
}
v_reusejp_1942_:
{
lean_object* v_newNode_1944_; uint8_t v___y_1946_; size_t v___x_1952_; uint8_t v___x_1953_; 
v_newNode_1944_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8___redArg(v___x_1943_, v_x_1889_, v_x_1890_);
v___x_1952_ = ((size_t)7ULL);
v___x_1953_ = lean_usize_dec_le(v___x_1952_, v_x_1888_);
if (v___x_1953_ == 0)
{
lean_object* v___x_1954_; lean_object* v___x_1955_; uint8_t v___x_1956_; 
v___x_1954_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1944_);
v___x_1955_ = lean_unsigned_to_nat(4u);
v___x_1956_ = lean_nat_dec_lt(v___x_1954_, v___x_1955_);
lean_dec(v___x_1954_);
v___y_1946_ = v___x_1956_;
goto v___jp_1945_;
}
else
{
v___y_1946_ = v___x_1953_;
goto v___jp_1945_;
}
v___jp_1945_:
{
if (v___y_1946_ == 0)
{
lean_object* v_ks_1947_; lean_object* v_vs_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; 
v_ks_1947_ = lean_ctor_get(v_newNode_1944_, 0);
lean_inc_ref(v_ks_1947_);
v_vs_1948_ = lean_ctor_get(v_newNode_1944_, 1);
lean_inc_ref(v_vs_1948_);
lean_dec_ref(v_newNode_1944_);
v___x_1949_ = lean_unsigned_to_nat(0u);
v___x_1950_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_1951_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg(v_x_1888_, v_ks_1947_, v_vs_1948_, v___x_1949_, v___x_1950_);
lean_dec_ref(v_vs_1948_);
lean_dec_ref(v_ks_1947_);
return v___x_1951_;
}
else
{
return v_newNode_1944_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg(size_t v_depth_1959_, lean_object* v_keys_1960_, lean_object* v_vals_1961_, lean_object* v_i_1962_, lean_object* v_entries_1963_){
_start:
{
lean_object* v___x_1964_; uint8_t v___x_1965_; 
v___x_1964_ = lean_array_get_size(v_keys_1960_);
v___x_1965_ = lean_nat_dec_lt(v_i_1962_, v___x_1964_);
if (v___x_1965_ == 0)
{
lean_dec(v_i_1962_);
return v_entries_1963_;
}
else
{
lean_object* v_k_1966_; lean_object* v_v_1967_; uint64_t v___x_1968_; size_t v_h_1969_; size_t v___x_1970_; lean_object* v___x_1971_; size_t v___x_1972_; size_t v___x_1973_; size_t v___x_1974_; size_t v_h_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; 
v_k_1966_ = lean_array_fget_borrowed(v_keys_1960_, v_i_1962_);
v_v_1967_ = lean_array_fget_borrowed(v_vals_1961_, v_i_1962_);
v___x_1968_ = l_Lean_instHashableMVarId_hash(v_k_1966_);
v_h_1969_ = lean_uint64_to_usize(v___x_1968_);
v___x_1970_ = ((size_t)5ULL);
v___x_1971_ = lean_unsigned_to_nat(1u);
v___x_1972_ = ((size_t)1ULL);
v___x_1973_ = lean_usize_sub(v_depth_1959_, v___x_1972_);
v___x_1974_ = lean_usize_mul(v___x_1970_, v___x_1973_);
v_h_1975_ = lean_usize_shift_right(v_h_1969_, v___x_1974_);
v___x_1976_ = lean_nat_add(v_i_1962_, v___x_1971_);
lean_dec(v_i_1962_);
lean_inc(v_v_1967_);
lean_inc(v_k_1966_);
v___x_1977_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(v_entries_1963_, v_h_1975_, v_depth_1959_, v_k_1966_, v_v_1967_);
v_i_1962_ = v___x_1976_;
v_entries_1963_ = v___x_1977_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg___boxed(lean_object* v_depth_1979_, lean_object* v_keys_1980_, lean_object* v_vals_1981_, lean_object* v_i_1982_, lean_object* v_entries_1983_){
_start:
{
size_t v_depth_boxed_1984_; lean_object* v_res_1985_; 
v_depth_boxed_1984_ = lean_unbox_usize(v_depth_1979_);
lean_dec(v_depth_1979_);
v_res_1985_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg(v_depth_boxed_1984_, v_keys_1980_, v_vals_1981_, v_i_1982_, v_entries_1983_);
lean_dec_ref(v_vals_1981_);
lean_dec_ref(v_keys_1980_);
return v_res_1985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_x_1986_, lean_object* v_x_1987_, lean_object* v_x_1988_, lean_object* v_x_1989_, lean_object* v_x_1990_){
_start:
{
size_t v_x_16794__boxed_1991_; size_t v_x_16795__boxed_1992_; lean_object* v_res_1993_; 
v_x_16794__boxed_1991_ = lean_unbox_usize(v_x_1987_);
lean_dec(v_x_1987_);
v_x_16795__boxed_1992_ = lean_unbox_usize(v_x_1988_);
lean_dec(v_x_1988_);
v_res_1993_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(v_x_1986_, v_x_16794__boxed_1991_, v_x_16795__boxed_1992_, v_x_1989_, v_x_1990_);
return v_res_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1___redArg(lean_object* v_x_1994_, lean_object* v_x_1995_, lean_object* v_x_1996_){
_start:
{
uint64_t v___x_1997_; size_t v___x_1998_; size_t v___x_1999_; lean_object* v___x_2000_; 
v___x_1997_ = l_Lean_instHashableMVarId_hash(v_x_1995_);
v___x_1998_ = lean_uint64_to_usize(v___x_1997_);
v___x_1999_ = ((size_t)1ULL);
v___x_2000_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(v_x_1994_, v___x_1998_, v___x_1999_, v_x_1995_, v_x_1996_);
return v___x_2000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg(lean_object* v_mvarId_2001_, lean_object* v_val_2002_, lean_object* v___y_2003_){
_start:
{
lean_object* v___x_2005_; lean_object* v_mctx_2006_; lean_object* v_cache_2007_; lean_object* v_zetaDeltaFVarIds_2008_; lean_object* v_postponed_2009_; lean_object* v_diag_2010_; lean_object* v___x_2012_; uint8_t v_isShared_2013_; uint8_t v_isSharedCheck_2038_; 
v___x_2005_ = lean_st_ref_take(v___y_2003_);
v_mctx_2006_ = lean_ctor_get(v___x_2005_, 0);
v_cache_2007_ = lean_ctor_get(v___x_2005_, 1);
v_zetaDeltaFVarIds_2008_ = lean_ctor_get(v___x_2005_, 2);
v_postponed_2009_ = lean_ctor_get(v___x_2005_, 3);
v_diag_2010_ = lean_ctor_get(v___x_2005_, 4);
v_isSharedCheck_2038_ = !lean_is_exclusive(v___x_2005_);
if (v_isSharedCheck_2038_ == 0)
{
v___x_2012_ = v___x_2005_;
v_isShared_2013_ = v_isSharedCheck_2038_;
goto v_resetjp_2011_;
}
else
{
lean_inc(v_diag_2010_);
lean_inc(v_postponed_2009_);
lean_inc(v_zetaDeltaFVarIds_2008_);
lean_inc(v_cache_2007_);
lean_inc(v_mctx_2006_);
lean_dec(v___x_2005_);
v___x_2012_ = lean_box(0);
v_isShared_2013_ = v_isSharedCheck_2038_;
goto v_resetjp_2011_;
}
v_resetjp_2011_:
{
lean_object* v_depth_2014_; lean_object* v_levelAssignDepth_2015_; lean_object* v_lmvarCounter_2016_; lean_object* v_mvarCounter_2017_; lean_object* v_lDecls_2018_; lean_object* v_decls_2019_; lean_object* v_userNames_2020_; lean_object* v_lAssignment_2021_; lean_object* v_eAssignment_2022_; lean_object* v_dAssignment_2023_; lean_object* v___x_2025_; uint8_t v_isShared_2026_; uint8_t v_isSharedCheck_2037_; 
v_depth_2014_ = lean_ctor_get(v_mctx_2006_, 0);
v_levelAssignDepth_2015_ = lean_ctor_get(v_mctx_2006_, 1);
v_lmvarCounter_2016_ = lean_ctor_get(v_mctx_2006_, 2);
v_mvarCounter_2017_ = lean_ctor_get(v_mctx_2006_, 3);
v_lDecls_2018_ = lean_ctor_get(v_mctx_2006_, 4);
v_decls_2019_ = lean_ctor_get(v_mctx_2006_, 5);
v_userNames_2020_ = lean_ctor_get(v_mctx_2006_, 6);
v_lAssignment_2021_ = lean_ctor_get(v_mctx_2006_, 7);
v_eAssignment_2022_ = lean_ctor_get(v_mctx_2006_, 8);
v_dAssignment_2023_ = lean_ctor_get(v_mctx_2006_, 9);
v_isSharedCheck_2037_ = !lean_is_exclusive(v_mctx_2006_);
if (v_isSharedCheck_2037_ == 0)
{
v___x_2025_ = v_mctx_2006_;
v_isShared_2026_ = v_isSharedCheck_2037_;
goto v_resetjp_2024_;
}
else
{
lean_inc(v_dAssignment_2023_);
lean_inc(v_eAssignment_2022_);
lean_inc(v_lAssignment_2021_);
lean_inc(v_userNames_2020_);
lean_inc(v_decls_2019_);
lean_inc(v_lDecls_2018_);
lean_inc(v_mvarCounter_2017_);
lean_inc(v_lmvarCounter_2016_);
lean_inc(v_levelAssignDepth_2015_);
lean_inc(v_depth_2014_);
lean_dec(v_mctx_2006_);
v___x_2025_ = lean_box(0);
v_isShared_2026_ = v_isSharedCheck_2037_;
goto v_resetjp_2024_;
}
v_resetjp_2024_:
{
lean_object* v___x_2027_; lean_object* v___x_2029_; 
v___x_2027_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1___redArg(v_eAssignment_2022_, v_mvarId_2001_, v_val_2002_);
if (v_isShared_2026_ == 0)
{
lean_ctor_set(v___x_2025_, 8, v___x_2027_);
v___x_2029_ = v___x_2025_;
goto v_reusejp_2028_;
}
else
{
lean_object* v_reuseFailAlloc_2036_; 
v_reuseFailAlloc_2036_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2036_, 0, v_depth_2014_);
lean_ctor_set(v_reuseFailAlloc_2036_, 1, v_levelAssignDepth_2015_);
lean_ctor_set(v_reuseFailAlloc_2036_, 2, v_lmvarCounter_2016_);
lean_ctor_set(v_reuseFailAlloc_2036_, 3, v_mvarCounter_2017_);
lean_ctor_set(v_reuseFailAlloc_2036_, 4, v_lDecls_2018_);
lean_ctor_set(v_reuseFailAlloc_2036_, 5, v_decls_2019_);
lean_ctor_set(v_reuseFailAlloc_2036_, 6, v_userNames_2020_);
lean_ctor_set(v_reuseFailAlloc_2036_, 7, v_lAssignment_2021_);
lean_ctor_set(v_reuseFailAlloc_2036_, 8, v___x_2027_);
lean_ctor_set(v_reuseFailAlloc_2036_, 9, v_dAssignment_2023_);
v___x_2029_ = v_reuseFailAlloc_2036_;
goto v_reusejp_2028_;
}
v_reusejp_2028_:
{
lean_object* v___x_2031_; 
if (v_isShared_2013_ == 0)
{
lean_ctor_set(v___x_2012_, 0, v___x_2029_);
v___x_2031_ = v___x_2012_;
goto v_reusejp_2030_;
}
else
{
lean_object* v_reuseFailAlloc_2035_; 
v_reuseFailAlloc_2035_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2035_, 0, v___x_2029_);
lean_ctor_set(v_reuseFailAlloc_2035_, 1, v_cache_2007_);
lean_ctor_set(v_reuseFailAlloc_2035_, 2, v_zetaDeltaFVarIds_2008_);
lean_ctor_set(v_reuseFailAlloc_2035_, 3, v_postponed_2009_);
lean_ctor_set(v_reuseFailAlloc_2035_, 4, v_diag_2010_);
v___x_2031_ = v_reuseFailAlloc_2035_;
goto v_reusejp_2030_;
}
v_reusejp_2030_:
{
lean_object* v___x_2032_; lean_object* v___x_2033_; lean_object* v___x_2034_; 
v___x_2032_ = lean_st_ref_set(v___y_2003_, v___x_2031_);
v___x_2033_ = lean_box(0);
v___x_2034_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2034_, 0, v___x_2033_);
return v___x_2034_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg___boxed(lean_object* v_mvarId_2039_, lean_object* v_val_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_){
_start:
{
lean_object* v_res_2043_; 
v_res_2043_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg(v_mvarId_2039_, v_val_2040_, v___y_2041_);
lean_dec(v___y_2041_);
return v_res_2043_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__3(void){
_start:
{
lean_object* v___x_2049_; lean_object* v___x_2050_; 
v___x_2049_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__2));
v___x_2050_ = l_String_toRawSubstring_x27(v___x_2049_);
return v___x_2050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0(lean_object* v___x_2057_, uint8_t v___x_2058_, lean_object* v___x_2059_, lean_object* v_a_2060_, uint8_t v___x_2061_, lean_object* v_a_2062_, uint8_t v___x_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_, lean_object* v___y_2068_, lean_object* v___y_2069_){
_start:
{
lean_object* v___x_2071_; 
v___x_2071_ = l_Lean_Meta_mkFreshExprMVar(v___x_2057_, v___x_2058_, v___x_2059_, v___y_2066_, v___y_2067_, v___y_2068_, v___y_2069_);
if (lean_obj_tag(v___x_2071_) == 0)
{
lean_object* v_a_2072_; lean_object* v___x_2073_; 
v_a_2072_ = lean_ctor_get(v___x_2071_, 0);
lean_inc_n(v_a_2072_, 2);
lean_dec_ref_known(v___x_2071_, 1);
v___x_2073_ = l_Lean_Elab_Term_exprToSyntax(v_a_2072_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_, v___y_2068_, v___y_2069_);
if (lean_obj_tag(v___x_2073_) == 0)
{
lean_object* v_a_2074_; lean_object* v_ref_2075_; lean_object* v_quotContext_2076_; lean_object* v_currMacroScope_2077_; lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; 
v_a_2074_ = lean_ctor_get(v___x_2073_, 0);
lean_inc(v_a_2074_);
lean_dec_ref_known(v___x_2073_, 1);
v_ref_2075_ = lean_ctor_get(v___y_2068_, 5);
v_quotContext_2076_ = lean_ctor_get(v___y_2068_, 10);
v_currMacroScope_2077_ = lean_ctor_get(v___y_2068_, 11);
v___x_2078_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__1));
lean_inc(v_currMacroScope_2077_);
lean_inc(v_quotContext_2076_);
v___x_2079_ = l_Lean_addMacroScope(v_quotContext_2076_, v___x_2078_, v_currMacroScope_2077_);
lean_inc(v_a_2060_);
v___x_2080_ = l_Lean_MVarId_getType(v_a_2060_, v___y_2066_, v___y_2067_, v___y_2068_, v___y_2069_);
if (lean_obj_tag(v___x_2080_) == 0)
{
lean_object* v_a_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; 
v_a_2081_ = lean_ctor_get(v___x_2080_, 0);
lean_inc(v_a_2081_);
lean_dec_ref_known(v___x_2080_, 1);
v___x_2082_ = l_Lean_SourceInfo_fromRef(v_ref_2075_, v___x_2061_);
v___x_2083_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__3);
v___x_2084_ = lean_box(0);
v___x_2085_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___closed__5));
lean_inc_n(v___x_2082_, 2);
v___x_2086_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2086_, 0, v___x_2082_);
lean_ctor_set(v___x_2086_, 1, v___x_2083_);
lean_ctor_set(v___x_2086_, 2, v___x_2079_);
lean_ctor_set(v___x_2086_, 3, v___x_2085_);
v___x_2087_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_2088_ = l_Lean_Syntax_node2(v___x_2082_, v___x_2087_, v_a_2062_, v_a_2074_);
v___x_2089_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__14));
v___x_2090_ = l_Lean_Syntax_node2(v___x_2082_, v___x_2089_, v___x_2086_, v___x_2088_);
v___x_2091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2091_, 0, v_a_2081_);
v___x_2092_ = lean_box(0);
v___x_2093_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_2090_, v___x_2091_, v___x_2063_, v___x_2063_, v___x_2092_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_, v___y_2068_, v___y_2069_);
lean_dec_ref(v___y_2068_);
if (lean_obj_tag(v___x_2093_) == 0)
{
lean_object* v_a_2094_; lean_object* v___x_2095_; 
v_a_2094_ = lean_ctor_get(v___x_2093_, 0);
lean_inc(v_a_2094_);
lean_dec_ref_known(v___x_2093_, 1);
v___x_2095_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg(v_a_2060_, v_a_2094_, v___y_2067_);
if (lean_obj_tag(v___x_2095_) == 0)
{
lean_object* v___x_2097_; uint8_t v_isShared_2098_; uint8_t v_isSharedCheck_2104_; 
v_isSharedCheck_2104_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2104_ == 0)
{
lean_object* v_unused_2105_; 
v_unused_2105_ = lean_ctor_get(v___x_2095_, 0);
lean_dec(v_unused_2105_);
v___x_2097_ = v___x_2095_;
v_isShared_2098_ = v_isSharedCheck_2104_;
goto v_resetjp_2096_;
}
else
{
lean_dec(v___x_2095_);
v___x_2097_ = lean_box(0);
v_isShared_2098_ = v_isSharedCheck_2104_;
goto v_resetjp_2096_;
}
v_resetjp_2096_:
{
lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2102_; 
v___x_2099_ = l_Lean_Expr_mvarId_x21(v_a_2072_);
lean_dec(v_a_2072_);
v___x_2100_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2100_, 0, v___x_2099_);
lean_ctor_set(v___x_2100_, 1, v___x_2084_);
if (v_isShared_2098_ == 0)
{
lean_ctor_set(v___x_2097_, 0, v___x_2100_);
v___x_2102_ = v___x_2097_;
goto v_reusejp_2101_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v___x_2100_);
v___x_2102_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2101_;
}
v_reusejp_2101_:
{
return v___x_2102_;
}
}
}
else
{
lean_object* v_a_2106_; lean_object* v___x_2108_; uint8_t v_isShared_2109_; uint8_t v_isSharedCheck_2113_; 
lean_dec(v_a_2072_);
v_a_2106_ = lean_ctor_get(v___x_2095_, 0);
v_isSharedCheck_2113_ = !lean_is_exclusive(v___x_2095_);
if (v_isSharedCheck_2113_ == 0)
{
v___x_2108_ = v___x_2095_;
v_isShared_2109_ = v_isSharedCheck_2113_;
goto v_resetjp_2107_;
}
else
{
lean_inc(v_a_2106_);
lean_dec(v___x_2095_);
v___x_2108_ = lean_box(0);
v_isShared_2109_ = v_isSharedCheck_2113_;
goto v_resetjp_2107_;
}
v_resetjp_2107_:
{
lean_object* v___x_2111_; 
if (v_isShared_2109_ == 0)
{
v___x_2111_ = v___x_2108_;
goto v_reusejp_2110_;
}
else
{
lean_object* v_reuseFailAlloc_2112_; 
v_reuseFailAlloc_2112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2112_, 0, v_a_2106_);
v___x_2111_ = v_reuseFailAlloc_2112_;
goto v_reusejp_2110_;
}
v_reusejp_2110_:
{
return v___x_2111_;
}
}
}
}
else
{
lean_object* v_a_2114_; lean_object* v___x_2116_; uint8_t v_isShared_2117_; uint8_t v_isSharedCheck_2121_; 
lean_dec(v_a_2072_);
lean_dec(v_a_2060_);
v_a_2114_ = lean_ctor_get(v___x_2093_, 0);
v_isSharedCheck_2121_ = !lean_is_exclusive(v___x_2093_);
if (v_isSharedCheck_2121_ == 0)
{
v___x_2116_ = v___x_2093_;
v_isShared_2117_ = v_isSharedCheck_2121_;
goto v_resetjp_2115_;
}
else
{
lean_inc(v_a_2114_);
lean_dec(v___x_2093_);
v___x_2116_ = lean_box(0);
v_isShared_2117_ = v_isSharedCheck_2121_;
goto v_resetjp_2115_;
}
v_resetjp_2115_:
{
lean_object* v___x_2119_; 
if (v_isShared_2117_ == 0)
{
v___x_2119_ = v___x_2116_;
goto v_reusejp_2118_;
}
else
{
lean_object* v_reuseFailAlloc_2120_; 
v_reuseFailAlloc_2120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2120_, 0, v_a_2114_);
v___x_2119_ = v_reuseFailAlloc_2120_;
goto v_reusejp_2118_;
}
v_reusejp_2118_:
{
return v___x_2119_;
}
}
}
}
else
{
lean_object* v_a_2122_; lean_object* v___x_2124_; uint8_t v_isShared_2125_; uint8_t v_isSharedCheck_2129_; 
lean_dec(v___x_2079_);
lean_dec(v_a_2074_);
lean_dec(v_a_2072_);
lean_dec_ref(v___y_2068_);
lean_dec(v_a_2062_);
lean_dec(v_a_2060_);
v_a_2122_ = lean_ctor_get(v___x_2080_, 0);
v_isSharedCheck_2129_ = !lean_is_exclusive(v___x_2080_);
if (v_isSharedCheck_2129_ == 0)
{
v___x_2124_ = v___x_2080_;
v_isShared_2125_ = v_isSharedCheck_2129_;
goto v_resetjp_2123_;
}
else
{
lean_inc(v_a_2122_);
lean_dec(v___x_2080_);
v___x_2124_ = lean_box(0);
v_isShared_2125_ = v_isSharedCheck_2129_;
goto v_resetjp_2123_;
}
v_resetjp_2123_:
{
lean_object* v___x_2127_; 
if (v_isShared_2125_ == 0)
{
v___x_2127_ = v___x_2124_;
goto v_reusejp_2126_;
}
else
{
lean_object* v_reuseFailAlloc_2128_; 
v_reuseFailAlloc_2128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2128_, 0, v_a_2122_);
v___x_2127_ = v_reuseFailAlloc_2128_;
goto v_reusejp_2126_;
}
v_reusejp_2126_:
{
return v___x_2127_;
}
}
}
}
else
{
lean_object* v_a_2130_; lean_object* v___x_2132_; uint8_t v_isShared_2133_; uint8_t v_isSharedCheck_2137_; 
lean_dec(v_a_2072_);
lean_dec_ref(v___y_2068_);
lean_dec(v_a_2062_);
lean_dec(v_a_2060_);
v_a_2130_ = lean_ctor_get(v___x_2073_, 0);
v_isSharedCheck_2137_ = !lean_is_exclusive(v___x_2073_);
if (v_isSharedCheck_2137_ == 0)
{
v___x_2132_ = v___x_2073_;
v_isShared_2133_ = v_isSharedCheck_2137_;
goto v_resetjp_2131_;
}
else
{
lean_inc(v_a_2130_);
lean_dec(v___x_2073_);
v___x_2132_ = lean_box(0);
v_isShared_2133_ = v_isSharedCheck_2137_;
goto v_resetjp_2131_;
}
v_resetjp_2131_:
{
lean_object* v___x_2135_; 
if (v_isShared_2133_ == 0)
{
v___x_2135_ = v___x_2132_;
goto v_reusejp_2134_;
}
else
{
lean_object* v_reuseFailAlloc_2136_; 
v_reuseFailAlloc_2136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2136_, 0, v_a_2130_);
v___x_2135_ = v_reuseFailAlloc_2136_;
goto v_reusejp_2134_;
}
v_reusejp_2134_:
{
return v___x_2135_;
}
}
}
}
else
{
lean_object* v_a_2138_; lean_object* v___x_2140_; uint8_t v_isShared_2141_; uint8_t v_isSharedCheck_2145_; 
lean_dec_ref(v___y_2068_);
lean_dec(v_a_2062_);
lean_dec(v_a_2060_);
v_a_2138_ = lean_ctor_get(v___x_2071_, 0);
v_isSharedCheck_2145_ = !lean_is_exclusive(v___x_2071_);
if (v_isSharedCheck_2145_ == 0)
{
v___x_2140_ = v___x_2071_;
v_isShared_2141_ = v_isSharedCheck_2145_;
goto v_resetjp_2139_;
}
else
{
lean_inc(v_a_2138_);
lean_dec(v___x_2071_);
v___x_2140_ = lean_box(0);
v_isShared_2141_ = v_isSharedCheck_2145_;
goto v_resetjp_2139_;
}
v_resetjp_2139_:
{
lean_object* v___x_2143_; 
if (v_isShared_2141_ == 0)
{
v___x_2143_ = v___x_2140_;
goto v_reusejp_2142_;
}
else
{
lean_object* v_reuseFailAlloc_2144_; 
v_reuseFailAlloc_2144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2144_, 0, v_a_2138_);
v___x_2143_ = v_reuseFailAlloc_2144_;
goto v_reusejp_2142_;
}
v_reusejp_2142_:
{
return v___x_2143_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___boxed(lean_object* v___x_2146_, lean_object* v___x_2147_, lean_object* v___x_2148_, lean_object* v_a_2149_, lean_object* v___x_2150_, lean_object* v_a_2151_, lean_object* v___x_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_){
_start:
{
uint8_t v___x_17047__boxed_2160_; uint8_t v___x_17050__boxed_2161_; uint8_t v___x_17051__boxed_2162_; lean_object* v_res_2163_; 
v___x_17047__boxed_2160_ = lean_unbox(v___x_2147_);
v___x_17050__boxed_2161_ = lean_unbox(v___x_2150_);
v___x_17051__boxed_2162_ = lean_unbox(v___x_2152_);
v_res_2163_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0(v___x_2146_, v___x_17047__boxed_2160_, v___x_2148_, v_a_2149_, v___x_17050__boxed_2161_, v_a_2151_, v___x_17051__boxed_2162_, v___y_2153_, v___y_2154_, v___y_2155_, v___y_2156_, v___y_2157_, v___y_2158_);
lean_dec(v___y_2158_);
lean_dec(v___y_2156_);
lean_dec_ref(v___y_2155_);
lean_dec(v___y_2154_);
lean_dec_ref(v___y_2153_);
return v_res_2163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3(uint8_t v___x_2164_, lean_object* v_as_2165_, size_t v_sz_2166_, size_t v_i_2167_, lean_object* v_b_2168_, lean_object* v___y_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_){
_start:
{
uint8_t v___x_2178_; 
v___x_2178_ = lean_usize_dec_lt(v_i_2167_, v_sz_2166_);
if (v___x_2178_ == 0)
{
lean_object* v___x_2179_; 
v___x_2179_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2179_, 0, v_b_2168_);
return v___x_2179_;
}
else
{
lean_object* v___x_2180_; 
v___x_2180_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2170_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_);
if (lean_obj_tag(v___x_2180_) == 0)
{
lean_object* v_a_2181_; uint8_t v___x_2182_; lean_object* v_a_2183_; lean_object* v___x_2184_; uint8_t v___x_2185_; lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___f_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; 
v_a_2181_ = lean_ctor_get(v___x_2180_, 0);
lean_inc_n(v_a_2181_, 2);
lean_dec_ref_known(v___x_2180_, 1);
v___x_2182_ = 0;
v_a_2183_ = lean_array_uget_borrowed(v_as_2165_, v_i_2167_);
v___x_2184_ = lean_box(0);
v___x_2185_ = 0;
v___x_2186_ = lean_box(0);
v___x_2187_ = lean_box(v___x_2185_);
v___x_2188_ = lean_box(v___x_2182_);
v___x_2189_ = lean_box(v___x_2164_);
lean_inc(v_a_2183_);
v___f_2190_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___lam__0___boxed), 14, 7);
lean_closure_set(v___f_2190_, 0, v___x_2184_);
lean_closure_set(v___f_2190_, 1, v___x_2187_);
lean_closure_set(v___f_2190_, 2, v___x_2186_);
lean_closure_set(v___f_2190_, 3, v_a_2181_);
lean_closure_set(v___f_2190_, 4, v___x_2188_);
lean_closure_set(v___f_2190_, 5, v_a_2183_);
lean_closure_set(v___f_2190_, 6, v___x_2189_);
v___x_2191_ = lean_box(v___x_2182_);
v___x_2192_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_runTermElab___boxed), 12, 3);
lean_closure_set(v___x_2192_, 0, lean_box(0));
lean_closure_set(v___x_2192_, 1, v___f_2190_);
lean_closure_set(v___x_2192_, 2, v___x_2191_);
v___x_2193_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__2___redArg(v_a_2181_, v___x_2192_, v___y_2169_, v___y_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_);
if (lean_obj_tag(v___x_2193_) == 0)
{
lean_object* v_a_2194_; lean_object* v___x_2195_; 
v_a_2194_ = lean_ctor_get(v___x_2193_, 0);
lean_inc(v_a_2194_);
lean_dec_ref_known(v___x_2193_, 1);
v___x_2195_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_2194_, v___y_2170_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_);
if (lean_obj_tag(v___x_2195_) == 0)
{
lean_object* v___x_2196_; size_t v___x_2197_; size_t v___x_2198_; 
lean_dec_ref_known(v___x_2195_, 1);
v___x_2196_ = lean_box(0);
v___x_2197_ = ((size_t)1ULL);
v___x_2198_ = lean_usize_add(v_i_2167_, v___x_2197_);
v_i_2167_ = v___x_2198_;
v_b_2168_ = v___x_2196_;
goto _start;
}
else
{
return v___x_2195_;
}
}
else
{
lean_object* v_a_2200_; lean_object* v___x_2202_; uint8_t v_isShared_2203_; uint8_t v_isSharedCheck_2207_; 
v_a_2200_ = lean_ctor_get(v___x_2193_, 0);
v_isSharedCheck_2207_ = !lean_is_exclusive(v___x_2193_);
if (v_isSharedCheck_2207_ == 0)
{
v___x_2202_ = v___x_2193_;
v_isShared_2203_ = v_isSharedCheck_2207_;
goto v_resetjp_2201_;
}
else
{
lean_inc(v_a_2200_);
lean_dec(v___x_2193_);
v___x_2202_ = lean_box(0);
v_isShared_2203_ = v_isSharedCheck_2207_;
goto v_resetjp_2201_;
}
v_resetjp_2201_:
{
lean_object* v___x_2205_; 
if (v_isShared_2203_ == 0)
{
v___x_2205_ = v___x_2202_;
goto v_reusejp_2204_;
}
else
{
lean_object* v_reuseFailAlloc_2206_; 
v_reuseFailAlloc_2206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2206_, 0, v_a_2200_);
v___x_2205_ = v_reuseFailAlloc_2206_;
goto v_reusejp_2204_;
}
v_reusejp_2204_:
{
return v___x_2205_;
}
}
}
}
else
{
lean_object* v_a_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2215_; 
v_a_2208_ = lean_ctor_get(v___x_2180_, 0);
v_isSharedCheck_2215_ = !lean_is_exclusive(v___x_2180_);
if (v_isSharedCheck_2215_ == 0)
{
v___x_2210_ = v___x_2180_;
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_a_2208_);
lean_dec(v___x_2180_);
v___x_2210_ = lean_box(0);
v_isShared_2211_ = v_isSharedCheck_2215_;
goto v_resetjp_2209_;
}
v_resetjp_2209_:
{
lean_object* v___x_2213_; 
if (v_isShared_2211_ == 0)
{
v___x_2213_ = v___x_2210_;
goto v_reusejp_2212_;
}
else
{
lean_object* v_reuseFailAlloc_2214_; 
v_reuseFailAlloc_2214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2214_, 0, v_a_2208_);
v___x_2213_ = v_reuseFailAlloc_2214_;
goto v_reusejp_2212_;
}
v_reusejp_2212_:
{
return v___x_2213_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3___boxed(lean_object* v___x_2216_, lean_object* v_as_2217_, lean_object* v_sz_2218_, lean_object* v_i_2219_, lean_object* v_b_2220_, lean_object* v___y_2221_, lean_object* v___y_2222_, lean_object* v___y_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_){
_start:
{
uint8_t v___x_17243__boxed_2230_; size_t v_sz_boxed_2231_; size_t v_i_boxed_2232_; lean_object* v_res_2233_; 
v___x_17243__boxed_2230_ = lean_unbox(v___x_2216_);
v_sz_boxed_2231_ = lean_unbox_usize(v_sz_2218_);
lean_dec(v_sz_2218_);
v_i_boxed_2232_ = lean_unbox_usize(v_i_2219_);
lean_dec(v_i_2219_);
v_res_2233_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3(v___x_17243__boxed_2230_, v_as_2217_, v_sz_boxed_2231_, v_i_boxed_2232_, v_b_2220_, v___y_2221_, v___y_2222_, v___y_2223_, v___y_2224_, v___y_2225_, v___y_2226_, v___y_2227_, v___y_2228_);
lean_dec(v___y_2228_);
lean_dec_ref(v___y_2227_);
lean_dec(v___y_2226_);
lean_dec_ref(v___y_2225_);
lean_dec(v___y_2224_);
lean_dec_ref(v___y_2223_);
lean_dec(v___y_2222_);
lean_dec_ref(v___y_2221_);
lean_dec_ref(v_as_2217_);
return v_res_2233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__4(size_t v_sz_2234_, size_t v_i_2235_, lean_object* v_bs_2236_){
_start:
{
uint8_t v___x_2237_; 
v___x_2237_ = lean_usize_dec_lt(v_i_2235_, v_sz_2234_);
if (v___x_2237_ == 0)
{
return v_bs_2236_;
}
else
{
lean_object* v_v_2238_; lean_object* v___x_2239_; lean_object* v_bs_x27_2240_; size_t v___x_2241_; size_t v___x_2242_; lean_object* v___x_2243_; 
v_v_2238_ = lean_array_uget(v_bs_2236_, v_i_2235_);
v___x_2239_ = lean_unsigned_to_nat(0u);
v_bs_x27_2240_ = lean_array_uset(v_bs_2236_, v_i_2235_, v___x_2239_);
v___x_2241_ = ((size_t)1ULL);
v___x_2242_ = lean_usize_add(v_i_2235_, v___x_2241_);
v___x_2243_ = lean_array_uset(v_bs_x27_2240_, v_i_2235_, v_v_2238_);
v_i_2235_ = v___x_2242_;
v_bs_2236_ = v___x_2243_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__4___boxed(lean_object* v_sz_2245_, lean_object* v_i_2246_, lean_object* v_bs_2247_){
_start:
{
size_t v_sz_boxed_2248_; size_t v_i_boxed_2249_; lean_object* v_res_2250_; 
v_sz_boxed_2248_ = lean_unbox_usize(v_sz_2245_);
lean_dec(v_sz_2245_);
v_i_boxed_2249_ = lean_unbox_usize(v_i_2246_);
lean_dec(v_i_2246_);
v_res_2250_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__4(v_sz_boxed_2248_, v_i_boxed_2249_, v_bs_2247_);
return v_res_2250_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__11(void){
_start:
{
lean_object* v___x_2262_; lean_object* v___x_2263_; 
v___x_2262_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__10));
v___x_2263_ = l_String_toRawSubstring_x27(v___x_2262_);
return v___x_2263_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__17(void){
_start:
{
lean_object* v___x_2270_; lean_object* v___x_2271_; 
v___x_2270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__16));
v___x_2271_ = l_String_toRawSubstring_x27(v___x_2270_);
return v___x_2271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2(uint8_t v___x_2283_, lean_object* v___x_2284_, size_t v_sz_2285_, size_t v___x_2286_, lean_object* v___x_2287_, lean_object* v___f_2288_, uint8_t v___x_2289_, lean_object* v___x_2290_, lean_object* v_usingArg_2291_, lean_object* v___f_2292_, lean_object* v_wth_2293_, lean_object* v___y_2294_, lean_object* v___y_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_){
_start:
{
lean_object* v___y_2304_; lean_object* v___y_2305_; lean_object* v___y_2306_; lean_object* v___y_2307_; lean_object* v___y_2308_; lean_object* v___y_2309_; lean_object* v___y_2310_; lean_object* v___y_2311_; lean_object* v___x_2331_; 
v___x_2331_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__3(v___x_2283_, v___x_2284_, v_sz_2285_, v___x_2286_, v___x_2287_, v___y_2294_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_, v___y_2301_);
if (lean_obj_tag(v___x_2331_) == 0)
{
lean_object* v___x_2332_; 
lean_dec_ref_known(v___x_2331_, 1);
v___x_2332_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2288_, v___y_2294_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_, v___y_2301_);
if (lean_obj_tag(v___x_2332_) == 0)
{
lean_object* v_ref_2333_; lean_object* v_quotContext_2334_; lean_object* v_currMacroScope_2335_; lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; lean_object* v___x_2392_; 
lean_dec_ref_known(v___x_2332_, 1);
v_ref_2333_ = lean_ctor_get(v___y_2300_, 5);
v_quotContext_2334_ = lean_ctor_get(v___y_2300_, 10);
v_currMacroScope_2335_ = lean_ctor_get(v___y_2300_, 11);
v___x_2336_ = l_Lean_SourceInfo_fromRef(v_ref_2333_, v___x_2289_);
v___x_2337_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__10));
v___x_2338_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__11));
v___x_2339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__1));
lean_inc_ref_n(v___x_2290_, 8);
v___x_2340_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2339_);
v___x_2341_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__2));
lean_inc_n(v___x_2336_, 21);
v___x_2342_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2342_, 0, v___x_2336_);
lean_ctor_set(v___x_2342_, 1, v___x_2341_);
v___x_2343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__3));
v___x_2344_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2343_);
v___x_2345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__4));
v___x_2346_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2345_);
v___x_2347_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__22));
v___x_2348_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__5));
v___x_2349_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2348_);
v___x_2350_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2350_, 0, v___x_2336_);
lean_ctor_set(v___x_2350_, 1, v___x_2348_);
v___x_2351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__6));
v___x_2352_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2351_);
v___x_2353_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__7));
v___x_2354_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2353_);
v___x_2355_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__8));
v___x_2356_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2355_);
v___x_2357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__9));
v___x_2358_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2358_, 0, v___x_2336_);
lean_ctor_set(v___x_2358_, 1, v___x_2357_);
v___x_2359_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__11, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__11_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__11);
v___x_2360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__12));
lean_inc_n(v_currMacroScope_2335_, 2);
lean_inc_n(v_quotContext_2334_, 2);
v___x_2361_ = l_Lean_addMacroScope(v_quotContext_2334_, v___x_2360_, v_currMacroScope_2335_);
v___x_2362_ = lean_box(0);
v___x_2363_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2363_, 0, v___x_2336_);
lean_ctor_set(v___x_2363_, 1, v___x_2359_);
lean_ctor_set(v___x_2363_, 2, v___x_2361_);
lean_ctor_set(v___x_2363_, 3, v___x_2362_);
v___x_2364_ = l_Lean_Syntax_node2(v___x_2336_, v___x_2356_, v___x_2358_, v___x_2363_);
v___x_2365_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2354_, v___x_2364_);
v___x_2366_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2347_, v___x_2365_);
v___x_2367_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2352_, v___x_2366_);
v___x_2368_ = lean_obj_once(&lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8, &lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8_once, _init_lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______delab__app__Filter__term_u2200_u1da0__In___x2c____1___lam__3___closed__8);
v___x_2369_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2369_, 0, v___x_2336_);
lean_ctor_set(v___x_2369_, 1, v___x_2347_);
lean_ctor_set(v___x_2369_, 2, v___x_2368_);
v___x_2370_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__13));
v___x_2371_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2371_, 0, v___x_2336_);
lean_ctor_set(v___x_2371_, 1, v___x_2370_);
v___x_2372_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2347_, v___x_2371_);
v___x_2373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__14));
v___x_2374_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2374_, 0, v___x_2336_);
lean_ctor_set(v___x_2374_, 1, v___x_2373_);
v___x_2375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__15));
v___x_2376_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2375_);
v___x_2377_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__17, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__17_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__17);
v___x_2378_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__19));
v___x_2379_ = l_Lean_addMacroScope(v_quotContext_2334_, v___x_2378_, v_currMacroScope_2335_);
v___x_2380_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__21));
v___x_2381_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_2381_, 0, v___x_2336_);
lean_ctor_set(v___x_2381_, 1, v___x_2377_);
lean_ctor_set(v___x_2381_, 2, v___x_2379_);
lean_ctor_set(v___x_2381_, 3, v___x_2380_);
lean_inc_ref_n(v___x_2369_, 3);
v___x_2382_ = l_Lean_Syntax_node3(v___x_2336_, v___x_2376_, v___x_2369_, v___x_2369_, v___x_2381_);
v___x_2383_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2347_, v___x_2382_);
v___x_2384_ = ((lean_object*)(lp_mathlib_Filter_unexpandEventuallyEq___closed__1));
v___x_2385_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2385_, 0, v___x_2336_);
lean_ctor_set(v___x_2385_, 1, v___x_2384_);
v___x_2386_ = l_Lean_Syntax_node3(v___x_2336_, v___x_2347_, v___x_2374_, v___x_2383_, v___x_2385_);
v___x_2387_ = l_Lean_Syntax_node6(v___x_2336_, v___x_2349_, v___x_2350_, v___x_2367_, v___x_2369_, v___x_2372_, v___x_2386_, v___x_2369_);
v___x_2388_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2347_, v___x_2387_);
v___x_2389_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2346_, v___x_2388_);
v___x_2390_ = l_Lean_Syntax_node1(v___x_2336_, v___x_2344_, v___x_2389_);
v___x_2391_ = l_Lean_Syntax_node2(v___x_2336_, v___x_2340_, v___x_2342_, v___x_2390_);
v___x_2392_ = l_Lean_Elab_Tactic_evalTactic(v___x_2391_, v___y_2294_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_, v___y_2301_);
if (lean_obj_tag(v___x_2392_) == 0)
{
lean_dec_ref_known(v___x_2392_, 1);
if (lean_obj_tag(v_wth_2293_) == 1)
{
lean_object* v_val_2393_; lean_object* v___x_2394_; 
v_val_2393_ = lean_ctor_get(v_wth_2293_, 0);
lean_inc(v_val_2393_);
lean_dec_ref_known(v_wth_2293_, 1);
lean_inc_ref(v___f_2292_);
lean_inc(v___y_2301_);
lean_inc_ref(v___y_2300_);
lean_inc(v___y_2299_);
lean_inc_ref(v___y_2298_);
lean_inc(v___y_2297_);
lean_inc_ref(v___y_2296_);
lean_inc(v___y_2295_);
lean_inc_ref(v___y_2294_);
v___x_2394_ = lean_apply_9(v___f_2292_, v___y_2294_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_, v___y_2301_, lean_box(0));
if (lean_obj_tag(v___x_2394_) == 0)
{
lean_object* v_a_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; lean_object* v___x_2398_; size_t v_sz_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; 
v_a_2395_ = lean_ctor_get(v___x_2394_, 0);
lean_inc_n(v_a_2395_, 3);
lean_dec_ref_known(v___x_2394_, 1);
v___x_2396_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__22));
lean_inc_ref(v___x_2290_);
v___x_2397_ = l_Lean_Name_mkStr4(v___x_2337_, v___x_2338_, v___x_2290_, v___x_2396_);
v___x_2398_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2398_, 0, v_a_2395_);
lean_ctor_set(v___x_2398_, 1, v___x_2396_);
v_sz_2399_ = lean_array_size(v_val_2393_);
v___x_2400_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__4(v_sz_2399_, v___x_2286_, v_val_2393_);
v___x_2401_ = l_Array_append___redArg(v___x_2368_, v___x_2400_);
lean_dec_ref(v___x_2400_);
v___x_2402_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2402_, 0, v_a_2395_);
lean_ctor_set(v___x_2402_, 1, v___x_2347_);
lean_ctor_set(v___x_2402_, 2, v___x_2401_);
v___x_2403_ = l_Lean_Syntax_node2(v_a_2395_, v___x_2397_, v___x_2398_, v___x_2402_);
v___x_2404_ = l_Lean_Elab_Tactic_evalTactic(v___x_2403_, v___y_2294_, v___y_2295_, v___y_2296_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_, v___y_2301_);
if (lean_obj_tag(v___x_2404_) == 0)
{
lean_dec_ref_known(v___x_2404_, 1);
v___y_2304_ = v___y_2294_;
v___y_2305_ = v___y_2295_;
v___y_2306_ = v___y_2296_;
v___y_2307_ = v___y_2297_;
v___y_2308_ = v___y_2298_;
v___y_2309_ = v___y_2299_;
v___y_2310_ = v___y_2300_;
v___y_2311_ = v___y_2301_;
goto v___jp_2303_;
}
else
{
lean_dec(v___y_2301_);
lean_dec_ref(v___y_2300_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
lean_dec(v___y_2295_);
lean_dec_ref(v___y_2294_);
lean_dec_ref(v___f_2292_);
lean_dec(v_usingArg_2291_);
lean_dec_ref(v___x_2290_);
return v___x_2404_;
}
}
else
{
lean_object* v_a_2405_; lean_object* v___x_2407_; uint8_t v_isShared_2408_; uint8_t v_isSharedCheck_2412_; 
lean_dec(v_val_2393_);
lean_dec(v___y_2301_);
lean_dec_ref(v___y_2300_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
lean_dec(v___y_2295_);
lean_dec_ref(v___y_2294_);
lean_dec_ref(v___f_2292_);
lean_dec(v_usingArg_2291_);
lean_dec_ref(v___x_2290_);
v_a_2405_ = lean_ctor_get(v___x_2394_, 0);
v_isSharedCheck_2412_ = !lean_is_exclusive(v___x_2394_);
if (v_isSharedCheck_2412_ == 0)
{
v___x_2407_ = v___x_2394_;
v_isShared_2408_ = v_isSharedCheck_2412_;
goto v_resetjp_2406_;
}
else
{
lean_inc(v_a_2405_);
lean_dec(v___x_2394_);
v___x_2407_ = lean_box(0);
v_isShared_2408_ = v_isSharedCheck_2412_;
goto v_resetjp_2406_;
}
v_resetjp_2406_:
{
lean_object* v___x_2410_; 
if (v_isShared_2408_ == 0)
{
v___x_2410_ = v___x_2407_;
goto v_reusejp_2409_;
}
else
{
lean_object* v_reuseFailAlloc_2411_; 
v_reuseFailAlloc_2411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2411_, 0, v_a_2405_);
v___x_2410_ = v_reuseFailAlloc_2411_;
goto v_reusejp_2409_;
}
v_reusejp_2409_:
{
return v___x_2410_;
}
}
}
}
else
{
lean_dec(v_wth_2293_);
v___y_2304_ = v___y_2294_;
v___y_2305_ = v___y_2295_;
v___y_2306_ = v___y_2296_;
v___y_2307_ = v___y_2297_;
v___y_2308_ = v___y_2298_;
v___y_2309_ = v___y_2299_;
v___y_2310_ = v___y_2300_;
v___y_2311_ = v___y_2301_;
goto v___jp_2303_;
}
}
else
{
lean_dec(v___y_2301_);
lean_dec_ref(v___y_2300_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
lean_dec(v___y_2295_);
lean_dec_ref(v___y_2294_);
lean_dec(v_wth_2293_);
lean_dec_ref(v___f_2292_);
lean_dec(v_usingArg_2291_);
lean_dec_ref(v___x_2290_);
return v___x_2392_;
}
}
else
{
lean_dec(v___y_2301_);
lean_dec_ref(v___y_2300_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
lean_dec(v___y_2295_);
lean_dec_ref(v___y_2294_);
lean_dec(v_wth_2293_);
lean_dec_ref(v___f_2292_);
lean_dec(v_usingArg_2291_);
lean_dec_ref(v___x_2290_);
return v___x_2332_;
}
}
else
{
lean_dec(v___y_2301_);
lean_dec_ref(v___y_2300_);
lean_dec(v___y_2299_);
lean_dec_ref(v___y_2298_);
lean_dec(v___y_2297_);
lean_dec_ref(v___y_2296_);
lean_dec(v___y_2295_);
lean_dec_ref(v___y_2294_);
lean_dec(v_wth_2293_);
lean_dec_ref(v___f_2292_);
lean_dec(v_usingArg_2291_);
lean_dec_ref(v___x_2290_);
lean_dec_ref(v___f_2288_);
return v___x_2331_;
}
v___jp_2303_:
{
if (lean_obj_tag(v_usingArg_2291_) == 1)
{
lean_object* v_val_2312_; lean_object* v___x_2313_; 
v_val_2312_ = lean_ctor_get(v_usingArg_2291_, 0);
lean_inc(v_val_2312_);
lean_dec_ref_known(v_usingArg_2291_, 1);
lean_inc(v___y_2311_);
lean_inc_ref(v___y_2310_);
lean_inc(v___y_2309_);
lean_inc_ref(v___y_2308_);
lean_inc(v___y_2307_);
lean_inc_ref(v___y_2306_);
lean_inc(v___y_2305_);
lean_inc_ref(v___y_2304_);
v___x_2313_ = lean_apply_9(v___f_2292_, v___y_2304_, v___y_2305_, v___y_2306_, v___y_2307_, v___y_2308_, v___y_2309_, v___y_2310_, v___y_2311_, lean_box(0));
if (lean_obj_tag(v___x_2313_) == 0)
{
lean_object* v_a_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; lean_object* v___x_2321_; 
v_a_2314_ = lean_ctor_get(v___x_2313_, 0);
lean_inc_n(v_a_2314_, 2);
lean_dec_ref_known(v___x_2313_, 1);
v___x_2315_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__10));
v___x_2316_ = ((lean_object*)(lp_mathlib_Filter___aux__Mathlib__Order__Filter__Defs______macroRules__Filter__term_u2200_u1da0__In___x2c____1___closed__11));
v___x_2317_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___closed__0));
v___x_2318_ = l_Lean_Name_mkStr4(v___x_2315_, v___x_2316_, v___x_2290_, v___x_2317_);
v___x_2319_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2319_, 0, v_a_2314_);
lean_ctor_set(v___x_2319_, 1, v___x_2317_);
v___x_2320_ = l_Lean_Syntax_node2(v_a_2314_, v___x_2318_, v___x_2319_, v_val_2312_);
v___x_2321_ = l_Lean_Elab_Tactic_evalTactic(v___x_2320_, v___y_2304_, v___y_2305_, v___y_2306_, v___y_2307_, v___y_2308_, v___y_2309_, v___y_2310_, v___y_2311_);
lean_dec(v___y_2311_);
lean_dec_ref(v___y_2310_);
lean_dec(v___y_2309_);
lean_dec_ref(v___y_2308_);
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2306_);
lean_dec(v___y_2305_);
lean_dec_ref(v___y_2304_);
return v___x_2321_;
}
else
{
lean_object* v_a_2322_; lean_object* v___x_2324_; uint8_t v_isShared_2325_; uint8_t v_isSharedCheck_2329_; 
lean_dec(v_val_2312_);
lean_dec(v___y_2311_);
lean_dec_ref(v___y_2310_);
lean_dec(v___y_2309_);
lean_dec_ref(v___y_2308_);
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2306_);
lean_dec(v___y_2305_);
lean_dec_ref(v___y_2304_);
lean_dec_ref(v___x_2290_);
v_a_2322_ = lean_ctor_get(v___x_2313_, 0);
v_isSharedCheck_2329_ = !lean_is_exclusive(v___x_2313_);
if (v_isSharedCheck_2329_ == 0)
{
v___x_2324_ = v___x_2313_;
v_isShared_2325_ = v_isSharedCheck_2329_;
goto v_resetjp_2323_;
}
else
{
lean_inc(v_a_2322_);
lean_dec(v___x_2313_);
v___x_2324_ = lean_box(0);
v_isShared_2325_ = v_isSharedCheck_2329_;
goto v_resetjp_2323_;
}
v_resetjp_2323_:
{
lean_object* v___x_2327_; 
if (v_isShared_2325_ == 0)
{
v___x_2327_ = v___x_2324_;
goto v_reusejp_2326_;
}
else
{
lean_object* v_reuseFailAlloc_2328_; 
v_reuseFailAlloc_2328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2328_, 0, v_a_2322_);
v___x_2327_ = v_reuseFailAlloc_2328_;
goto v_reusejp_2326_;
}
v_reusejp_2326_:
{
return v___x_2327_;
}
}
}
}
else
{
lean_object* v___x_2330_; 
lean_dec(v___y_2311_);
lean_dec_ref(v___y_2310_);
lean_dec(v___y_2309_);
lean_dec_ref(v___y_2308_);
lean_dec(v___y_2307_);
lean_dec_ref(v___y_2306_);
lean_dec(v___y_2305_);
lean_dec_ref(v___y_2304_);
lean_dec_ref(v___f_2292_);
lean_dec(v_usingArg_2291_);
lean_dec_ref(v___x_2290_);
v___x_2330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2330_, 0, v___x_2287_);
return v___x_2330_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___boxed(lean_object** _args){
lean_object* v___x_2413_ = _args[0];
lean_object* v___x_2414_ = _args[1];
lean_object* v_sz_2415_ = _args[2];
lean_object* v___x_2416_ = _args[3];
lean_object* v___x_2417_ = _args[4];
lean_object* v___f_2418_ = _args[5];
lean_object* v___x_2419_ = _args[6];
lean_object* v___x_2420_ = _args[7];
lean_object* v_usingArg_2421_ = _args[8];
lean_object* v___f_2422_ = _args[9];
lean_object* v_wth_2423_ = _args[10];
lean_object* v___y_2424_ = _args[11];
lean_object* v___y_2425_ = _args[12];
lean_object* v___y_2426_ = _args[13];
lean_object* v___y_2427_ = _args[14];
lean_object* v___y_2428_ = _args[15];
lean_object* v___y_2429_ = _args[16];
lean_object* v___y_2430_ = _args[17];
lean_object* v___y_2431_ = _args[18];
lean_object* v___y_2432_ = _args[19];
_start:
{
uint8_t v___x_17435__boxed_2433_; size_t v_sz_boxed_2434_; size_t v___x_17437__boxed_2435_; uint8_t v___x_17440__boxed_2436_; lean_object* v_res_2437_; 
v___x_17435__boxed_2433_ = lean_unbox(v___x_2413_);
v_sz_boxed_2434_ = lean_unbox_usize(v_sz_2415_);
lean_dec(v_sz_2415_);
v___x_17437__boxed_2435_ = lean_unbox_usize(v___x_2416_);
lean_dec(v___x_2416_);
v___x_17440__boxed_2436_ = lean_unbox(v___x_2419_);
v_res_2437_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2(v___x_17435__boxed_2433_, v___x_2414_, v_sz_boxed_2434_, v___x_17437__boxed_2435_, v___x_2417_, v___f_2418_, v___x_17440__boxed_2436_, v___x_2420_, v_usingArg_2421_, v___f_2422_, v_wth_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_, v___y_2428_, v___y_2429_, v___y_2430_, v___y_2431_);
lean_dec_ref(v___x_2414_);
return v_res_2437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__5(size_t v_sz_2438_, size_t v_i_2439_, lean_object* v_bs_2440_){
_start:
{
uint8_t v___x_2441_; 
v___x_2441_ = lean_usize_dec_lt(v_i_2439_, v_sz_2438_);
if (v___x_2441_ == 0)
{
lean_object* v___x_2442_; 
v___x_2442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2442_, 0, v_bs_2440_);
return v___x_2442_;
}
else
{
lean_object* v_v_2443_; lean_object* v___x_2444_; lean_object* v_bs_x27_2445_; size_t v___x_2446_; size_t v___x_2447_; lean_object* v___x_2448_; 
v_v_2443_ = lean_array_uget(v_bs_2440_, v_i_2439_);
v___x_2444_ = lean_unsigned_to_nat(0u);
v_bs_x27_2445_ = lean_array_uset(v_bs_2440_, v_i_2439_, v___x_2444_);
v___x_2446_ = ((size_t)1ULL);
v___x_2447_ = lean_usize_add(v_i_2439_, v___x_2446_);
v___x_2448_ = lean_array_uset(v_bs_x27_2445_, v_i_2439_, v_v_2443_);
v_i_2439_ = v___x_2447_;
v_bs_2440_ = v___x_2448_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__5___boxed(lean_object* v_sz_2450_, lean_object* v_i_2451_, lean_object* v_bs_2452_){
_start:
{
size_t v_sz_boxed_2453_; size_t v_i_boxed_2454_; lean_object* v_res_2455_; 
v_sz_boxed_2453_ = lean_unbox_usize(v_sz_2450_);
lean_dec(v_sz_2450_);
v_i_boxed_2454_ = lean_unbox_usize(v_i_2451_);
lean_dec(v_i_2451_);
v_res_2455_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__5(v_sz_boxed_2453_, v_i_boxed_2454_, v_bs_2452_);
return v_res_2455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6(uint8_t v___x_2456_, uint8_t v___x_2457_, lean_object* v_as_2458_, size_t v_i_2459_, size_t v_stop_2460_, lean_object* v_b_2461_){
_start:
{
lean_object* v___y_2463_; uint8_t v___x_2467_; 
v___x_2467_ = lean_usize_dec_eq(v_i_2459_, v_stop_2460_);
if (v___x_2467_ == 0)
{
lean_object* v_fst_2468_; uint8_t v___x_2469_; 
v_fst_2468_ = lean_ctor_get(v_b_2461_, 0);
v___x_2469_ = lean_unbox(v_fst_2468_);
if (v___x_2469_ == 0)
{
lean_object* v_snd_2470_; lean_object* v___x_2472_; uint8_t v_isShared_2473_; uint8_t v_isSharedCheck_2478_; 
v_snd_2470_ = lean_ctor_get(v_b_2461_, 1);
v_isSharedCheck_2478_ = !lean_is_exclusive(v_b_2461_);
if (v_isSharedCheck_2478_ == 0)
{
lean_object* v_unused_2479_; 
v_unused_2479_ = lean_ctor_get(v_b_2461_, 0);
lean_dec(v_unused_2479_);
v___x_2472_ = v_b_2461_;
v_isShared_2473_ = v_isSharedCheck_2478_;
goto v_resetjp_2471_;
}
else
{
lean_inc(v_snd_2470_);
lean_dec(v_b_2461_);
v___x_2472_ = lean_box(0);
v_isShared_2473_ = v_isSharedCheck_2478_;
goto v_resetjp_2471_;
}
v_resetjp_2471_:
{
lean_object* v___x_2474_; lean_object* v___x_2476_; 
v___x_2474_ = lean_box(v___x_2456_);
if (v_isShared_2473_ == 0)
{
lean_ctor_set(v___x_2472_, 0, v___x_2474_);
v___x_2476_ = v___x_2472_;
goto v_reusejp_2475_;
}
else
{
lean_object* v_reuseFailAlloc_2477_; 
v_reuseFailAlloc_2477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2477_, 0, v___x_2474_);
lean_ctor_set(v_reuseFailAlloc_2477_, 1, v_snd_2470_);
v___x_2476_ = v_reuseFailAlloc_2477_;
goto v_reusejp_2475_;
}
v_reusejp_2475_:
{
v___y_2463_ = v___x_2476_;
goto v___jp_2462_;
}
}
}
else
{
lean_object* v_snd_2480_; lean_object* v___x_2482_; uint8_t v_isShared_2483_; uint8_t v_isSharedCheck_2490_; 
v_snd_2480_ = lean_ctor_get(v_b_2461_, 1);
v_isSharedCheck_2490_ = !lean_is_exclusive(v_b_2461_);
if (v_isSharedCheck_2490_ == 0)
{
lean_object* v_unused_2491_; 
v_unused_2491_ = lean_ctor_get(v_b_2461_, 0);
lean_dec(v_unused_2491_);
v___x_2482_ = v_b_2461_;
v_isShared_2483_ = v_isSharedCheck_2490_;
goto v_resetjp_2481_;
}
else
{
lean_inc(v_snd_2480_);
lean_dec(v_b_2461_);
v___x_2482_ = lean_box(0);
v_isShared_2483_ = v_isSharedCheck_2490_;
goto v_resetjp_2481_;
}
v_resetjp_2481_:
{
lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2488_; 
v___x_2484_ = lean_array_uget_borrowed(v_as_2458_, v_i_2459_);
lean_inc(v___x_2484_);
v___x_2485_ = lean_array_push(v_snd_2480_, v___x_2484_);
v___x_2486_ = lean_box(v___x_2457_);
if (v_isShared_2483_ == 0)
{
lean_ctor_set(v___x_2482_, 1, v___x_2485_);
lean_ctor_set(v___x_2482_, 0, v___x_2486_);
v___x_2488_ = v___x_2482_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2489_; 
v_reuseFailAlloc_2489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2489_, 0, v___x_2486_);
lean_ctor_set(v_reuseFailAlloc_2489_, 1, v___x_2485_);
v___x_2488_ = v_reuseFailAlloc_2489_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
v___y_2463_ = v___x_2488_;
goto v___jp_2462_;
}
}
}
}
else
{
return v_b_2461_;
}
v___jp_2462_:
{
size_t v___x_2464_; size_t v___x_2465_; 
v___x_2464_ = ((size_t)1ULL);
v___x_2465_ = lean_usize_add(v_i_2459_, v___x_2464_);
v_i_2459_ = v___x_2465_;
v_b_2461_ = v___y_2463_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6___boxed(lean_object* v___x_2492_, lean_object* v___x_2493_, lean_object* v_as_2494_, lean_object* v_i_2495_, lean_object* v_stop_2496_, lean_object* v_b_2497_){
_start:
{
uint8_t v___x_17750__boxed_2498_; uint8_t v___x_17751__boxed_2499_; size_t v_i_boxed_2500_; size_t v_stop_boxed_2501_; lean_object* v_res_2502_; 
v___x_17750__boxed_2498_ = lean_unbox(v___x_2492_);
v___x_17751__boxed_2499_ = lean_unbox(v___x_2493_);
v_i_boxed_2500_ = lean_unbox_usize(v_i_2495_);
lean_dec(v_i_2495_);
v_stop_boxed_2501_ = lean_unbox_usize(v_stop_2496_);
lean_dec(v_stop_2496_);
v_res_2502_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6(v___x_17750__boxed_2498_, v___x_17751__boxed_2499_, v_as_2494_, v_i_boxed_2500_, v_stop_boxed_2501_, v_b_2497_);
lean_dec_ref(v_as_2494_);
return v_res_2502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1(lean_object* v_x_2510_, lean_object* v_a_2511_, lean_object* v_a_2512_, lean_object* v_a_2513_, lean_object* v_a_2514_, lean_object* v_a_2515_, lean_object* v_a_2516_, lean_object* v_a_2517_, lean_object* v_a_2518_){
_start:
{
lean_object* v___x_2520_; lean_object* v___x_2521_; uint8_t v___x_2522_; 
v___x_2520_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_filterUpwards___closed__0));
v___x_2521_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_filterUpwards___closed__2));
lean_inc(v_x_2510_);
v___x_2522_ = l_Lean_Syntax_isOfKind(v_x_2510_, v___x_2521_);
if (v___x_2522_ == 0)
{
lean_object* v___x_2523_; 
lean_dec(v_x_2510_);
v___x_2523_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v___x_2523_;
}
else
{
lean_object* v___f_2524_; lean_object* v___y_2526_; lean_object* v___y_2527_; lean_object* v___y_2528_; uint8_t v___y_2529_; lean_object* v___y_2530_; lean_object* v___y_2531_; lean_object* v___y_2532_; lean_object* v___y_2533_; lean_object* v___y_2534_; lean_object* v___y_2535_; lean_object* v___y_2536_; lean_object* v___y_2537_; lean_object* v___y_2538_; lean_object* v___y_2549_; lean_object* v___y_2550_; lean_object* v___y_2551_; lean_object* v___y_2552_; lean_object* v___y_2553_; lean_object* v___y_2554_; lean_object* v___y_2555_; lean_object* v___y_2556_; lean_object* v___y_2557_; lean_object* v___y_2558_; lean_object* v_usingArg_2559_; lean_object* v___x_2566_; lean_object* v___y_2568_; lean_object* v___y_2569_; lean_object* v_wth_2570_; lean_object* v___y_2571_; lean_object* v___y_2572_; lean_object* v___y_2573_; lean_object* v___y_2574_; lean_object* v___y_2575_; lean_object* v___y_2576_; lean_object* v___y_2577_; lean_object* v___y_2578_; lean_object* v_args_2588_; lean_object* v___y_2589_; lean_object* v___y_2590_; lean_object* v___y_2591_; lean_object* v___y_2592_; lean_object* v___y_2593_; lean_object* v___y_2594_; lean_object* v___y_2595_; lean_object* v___y_2596_; lean_object* v___y_2607_; lean_object* v___x_2612_; uint8_t v___x_2613_; 
v___f_2524_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__0));
v___x_2566_ = lean_unsigned_to_nat(1u);
v___x_2612_ = l_Lean_Syntax_getArg(v_x_2510_, v___x_2566_);
v___x_2613_ = l_Lean_Syntax_isNone(v___x_2612_);
if (v___x_2613_ == 0)
{
lean_object* v___x_2614_; uint8_t v___x_2615_; 
v___x_2614_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_2612_);
v___x_2615_ = l_Lean_Syntax_matchesNull(v___x_2612_, v___x_2614_);
if (v___x_2615_ == 0)
{
lean_object* v___x_2616_; 
lean_dec(v___x_2612_);
lean_dec(v_x_2510_);
v___x_2616_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v___x_2616_;
}
else
{
lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; uint8_t v___x_2622_; 
v___x_2617_ = l_Lean_Syntax_getArg(v___x_2612_, v___x_2566_);
lean_dec(v___x_2612_);
v___x_2618_ = l_Lean_Syntax_getArgs(v___x_2617_);
lean_dec(v___x_2617_);
v___x_2619_ = lean_unsigned_to_nat(0u);
v___x_2620_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__2));
v___x_2621_ = lean_array_get_size(v___x_2618_);
v___x_2622_ = lean_nat_dec_lt(v___x_2619_, v___x_2621_);
if (v___x_2622_ == 0)
{
lean_dec_ref(v___x_2618_);
v___y_2607_ = v___x_2620_;
goto v___jp_2606_;
}
else
{
lean_object* v___x_2623_; lean_object* v___x_2624_; uint8_t v___x_2625_; 
v___x_2623_ = lean_box(v___x_2615_);
v___x_2624_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2624_, 0, v___x_2623_);
lean_ctor_set(v___x_2624_, 1, v___x_2620_);
v___x_2625_ = lean_nat_dec_le(v___x_2621_, v___x_2621_);
if (v___x_2625_ == 0)
{
if (v___x_2622_ == 0)
{
lean_dec_ref_known(v___x_2624_, 2);
lean_dec_ref(v___x_2618_);
v___y_2607_ = v___x_2620_;
goto v___jp_2606_;
}
else
{
size_t v___x_2626_; size_t v___x_2627_; lean_object* v___x_2628_; lean_object* v_snd_2629_; 
v___x_2626_ = ((size_t)0ULL);
v___x_2627_ = lean_usize_of_nat(v___x_2621_);
v___x_2628_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6(v___x_2615_, v___x_2613_, v___x_2618_, v___x_2626_, v___x_2627_, v___x_2624_);
lean_dec_ref(v___x_2618_);
v_snd_2629_ = lean_ctor_get(v___x_2628_, 1);
lean_inc(v_snd_2629_);
lean_dec_ref(v___x_2628_);
v___y_2607_ = v_snd_2629_;
goto v___jp_2606_;
}
}
else
{
size_t v___x_2630_; size_t v___x_2631_; lean_object* v___x_2632_; lean_object* v_snd_2633_; 
v___x_2630_ = ((size_t)0ULL);
v___x_2631_ = lean_usize_of_nat(v___x_2621_);
v___x_2632_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__6(v___x_2615_, v___x_2613_, v___x_2618_, v___x_2630_, v___x_2631_, v___x_2624_);
lean_dec_ref(v___x_2618_);
v_snd_2633_ = lean_ctor_get(v___x_2632_, 1);
lean_inc(v_snd_2633_);
lean_dec_ref(v___x_2632_);
v___y_2607_ = v_snd_2633_;
goto v___jp_2606_;
}
}
}
}
else
{
lean_object* v___x_2634_; 
lean_dec(v___x_2612_);
v___x_2634_ = lean_box(0);
v_args_2588_ = v___x_2634_;
v___y_2589_ = v_a_2511_;
v___y_2590_ = v_a_2512_;
v___y_2591_ = v_a_2513_;
v___y_2592_ = v_a_2514_;
v___y_2593_ = v_a_2515_;
v___y_2594_ = v_a_2516_;
v___y_2595_ = v_a_2517_;
v___y_2596_ = v_a_2518_;
goto v___jp_2587_;
}
v___jp_2525_:
{
lean_object* v___x_2539_; lean_object* v___x_2540_; size_t v_sz_2541_; lean_object* v___x_2542_; lean_object* v___x_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; lean_object* v___f_2546_; lean_object* v___x_2547_; 
v___x_2539_ = l_Array_reverse___redArg(v___y_2538_);
v___x_2540_ = lean_box(0);
v_sz_2541_ = lean_array_size(v___x_2539_);
v___x_2542_ = lean_box(v___x_2522_);
v___x_2543_ = lean_box_usize(v_sz_2541_);
v___x_2544_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___boxed__const__1));
v___x_2545_ = lean_box(v___y_2529_);
v___f_2546_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__2___boxed), 20, 11);
lean_closure_set(v___f_2546_, 0, v___x_2542_);
lean_closure_set(v___f_2546_, 1, v___x_2539_);
lean_closure_set(v___f_2546_, 2, v___x_2543_);
lean_closure_set(v___f_2546_, 3, v___x_2544_);
lean_closure_set(v___f_2546_, 4, v___x_2540_);
lean_closure_set(v___f_2546_, 5, v___y_2526_);
lean_closure_set(v___f_2546_, 6, v___x_2545_);
lean_closure_set(v___f_2546_, 7, v___x_2520_);
lean_closure_set(v___f_2546_, 8, v___y_2535_);
lean_closure_set(v___f_2546_, 9, v___f_2524_);
lean_closure_set(v___f_2546_, 10, v___y_2530_);
v___x_2547_ = l_Lean_Elab_Tactic_focus___redArg(v___f_2546_, v___y_2533_, v___y_2531_, v___y_2527_, v___y_2537_, v___y_2534_, v___y_2532_, v___y_2536_, v___y_2528_);
return v___x_2547_;
}
v___jp_2548_:
{
uint8_t v___x_2560_; uint8_t v___x_2561_; lean_object* v_config_2562_; lean_object* v___f_2563_; 
v___x_2560_ = 1;
v___x_2561_ = 0;
v_config_2562_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v_config_2562_, 0, v___x_2560_);
lean_ctor_set_uint8(v_config_2562_, 1, v___x_2522_);
lean_ctor_set_uint8(v_config_2562_, 2, v___x_2561_);
lean_ctor_set_uint8(v_config_2562_, 3, v___x_2522_);
v___f_2563_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___lam__1___boxed), 10, 1);
lean_closure_set(v___f_2563_, 0, v_config_2562_);
if (lean_obj_tag(v___y_2551_) == 0)
{
lean_object* v___x_2564_; 
v___x_2564_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___closed__1));
v___y_2526_ = v___f_2563_;
v___y_2527_ = v___y_2552_;
v___y_2528_ = v___y_2553_;
v___y_2529_ = v___x_2561_;
v___y_2530_ = v___y_2556_;
v___y_2531_ = v___y_2549_;
v___y_2532_ = v___y_2550_;
v___y_2533_ = v___y_2554_;
v___y_2534_ = v___y_2555_;
v___y_2535_ = v_usingArg_2559_;
v___y_2536_ = v___y_2557_;
v___y_2537_ = v___y_2558_;
v___y_2538_ = v___x_2564_;
goto v___jp_2525_;
}
else
{
lean_object* v_val_2565_; 
v_val_2565_ = lean_ctor_get(v___y_2551_, 0);
lean_inc(v_val_2565_);
lean_dec_ref_known(v___y_2551_, 1);
v___y_2526_ = v___f_2563_;
v___y_2527_ = v___y_2552_;
v___y_2528_ = v___y_2553_;
v___y_2529_ = v___x_2561_;
v___y_2530_ = v___y_2556_;
v___y_2531_ = v___y_2549_;
v___y_2532_ = v___y_2550_;
v___y_2533_ = v___y_2554_;
v___y_2534_ = v___y_2555_;
v___y_2535_ = v_usingArg_2559_;
v___y_2536_ = v___y_2557_;
v___y_2537_ = v___y_2558_;
v___y_2538_ = v_val_2565_;
goto v___jp_2525_;
}
}
v___jp_2567_:
{
lean_object* v___x_2579_; lean_object* v___x_2580_; uint8_t v___x_2581_; 
v___x_2579_ = lean_unsigned_to_nat(3u);
v___x_2580_ = l_Lean_Syntax_getArg(v_x_2510_, v___x_2579_);
lean_dec(v_x_2510_);
v___x_2581_ = l_Lean_Syntax_isNone(v___x_2580_);
if (v___x_2581_ == 0)
{
uint8_t v___x_2582_; 
lean_inc(v___x_2580_);
v___x_2582_ = l_Lean_Syntax_matchesNull(v___x_2580_, v___y_2569_);
if (v___x_2582_ == 0)
{
lean_object* v___x_2583_; 
lean_dec(v___x_2580_);
lean_dec(v_wth_2570_);
lean_dec(v___y_2568_);
v___x_2583_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v___x_2583_;
}
else
{
lean_object* v_usingArg_2584_; lean_object* v___x_2585_; 
v_usingArg_2584_ = l_Lean_Syntax_getArg(v___x_2580_, v___x_2566_);
lean_dec(v___x_2580_);
v___x_2585_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2585_, 0, v_usingArg_2584_);
v___y_2549_ = v___y_2572_;
v___y_2550_ = v___y_2576_;
v___y_2551_ = v___y_2568_;
v___y_2552_ = v___y_2573_;
v___y_2553_ = v___y_2578_;
v___y_2554_ = v___y_2571_;
v___y_2555_ = v___y_2575_;
v___y_2556_ = v_wth_2570_;
v___y_2557_ = v___y_2577_;
v___y_2558_ = v___y_2574_;
v_usingArg_2559_ = v___x_2585_;
goto v___jp_2548_;
}
}
else
{
lean_object* v___x_2586_; 
lean_dec(v___x_2580_);
v___x_2586_ = lean_box(0);
v___y_2549_ = v___y_2572_;
v___y_2550_ = v___y_2576_;
v___y_2551_ = v___y_2568_;
v___y_2552_ = v___y_2573_;
v___y_2553_ = v___y_2578_;
v___y_2554_ = v___y_2571_;
v___y_2555_ = v___y_2575_;
v___y_2556_ = v_wth_2570_;
v___y_2557_ = v___y_2577_;
v___y_2558_ = v___y_2574_;
v_usingArg_2559_ = v___x_2586_;
goto v___jp_2548_;
}
}
v___jp_2587_:
{
lean_object* v___x_2597_; lean_object* v___x_2598_; uint8_t v___x_2599_; 
v___x_2597_ = lean_unsigned_to_nat(2u);
v___x_2598_ = l_Lean_Syntax_getArg(v_x_2510_, v___x_2597_);
v___x_2599_ = l_Lean_Syntax_isNone(v___x_2598_);
if (v___x_2599_ == 0)
{
uint8_t v___x_2600_; 
lean_inc(v___x_2598_);
v___x_2600_ = l_Lean_Syntax_matchesNull(v___x_2598_, v___x_2597_);
if (v___x_2600_ == 0)
{
lean_object* v___x_2601_; 
lean_dec(v___x_2598_);
lean_dec(v_args_2588_);
lean_dec(v_x_2510_);
v___x_2601_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v___x_2601_;
}
else
{
lean_object* v___x_2602_; lean_object* v_wth_2603_; lean_object* v___x_2604_; 
v___x_2602_ = l_Lean_Syntax_getArg(v___x_2598_, v___x_2566_);
lean_dec(v___x_2598_);
v_wth_2603_ = l_Lean_Syntax_getArgs(v___x_2602_);
lean_dec(v___x_2602_);
v___x_2604_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2604_, 0, v_wth_2603_);
v___y_2568_ = v_args_2588_;
v___y_2569_ = v___x_2597_;
v_wth_2570_ = v___x_2604_;
v___y_2571_ = v___y_2589_;
v___y_2572_ = v___y_2590_;
v___y_2573_ = v___y_2591_;
v___y_2574_ = v___y_2592_;
v___y_2575_ = v___y_2593_;
v___y_2576_ = v___y_2594_;
v___y_2577_ = v___y_2595_;
v___y_2578_ = v___y_2596_;
goto v___jp_2567_;
}
}
else
{
lean_object* v___x_2605_; 
lean_dec(v___x_2598_);
v___x_2605_ = lean_box(0);
v___y_2568_ = v_args_2588_;
v___y_2569_ = v___x_2597_;
v_wth_2570_ = v___x_2605_;
v___y_2571_ = v___y_2589_;
v___y_2572_ = v___y_2590_;
v___y_2573_ = v___y_2591_;
v___y_2574_ = v___y_2592_;
v___y_2575_ = v___y_2593_;
v___y_2576_ = v___y_2594_;
v___y_2577_ = v___y_2595_;
v___y_2578_ = v___y_2596_;
goto v___jp_2567_;
}
}
v___jp_2606_:
{
size_t v_sz_2608_; size_t v___x_2609_; lean_object* v___x_2610_; 
v_sz_2608_ = lean_array_size(v___y_2607_);
v___x_2609_ = ((size_t)0ULL);
v___x_2610_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__5(v_sz_2608_, v___x_2609_, v___y_2607_);
if (lean_obj_tag(v___x_2610_) == 0)
{
lean_object* v___x_2611_; 
lean_dec(v_x_2510_);
v___x_2611_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__0___redArg();
return v___x_2611_;
}
else
{
v_args_2588_ = v___x_2610_;
v___y_2589_ = v_a_2511_;
v___y_2590_ = v_a_2512_;
v___y_2591_ = v_a_2513_;
v___y_2592_ = v_a_2514_;
v___y_2593_ = v_a_2515_;
v___y_2594_ = v_a_2516_;
v___y_2595_ = v_a_2517_;
v___y_2596_ = v_a_2518_;
goto v___jp_2587_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1___boxed(lean_object* v_x_2635_, lean_object* v_a_2636_, lean_object* v_a_2637_, lean_object* v_a_2638_, lean_object* v_a_2639_, lean_object* v_a_2640_, lean_object* v_a_2641_, lean_object* v_a_2642_, lean_object* v_a_2643_, lean_object* v_a_2644_){
_start:
{
lean_object* v_res_2645_; 
v_res_2645_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1(v_x_2635_, v_a_2636_, v_a_2637_, v_a_2638_, v_a_2639_, v_a_2640_, v_a_2641_, v_a_2642_, v_a_2643_);
lean_dec(v_a_2643_);
lean_dec_ref(v_a_2642_);
lean_dec(v_a_2641_);
lean_dec_ref(v_a_2640_);
lean_dec(v_a_2639_);
lean_dec_ref(v_a_2638_);
lean_dec(v_a_2637_);
lean_dec_ref(v_a_2636_);
return v_res_2645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1(lean_object* v_mvarId_2646_, lean_object* v_val_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_){
_start:
{
lean_object* v___x_2655_; 
v___x_2655_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___redArg(v_mvarId_2646_, v_val_2647_, v___y_2651_);
return v___x_2655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1___boxed(lean_object* v_mvarId_2656_, lean_object* v_val_2657_, lean_object* v___y_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_){
_start:
{
lean_object* v_res_2665_; 
v_res_2665_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1(v_mvarId_2656_, v_val_2657_, v___y_2658_, v___y_2659_, v___y_2660_, v___y_2661_, v___y_2662_, v___y_2663_);
lean_dec(v___y_2663_);
lean_dec_ref(v___y_2662_);
lean_dec(v___y_2661_);
lean_dec_ref(v___y_2660_);
lean_dec(v___y_2659_);
lean_dec_ref(v___y_2658_);
return v_res_2665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1(lean_object* v_00_u03b2_2666_, lean_object* v_x_2667_, lean_object* v_x_2668_, lean_object* v_x_2669_){
_start:
{
lean_object* v___x_2670_; 
v___x_2670_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1___redArg(v_x_2667_, v_x_2668_, v_x_2669_);
return v___x_2670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3(lean_object* v_00_u03b2_2671_, lean_object* v_x_2672_, size_t v_x_2673_, size_t v_x_2674_, lean_object* v_x_2675_, lean_object* v_x_2676_){
_start:
{
lean_object* v___x_2677_; 
v___x_2677_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___redArg(v_x_2672_, v_x_2673_, v_x_2674_, v_x_2675_, v_x_2676_);
return v___x_2677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b2_2678_, lean_object* v_x_2679_, lean_object* v_x_2680_, lean_object* v_x_2681_, lean_object* v_x_2682_, lean_object* v_x_2683_){
_start:
{
size_t v_x_18116__boxed_2684_; size_t v_x_18117__boxed_2685_; lean_object* v_res_2686_; 
v_x_18116__boxed_2684_ = lean_unbox_usize(v_x_2680_);
lean_dec(v_x_2680_);
v_x_18117__boxed_2685_ = lean_unbox_usize(v_x_2681_);
lean_dec(v_x_2681_);
v_res_2686_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3(v_00_u03b2_2678_, v_x_2679_, v_x_18116__boxed_2684_, v_x_18117__boxed_2685_, v_x_2682_, v_x_2683_);
return v_res_2686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8(lean_object* v_00_u03b2_2687_, lean_object* v_n_2688_, lean_object* v_k_2689_, lean_object* v_v_2690_){
_start:
{
lean_object* v___x_2691_; 
v___x_2691_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8___redArg(v_n_2688_, v_k_2689_, v_v_2690_);
return v___x_2691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9(lean_object* v_00_u03b2_2692_, size_t v_depth_2693_, lean_object* v_keys_2694_, lean_object* v_vals_2695_, lean_object* v_heq_2696_, lean_object* v_i_2697_, lean_object* v_entries_2698_){
_start:
{
lean_object* v___x_2699_; 
v___x_2699_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___redArg(v_depth_2693_, v_keys_2694_, v_vals_2695_, v_i_2697_, v_entries_2698_);
return v___x_2699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9___boxed(lean_object* v_00_u03b2_2700_, lean_object* v_depth_2701_, lean_object* v_keys_2702_, lean_object* v_vals_2703_, lean_object* v_heq_2704_, lean_object* v_i_2705_, lean_object* v_entries_2706_){
_start:
{
size_t v_depth_boxed_2707_; lean_object* v_res_2708_; 
v_depth_boxed_2707_ = lean_unbox_usize(v_depth_2701_);
lean_dec(v_depth_2701_);
v_res_2708_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__9(v_00_u03b2_2700_, v_depth_boxed_2707_, v_keys_2702_, v_vals_2703_, v_heq_2704_, v_i_2705_, v_entries_2706_);
lean_dec_ref(v_vals_2703_);
lean_dec_ref(v_keys_2702_);
return v_res_2708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8_spec__9(lean_object* v_00_u03b2_2709_, lean_object* v_x_2710_, lean_object* v_x_2711_, lean_object* v_x_2712_, lean_object* v_x_2713_){
_start:
{
lean_object* v___x_2714_; 
v___x_2714_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic___aux__Mathlib__Order__Filter__Defs______elabRules__Mathlib__Tactic__filterUpwards__1_spec__1_spec__1_spec__3_spec__8_spec__9___redArg(v_x_2710_, v_x_2711_, v_x_2712_, v_x_2713_);
return v___x_2714_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Filter_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Filter_term_u2200_u1da0__In___x2c__ = _init_lp_mathlib_Filter_term_u2200_u1da0__In___x2c__();
lean_mark_persistent(lp_mathlib_Filter_term_u2200_u1da0__In___x2c__);
lp_mathlib_Filter_term_u2203_u1da0__In___x2c__ = _init_lp_mathlib_Filter_term_u2203_u1da0__In___x2c__();
lean_mark_persistent(lp_mathlib_Filter_term_u2203_u1da0__In___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Insert(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SetNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Filter_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Insert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SetNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BooleanAlgebra_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Filter_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Filter_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
