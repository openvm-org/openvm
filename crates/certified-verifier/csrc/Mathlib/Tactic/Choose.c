// Lean compiler output
// Module: Mathlib.Tactic.Choose
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Basic
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
lean_object* l_Lean_Elab_Term_addLocalVarInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_changeLocalDecl(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized;
extern lean_object* l_Lean_binderIdent;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_intro(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_assert(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_replaceFVar(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkArrow(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getTag(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Expr_getBinderName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "sometimes"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__1_value),LEAN_SCALAR_PTR_LITERAL(133, 192, 12, 45, 216, 70, 11, 114)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "sometimes_spec"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__3_value),LEAN_SCALAR_PTR_LITERAL(77, 45, 26, 135, 1, 73, 146, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mk__sometimes(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mk__sometimes___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_success_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_success_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_failure_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_failure_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_merge(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg_default___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg_default = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_instInhabitedChooseArg_default___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "chooseBinder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Choose"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(48, 63, 246, 86, 229, 66, 75, 131)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(171, 127, 177, 87, 200, 154, 61, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__5_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_chooseBinder;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__3_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "extBinderParenthesized"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(207, 166, 79, 161, 194, 16, 7, 156)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 107, .m_capacity = 107, .m_length = 106, .m_data = "binder predicates like '< n' are not supported by choose; use a type annotation like '(h : x < n)' instead"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Nonempty"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 191, 110, 220, 210, 100, 152, 183)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__1_value),LEAN_SCALAR_PTR_LITERAL(113, 209, 180, 93, 84, 117, 67, 110)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 71, .m_data = "expected a term of the shape `∀ xs, ∃ a, p xs a` or `∀ xs, p xs ∧ q xs`"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__1;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__4_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__5 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__6_value),LEAN_SCALAR_PTR_LITERAL(12, 252, 227, 83, 88, 185, 40, 148)}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__7 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__8;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "right"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__9 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__9_value),LEAN_SCALAR_PTR_LITERAL(18, 204, 165, 192, 253, 41, 237, 145)}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__10 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__11;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Classical"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__12 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "choose"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__12_value),LEAN_SCALAR_PTR_LITERAL(40, 236, 220, 79, 38, 141, 161, 150)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13_value),LEAN_SCALAR_PTR_LITERAL(195, 210, 46, 118, 49, 85, 209, 180)}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__14 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "choose_spec"};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__15 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__12_value),LEAN_SCALAR_PTR_LITERAL(40, 236, 220, 79, 38, 141, 161, 150)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__15_value),LEAN_SCALAR_PTR_LITERAL(193, 24, 211, 87, 203, 104, 217, 124)}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__16 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__17 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 191, 110, 220, 210, 100, 152, 183)}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__18 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__18_value;
static const lean_array_object lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__19 = (const lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__19_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "type mismatch for '"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "'\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "expect list of variables"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "choose!: failed to synthesize any nonempty instances"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(48, 63, 246, 86, 229, 66, 75, 131)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value_aux_2),((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13_value),LEAN_SCALAR_PTR_LITERAL(27, 85, 110, 149, 121, 194, 145, 199)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__10_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__12_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__15_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__21;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__24_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__25_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_choose___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__28_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__29;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_choose___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose___closed__30;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "tacticChoose!___Using_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(48, 63, 246, 86, 229, 66, 75, 131)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(150, 202, 41, 65, 64, 16, 208, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "choose!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using__;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "using"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__3_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes(lean_object* v_u_10_, lean_object* v_00_u03b1_11_, lean_object* v_nonemp_12_, lean_object* v_p_13_, lean_object* v_x_14_, lean_object* v_x_15_, lean_object* v_a_16_, lean_object* v_a_17_, lean_object* v_a_18_, lean_object* v_a_19_){
_start:
{
if (lean_obj_tag(v_x_14_) == 0)
{
lean_object* v___x_21_; 
lean_dec_ref(v_p_13_);
lean_dec_ref(v_nonemp_12_);
lean_dec_ref(v_00_u03b1_11_);
lean_dec(v_u_10_);
v___x_21_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_21_, 0, v_x_15_);
return v___x_21_;
}
else
{
lean_object* v_head_22_; lean_object* v_tail_23_; lean_object* v___x_25_; uint8_t v_isShared_26_; uint8_t v_isSharedCheck_104_; 
v_head_22_ = lean_ctor_get(v_x_14_, 0);
v_tail_23_ = lean_ctor_get(v_x_14_, 1);
v_isSharedCheck_104_ = !lean_is_exclusive(v_x_14_);
if (v_isSharedCheck_104_ == 0)
{
v___x_25_ = v_x_14_;
v_isShared_26_ = v_isSharedCheck_104_;
goto v_resetjp_24_;
}
else
{
lean_inc(v_tail_23_);
lean_inc(v_head_22_);
lean_dec(v_x_14_);
v___x_25_ = lean_box(0);
v_isShared_26_ = v_isSharedCheck_104_;
goto v_resetjp_24_;
}
v_resetjp_24_:
{
lean_object* v___x_27_; 
lean_inc_ref(v_p_13_);
lean_inc_ref(v_nonemp_12_);
lean_inc_ref(v_00_u03b1_11_);
lean_inc(v_u_10_);
v___x_27_ = lp_mathlib_Mathlib_Tactic_Choose_mkSometimes(v_u_10_, v_00_u03b1_11_, v_nonemp_12_, v_p_13_, v_tail_23_, v_x_15_, v_a_16_, v_a_17_, v_a_18_, v_a_19_);
if (lean_obj_tag(v___x_27_) == 0)
{
lean_object* v_a_28_; lean_object* v_fst_29_; lean_object* v_snd_30_; lean_object* v___x_31_; 
v_a_28_ = lean_ctor_get(v___x_27_, 0);
lean_inc(v_a_28_);
lean_dec_ref_known(v___x_27_, 1);
v_fst_29_ = lean_ctor_get(v_a_28_, 0);
v_snd_30_ = lean_ctor_get(v_a_28_, 1);
lean_inc(v_a_19_);
lean_inc_ref(v_a_18_);
lean_inc(v_a_17_);
lean_inc_ref(v_a_16_);
lean_inc(v_head_22_);
v___x_31_ = lean_infer_type(v_head_22_, v_a_16_, v_a_17_, v_a_18_, v_a_19_);
if (lean_obj_tag(v___x_31_) == 0)
{
lean_object* v_a_32_; lean_object* v___x_33_; 
v_a_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc_n(v_a_32_, 2);
lean_dec_ref_known(v___x_31_, 1);
v___x_33_ = l_Lean_Meta_isProp(v_a_32_, v_a_16_, v_a_17_, v_a_18_, v_a_19_);
if (lean_obj_tag(v___x_33_) == 0)
{
lean_object* v_a_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_87_; 
v_a_34_ = lean_ctor_get(v___x_33_, 0);
v_isSharedCheck_87_ = !lean_is_exclusive(v___x_33_);
if (v_isSharedCheck_87_ == 0)
{
v___x_36_ = v___x_33_;
v_isShared_37_ = v_isSharedCheck_87_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_a_34_);
lean_dec(v___x_33_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_87_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
uint8_t v___x_38_; 
v___x_38_ = lean_unbox(v_a_34_);
if (v___x_38_ == 0)
{
lean_object* v___x_40_; 
lean_dec(v_a_34_);
lean_dec(v_a_32_);
lean_del_object(v___x_25_);
lean_dec(v_head_22_);
lean_dec_ref(v_p_13_);
lean_dec_ref(v_nonemp_12_);
lean_dec_ref(v_00_u03b1_11_);
lean_dec(v_u_10_);
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 0, v_a_28_);
v___x_40_ = v___x_36_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v_a_28_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
else
{
lean_object* v___x_43_; uint8_t v_isShared_44_; uint8_t v_isSharedCheck_84_; 
lean_inc(v_snd_30_);
lean_inc(v_fst_29_);
lean_del_object(v___x_36_);
v_isSharedCheck_84_ = !lean_is_exclusive(v_a_28_);
if (v_isSharedCheck_84_ == 0)
{
lean_object* v_unused_85_; lean_object* v_unused_86_; 
v_unused_85_ = lean_ctor_get(v_a_28_, 1);
lean_dec(v_unused_85_);
v_unused_86_ = lean_ctor_get(v_a_28_, 0);
lean_dec(v_unused_86_);
v___x_43_ = v_a_28_;
v_isShared_44_ = v_isSharedCheck_84_;
goto v_resetjp_42_;
}
else
{
lean_dec(v_a_28_);
v___x_43_ = lean_box(0);
v_isShared_44_ = v_isSharedCheck_84_;
goto v_resetjp_42_;
}
v_resetjp_42_:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; uint8_t v___x_48_; uint8_t v___x_49_; uint8_t v___x_50_; uint8_t v___x_51_; lean_object* v___x_52_; 
v___x_45_ = lean_unsigned_to_nat(1u);
v___x_46_ = lean_mk_empty_array_with_capacity(v___x_45_);
lean_inc(v_head_22_);
v___x_47_ = lean_array_push(v___x_46_, v_head_22_);
v___x_48_ = 0;
v___x_49_ = 1;
v___x_50_ = lean_unbox(v_a_34_);
v___x_51_ = lean_unbox(v_a_34_);
lean_dec(v_a_34_);
v___x_52_ = l_Lean_Meta_mkLambdaFVars(v___x_47_, v_fst_29_, v___x_48_, v___x_50_, v___x_48_, v___x_51_, v___x_49_, v_a_16_, v_a_17_, v_a_18_, v_a_19_);
lean_dec_ref(v___x_47_);
if (lean_obj_tag(v___x_52_) == 0)
{
lean_object* v_a_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_75_; 
v_a_53_ = lean_ctor_get(v___x_52_, 0);
v_isSharedCheck_75_ = !lean_is_exclusive(v___x_52_);
if (v_isSharedCheck_75_ == 0)
{
v___x_55_ = v___x_52_;
v_isShared_56_ = v_isSharedCheck_75_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_a_53_);
lean_dec(v___x_52_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_75_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_61_; 
v___x_57_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__2));
v___x_58_ = lean_box(0);
v___x_59_ = lean_box(0);
if (v_isShared_26_ == 0)
{
lean_ctor_set(v___x_25_, 1, v___x_59_);
lean_ctor_set(v___x_25_, 0, v_u_10_);
v___x_61_ = v___x_25_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_74_; 
v_reuseFailAlloc_74_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_74_, 0, v_u_10_);
lean_ctor_set(v_reuseFailAlloc_74_, 1, v___x_59_);
v___x_61_ = v_reuseFailAlloc_74_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_69_; 
lean_inc_ref(v___x_61_);
v___x_62_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_58_);
lean_ctor_set(v___x_62_, 1, v___x_61_);
v___x_63_ = l_Lean_Expr_const___override(v___x_57_, v___x_62_);
lean_inc(v_a_53_);
lean_inc_ref(v_nonemp_12_);
lean_inc_ref(v_00_u03b1_11_);
lean_inc(v_a_32_);
v___x_64_ = l_Lean_mkApp4(v___x_63_, v_a_32_, v_00_u03b1_11_, v_nonemp_12_, v_a_53_);
v___x_65_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___closed__4));
v___x_66_ = l_Lean_Expr_const___override(v___x_65_, v___x_61_);
v___x_67_ = l_Lean_mkApp7(v___x_66_, v_a_32_, v_00_u03b1_11_, v_nonemp_12_, v_p_13_, v_a_53_, v_head_22_, v_snd_30_);
if (v_isShared_44_ == 0)
{
lean_ctor_set(v___x_43_, 1, v___x_67_);
lean_ctor_set(v___x_43_, 0, v___x_64_);
v___x_69_ = v___x_43_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_73_; 
v_reuseFailAlloc_73_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_73_, 0, v___x_64_);
lean_ctor_set(v_reuseFailAlloc_73_, 1, v___x_67_);
v___x_69_ = v_reuseFailAlloc_73_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
lean_object* v___x_71_; 
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 0, v___x_69_);
v___x_71_ = v___x_55_;
goto v_reusejp_70_;
}
else
{
lean_object* v_reuseFailAlloc_72_; 
v_reuseFailAlloc_72_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_72_, 0, v___x_69_);
v___x_71_ = v_reuseFailAlloc_72_;
goto v_reusejp_70_;
}
v_reusejp_70_:
{
return v___x_71_;
}
}
}
}
}
else
{
lean_object* v_a_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_83_; 
lean_del_object(v___x_43_);
lean_dec(v_a_32_);
lean_dec(v_snd_30_);
lean_del_object(v___x_25_);
lean_dec(v_head_22_);
lean_dec_ref(v_p_13_);
lean_dec_ref(v_nonemp_12_);
lean_dec_ref(v_00_u03b1_11_);
lean_dec(v_u_10_);
v_a_76_ = lean_ctor_get(v___x_52_, 0);
v_isSharedCheck_83_ = !lean_is_exclusive(v___x_52_);
if (v_isSharedCheck_83_ == 0)
{
v___x_78_ = v___x_52_;
v_isShared_79_ = v_isSharedCheck_83_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_a_76_);
lean_dec(v___x_52_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_83_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_81_; 
if (v_isShared_79_ == 0)
{
v___x_81_ = v___x_78_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v_a_76_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_88_; lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_95_; 
lean_dec(v_a_32_);
lean_dec(v_a_28_);
lean_del_object(v___x_25_);
lean_dec(v_head_22_);
lean_dec_ref(v_p_13_);
lean_dec_ref(v_nonemp_12_);
lean_dec_ref(v_00_u03b1_11_);
lean_dec(v_u_10_);
v_a_88_ = lean_ctor_get(v___x_33_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v___x_33_);
if (v_isSharedCheck_95_ == 0)
{
v___x_90_ = v___x_33_;
v_isShared_91_ = v_isSharedCheck_95_;
goto v_resetjp_89_;
}
else
{
lean_inc(v_a_88_);
lean_dec(v___x_33_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_95_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v___x_93_; 
if (v_isShared_91_ == 0)
{
v___x_93_ = v___x_90_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v_a_88_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
else
{
lean_object* v_a_96_; lean_object* v___x_98_; uint8_t v_isShared_99_; uint8_t v_isSharedCheck_103_; 
lean_dec(v_a_28_);
lean_del_object(v___x_25_);
lean_dec(v_head_22_);
lean_dec_ref(v_p_13_);
lean_dec_ref(v_nonemp_12_);
lean_dec_ref(v_00_u03b1_11_);
lean_dec(v_u_10_);
v_a_96_ = lean_ctor_get(v___x_31_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_31_);
if (v_isSharedCheck_103_ == 0)
{
v___x_98_ = v___x_31_;
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
else
{
lean_inc(v_a_96_);
lean_dec(v___x_31_);
v___x_98_ = lean_box(0);
v_isShared_99_ = v_isSharedCheck_103_;
goto v_resetjp_97_;
}
v_resetjp_97_:
{
lean_object* v___x_101_; 
if (v_isShared_99_ == 0)
{
v___x_101_ = v___x_98_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_a_96_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
}
else
{
lean_del_object(v___x_25_);
lean_dec(v_head_22_);
lean_dec_ref(v_p_13_);
lean_dec_ref(v_nonemp_12_);
lean_dec_ref(v_00_u03b1_11_);
lean_dec(v_u_10_);
return v___x_27_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkSometimes___boxed(lean_object* v_u_105_, lean_object* v_00_u03b1_106_, lean_object* v_nonemp_107_, lean_object* v_p_108_, lean_object* v_x_109_, lean_object* v_x_110_, lean_object* v_a_111_, lean_object* v_a_112_, lean_object* v_a_113_, lean_object* v_a_114_, lean_object* v_a_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Mathlib_Tactic_Choose_mkSometimes(v_u_105_, v_00_u03b1_106_, v_nonemp_107_, v_p_108_, v_x_109_, v_x_110_, v_a_111_, v_a_112_, v_a_113_, v_a_114_);
lean_dec(v_a_114_);
lean_dec_ref(v_a_113_);
lean_dec(v_a_112_);
lean_dec_ref(v_a_111_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mk__sometimes(lean_object* v_u_117_, lean_object* v_00_u03b1_118_, lean_object* v_nonemp_119_, lean_object* v_p_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lp_mathlib_Mathlib_Tactic_Choose_mkSometimes(v_u_117_, v_00_u03b1_118_, v_nonemp_119_, v_p_120_, v_a_121_, v_a_122_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mk__sometimes___boxed(lean_object* v_u_129_, lean_object* v_00_u03b1_130_, lean_object* v_nonemp_131_, lean_object* v_p_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_, lean_object* v_a_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_Mathlib_Tactic_Choose_mk__sometimes(v_u_129_, v_00_u03b1_130_, v_nonemp_131_, v_p_132_, v_a_133_, v_a_134_, v_a_135_, v_a_136_, v_a_137_, v_a_138_);
lean_dec(v_a_138_);
lean_dec_ref(v_a_137_);
lean_dec(v_a_136_);
lean_dec_ref(v_a_135_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorIdx(lean_object* v_x_141_){
_start:
{
if (lean_obj_tag(v_x_141_) == 0)
{
lean_object* v___x_142_; 
v___x_142_ = lean_unsigned_to_nat(0u);
return v___x_142_;
}
else
{
lean_object* v___x_143_; 
v___x_143_ = lean_unsigned_to_nat(1u);
return v___x_143_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorIdx___boxed(lean_object* v_x_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorIdx(v_x_144_);
lean_dec(v_x_144_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(lean_object* v_t_146_, lean_object* v_k_147_){
_start:
{
if (lean_obj_tag(v_t_146_) == 0)
{
return v_k_147_;
}
else
{
lean_object* v_ts_148_; lean_object* v___x_149_; 
v_ts_148_ = lean_ctor_get(v_t_146_, 0);
lean_inc(v_ts_148_);
lean_dec_ref_known(v_t_146_, 1);
v___x_149_ = lean_apply_1(v_k_147_, v_ts_148_);
return v___x_149_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim(lean_object* v_motive_150_, lean_object* v_ctorIdx_151_, lean_object* v_t_152_, lean_object* v_h_153_, lean_object* v_k_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(v_t_152_, v_k_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___boxed(lean_object* v_motive_156_, lean_object* v_ctorIdx_157_, lean_object* v_t_158_, lean_object* v_h_159_, lean_object* v_k_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim(v_motive_156_, v_ctorIdx_157_, v_t_158_, v_h_159_, v_k_160_);
lean_dec(v_ctorIdx_157_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_success_elim___redArg(lean_object* v_t_162_, lean_object* v_success_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(v_t_162_, v_success_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_success_elim(lean_object* v_motive_165_, lean_object* v_t_166_, lean_object* v_h_167_, lean_object* v_success_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(v_t_166_, v_success_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_failure_elim___redArg(lean_object* v_t_170_, lean_object* v_failure_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(v_t_170_, v_failure_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_failure_elim(lean_object* v_motive_173_, lean_object* v_t_174_, lean_object* v_h_175_, lean_object* v_failure_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_ctorElim___redArg(v_t_174_, v_failure_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_merge(lean_object* v_x_178_, lean_object* v_x_179_){
_start:
{
if (lean_obj_tag(v_x_178_) == 0)
{
lean_dec(v_x_179_);
return v_x_178_;
}
else
{
if (lean_obj_tag(v_x_179_) == 0)
{
lean_dec_ref_known(v_x_178_, 1);
return v_x_179_;
}
else
{
lean_object* v_ts_180_; lean_object* v_ts_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_189_; 
v_ts_180_ = lean_ctor_get(v_x_178_, 0);
lean_inc(v_ts_180_);
lean_dec_ref_known(v_x_178_, 1);
v_ts_181_ = lean_ctor_get(v_x_179_, 0);
v_isSharedCheck_189_ = !lean_is_exclusive(v_x_179_);
if (v_isSharedCheck_189_ == 0)
{
v___x_183_ = v_x_179_;
v_isShared_184_ = v_isSharedCheck_189_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_ts_181_);
lean_dec(v_x_179_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_189_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v___x_185_; lean_object* v___x_187_; 
v___x_185_ = l_List_appendTR___redArg(v_ts_180_, v_ts_181_);
if (v_isShared_184_ == 0)
{
lean_ctor_set(v___x_183_, 0, v___x_185_);
v___x_187_ = v___x_183_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v___x_185_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom(lean_object* v_orig_193_, lean_object* v_base_194_, lean_object* v_a_195_, lean_object* v_a_196_){
_start:
{
lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_199_ = lean_name_eq(v_orig_193_, v___x_198_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; 
lean_dec(v_base_194_);
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v_orig_193_);
return v___x_200_;
}
else
{
lean_object* v___x_201_; 
lean_dec(v_orig_193_);
v___x_201_ = l_Lean_Core_mkFreshUserName(v_base_194_, v_a_195_, v_a_196_);
return v___x_201_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___boxed(lean_object* v_orig_202_, lean_object* v_base_203_, lean_object* v_a_204_, lean_object* v_a_205_, lean_object* v_a_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom(v_orig_202_, v_base_203_, v_a_204_, v_a_205_);
lean_dec(v_a_205_);
lean_dec_ref(v_a_204_);
return v_res_207_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__7(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_226_ = lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized;
v___x_227_ = l_Lean_binderIdent;
v___x_228_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__6));
v___x_229_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v___x_227_);
lean_ctor_set(v___x_229_, 2, v___x_226_);
return v___x_229_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__8(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_230_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__7, &lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__7);
v___x_231_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4));
v___x_232_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__0));
v___x_233_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
lean_ctor_set(v___x_233_, 1, v___x_231_);
lean_ctor_set(v___x_233_, 2, v___x_230_);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_chooseBinder(void){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__8, &lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__8);
return v___x_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent(lean_object* v_id_243_){
_start:
{
lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_244_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2));
lean_inc(v_id_243_);
v___x_245_ = l_Lean_Syntax_isOfKind(v_id_243_, v___x_244_);
if (v___x_245_ == 0)
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_247_ = lean_box(0);
v___x_248_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_248_, 0, v_id_243_);
lean_ctor_set(v___x_248_, 1, v___x_246_);
lean_ctor_set(v___x_248_, 2, v___x_247_);
return v___x_248_;
}
else
{
lean_object* v___x_249_; lean_object* v_h_250_; lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_249_ = lean_unsigned_to_nat(0u);
v_h_250_ = l_Lean_Syntax_getArg(v_id_243_, v___x_249_);
v___x_251_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__4));
lean_inc(v_h_250_);
v___x_252_ = l_Lean_Syntax_isOfKind(v_h_250_, v___x_251_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
lean_dec(v_h_250_);
v___x_253_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_254_ = lean_box(0);
v___x_255_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_255_, 0, v_id_243_);
lean_ctor_set(v___x_255_, 1, v___x_253_);
lean_ctor_set(v___x_255_, 2, v___x_254_);
return v___x_255_;
}
else
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
lean_dec(v_id_243_);
v___x_256_ = l_Lean_TSyntax_getId(v_h_250_);
v___x_257_ = lean_box(0);
v___x_258_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_258_, 0, v_h_250_);
lean_ctor_set(v___x_258_, 1, v___x_256_);
lean_ctor_set(v___x_258_, 2, v___x_257_);
return v___x_258_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1(lean_object* v_msgData_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v___x_265_; lean_object* v_env_266_; lean_object* v___x_267_; lean_object* v_mctx_268_; lean_object* v_lctx_269_; lean_object* v_options_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_265_ = lean_st_ref_get(v___y_263_);
v_env_266_ = lean_ctor_get(v___x_265_, 0);
lean_inc_ref(v_env_266_);
lean_dec(v___x_265_);
v___x_267_ = lean_st_ref_get(v___y_261_);
v_mctx_268_ = lean_ctor_get(v___x_267_, 0);
lean_inc_ref(v_mctx_268_);
lean_dec(v___x_267_);
v_lctx_269_ = lean_ctor_get(v___y_260_, 2);
v_options_270_ = lean_ctor_get(v___y_262_, 2);
lean_inc_ref(v_options_270_);
lean_inc_ref(v_lctx_269_);
v___x_271_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_271_, 0, v_env_266_);
lean_ctor_set(v___x_271_, 1, v_mctx_268_);
lean_ctor_set(v___x_271_, 2, v_lctx_269_);
lean_ctor_set(v___x_271_, 3, v_options_270_);
v___x_272_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
lean_ctor_set(v___x_272_, 1, v_msgData_259_);
v___x_273_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_273_, 0, v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1___boxed(lean_object* v_msgData_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1(v_msgData_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_);
lean_dec(v___y_278_);
lean_dec_ref(v___y_277_);
lean_dec(v___y_276_);
lean_dec_ref(v___y_275_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg(lean_object* v_msg_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_){
_start:
{
lean_object* v_ref_287_; lean_object* v___x_288_; lean_object* v_a_289_; lean_object* v___x_291_; uint8_t v_isShared_292_; uint8_t v_isSharedCheck_297_; 
v_ref_287_ = lean_ctor_get(v___y_284_, 5);
v___x_288_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1(v_msg_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_);
v_a_289_ = lean_ctor_get(v___x_288_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v___x_288_);
if (v_isSharedCheck_297_ == 0)
{
v___x_291_ = v___x_288_;
v_isShared_292_ = v_isSharedCheck_297_;
goto v_resetjp_290_;
}
else
{
lean_inc(v_a_289_);
lean_dec(v___x_288_);
v___x_291_ = lean_box(0);
v_isShared_292_ = v_isSharedCheck_297_;
goto v_resetjp_290_;
}
v_resetjp_290_:
{
lean_object* v___x_293_; lean_object* v___x_295_; 
lean_inc(v_ref_287_);
v___x_293_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_293_, 0, v_ref_287_);
lean_ctor_set(v___x_293_, 1, v_a_289_);
if (v_isShared_292_ == 0)
{
lean_ctor_set_tag(v___x_291_, 1);
lean_ctor_set(v___x_291_, 0, v___x_293_);
v___x_295_ = v___x_291_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v___x_293_);
v___x_295_ = v_reuseFailAlloc_296_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
return v___x_295_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg___boxed(lean_object* v_msg_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg(v_msg_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg(lean_object* v_ref_305_, lean_object* v_msg_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_){
_start:
{
lean_object* v_fileName_312_; lean_object* v_fileMap_313_; lean_object* v_options_314_; lean_object* v_currRecDepth_315_; lean_object* v_maxRecDepth_316_; lean_object* v_ref_317_; lean_object* v_currNamespace_318_; lean_object* v_openDecls_319_; lean_object* v_initHeartbeats_320_; lean_object* v_maxHeartbeats_321_; lean_object* v_quotContext_322_; lean_object* v_currMacroScope_323_; uint8_t v_diag_324_; lean_object* v_cancelTk_x3f_325_; uint8_t v_suppressElabErrors_326_; lean_object* v_inheritedTraceOptions_327_; lean_object* v_ref_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v_fileName_312_ = lean_ctor_get(v___y_309_, 0);
v_fileMap_313_ = lean_ctor_get(v___y_309_, 1);
v_options_314_ = lean_ctor_get(v___y_309_, 2);
v_currRecDepth_315_ = lean_ctor_get(v___y_309_, 3);
v_maxRecDepth_316_ = lean_ctor_get(v___y_309_, 4);
v_ref_317_ = lean_ctor_get(v___y_309_, 5);
v_currNamespace_318_ = lean_ctor_get(v___y_309_, 6);
v_openDecls_319_ = lean_ctor_get(v___y_309_, 7);
v_initHeartbeats_320_ = lean_ctor_get(v___y_309_, 8);
v_maxHeartbeats_321_ = lean_ctor_get(v___y_309_, 9);
v_quotContext_322_ = lean_ctor_get(v___y_309_, 10);
v_currMacroScope_323_ = lean_ctor_get(v___y_309_, 11);
v_diag_324_ = lean_ctor_get_uint8(v___y_309_, sizeof(void*)*14);
v_cancelTk_x3f_325_ = lean_ctor_get(v___y_309_, 12);
v_suppressElabErrors_326_ = lean_ctor_get_uint8(v___y_309_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_327_ = lean_ctor_get(v___y_309_, 13);
v_ref_328_ = l_Lean_replaceRef(v_ref_305_, v_ref_317_);
lean_inc_ref(v_inheritedTraceOptions_327_);
lean_inc(v_cancelTk_x3f_325_);
lean_inc(v_currMacroScope_323_);
lean_inc(v_quotContext_322_);
lean_inc(v_maxHeartbeats_321_);
lean_inc(v_initHeartbeats_320_);
lean_inc(v_openDecls_319_);
lean_inc(v_currNamespace_318_);
lean_inc(v_maxRecDepth_316_);
lean_inc(v_currRecDepth_315_);
lean_inc_ref(v_options_314_);
lean_inc_ref(v_fileMap_313_);
lean_inc_ref(v_fileName_312_);
v___x_329_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_329_, 0, v_fileName_312_);
lean_ctor_set(v___x_329_, 1, v_fileMap_313_);
lean_ctor_set(v___x_329_, 2, v_options_314_);
lean_ctor_set(v___x_329_, 3, v_currRecDepth_315_);
lean_ctor_set(v___x_329_, 4, v_maxRecDepth_316_);
lean_ctor_set(v___x_329_, 5, v_ref_328_);
lean_ctor_set(v___x_329_, 6, v_currNamespace_318_);
lean_ctor_set(v___x_329_, 7, v_openDecls_319_);
lean_ctor_set(v___x_329_, 8, v_initHeartbeats_320_);
lean_ctor_set(v___x_329_, 9, v_maxHeartbeats_321_);
lean_ctor_set(v___x_329_, 10, v_quotContext_322_);
lean_ctor_set(v___x_329_, 11, v_currMacroScope_323_);
lean_ctor_set(v___x_329_, 12, v_cancelTk_x3f_325_);
lean_ctor_set(v___x_329_, 13, v_inheritedTraceOptions_327_);
lean_ctor_set_uint8(v___x_329_, sizeof(void*)*14, v_diag_324_);
lean_ctor_set_uint8(v___x_329_, sizeof(void*)*14 + 1, v_suppressElabErrors_326_);
v___x_330_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg(v_msg_306_, v___y_307_, v___y_308_, v___x_329_, v___y_310_);
lean_dec_ref_known(v___x_329_, 14);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg___boxed(lean_object* v_ref_331_, lean_object* v_msg_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg(v_ref_331_, v_msg_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
lean_dec(v___y_336_);
lean_dec_ref(v___y_335_);
lean_dec(v___y_334_);
lean_dec_ref(v___y_333_);
lean_dec(v_ref_331_);
return v_res_338_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__9(void){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_355_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__8));
v___x_356_ = l_Lean_stringToMessageData(v___x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg(lean_object* v_stx_357_, lean_object* v_a_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_){
_start:
{
lean_object* v___x_363_; uint8_t v___x_364_; 
v___x_363_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4));
lean_inc(v_stx_357_);
v___x_364_ = l_Lean_Syntax_isOfKind(v_stx_357_, v___x_363_);
if (v___x_364_ == 0)
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_365_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_366_ = lean_box(0);
v___x_367_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_367_, 0, v_stx_357_);
lean_ctor_set(v___x_367_, 1, v___x_365_);
lean_ctor_set(v___x_367_, 2, v___x_366_);
v___x_368_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_368_, 0, v___x_367_);
return v___x_368_;
}
else
{
lean_object* v___x_369_; lean_object* v_id_370_; lean_object* v___x_371_; uint8_t v___x_372_; 
v___x_369_ = lean_unsigned_to_nat(0u);
v_id_370_ = l_Lean_Syntax_getArg(v_stx_357_, v___x_369_);
v___x_371_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent___closed__2));
lean_inc(v_id_370_);
v___x_372_ = l_Lean_Syntax_isOfKind(v_id_370_, v___x_371_);
if (v___x_372_ == 0)
{
lean_object* v___x_373_; uint8_t v___x_374_; 
v___x_373_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__3));
lean_inc(v_id_370_);
v___x_374_ = l_Lean_Syntax_isOfKind(v_id_370_, v___x_373_);
if (v___x_374_ == 0)
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
lean_dec(v_id_370_);
v___x_375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_376_ = lean_box(0);
v___x_377_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_377_, 0, v_stx_357_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
lean_ctor_set(v___x_377_, 2, v___x_376_);
v___x_378_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
return v___x_378_;
}
else
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; uint8_t v___x_382_; 
v___x_379_ = lean_unsigned_to_nat(1u);
v___x_380_ = l_Lean_Syntax_getArg(v_id_370_, v___x_379_);
lean_dec(v_id_370_);
v___x_381_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__5));
lean_inc(v___x_380_);
v___x_382_ = l_Lean_Syntax_isOfKind(v___x_380_, v___x_381_);
if (v___x_382_ == 0)
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
lean_dec(v___x_380_);
v___x_383_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_384_ = lean_box(0);
v___x_385_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_385_, 0, v_stx_357_);
lean_ctor_set(v___x_385_, 1, v___x_383_);
lean_ctor_set(v___x_385_, 2, v___x_384_);
v___x_386_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
return v___x_386_;
}
else
{
lean_object* v_id_387_; uint8_t v___x_388_; 
v_id_387_ = l_Lean_Syntax_getArg(v___x_380_, v___x_369_);
lean_inc(v_id_387_);
v___x_388_ = l_Lean_Syntax_isOfKind(v_id_387_, v___x_371_);
if (v___x_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec(v_id_387_);
lean_dec(v___x_380_);
v___x_389_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_390_ = lean_box(0);
v___x_391_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_391_, 0, v_stx_357_);
lean_ctor_set(v___x_391_, 1, v___x_389_);
lean_ctor_set(v___x_391_, 2, v___x_390_);
v___x_392_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
return v___x_392_;
}
else
{
lean_object* v___x_393_; uint8_t v___x_394_; 
v___x_393_ = l_Lean_Syntax_getArg(v___x_380_, v___x_379_);
lean_dec(v___x_380_);
lean_inc(v___x_393_);
v___x_394_ = l_Lean_Syntax_matchesNull(v___x_393_, v___x_379_);
if (v___x_394_ == 0)
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
lean_dec(v___x_393_);
lean_dec(v_id_387_);
v___x_395_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_396_ = lean_box(0);
v___x_397_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_397_, 0, v_stx_357_);
lean_ctor_set(v___x_397_, 1, v___x_395_);
lean_ctor_set(v___x_397_, 2, v___x_396_);
v___x_398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
return v___x_398_;
}
else
{
lean_object* v___x_399_; lean_object* v___x_400_; uint8_t v___x_401_; 
lean_dec(v_stx_357_);
v___x_399_ = l_Lean_Syntax_getArg(v___x_393_, v___x_369_);
lean_dec(v___x_393_);
v___x_400_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__7));
lean_inc(v___x_399_);
v___x_401_ = l_Lean_Syntax_isOfKind(v___x_399_, v___x_400_);
if (v___x_401_ == 0)
{
lean_object* v___x_402_; lean_object* v___x_403_; 
lean_dec(v_id_387_);
v___x_402_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__9, &lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___closed__9);
v___x_403_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg(v___x_399_, v___x_402_, v_a_358_, v_a_359_, v_a_360_, v_a_361_);
lean_dec(v___x_399_);
return v___x_403_;
}
else
{
lean_object* v___x_404_; lean_object* v_ref_405_; lean_object* v_name_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_416_; 
v___x_404_ = lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent(v_id_387_);
v_ref_405_ = lean_ctor_get(v___x_404_, 0);
v_name_406_ = lean_ctor_get(v___x_404_, 1);
v_isSharedCheck_416_ = !lean_is_exclusive(v___x_404_);
if (v_isSharedCheck_416_ == 0)
{
lean_object* v_unused_417_; 
v_unused_417_ = lean_ctor_get(v___x_404_, 2);
lean_dec(v_unused_417_);
v___x_408_ = v___x_404_;
v_isShared_409_ = v_isSharedCheck_416_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_name_406_);
lean_inc(v_ref_405_);
lean_dec(v___x_404_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_416_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v_ty_410_; lean_object* v___x_411_; lean_object* v___x_413_; 
v_ty_410_ = l_Lean_Syntax_getArg(v___x_399_, v___x_379_);
lean_dec(v___x_399_);
v___x_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_411_, 0, v_ty_410_);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 2, v___x_411_);
v___x_413_ = v___x_408_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_ref_405_);
lean_ctor_set(v_reuseFailAlloc_415_, 1, v_name_406_);
lean_ctor_set(v_reuseFailAlloc_415_, 2, v___x_411_);
v___x_413_ = v_reuseFailAlloc_415_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
lean_object* v___x_414_; 
v___x_414_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_414_, 0, v___x_413_);
return v___x_414_;
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
lean_object* v___x_418_; lean_object* v___x_419_; 
lean_dec(v_stx_357_);
v___x_418_ = lp_mathlib___private_Mathlib_Tactic_Choose_0__Mathlib_Tactic_Choose_parseChooseArg_parseBinderIdent(v_id_370_);
v___x_419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
return v___x_419_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg___boxed(lean_object* v_stx_420_, lean_object* v_a_421_, lean_object* v_a_422_, lean_object* v_a_423_, lean_object* v_a_424_, lean_object* v_a_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg(v_stx_420_, v_a_421_, v_a_422_, v_a_423_, v_a_424_);
lean_dec(v_a_424_);
lean_dec_ref(v_a_423_);
lean_dec(v_a_422_);
lean_dec_ref(v_a_421_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0(lean_object* v_00_u03b1_427_, lean_object* v_ref_428_, lean_object* v_msg_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_, lean_object* v___y_433_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___redArg(v_ref_428_, v_msg_429_, v___y_430_, v___y_431_, v___y_432_, v___y_433_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0___boxed(lean_object* v_00_u03b1_436_, lean_object* v_ref_437_, lean_object* v_msg_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0(v_00_u03b1_436_, v_ref_437_, v_msg_438_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v_ref_437_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0(lean_object* v_00_u03b1_445_, lean_object* v_msg_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg(v_msg_446_, v___y_447_, v___y_448_, v___y_449_, v___y_450_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___boxed(lean_object* v_00_u03b1_453_, lean_object* v_msg_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0(v_00_u03b1_453_, v_msg_454_, v___y_455_, v___y_456_, v___y_457_, v___y_458_);
lean_dec(v___y_458_);
lean_dec_ref(v___y_457_);
lean_dec(v___y_456_);
lean_dec_ref(v___y_455_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg(lean_object* v_e_461_, lean_object* v___y_462_){
_start:
{
uint8_t v___x_464_; 
v___x_464_ = l_Lean_Expr_hasMVar(v_e_461_);
if (v___x_464_ == 0)
{
lean_object* v___x_465_; 
v___x_465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_465_, 0, v_e_461_);
return v___x_465_;
}
else
{
lean_object* v___x_466_; lean_object* v_mctx_467_; lean_object* v___x_468_; lean_object* v_fst_469_; lean_object* v_snd_470_; lean_object* v___x_471_; lean_object* v_cache_472_; lean_object* v_zetaDeltaFVarIds_473_; lean_object* v_postponed_474_; lean_object* v_diag_475_; lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_484_; 
v___x_466_ = lean_st_ref_get(v___y_462_);
v_mctx_467_ = lean_ctor_get(v___x_466_, 0);
lean_inc_ref(v_mctx_467_);
lean_dec(v___x_466_);
v___x_468_ = l_Lean_instantiateMVarsCore(v_mctx_467_, v_e_461_);
v_fst_469_ = lean_ctor_get(v___x_468_, 0);
lean_inc(v_fst_469_);
v_snd_470_ = lean_ctor_get(v___x_468_, 1);
lean_inc(v_snd_470_);
lean_dec_ref(v___x_468_);
v___x_471_ = lean_st_ref_take(v___y_462_);
v_cache_472_ = lean_ctor_get(v___x_471_, 1);
v_zetaDeltaFVarIds_473_ = lean_ctor_get(v___x_471_, 2);
v_postponed_474_ = lean_ctor_get(v___x_471_, 3);
v_diag_475_ = lean_ctor_get(v___x_471_, 4);
v_isSharedCheck_484_ = !lean_is_exclusive(v___x_471_);
if (v_isSharedCheck_484_ == 0)
{
lean_object* v_unused_485_; 
v_unused_485_ = lean_ctor_get(v___x_471_, 0);
lean_dec(v_unused_485_);
v___x_477_ = v___x_471_;
v_isShared_478_ = v_isSharedCheck_484_;
goto v_resetjp_476_;
}
else
{
lean_inc(v_diag_475_);
lean_inc(v_postponed_474_);
lean_inc(v_zetaDeltaFVarIds_473_);
lean_inc(v_cache_472_);
lean_dec(v___x_471_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_484_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
lean_object* v___x_480_; 
if (v_isShared_478_ == 0)
{
lean_ctor_set(v___x_477_, 0, v_snd_470_);
v___x_480_ = v___x_477_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v_snd_470_);
lean_ctor_set(v_reuseFailAlloc_483_, 1, v_cache_472_);
lean_ctor_set(v_reuseFailAlloc_483_, 2, v_zetaDeltaFVarIds_473_);
lean_ctor_set(v_reuseFailAlloc_483_, 3, v_postponed_474_);
lean_ctor_set(v_reuseFailAlloc_483_, 4, v_diag_475_);
v___x_480_ = v_reuseFailAlloc_483_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_481_ = lean_st_ref_set(v___y_462_, v___x_480_);
v___x_482_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_482_, 0, v_fst_469_);
return v___x_482_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg___boxed(lean_object* v_e_486_, lean_object* v___y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v_res_489_; 
v_res_489_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg(v_e_486_, v___y_487_);
lean_dec(v___y_487_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0(lean_object* v_e_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg(v_e_490_, v___y_492_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___boxed(lean_object* v_e_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0(v_e_497_, v___y_498_, v___y_499_, v___y_500_, v___y_501_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
lean_dec(v___y_499_);
lean_dec_ref(v___y_498_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(lean_object* v_mvarId_504_, lean_object* v_x_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_504_, v_x_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
if (lean_obj_tag(v___x_511_) == 0)
{
lean_object* v_a_512_; lean_object* v___x_514_; uint8_t v_isShared_515_; uint8_t v_isSharedCheck_519_; 
v_a_512_ = lean_ctor_get(v___x_511_, 0);
v_isSharedCheck_519_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_519_ == 0)
{
v___x_514_ = v___x_511_;
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
else
{
lean_inc(v_a_512_);
lean_dec(v___x_511_);
v___x_514_ = lean_box(0);
v_isShared_515_ = v_isSharedCheck_519_;
goto v_resetjp_513_;
}
v_resetjp_513_:
{
lean_object* v___x_517_; 
if (v_isShared_515_ == 0)
{
v___x_517_ = v___x_514_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_a_512_);
v___x_517_ = v_reuseFailAlloc_518_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
return v___x_517_;
}
}
}
else
{
lean_object* v_a_520_; lean_object* v___x_522_; uint8_t v_isShared_523_; uint8_t v_isSharedCheck_527_; 
v_a_520_ = lean_ctor_get(v___x_511_, 0);
v_isSharedCheck_527_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_527_ == 0)
{
v___x_522_ = v___x_511_;
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
else
{
lean_inc(v_a_520_);
lean_dec(v___x_511_);
v___x_522_ = lean_box(0);
v_isShared_523_ = v_isSharedCheck_527_;
goto v_resetjp_521_;
}
v_resetjp_521_:
{
lean_object* v___x_525_; 
if (v_isShared_523_ == 0)
{
v___x_525_ = v___x_522_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v_a_520_);
v___x_525_ = v_reuseFailAlloc_526_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
return v___x_525_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg___boxed(lean_object* v_mvarId_528_, lean_object* v_x_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(v_mvarId_528_, v_x_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_);
lean_dec(v___y_533_);
lean_dec_ref(v___y_532_);
lean_dec(v___y_531_);
lean_dec_ref(v___y_530_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3(lean_object* v_00_u03b1_536_, lean_object* v_mvarId_537_, lean_object* v_x_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(v_mvarId_537_, v_x_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___boxed(lean_object* v_00_u03b1_545_, lean_object* v_mvarId_546_, lean_object* v_x_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_){
_start:
{
lean_object* v_res_553_; 
v_res_553_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3(v_00_u03b1_545_, v_mvarId_546_, v_x_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_);
lean_dec(v___y_551_);
lean_dec_ref(v___y_550_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
return v_res_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___lam__0(lean_object* v_k_554_, lean_object* v_b_555_, lean_object* v_c_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_){
_start:
{
lean_object* v___x_562_; 
lean_inc(v___y_560_);
lean_inc_ref(v___y_559_);
lean_inc(v___y_558_);
lean_inc_ref(v___y_557_);
v___x_562_ = lean_apply_7(v_k_554_, v_b_555_, v_c_556_, v___y_557_, v___y_558_, v___y_559_, v___y_560_, lean_box(0));
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___lam__0___boxed(lean_object* v_k_563_, lean_object* v_b_564_, lean_object* v_c_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___lam__0(v_k_563_, v_b_564_, v_c_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_);
lean_dec(v___y_569_);
lean_dec_ref(v___y_568_);
lean_dec(v___y_567_);
lean_dec_ref(v___y_566_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg(lean_object* v_type_572_, lean_object* v_k_573_, uint8_t v_cleanupAnnotations_574_, uint8_t v_whnfType_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_){
_start:
{
lean_object* v___f_581_; lean_object* v___x_582_; 
v___f_581_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_581_, 0, v_k_573_);
v___x_582_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_572_, v___f_581_, v_cleanupAnnotations_574_, v_whnfType_575_, v___y_576_, v___y_577_, v___y_578_, v___y_579_);
if (lean_obj_tag(v___x_582_) == 0)
{
lean_object* v_a_583_; lean_object* v___x_585_; uint8_t v_isShared_586_; uint8_t v_isSharedCheck_590_; 
v_a_583_ = lean_ctor_get(v___x_582_, 0);
v_isSharedCheck_590_ = !lean_is_exclusive(v___x_582_);
if (v_isSharedCheck_590_ == 0)
{
v___x_585_ = v___x_582_;
v_isShared_586_ = v_isSharedCheck_590_;
goto v_resetjp_584_;
}
else
{
lean_inc(v_a_583_);
lean_dec(v___x_582_);
v___x_585_ = lean_box(0);
v_isShared_586_ = v_isSharedCheck_590_;
goto v_resetjp_584_;
}
v_resetjp_584_:
{
lean_object* v___x_588_; 
if (v_isShared_586_ == 0)
{
v___x_588_ = v___x_585_;
goto v_reusejp_587_;
}
else
{
lean_object* v_reuseFailAlloc_589_; 
v_reuseFailAlloc_589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_589_, 0, v_a_583_);
v___x_588_ = v_reuseFailAlloc_589_;
goto v_reusejp_587_;
}
v_reusejp_587_:
{
return v___x_588_;
}
}
}
else
{
lean_object* v_a_591_; lean_object* v___x_593_; uint8_t v_isShared_594_; uint8_t v_isSharedCheck_598_; 
v_a_591_ = lean_ctor_get(v___x_582_, 0);
v_isSharedCheck_598_ = !lean_is_exclusive(v___x_582_);
if (v_isSharedCheck_598_ == 0)
{
v___x_593_ = v___x_582_;
v_isShared_594_ = v_isSharedCheck_598_;
goto v_resetjp_592_;
}
else
{
lean_inc(v_a_591_);
lean_dec(v___x_582_);
v___x_593_ = lean_box(0);
v_isShared_594_ = v_isSharedCheck_598_;
goto v_resetjp_592_;
}
v_resetjp_592_:
{
lean_object* v___x_596_; 
if (v_isShared_594_ == 0)
{
v___x_596_ = v___x_593_;
goto v_reusejp_595_;
}
else
{
lean_object* v_reuseFailAlloc_597_; 
v_reuseFailAlloc_597_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_597_, 0, v_a_591_);
v___x_596_ = v_reuseFailAlloc_597_;
goto v_reusejp_595_;
}
v_reusejp_595_:
{
return v___x_596_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg___boxed(lean_object* v_type_599_, lean_object* v_k_600_, lean_object* v_cleanupAnnotations_601_, lean_object* v_whnfType_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_608_; uint8_t v_whnfType_boxed_609_; lean_object* v_res_610_; 
v_cleanupAnnotations_boxed_608_ = lean_unbox(v_cleanupAnnotations_601_);
v_whnfType_boxed_609_ = lean_unbox(v_whnfType_602_);
v_res_610_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg(v_type_599_, v_k_600_, v_cleanupAnnotations_boxed_608_, v_whnfType_boxed_609_, v___y_603_, v___y_604_, v___y_605_, v___y_606_);
lean_dec(v___y_606_);
lean_dec_ref(v___y_605_);
lean_dec(v___y_604_);
lean_dec_ref(v___y_603_);
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7(lean_object* v_00_u03b1_611_, lean_object* v_type_612_, lean_object* v_k_613_, uint8_t v_cleanupAnnotations_614_, uint8_t v_whnfType_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v___x_621_; 
v___x_621_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg(v_type_612_, v_k_613_, v_cleanupAnnotations_614_, v_whnfType_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_);
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___boxed(lean_object* v_00_u03b1_622_, lean_object* v_type_623_, lean_object* v_k_624_, lean_object* v_cleanupAnnotations_625_, lean_object* v_whnfType_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_632_; uint8_t v_whnfType_boxed_633_; lean_object* v_res_634_; 
v_cleanupAnnotations_boxed_632_ = lean_unbox(v_cleanupAnnotations_625_);
v_whnfType_boxed_633_ = lean_unbox(v_whnfType_626_);
v_res_634_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7(v_00_u03b1_622_, v_type_623_, v_k_624_, v_cleanupAnnotations_boxed_632_, v_whnfType_boxed_633_, v___y_627_, v___y_628_, v___y_629_, v___y_630_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec_ref(v___y_627_);
return v_res_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10_spec__11___redArg(lean_object* v_x_635_, lean_object* v_x_636_, lean_object* v_x_637_, lean_object* v_x_638_){
_start:
{
lean_object* v_ks_639_; lean_object* v_vs_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_664_; 
v_ks_639_ = lean_ctor_get(v_x_635_, 0);
v_vs_640_ = lean_ctor_get(v_x_635_, 1);
v_isSharedCheck_664_ = !lean_is_exclusive(v_x_635_);
if (v_isSharedCheck_664_ == 0)
{
v___x_642_ = v_x_635_;
v_isShared_643_ = v_isSharedCheck_664_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_vs_640_);
lean_inc(v_ks_639_);
lean_dec(v_x_635_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_664_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_644_; uint8_t v___x_645_; 
v___x_644_ = lean_array_get_size(v_ks_639_);
v___x_645_ = lean_nat_dec_lt(v_x_636_, v___x_644_);
if (v___x_645_ == 0)
{
lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_649_; 
lean_dec(v_x_636_);
v___x_646_ = lean_array_push(v_ks_639_, v_x_637_);
v___x_647_ = lean_array_push(v_vs_640_, v_x_638_);
if (v_isShared_643_ == 0)
{
lean_ctor_set(v___x_642_, 1, v___x_647_);
lean_ctor_set(v___x_642_, 0, v___x_646_);
v___x_649_ = v___x_642_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_646_);
lean_ctor_set(v_reuseFailAlloc_650_, 1, v___x_647_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
else
{
lean_object* v_k_x27_651_; uint8_t v___x_652_; 
v_k_x27_651_ = lean_array_fget_borrowed(v_ks_639_, v_x_636_);
v___x_652_ = l_Lean_instBEqMVarId_beq(v_x_637_, v_k_x27_651_);
if (v___x_652_ == 0)
{
lean_object* v___x_654_; 
if (v_isShared_643_ == 0)
{
v___x_654_ = v___x_642_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_658_; 
v_reuseFailAlloc_658_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_658_, 0, v_ks_639_);
lean_ctor_set(v_reuseFailAlloc_658_, 1, v_vs_640_);
v___x_654_ = v_reuseFailAlloc_658_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
lean_object* v___x_655_; lean_object* v___x_656_; 
v___x_655_ = lean_unsigned_to_nat(1u);
v___x_656_ = lean_nat_add(v_x_636_, v___x_655_);
lean_dec(v_x_636_);
v_x_635_ = v___x_654_;
v_x_636_ = v___x_656_;
goto _start;
}
}
else
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_662_; 
v___x_659_ = lean_array_fset(v_ks_639_, v_x_636_, v_x_637_);
v___x_660_ = lean_array_fset(v_vs_640_, v_x_636_, v_x_638_);
lean_dec(v_x_636_);
if (v_isShared_643_ == 0)
{
lean_ctor_set(v___x_642_, 1, v___x_660_);
lean_ctor_set(v___x_642_, 0, v___x_659_);
v___x_662_ = v___x_642_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v___x_659_);
lean_ctor_set(v_reuseFailAlloc_663_, 1, v___x_660_);
v___x_662_ = v_reuseFailAlloc_663_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
return v___x_662_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10___redArg(lean_object* v_n_665_, lean_object* v_k_666_, lean_object* v_v_667_){
_start:
{
lean_object* v___x_668_; lean_object* v___x_669_; 
v___x_668_ = lean_unsigned_to_nat(0u);
v___x_669_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10_spec__11___redArg(v_n_665_, v___x_668_, v_k_666_, v_v_667_);
return v___x_669_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_670_; 
v___x_670_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(lean_object* v_x_671_, size_t v_x_672_, size_t v_x_673_, lean_object* v_x_674_, lean_object* v_x_675_){
_start:
{
if (lean_obj_tag(v_x_671_) == 0)
{
lean_object* v_es_676_; size_t v___x_677_; size_t v___x_678_; lean_object* v_j_679_; lean_object* v___x_680_; uint8_t v___x_681_; 
v_es_676_ = lean_ctor_get(v_x_671_, 0);
v___x_677_ = ((size_t)31ULL);
v___x_678_ = lean_usize_land(v_x_672_, v___x_677_);
v_j_679_ = lean_usize_to_nat(v___x_678_);
v___x_680_ = lean_array_get_size(v_es_676_);
v___x_681_ = lean_nat_dec_lt(v_j_679_, v___x_680_);
if (v___x_681_ == 0)
{
lean_dec(v_j_679_);
lean_dec(v_x_675_);
lean_dec(v_x_674_);
return v_x_671_;
}
else
{
lean_object* v___x_683_; uint8_t v_isShared_684_; uint8_t v_isSharedCheck_720_; 
lean_inc_ref(v_es_676_);
v_isSharedCheck_720_ = !lean_is_exclusive(v_x_671_);
if (v_isSharedCheck_720_ == 0)
{
lean_object* v_unused_721_; 
v_unused_721_ = lean_ctor_get(v_x_671_, 0);
lean_dec(v_unused_721_);
v___x_683_ = v_x_671_;
v_isShared_684_ = v_isSharedCheck_720_;
goto v_resetjp_682_;
}
else
{
lean_dec(v_x_671_);
v___x_683_ = lean_box(0);
v_isShared_684_ = v_isSharedCheck_720_;
goto v_resetjp_682_;
}
v_resetjp_682_:
{
lean_object* v_v_685_; lean_object* v___x_686_; lean_object* v_xs_x27_687_; lean_object* v___y_689_; 
v_v_685_ = lean_array_fget(v_es_676_, v_j_679_);
v___x_686_ = lean_box(0);
v_xs_x27_687_ = lean_array_fset(v_es_676_, v_j_679_, v___x_686_);
switch(lean_obj_tag(v_v_685_))
{
case 0:
{
lean_object* v_key_694_; lean_object* v_val_695_; lean_object* v___x_697_; uint8_t v_isShared_698_; uint8_t v_isSharedCheck_705_; 
v_key_694_ = lean_ctor_get(v_v_685_, 0);
v_val_695_ = lean_ctor_get(v_v_685_, 1);
v_isSharedCheck_705_ = !lean_is_exclusive(v_v_685_);
if (v_isSharedCheck_705_ == 0)
{
v___x_697_ = v_v_685_;
v_isShared_698_ = v_isSharedCheck_705_;
goto v_resetjp_696_;
}
else
{
lean_inc(v_val_695_);
lean_inc(v_key_694_);
lean_dec(v_v_685_);
v___x_697_ = lean_box(0);
v_isShared_698_ = v_isSharedCheck_705_;
goto v_resetjp_696_;
}
v_resetjp_696_:
{
uint8_t v___x_699_; 
v___x_699_ = l_Lean_instBEqMVarId_beq(v_x_674_, v_key_694_);
if (v___x_699_ == 0)
{
lean_object* v___x_700_; lean_object* v___x_701_; 
lean_del_object(v___x_697_);
v___x_700_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_694_, v_val_695_, v_x_674_, v_x_675_);
v___x_701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_701_, 0, v___x_700_);
v___y_689_ = v___x_701_;
goto v___jp_688_;
}
else
{
lean_object* v___x_703_; 
lean_dec(v_val_695_);
lean_dec(v_key_694_);
if (v_isShared_698_ == 0)
{
lean_ctor_set(v___x_697_, 1, v_x_675_);
lean_ctor_set(v___x_697_, 0, v_x_674_);
v___x_703_ = v___x_697_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v_x_674_);
lean_ctor_set(v_reuseFailAlloc_704_, 1, v_x_675_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
v___y_689_ = v___x_703_;
goto v___jp_688_;
}
}
}
}
case 1:
{
lean_object* v_node_706_; lean_object* v___x_708_; uint8_t v_isShared_709_; uint8_t v_isSharedCheck_718_; 
v_node_706_ = lean_ctor_get(v_v_685_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v_v_685_);
if (v_isSharedCheck_718_ == 0)
{
v___x_708_ = v_v_685_;
v_isShared_709_ = v_isSharedCheck_718_;
goto v_resetjp_707_;
}
else
{
lean_inc(v_node_706_);
lean_dec(v_v_685_);
v___x_708_ = lean_box(0);
v_isShared_709_ = v_isSharedCheck_718_;
goto v_resetjp_707_;
}
v_resetjp_707_:
{
size_t v___x_710_; size_t v___x_711_; size_t v___x_712_; size_t v___x_713_; lean_object* v___x_714_; lean_object* v___x_716_; 
v___x_710_ = ((size_t)5ULL);
v___x_711_ = lean_usize_shift_right(v_x_672_, v___x_710_);
v___x_712_ = ((size_t)1ULL);
v___x_713_ = lean_usize_add(v_x_673_, v___x_712_);
v___x_714_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(v_node_706_, v___x_711_, v___x_713_, v_x_674_, v_x_675_);
if (v_isShared_709_ == 0)
{
lean_ctor_set(v___x_708_, 0, v___x_714_);
v___x_716_ = v___x_708_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v___x_714_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
v___y_689_ = v___x_716_;
goto v___jp_688_;
}
}
}
default: 
{
lean_object* v___x_719_; 
v___x_719_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_719_, 0, v_x_674_);
lean_ctor_set(v___x_719_, 1, v_x_675_);
v___y_689_ = v___x_719_;
goto v___jp_688_;
}
}
v___jp_688_:
{
lean_object* v___x_690_; lean_object* v___x_692_; 
v___x_690_ = lean_array_fset(v_xs_x27_687_, v_j_679_, v___y_689_);
lean_dec(v_j_679_);
if (v_isShared_684_ == 0)
{
lean_ctor_set(v___x_683_, 0, v___x_690_);
v___x_692_ = v___x_683_;
goto v_reusejp_691_;
}
else
{
lean_object* v_reuseFailAlloc_693_; 
v_reuseFailAlloc_693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_693_, 0, v___x_690_);
v___x_692_ = v_reuseFailAlloc_693_;
goto v_reusejp_691_;
}
v_reusejp_691_:
{
return v___x_692_;
}
}
}
}
}
else
{
lean_object* v_ks_722_; lean_object* v_vs_723_; lean_object* v___x_725_; uint8_t v_isShared_726_; uint8_t v_isSharedCheck_743_; 
v_ks_722_ = lean_ctor_get(v_x_671_, 0);
v_vs_723_ = lean_ctor_get(v_x_671_, 1);
v_isSharedCheck_743_ = !lean_is_exclusive(v_x_671_);
if (v_isSharedCheck_743_ == 0)
{
v___x_725_ = v_x_671_;
v_isShared_726_ = v_isSharedCheck_743_;
goto v_resetjp_724_;
}
else
{
lean_inc(v_vs_723_);
lean_inc(v_ks_722_);
lean_dec(v_x_671_);
v___x_725_ = lean_box(0);
v_isShared_726_ = v_isSharedCheck_743_;
goto v_resetjp_724_;
}
v_resetjp_724_:
{
lean_object* v___x_728_; 
if (v_isShared_726_ == 0)
{
v___x_728_ = v___x_725_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_ks_722_);
lean_ctor_set(v_reuseFailAlloc_742_, 1, v_vs_723_);
v___x_728_ = v_reuseFailAlloc_742_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
lean_object* v_newNode_729_; uint8_t v___y_731_; size_t v___x_737_; uint8_t v___x_738_; 
v_newNode_729_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10___redArg(v___x_728_, v_x_674_, v_x_675_);
v___x_737_ = ((size_t)7ULL);
v___x_738_ = lean_usize_dec_le(v___x_737_, v_x_673_);
if (v___x_738_ == 0)
{
lean_object* v___x_739_; lean_object* v___x_740_; uint8_t v___x_741_; 
v___x_739_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_729_);
v___x_740_ = lean_unsigned_to_nat(4u);
v___x_741_ = lean_nat_dec_lt(v___x_739_, v___x_740_);
lean_dec(v___x_739_);
v___y_731_ = v___x_741_;
goto v___jp_730_;
}
else
{
v___y_731_ = v___x_738_;
goto v___jp_730_;
}
v___jp_730_:
{
if (v___y_731_ == 0)
{
lean_object* v_ks_732_; lean_object* v_vs_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; 
v_ks_732_ = lean_ctor_get(v_newNode_729_, 0);
lean_inc_ref(v_ks_732_);
v_vs_733_ = lean_ctor_get(v_newNode_729_, 1);
lean_inc_ref(v_vs_733_);
lean_dec_ref(v_newNode_729_);
v___x_734_ = lean_unsigned_to_nat(0u);
v___x_735_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___closed__0);
v___x_736_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg(v_x_673_, v_ks_732_, v_vs_733_, v___x_734_, v___x_735_);
lean_dec_ref(v_vs_733_);
lean_dec_ref(v_ks_732_);
return v___x_736_;
}
else
{
return v_newNode_729_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg(size_t v_depth_744_, lean_object* v_keys_745_, lean_object* v_vals_746_, lean_object* v_i_747_, lean_object* v_entries_748_){
_start:
{
lean_object* v___x_749_; uint8_t v___x_750_; 
v___x_749_ = lean_array_get_size(v_keys_745_);
v___x_750_ = lean_nat_dec_lt(v_i_747_, v___x_749_);
if (v___x_750_ == 0)
{
lean_dec(v_i_747_);
return v_entries_748_;
}
else
{
lean_object* v_k_751_; lean_object* v_v_752_; uint64_t v___x_753_; size_t v_h_754_; size_t v___x_755_; lean_object* v___x_756_; size_t v___x_757_; size_t v___x_758_; size_t v___x_759_; size_t v_h_760_; lean_object* v___x_761_; lean_object* v___x_762_; 
v_k_751_ = lean_array_fget_borrowed(v_keys_745_, v_i_747_);
v_v_752_ = lean_array_fget_borrowed(v_vals_746_, v_i_747_);
v___x_753_ = l_Lean_instHashableMVarId_hash(v_k_751_);
v_h_754_ = lean_uint64_to_usize(v___x_753_);
v___x_755_ = ((size_t)5ULL);
v___x_756_ = lean_unsigned_to_nat(1u);
v___x_757_ = ((size_t)1ULL);
v___x_758_ = lean_usize_sub(v_depth_744_, v___x_757_);
v___x_759_ = lean_usize_mul(v___x_755_, v___x_758_);
v_h_760_ = lean_usize_shift_right(v_h_754_, v___x_759_);
v___x_761_ = lean_nat_add(v_i_747_, v___x_756_);
lean_dec(v_i_747_);
lean_inc(v_v_752_);
lean_inc(v_k_751_);
v___x_762_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(v_entries_748_, v_h_760_, v_depth_744_, v_k_751_, v_v_752_);
v_i_747_ = v___x_761_;
v_entries_748_ = v___x_762_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg___boxed(lean_object* v_depth_764_, lean_object* v_keys_765_, lean_object* v_vals_766_, lean_object* v_i_767_, lean_object* v_entries_768_){
_start:
{
size_t v_depth_boxed_769_; lean_object* v_res_770_; 
v_depth_boxed_769_ = lean_unbox_usize(v_depth_764_);
lean_dec(v_depth_764_);
v_res_770_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg(v_depth_boxed_769_, v_keys_765_, v_vals_766_, v_i_767_, v_entries_768_);
lean_dec_ref(v_vals_766_);
lean_dec_ref(v_keys_765_);
return v_res_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_x_771_, lean_object* v_x_772_, lean_object* v_x_773_, lean_object* v_x_774_, lean_object* v_x_775_){
_start:
{
size_t v_x_18805__boxed_776_; size_t v_x_18806__boxed_777_; lean_object* v_res_778_; 
v_x_18805__boxed_776_ = lean_unbox_usize(v_x_772_);
lean_dec(v_x_772_);
v_x_18806__boxed_777_ = lean_unbox_usize(v_x_773_);
lean_dec(v_x_773_);
v_res_778_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(v_x_771_, v_x_18805__boxed_776_, v_x_18806__boxed_777_, v_x_774_, v_x_775_);
return v_res_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1___redArg(lean_object* v_x_779_, lean_object* v_x_780_, lean_object* v_x_781_){
_start:
{
uint64_t v___x_782_; size_t v___x_783_; size_t v___x_784_; lean_object* v___x_785_; 
v___x_782_ = l_Lean_instHashableMVarId_hash(v_x_780_);
v___x_783_ = lean_uint64_to_usize(v___x_782_);
v___x_784_ = ((size_t)1ULL);
v___x_785_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(v_x_779_, v___x_783_, v___x_784_, v_x_780_, v_x_781_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg(lean_object* v_mvarId_786_, lean_object* v_val_787_, lean_object* v___y_788_){
_start:
{
lean_object* v___x_790_; lean_object* v_mctx_791_; lean_object* v_cache_792_; lean_object* v_zetaDeltaFVarIds_793_; lean_object* v_postponed_794_; lean_object* v_diag_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_823_; 
v___x_790_ = lean_st_ref_take(v___y_788_);
v_mctx_791_ = lean_ctor_get(v___x_790_, 0);
v_cache_792_ = lean_ctor_get(v___x_790_, 1);
v_zetaDeltaFVarIds_793_ = lean_ctor_get(v___x_790_, 2);
v_postponed_794_ = lean_ctor_get(v___x_790_, 3);
v_diag_795_ = lean_ctor_get(v___x_790_, 4);
v_isSharedCheck_823_ = !lean_is_exclusive(v___x_790_);
if (v_isSharedCheck_823_ == 0)
{
v___x_797_ = v___x_790_;
v_isShared_798_ = v_isSharedCheck_823_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_diag_795_);
lean_inc(v_postponed_794_);
lean_inc(v_zetaDeltaFVarIds_793_);
lean_inc(v_cache_792_);
lean_inc(v_mctx_791_);
lean_dec(v___x_790_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_823_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v_depth_799_; lean_object* v_levelAssignDepth_800_; lean_object* v_lmvarCounter_801_; lean_object* v_mvarCounter_802_; lean_object* v_lDecls_803_; lean_object* v_decls_804_; lean_object* v_userNames_805_; lean_object* v_lAssignment_806_; lean_object* v_eAssignment_807_; lean_object* v_dAssignment_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_822_; 
v_depth_799_ = lean_ctor_get(v_mctx_791_, 0);
v_levelAssignDepth_800_ = lean_ctor_get(v_mctx_791_, 1);
v_lmvarCounter_801_ = lean_ctor_get(v_mctx_791_, 2);
v_mvarCounter_802_ = lean_ctor_get(v_mctx_791_, 3);
v_lDecls_803_ = lean_ctor_get(v_mctx_791_, 4);
v_decls_804_ = lean_ctor_get(v_mctx_791_, 5);
v_userNames_805_ = lean_ctor_get(v_mctx_791_, 6);
v_lAssignment_806_ = lean_ctor_get(v_mctx_791_, 7);
v_eAssignment_807_ = lean_ctor_get(v_mctx_791_, 8);
v_dAssignment_808_ = lean_ctor_get(v_mctx_791_, 9);
v_isSharedCheck_822_ = !lean_is_exclusive(v_mctx_791_);
if (v_isSharedCheck_822_ == 0)
{
v___x_810_ = v_mctx_791_;
v_isShared_811_ = v_isSharedCheck_822_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_dAssignment_808_);
lean_inc(v_eAssignment_807_);
lean_inc(v_lAssignment_806_);
lean_inc(v_userNames_805_);
lean_inc(v_decls_804_);
lean_inc(v_lDecls_803_);
lean_inc(v_mvarCounter_802_);
lean_inc(v_lmvarCounter_801_);
lean_inc(v_levelAssignDepth_800_);
lean_inc(v_depth_799_);
lean_dec(v_mctx_791_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_822_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___x_812_; lean_object* v___x_814_; 
v___x_812_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1___redArg(v_eAssignment_807_, v_mvarId_786_, v_val_787_);
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 8, v___x_812_);
v___x_814_ = v___x_810_;
goto v_reusejp_813_;
}
else
{
lean_object* v_reuseFailAlloc_821_; 
v_reuseFailAlloc_821_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_821_, 0, v_depth_799_);
lean_ctor_set(v_reuseFailAlloc_821_, 1, v_levelAssignDepth_800_);
lean_ctor_set(v_reuseFailAlloc_821_, 2, v_lmvarCounter_801_);
lean_ctor_set(v_reuseFailAlloc_821_, 3, v_mvarCounter_802_);
lean_ctor_set(v_reuseFailAlloc_821_, 4, v_lDecls_803_);
lean_ctor_set(v_reuseFailAlloc_821_, 5, v_decls_804_);
lean_ctor_set(v_reuseFailAlloc_821_, 6, v_userNames_805_);
lean_ctor_set(v_reuseFailAlloc_821_, 7, v_lAssignment_806_);
lean_ctor_set(v_reuseFailAlloc_821_, 8, v___x_812_);
lean_ctor_set(v_reuseFailAlloc_821_, 9, v_dAssignment_808_);
v___x_814_ = v_reuseFailAlloc_821_;
goto v_reusejp_813_;
}
v_reusejp_813_:
{
lean_object* v___x_816_; 
if (v_isShared_798_ == 0)
{
lean_ctor_set(v___x_797_, 0, v___x_814_);
v___x_816_ = v___x_797_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_820_; 
v_reuseFailAlloc_820_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_820_, 0, v___x_814_);
lean_ctor_set(v_reuseFailAlloc_820_, 1, v_cache_792_);
lean_ctor_set(v_reuseFailAlloc_820_, 2, v_zetaDeltaFVarIds_793_);
lean_ctor_set(v_reuseFailAlloc_820_, 3, v_postponed_794_);
lean_ctor_set(v_reuseFailAlloc_820_, 4, v_diag_795_);
v___x_816_ = v_reuseFailAlloc_820_;
goto v_reusejp_815_;
}
v_reusejp_815_:
{
lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_817_ = lean_st_ref_set(v___y_788_, v___x_816_);
v___x_818_ = lean_box(0);
v___x_819_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_819_, 0, v___x_818_);
return v___x_819_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg___boxed(lean_object* v_mvarId_824_, lean_object* v_val_825_, lean_object* v___y_826_, lean_object* v___y_827_){
_start:
{
lean_object* v_res_828_; 
v_res_828_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg(v_mvarId_824_, v_val_825_, v___y_826_);
lean_dec(v___y_826_);
return v_res_828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__0(lean_object* v_fst_829_, lean_object* v_a_830_, lean_object* v_d_831_, lean_object* v___x_832_, uint8_t v___x_833_, uint8_t v___x_834_, uint8_t v___x_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_d_x27_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_){
_start:
{
lean_object* v___x_844_; 
lean_inc(v_fst_829_);
v___x_844_ = l_Lean_MVarId_getType(v_fst_829_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
if (lean_obj_tag(v___x_844_) == 0)
{
lean_object* v_a_845_; lean_object* v___x_846_; lean_object* v___x_847_; 
v_a_845_ = lean_ctor_get(v___x_844_, 0);
lean_inc(v_a_845_);
lean_dec_ref_known(v___x_844_, 1);
v___x_846_ = l_Lean_Expr_replaceFVar(v_a_830_, v_d_831_, v_d_x27_838_);
v___x_847_ = l_Lean_mkArrow(v___x_846_, v_a_845_, v___y_841_, v___y_842_);
if (lean_obj_tag(v___x_847_) == 0)
{
lean_object* v_a_848_; lean_object* v___x_849_; 
v_a_848_ = lean_ctor_get(v___x_847_, 0);
lean_inc(v_a_848_);
lean_dec_ref_known(v___x_847_, 1);
lean_inc(v_fst_829_);
v___x_849_ = l_Lean_MVarId_getTag(v_fst_829_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
if (lean_obj_tag(v___x_849_) == 0)
{
lean_object* v_a_850_; lean_object* v___x_851_; 
v_a_850_ = lean_ctor_get(v___x_849_, 0);
lean_inc(v_a_850_);
lean_dec_ref_known(v___x_849_, 1);
v___x_851_ = l_Lean_Meta_mkFreshExprSyntheticOpaqueMVar(v_a_848_, v_a_850_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
if (lean_obj_tag(v___x_851_) == 0)
{
lean_object* v_a_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v_a_852_ = lean_ctor_get(v___x_851_, 0);
lean_inc_n(v_a_852_, 2);
lean_dec_ref_known(v___x_851_, 1);
v___x_853_ = lean_mk_empty_array_with_capacity(v___x_832_);
lean_inc_ref(v_d_x27_838_);
v___x_854_ = lean_array_push(v___x_853_, v_d_x27_838_);
v___x_855_ = l_Lean_Meta_mkLambdaFVars(v___x_854_, v_a_852_, v___x_833_, v___x_834_, v___x_833_, v___x_834_, v___x_835_, v___y_839_, v___y_840_, v___y_841_, v___y_842_);
lean_dec_ref(v___x_854_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v_a_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_867_; 
v_a_856_ = lean_ctor_get(v___x_855_, 0);
lean_inc(v_a_856_);
lean_dec_ref_known(v___x_855_, 1);
v___x_857_ = l_Lean_mkAppB(v_a_856_, v_a_836_, v_a_837_);
v___x_858_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg(v_fst_829_, v___x_857_, v___y_840_);
v_isSharedCheck_867_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_867_ == 0)
{
lean_object* v_unused_868_; 
v_unused_868_ = lean_ctor_get(v___x_858_, 0);
lean_dec(v_unused_868_);
v___x_860_ = v___x_858_;
v_isShared_861_ = v_isSharedCheck_867_;
goto v_resetjp_859_;
}
else
{
lean_dec(v___x_858_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_867_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_865_; 
v___x_862_ = l_Lean_Expr_mvarId_x21(v_a_852_);
lean_dec(v_a_852_);
v___x_863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_863_, 0, v_d_x27_838_);
lean_ctor_set(v___x_863_, 1, v___x_862_);
if (v_isShared_861_ == 0)
{
lean_ctor_set(v___x_860_, 0, v___x_863_);
v___x_865_ = v___x_860_;
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
lean_object* v_a_869_; lean_object* v___x_871_; uint8_t v_isShared_872_; uint8_t v_isSharedCheck_876_; 
lean_dec(v_a_852_);
lean_dec_ref(v_d_x27_838_);
lean_dec_ref(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec(v_fst_829_);
v_a_869_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_876_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_876_ == 0)
{
v___x_871_ = v___x_855_;
v_isShared_872_ = v_isSharedCheck_876_;
goto v_resetjp_870_;
}
else
{
lean_inc(v_a_869_);
lean_dec(v___x_855_);
v___x_871_ = lean_box(0);
v_isShared_872_ = v_isSharedCheck_876_;
goto v_resetjp_870_;
}
v_resetjp_870_:
{
lean_object* v___x_874_; 
if (v_isShared_872_ == 0)
{
v___x_874_ = v___x_871_;
goto v_reusejp_873_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v_a_869_);
v___x_874_ = v_reuseFailAlloc_875_;
goto v_reusejp_873_;
}
v_reusejp_873_:
{
return v___x_874_;
}
}
}
}
else
{
lean_object* v_a_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_884_; 
lean_dec_ref(v_d_x27_838_);
lean_dec_ref(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec(v_fst_829_);
v_a_877_ = lean_ctor_get(v___x_851_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_851_);
if (v_isSharedCheck_884_ == 0)
{
v___x_879_ = v___x_851_;
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_a_877_);
lean_dec(v___x_851_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
lean_object* v___x_882_; 
if (v_isShared_880_ == 0)
{
v___x_882_ = v___x_879_;
goto v_reusejp_881_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v_a_877_);
v___x_882_ = v_reuseFailAlloc_883_;
goto v_reusejp_881_;
}
v_reusejp_881_:
{
return v___x_882_;
}
}
}
}
else
{
lean_object* v_a_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_892_; 
lean_dec(v_a_848_);
lean_dec_ref(v_d_x27_838_);
lean_dec_ref(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec(v_fst_829_);
v_a_885_ = lean_ctor_get(v___x_849_, 0);
v_isSharedCheck_892_ = !lean_is_exclusive(v___x_849_);
if (v_isSharedCheck_892_ == 0)
{
v___x_887_ = v___x_849_;
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_a_885_);
lean_dec(v___x_849_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_892_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_890_; 
if (v_isShared_888_ == 0)
{
v___x_890_ = v___x_887_;
goto v_reusejp_889_;
}
else
{
lean_object* v_reuseFailAlloc_891_; 
v_reuseFailAlloc_891_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_891_, 0, v_a_885_);
v___x_890_ = v_reuseFailAlloc_891_;
goto v_reusejp_889_;
}
v_reusejp_889_:
{
return v___x_890_;
}
}
}
}
else
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_900_; 
lean_dec_ref(v_d_x27_838_);
lean_dec_ref(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec(v_fst_829_);
v_a_893_ = lean_ctor_get(v___x_847_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___x_847_);
if (v_isSharedCheck_900_ == 0)
{
v___x_895_ = v___x_847_;
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_847_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v___x_898_; 
if (v_isShared_896_ == 0)
{
v___x_898_ = v___x_895_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_a_893_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
}
}
else
{
lean_object* v_a_901_; lean_object* v___x_903_; uint8_t v_isShared_904_; uint8_t v_isSharedCheck_908_; 
lean_dec_ref(v_d_x27_838_);
lean_dec_ref(v_a_837_);
lean_dec_ref(v_a_836_);
lean_dec_ref(v_d_831_);
lean_dec(v_fst_829_);
v_a_901_ = lean_ctor_get(v___x_844_, 0);
v_isSharedCheck_908_ = !lean_is_exclusive(v___x_844_);
if (v_isSharedCheck_908_ == 0)
{
v___x_903_ = v___x_844_;
v_isShared_904_ = v_isSharedCheck_908_;
goto v_resetjp_902_;
}
else
{
lean_inc(v_a_901_);
lean_dec(v___x_844_);
v___x_903_ = lean_box(0);
v_isShared_904_ = v_isSharedCheck_908_;
goto v_resetjp_902_;
}
v_resetjp_902_:
{
lean_object* v___x_906_; 
if (v_isShared_904_ == 0)
{
v___x_906_ = v___x_903_;
goto v_reusejp_905_;
}
else
{
lean_object* v_reuseFailAlloc_907_; 
v_reuseFailAlloc_907_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_907_, 0, v_a_901_);
v___x_906_ = v_reuseFailAlloc_907_;
goto v_reusejp_905_;
}
v_reusejp_905_:
{
return v___x_906_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__0___boxed(lean_object* v_fst_909_, lean_object* v_a_910_, lean_object* v_d_911_, lean_object* v___x_912_, lean_object* v___x_913_, lean_object* v___x_914_, lean_object* v___x_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_d_x27_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_){
_start:
{
uint8_t v___x_19021__boxed_924_; uint8_t v___x_19022__boxed_925_; uint8_t v___x_19023__boxed_926_; lean_object* v_res_927_; 
v___x_19021__boxed_924_ = lean_unbox(v___x_913_);
v___x_19022__boxed_925_ = lean_unbox(v___x_914_);
v___x_19023__boxed_926_ = lean_unbox(v___x_915_);
v_res_927_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__0(v_fst_909_, v_a_910_, v_d_911_, v___x_912_, v___x_19021__boxed_924_, v___x_19022__boxed_925_, v___x_19023__boxed_926_, v_a_916_, v_a_917_, v_d_x27_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_);
lean_dec(v___y_922_);
lean_dec_ref(v___y_921_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
lean_dec(v___x_912_);
lean_dec_ref(v_a_910_);
return v_res_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___lam__0(lean_object* v_k_928_, lean_object* v_b_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
lean_object* v___x_935_; 
lean_inc(v___y_933_);
lean_inc_ref(v___y_932_);
lean_inc(v___y_931_);
lean_inc_ref(v___y_930_);
v___x_935_ = lean_apply_6(v_k_928_, v_b_929_, v___y_930_, v___y_931_, v___y_932_, v___y_933_, lean_box(0));
return v___x_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v_k_936_, lean_object* v_b_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_){
_start:
{
lean_object* v_res_943_; 
v_res_943_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___lam__0(v_k_936_, v_b_937_, v___y_938_, v___y_939_, v___y_940_, v___y_941_);
lean_dec(v___y_941_);
lean_dec_ref(v___y_940_);
lean_dec(v___y_939_);
lean_dec_ref(v___y_938_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg(lean_object* v_name_944_, uint8_t v_bi_945_, lean_object* v_type_946_, lean_object* v_k_947_, uint8_t v_kind_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_){
_start:
{
lean_object* v___f_954_; lean_object* v___x_955_; 
v___f_954_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_954_, 0, v_k_947_);
v___x_955_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_944_, v_bi_945_, v_type_946_, v___f_954_, v_kind_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
if (lean_obj_tag(v___x_955_) == 0)
{
lean_object* v_a_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_963_; 
v_a_956_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_963_ == 0)
{
v___x_958_ = v___x_955_;
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_a_956_);
lean_dec(v___x_955_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_963_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
if (v_isShared_959_ == 0)
{
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v_a_956_);
v___x_961_ = v_reuseFailAlloc_962_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
return v___x_961_;
}
}
}
else
{
lean_object* v_a_964_; lean_object* v___x_966_; uint8_t v_isShared_967_; uint8_t v_isSharedCheck_971_; 
v_a_964_ = lean_ctor_get(v___x_955_, 0);
v_isSharedCheck_971_ = !lean_is_exclusive(v___x_955_);
if (v_isSharedCheck_971_ == 0)
{
v___x_966_ = v___x_955_;
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
else
{
lean_inc(v_a_964_);
lean_dec(v___x_955_);
v___x_966_ = lean_box(0);
v_isShared_967_ = v_isSharedCheck_971_;
goto v_resetjp_965_;
}
v_resetjp_965_:
{
lean_object* v___x_969_; 
if (v_isShared_967_ == 0)
{
v___x_969_ = v___x_966_;
goto v_reusejp_968_;
}
else
{
lean_object* v_reuseFailAlloc_970_; 
v_reuseFailAlloc_970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_970_, 0, v_a_964_);
v___x_969_ = v_reuseFailAlloc_970_;
goto v_reusejp_968_;
}
v_reusejp_968_:
{
return v___x_969_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg___boxed(lean_object* v_name_972_, lean_object* v_bi_973_, lean_object* v_type_974_, lean_object* v_k_975_, lean_object* v_kind_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
uint8_t v_bi_boxed_982_; uint8_t v_kind_boxed_983_; lean_object* v_res_984_; 
v_bi_boxed_982_ = lean_unbox(v_bi_973_);
v_kind_boxed_983_ = lean_unbox(v_kind_976_);
v_res_984_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg(v_name_972_, v_bi_boxed_982_, v_type_974_, v_k_975_, v_kind_boxed_983_, v___y_977_, v___y_978_, v___y_979_, v___y_980_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
lean_dec(v___y_978_);
lean_dec_ref(v___y_977_);
return v_res_984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg(lean_object* v_name_985_, lean_object* v_type_986_, lean_object* v_k_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_){
_start:
{
uint8_t v___x_993_; uint8_t v___x_994_; lean_object* v___x_995_; 
v___x_993_ = 0;
v___x_994_ = 0;
v___x_995_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg(v_name_985_, v___x_993_, v_type_986_, v_k_987_, v___x_994_, v___y_988_, v___y_989_, v___y_990_, v___y_991_);
return v___x_995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg___boxed(lean_object* v_name_996_, lean_object* v_type_997_, lean_object* v_k_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_){
_start:
{
lean_object* v_res_1004_; 
v_res_1004_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg(v_name_996_, v_type_997_, v_k_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_);
lean_dec(v___y_1002_);
lean_dec_ref(v___y_1001_);
lean_dec(v___y_1000_);
lean_dec_ref(v___y_999_);
return v_res_1004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2(lean_object* v_00_u03b1_1005_, lean_object* v_name_1006_, lean_object* v_type_1007_, lean_object* v_k_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_){
_start:
{
lean_object* v___x_1014_; 
v___x_1014_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg(v_name_1006_, v_type_1007_, v_k_1008_, v___y_1009_, v___y_1010_, v___y_1011_, v___y_1012_);
return v___x_1014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___boxed(lean_object* v_00_u03b1_1015_, lean_object* v_name_1016_, lean_object* v_type_1017_, lean_object* v_k_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
lean_object* v_res_1024_; 
v_res_1024_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2(v_00_u03b1_1015_, v_name_1016_, v_type_1017_, v_k_1018_, v___y_1019_, v___y_1020_, v___y_1021_, v___y_1022_);
lean_dec(v___y_1022_);
lean_dec_ref(v___y_1021_);
lean_dec(v___y_1020_);
lean_dec_ref(v___y_1019_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__1(lean_object* v_ctx_x27_1025_, lean_object* v___x_1026_, lean_object* v_ctx_1027_, uint8_t v___x_1028_, uint8_t v___x_1029_, uint8_t v___x_1030_, lean_object* v_fst_1031_, lean_object* v___x_1032_, lean_object* v_a_1033_, lean_object* v_a_1034_, lean_object* v_a_1035_, lean_object* v_a_1036_, lean_object* v_d_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; 
lean_inc_ref(v_d_1037_);
v___x_1043_ = l_Lean_mkAppN(v_d_1037_, v_ctx_x27_1025_);
v___x_1044_ = l_Lean_Expr_app___override(v___x_1026_, v___x_1043_);
v___x_1045_ = l_Lean_Expr_headBeta(v___x_1044_);
v___x_1046_ = l_Lean_Meta_mkForallFVars(v_ctx_1027_, v___x_1045_, v___x_1028_, v___x_1029_, v___x_1029_, v___x_1030_, v___y_1038_, v___y_1039_, v___y_1040_, v___y_1041_);
if (lean_obj_tag(v___x_1046_) == 0)
{
lean_object* v_a_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___f_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v_a_1047_ = lean_ctor_get(v___x_1046_, 0);
lean_inc(v_a_1047_);
lean_dec_ref_known(v___x_1046_, 1);
v___x_1048_ = lean_box(v___x_1028_);
v___x_1049_ = lean_box(v___x_1029_);
v___x_1050_ = lean_box(v___x_1030_);
lean_inc(v_fst_1031_);
v___f_1051_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__0___boxed), 15, 9);
lean_closure_set(v___f_1051_, 0, v_fst_1031_);
lean_closure_set(v___f_1051_, 1, v_a_1047_);
lean_closure_set(v___f_1051_, 2, v_d_1037_);
lean_closure_set(v___f_1051_, 3, v___x_1032_);
lean_closure_set(v___f_1051_, 4, v___x_1048_);
lean_closure_set(v___f_1051_, 5, v___x_1049_);
lean_closure_set(v___f_1051_, 6, v___x_1050_);
lean_closure_set(v___f_1051_, 7, v_a_1033_);
lean_closure_set(v___f_1051_, 8, v_a_1034_);
v___x_1052_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___boxed), 9, 4);
lean_closure_set(v___x_1052_, 0, lean_box(0));
lean_closure_set(v___x_1052_, 1, v_a_1035_);
lean_closure_set(v___x_1052_, 2, v_a_1036_);
lean_closure_set(v___x_1052_, 3, v___f_1051_);
v___x_1053_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(v_fst_1031_, v___x_1052_, v___y_1038_, v___y_1039_, v___y_1040_, v___y_1041_);
return v___x_1053_;
}
else
{
lean_object* v_a_1054_; lean_object* v___x_1056_; uint8_t v_isShared_1057_; uint8_t v_isSharedCheck_1061_; 
lean_dec_ref(v_d_1037_);
lean_dec_ref(v_a_1036_);
lean_dec(v_a_1035_);
lean_dec_ref(v_a_1034_);
lean_dec_ref(v_a_1033_);
lean_dec(v___x_1032_);
lean_dec(v_fst_1031_);
v_a_1054_ = lean_ctor_get(v___x_1046_, 0);
v_isSharedCheck_1061_ = !lean_is_exclusive(v___x_1046_);
if (v_isSharedCheck_1061_ == 0)
{
v___x_1056_ = v___x_1046_;
v_isShared_1057_ = v_isSharedCheck_1061_;
goto v_resetjp_1055_;
}
else
{
lean_inc(v_a_1054_);
lean_dec(v___x_1046_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__1___boxed(lean_object** _args){
lean_object* v_ctx_x27_1062_ = _args[0];
lean_object* v___x_1063_ = _args[1];
lean_object* v_ctx_1064_ = _args[2];
lean_object* v___x_1065_ = _args[3];
lean_object* v___x_1066_ = _args[4];
lean_object* v___x_1067_ = _args[5];
lean_object* v_fst_1068_ = _args[6];
lean_object* v___x_1069_ = _args[7];
lean_object* v_a_1070_ = _args[8];
lean_object* v_a_1071_ = _args[9];
lean_object* v_a_1072_ = _args[10];
lean_object* v_a_1073_ = _args[11];
lean_object* v_d_1074_ = _args[12];
lean_object* v___y_1075_ = _args[13];
lean_object* v___y_1076_ = _args[14];
lean_object* v___y_1077_ = _args[15];
lean_object* v___y_1078_ = _args[16];
lean_object* v___y_1079_ = _args[17];
_start:
{
uint8_t v___x_19296__boxed_1080_; uint8_t v___x_19297__boxed_1081_; uint8_t v___x_19298__boxed_1082_; lean_object* v_res_1083_; 
v___x_19296__boxed_1080_ = lean_unbox(v___x_1065_);
v___x_19297__boxed_1081_ = lean_unbox(v___x_1066_);
v___x_19298__boxed_1082_ = lean_unbox(v___x_1067_);
v_res_1083_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__1(v_ctx_x27_1062_, v___x_1063_, v_ctx_1064_, v___x_19296__boxed_1080_, v___x_19297__boxed_1081_, v___x_19298__boxed_1082_, v_fst_1068_, v___x_1069_, v_a_1070_, v_a_1071_, v_a_1072_, v_a_1073_, v_d_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_);
lean_dec(v___y_1078_);
lean_dec_ref(v___y_1077_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
lean_dec_ref(v_ctx_1064_);
lean_dec_ref(v_ctx_x27_1062_);
return v_res_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__2(lean_object* v_snd_1084_, lean_object* v___x_1085_, lean_object* v_a_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_){
_start:
{
lean_object* v___x_1092_; 
lean_inc(v_snd_1084_);
v___x_1092_ = l_Lean_MVarId_getType(v_snd_1084_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
if (lean_obj_tag(v___x_1092_) == 0)
{
lean_object* v_a_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
v_a_1093_ = lean_ctor_get(v___x_1092_, 0);
lean_inc(v_a_1093_);
lean_dec_ref_known(v___x_1092_, 1);
v___x_1094_ = lean_box(0);
v___x_1095_ = l_Lean_Meta_synthInstance_x3f(v_a_1093_, v___x_1094_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_);
if (lean_obj_tag(v___x_1095_) == 0)
{
lean_object* v_a_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1127_; 
v_a_1096_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1127_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1127_ == 0)
{
v___x_1098_ = v___x_1095_;
v_isShared_1099_ = v_isSharedCheck_1127_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_a_1096_);
lean_dec(v___x_1095_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1127_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
if (lean_obj_tag(v_a_1096_) == 0)
{
lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1105_; 
lean_dec_ref(v_a_1086_);
lean_dec(v_snd_1084_);
v___x_1100_ = lean_box(0);
v___x_1101_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1101_, 0, v___x_1085_);
lean_ctor_set(v___x_1101_, 1, v___x_1100_);
v___x_1102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1102_, 0, v___x_1101_);
v___x_1103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1103_, 0, v___x_1102_);
lean_ctor_set(v___x_1103_, 1, v_a_1096_);
if (v_isShared_1099_ == 0)
{
lean_ctor_set(v___x_1098_, 0, v___x_1103_);
v___x_1105_ = v___x_1098_;
goto v_reusejp_1104_;
}
else
{
lean_object* v_reuseFailAlloc_1106_; 
v_reuseFailAlloc_1106_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1106_, 0, v___x_1103_);
v___x_1105_ = v_reuseFailAlloc_1106_;
goto v_reusejp_1104_;
}
v_reusejp_1104_:
{
return v___x_1105_;
}
}
else
{
lean_object* v_val_1107_; lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1126_; 
lean_del_object(v___x_1098_);
lean_dec_ref(v___x_1085_);
v_val_1107_ = lean_ctor_get(v_a_1096_, 0);
v_isSharedCheck_1126_ = !lean_is_exclusive(v_a_1096_);
if (v_isSharedCheck_1126_ == 0)
{
v___x_1109_ = v_a_1096_;
v_isShared_1110_ = v_isSharedCheck_1126_;
goto v_resetjp_1108_;
}
else
{
lean_inc(v_val_1107_);
lean_dec(v_a_1096_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1126_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v_a_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1125_; 
v___x_1111_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg(v_snd_1084_, v_val_1107_, v___y_1088_);
lean_dec_ref(v___x_1111_);
v___x_1112_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg(v_a_1086_, v___y_1088_);
v_a_1113_ = lean_ctor_get(v___x_1112_, 0);
v_isSharedCheck_1125_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1125_ == 0)
{
v___x_1115_ = v___x_1112_;
v_isShared_1116_ = v_isSharedCheck_1125_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_a_1113_);
lean_dec(v___x_1112_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1125_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1117_; lean_object* v___x_1119_; 
v___x_1117_ = lean_box(0);
if (v_isShared_1110_ == 0)
{
lean_ctor_set(v___x_1109_, 0, v_a_1113_);
v___x_1119_ = v___x_1109_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1124_; 
v_reuseFailAlloc_1124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1124_, 0, v_a_1113_);
v___x_1119_ = v_reuseFailAlloc_1124_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
lean_object* v___x_1120_; lean_object* v___x_1122_; 
v___x_1120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1120_, 0, v___x_1117_);
lean_ctor_set(v___x_1120_, 1, v___x_1119_);
if (v_isShared_1116_ == 0)
{
lean_ctor_set(v___x_1115_, 0, v___x_1120_);
v___x_1122_ = v___x_1115_;
goto v_reusejp_1121_;
}
else
{
lean_object* v_reuseFailAlloc_1123_; 
v_reuseFailAlloc_1123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1123_, 0, v___x_1120_);
v___x_1122_ = v_reuseFailAlloc_1123_;
goto v_reusejp_1121_;
}
v_reusejp_1121_:
{
return v___x_1122_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1128_; lean_object* v___x_1130_; uint8_t v_isShared_1131_; uint8_t v_isSharedCheck_1135_; 
lean_dec_ref(v_a_1086_);
lean_dec_ref(v___x_1085_);
lean_dec(v_snd_1084_);
v_a_1128_ = lean_ctor_get(v___x_1095_, 0);
v_isSharedCheck_1135_ = !lean_is_exclusive(v___x_1095_);
if (v_isSharedCheck_1135_ == 0)
{
v___x_1130_ = v___x_1095_;
v_isShared_1131_ = v_isSharedCheck_1135_;
goto v_resetjp_1129_;
}
else
{
lean_inc(v_a_1128_);
lean_dec(v___x_1095_);
v___x_1130_ = lean_box(0);
v_isShared_1131_ = v_isSharedCheck_1135_;
goto v_resetjp_1129_;
}
v_resetjp_1129_:
{
lean_object* v___x_1133_; 
if (v_isShared_1131_ == 0)
{
v___x_1133_ = v___x_1130_;
goto v_reusejp_1132_;
}
else
{
lean_object* v_reuseFailAlloc_1134_; 
v_reuseFailAlloc_1134_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1134_, 0, v_a_1128_);
v___x_1133_ = v_reuseFailAlloc_1134_;
goto v_reusejp_1132_;
}
v_reusejp_1132_:
{
return v___x_1133_;
}
}
}
}
else
{
lean_object* v_a_1136_; lean_object* v___x_1138_; uint8_t v_isShared_1139_; uint8_t v_isSharedCheck_1143_; 
lean_dec_ref(v_a_1086_);
lean_dec_ref(v___x_1085_);
lean_dec(v_snd_1084_);
v_a_1136_ = lean_ctor_get(v___x_1092_, 0);
v_isSharedCheck_1143_ = !lean_is_exclusive(v___x_1092_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1138_ = v___x_1092_;
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
else
{
lean_inc(v_a_1136_);
lean_dec(v___x_1092_);
v___x_1138_ = lean_box(0);
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
v_resetjp_1137_:
{
lean_object* v___x_1141_; 
if (v_isShared_1139_ == 0)
{
v___x_1141_ = v___x_1138_;
goto v_reusejp_1140_;
}
else
{
lean_object* v_reuseFailAlloc_1142_; 
v_reuseFailAlloc_1142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1142_, 0, v_a_1136_);
v___x_1141_ = v_reuseFailAlloc_1142_;
goto v_reusejp_1140_;
}
v_reusejp_1140_:
{
return v___x_1141_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__2___boxed(lean_object* v_snd_1144_, lean_object* v___x_1145_, lean_object* v_a_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_){
_start:
{
lean_object* v_res_1152_; 
v_res_1152_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__2(v_snd_1144_, v___x_1145_, v_a_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_);
lean_dec(v___y_1150_);
lean_dec_ref(v___y_1149_);
lean_dec(v___y_1148_);
lean_dec_ref(v___y_1147_);
return v_res_1152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4(lean_object* v_as_1153_, size_t v_i_1154_, size_t v_stop_1155_, lean_object* v_b_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_){
_start:
{
lean_object* v_a_1163_; uint8_t v___x_1167_; 
v___x_1167_ = lean_usize_dec_eq(v_i_1154_, v_stop_1155_);
if (v___x_1167_ == 0)
{
lean_object* v___x_1168_; lean_object* v___x_1171_; 
v___x_1168_ = lean_array_uget_borrowed(v_as_1153_, v_i_1154_);
lean_inc(v___x_1168_);
v___x_1171_ = l_Lean_Meta_isProof(v___x_1168_, v___y_1157_, v___y_1158_, v___y_1159_, v___y_1160_);
if (lean_obj_tag(v___x_1171_) == 0)
{
lean_object* v_a_1172_; uint8_t v___x_1173_; 
v_a_1172_ = lean_ctor_get(v___x_1171_, 0);
lean_inc(v_a_1172_);
lean_dec_ref_known(v___x_1171_, 1);
v___x_1173_ = lean_unbox(v_a_1172_);
lean_dec(v_a_1172_);
if (v___x_1173_ == 0)
{
goto v___jp_1169_;
}
else
{
v_a_1163_ = v_b_1156_;
goto v___jp_1162_;
}
}
else
{
if (lean_obj_tag(v___x_1171_) == 0)
{
lean_object* v_a_1174_; uint8_t v___x_1175_; 
v_a_1174_ = lean_ctor_get(v___x_1171_, 0);
lean_inc(v_a_1174_);
lean_dec_ref_known(v___x_1171_, 1);
v___x_1175_ = lean_unbox(v_a_1174_);
lean_dec(v_a_1174_);
if (v___x_1175_ == 0)
{
v_a_1163_ = v_b_1156_;
goto v___jp_1162_;
}
else
{
goto v___jp_1169_;
}
}
else
{
lean_object* v_a_1176_; lean_object* v___x_1178_; uint8_t v_isShared_1179_; uint8_t v_isSharedCheck_1183_; 
lean_dec_ref(v_b_1156_);
v_a_1176_ = lean_ctor_get(v___x_1171_, 0);
v_isSharedCheck_1183_ = !lean_is_exclusive(v___x_1171_);
if (v_isSharedCheck_1183_ == 0)
{
v___x_1178_ = v___x_1171_;
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
else
{
lean_inc(v_a_1176_);
lean_dec(v___x_1171_);
v___x_1178_ = lean_box(0);
v_isShared_1179_ = v_isSharedCheck_1183_;
goto v_resetjp_1177_;
}
v_resetjp_1177_:
{
lean_object* v___x_1181_; 
if (v_isShared_1179_ == 0)
{
v___x_1181_ = v___x_1178_;
goto v_reusejp_1180_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_a_1176_);
v___x_1181_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1180_;
}
v_reusejp_1180_:
{
return v___x_1181_;
}
}
}
}
v___jp_1169_:
{
lean_object* v___x_1170_; 
lean_inc(v___x_1168_);
v___x_1170_ = lean_array_push(v_b_1156_, v___x_1168_);
v_a_1163_ = v___x_1170_;
goto v___jp_1162_;
}
}
else
{
lean_object* v___x_1184_; 
v___x_1184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1184_, 0, v_b_1156_);
return v___x_1184_;
}
v___jp_1162_:
{
size_t v___x_1164_; size_t v___x_1165_; 
v___x_1164_ = ((size_t)1ULL);
v___x_1165_ = lean_usize_add(v_i_1154_, v___x_1164_);
v_i_1154_ = v___x_1165_;
v_b_1156_ = v_a_1163_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4___boxed(lean_object* v_as_1185_, lean_object* v_i_1186_, lean_object* v_stop_1187_, lean_object* v_b_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_){
_start:
{
size_t v_i_boxed_1194_; size_t v_stop_boxed_1195_; lean_object* v_res_1196_; 
v_i_boxed_1194_ = lean_unbox_usize(v_i_1186_);
lean_dec(v_i_1186_);
v_stop_boxed_1195_ = lean_unbox_usize(v_stop_1187_);
lean_dec(v_stop_1187_);
v_res_1196_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4(v_as_1185_, v_i_boxed_1194_, v_stop_boxed_1195_, v_b_1188_, v___y_1189_, v___y_1190_, v___y_1191_, v___y_1192_);
lean_dec(v___y_1192_);
lean_dec_ref(v___y_1191_);
lean_dec(v___y_1190_);
lean_dec_ref(v___y_1189_);
lean_dec_ref(v_as_1185_);
return v_res_1196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5(lean_object* v___x_1202_, lean_object* v_us_1203_, lean_object* v_pre_1204_, lean_object* v_as_1205_, size_t v_sz_1206_, size_t v_i_1207_, lean_object* v_b_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_){
_start:
{
lean_object* v_a_1215_; uint8_t v___x_1219_; 
v___x_1219_ = lean_usize_dec_lt(v_i_1207_, v_sz_1206_);
if (v___x_1219_ == 0)
{
lean_object* v___x_1220_; 
lean_dec(v_pre_1204_);
lean_dec(v_us_1203_);
lean_dec_ref(v___x_1202_);
v___x_1220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1220_, 0, v_b_1208_);
return v___x_1220_;
}
else
{
lean_object* v_a_1221_; lean_object* v___x_1222_; 
v_a_1221_ = lean_array_uget_borrowed(v_as_1205_, v_i_1207_);
lean_inc(v_a_1221_);
v___x_1222_ = l_Lean_Meta_isProof(v_a_1221_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
if (lean_obj_tag(v___x_1222_) == 0)
{
lean_object* v_a_1223_; uint8_t v___x_1224_; 
v_a_1223_ = lean_ctor_get(v___x_1222_, 0);
lean_inc(v_a_1223_);
lean_dec_ref_known(v___x_1222_, 1);
v___x_1224_ = lean_unbox(v_a_1223_);
lean_dec(v_a_1223_);
if (v___x_1224_ == 0)
{
lean_object* v___x_1225_; 
lean_inc(v___y_1212_);
lean_inc_ref(v___y_1211_);
lean_inc(v___y_1210_);
lean_inc_ref(v___y_1209_);
lean_inc(v_a_1221_);
v___x_1225_ = lean_infer_type(v_a_1221_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
if (lean_obj_tag(v___x_1225_) == 0)
{
lean_object* v_a_1226_; lean_object* v___x_1227_; 
v_a_1226_ = lean_ctor_get(v___x_1225_, 0);
lean_inc(v_a_1226_);
lean_dec_ref_known(v___x_1225_, 1);
lean_inc(v___y_1212_);
lean_inc_ref(v___y_1211_);
lean_inc(v___y_1210_);
lean_inc_ref(v___y_1209_);
v___x_1227_ = lean_whnf(v_a_1226_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
if (lean_obj_tag(v___x_1227_) == 0)
{
lean_object* v_a_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; 
v_a_1228_ = lean_ctor_get(v___x_1227_, 0);
lean_inc_n(v_a_1228_, 2);
lean_dec_ref_known(v___x_1227_, 1);
lean_inc_ref(v___x_1202_);
v___x_1229_ = l_Lean_Expr_app___override(v___x_1202_, v_a_1228_);
v___x_1230_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___closed__2));
lean_inc(v_us_1203_);
v___x_1231_ = l_Lean_Expr_const___override(v___x_1230_, v_us_1203_);
lean_inc(v_a_1221_);
v___x_1232_ = l_Lean_mkAppB(v___x_1231_, v_a_1228_, v_a_1221_);
lean_inc(v_pre_1204_);
v___x_1233_ = l_Lean_MVarId_assert(v_b_1208_, v_pre_1204_, v___x_1229_, v___x_1232_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
if (lean_obj_tag(v___x_1233_) == 0)
{
lean_object* v_a_1234_; 
v_a_1234_ = lean_ctor_get(v___x_1233_, 0);
lean_inc(v_a_1234_);
lean_dec_ref_known(v___x_1233_, 1);
v_a_1215_ = v_a_1234_;
goto v___jp_1214_;
}
else
{
lean_dec(v_pre_1204_);
lean_dec(v_us_1203_);
lean_dec_ref(v___x_1202_);
return v___x_1233_;
}
}
else
{
lean_object* v_a_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1242_; 
lean_dec(v_b_1208_);
lean_dec(v_pre_1204_);
lean_dec(v_us_1203_);
lean_dec_ref(v___x_1202_);
v_a_1235_ = lean_ctor_get(v___x_1227_, 0);
v_isSharedCheck_1242_ = !lean_is_exclusive(v___x_1227_);
if (v_isSharedCheck_1242_ == 0)
{
v___x_1237_ = v___x_1227_;
v_isShared_1238_ = v_isSharedCheck_1242_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_a_1235_);
lean_dec(v___x_1227_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1242_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v___x_1240_; 
if (v_isShared_1238_ == 0)
{
v___x_1240_ = v___x_1237_;
goto v_reusejp_1239_;
}
else
{
lean_object* v_reuseFailAlloc_1241_; 
v_reuseFailAlloc_1241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1241_, 0, v_a_1235_);
v___x_1240_ = v_reuseFailAlloc_1241_;
goto v_reusejp_1239_;
}
v_reusejp_1239_:
{
return v___x_1240_;
}
}
}
}
else
{
lean_object* v_a_1243_; lean_object* v___x_1245_; uint8_t v_isShared_1246_; uint8_t v_isSharedCheck_1250_; 
lean_dec(v_b_1208_);
lean_dec(v_pre_1204_);
lean_dec(v_us_1203_);
lean_dec_ref(v___x_1202_);
v_a_1243_ = lean_ctor_get(v___x_1225_, 0);
v_isSharedCheck_1250_ = !lean_is_exclusive(v___x_1225_);
if (v_isSharedCheck_1250_ == 0)
{
v___x_1245_ = v___x_1225_;
v_isShared_1246_ = v_isSharedCheck_1250_;
goto v_resetjp_1244_;
}
else
{
lean_inc(v_a_1243_);
lean_dec(v___x_1225_);
v___x_1245_ = lean_box(0);
v_isShared_1246_ = v_isSharedCheck_1250_;
goto v_resetjp_1244_;
}
v_resetjp_1244_:
{
lean_object* v___x_1248_; 
if (v_isShared_1246_ == 0)
{
v___x_1248_ = v___x_1245_;
goto v_reusejp_1247_;
}
else
{
lean_object* v_reuseFailAlloc_1249_; 
v_reuseFailAlloc_1249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1249_, 0, v_a_1243_);
v___x_1248_ = v_reuseFailAlloc_1249_;
goto v_reusejp_1247_;
}
v_reusejp_1247_:
{
return v___x_1248_;
}
}
}
}
else
{
v_a_1215_ = v_b_1208_;
goto v___jp_1214_;
}
}
else
{
lean_object* v_a_1251_; lean_object* v___x_1253_; uint8_t v_isShared_1254_; uint8_t v_isSharedCheck_1258_; 
lean_dec(v_b_1208_);
lean_dec(v_pre_1204_);
lean_dec(v_us_1203_);
lean_dec_ref(v___x_1202_);
v_a_1251_ = lean_ctor_get(v___x_1222_, 0);
v_isSharedCheck_1258_ = !lean_is_exclusive(v___x_1222_);
if (v_isSharedCheck_1258_ == 0)
{
v___x_1253_ = v___x_1222_;
v_isShared_1254_ = v_isSharedCheck_1258_;
goto v_resetjp_1252_;
}
else
{
lean_inc(v_a_1251_);
lean_dec(v___x_1222_);
v___x_1253_ = lean_box(0);
v_isShared_1254_ = v_isSharedCheck_1258_;
goto v_resetjp_1252_;
}
v_resetjp_1252_:
{
lean_object* v___x_1256_; 
if (v_isShared_1254_ == 0)
{
v___x_1256_ = v___x_1253_;
goto v_reusejp_1255_;
}
else
{
lean_object* v_reuseFailAlloc_1257_; 
v_reuseFailAlloc_1257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1257_, 0, v_a_1251_);
v___x_1256_ = v_reuseFailAlloc_1257_;
goto v_reusejp_1255_;
}
v_reusejp_1255_:
{
return v___x_1256_;
}
}
}
}
v___jp_1214_:
{
size_t v___x_1216_; size_t v___x_1217_; 
v___x_1216_ = ((size_t)1ULL);
v___x_1217_ = lean_usize_add(v_i_1207_, v___x_1216_);
v_i_1207_ = v___x_1217_;
v_b_1208_ = v_a_1215_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5___boxed(lean_object* v___x_1259_, lean_object* v_us_1260_, lean_object* v_pre_1261_, lean_object* v_as_1262_, lean_object* v_sz_1263_, lean_object* v_i_1264_, lean_object* v_b_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_){
_start:
{
size_t v_sz_boxed_1271_; size_t v_i_boxed_1272_; lean_object* v_res_1273_; 
v_sz_boxed_1271_ = lean_unbox_usize(v_sz_1263_);
lean_dec(v_sz_1263_);
v_i_boxed_1272_ = lean_unbox_usize(v_i_1264_);
lean_dec(v_i_1264_);
v_res_1273_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5(v___x_1259_, v_us_1260_, v_pre_1261_, v_as_1262_, v_sz_boxed_1271_, v_i_boxed_1272_, v_b_1265_, v___y_1266_, v___y_1267_, v___y_1268_, v___y_1269_);
lean_dec(v___y_1269_);
lean_dec_ref(v___y_1268_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec_ref(v_as_1262_);
return v_res_1273_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__1(void){
_start:
{
lean_object* v___x_1275_; lean_object* v___x_1276_; 
v___x_1275_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__0));
v___x_1276_ = l_Lean_stringToMessageData(v___x_1275_);
return v___x_1276_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__8(void){
_start:
{
lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1286_ = lean_box(0);
v___x_1287_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__7));
v___x_1288_ = l_Lean_Expr_const___override(v___x_1287_, v___x_1286_);
return v___x_1288_;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__11(void){
_start:
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v___x_1293_ = lean_box(0);
v___x_1294_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__10));
v___x_1295_ = l_Lean_Expr_const___override(v___x_1294_, v___x_1293_);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6(lean_object* v_data_1311_, lean_object* v_a_1312_, lean_object* v_ctx_1313_, lean_object* v_fst_1314_, uint8_t v_nondep_1315_, lean_object* v_x_1316_, lean_object* v_x_1317_, lean_object* v_x_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
lean_object* v___y_1325_; lean_object* v___y_1326_; lean_object* v_g_1327_; lean_object* v___y_1332_; lean_object* v___y_1333_; lean_object* v___y_1334_; lean_object* v___y_1335_; 
if (lean_obj_tag(v_x_1316_) == 5)
{
lean_object* v_fn_1338_; lean_object* v_arg_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; 
v_fn_1338_ = lean_ctor_get(v_x_1316_, 0);
lean_inc_ref(v_fn_1338_);
v_arg_1339_ = lean_ctor_get(v_x_1316_, 1);
lean_inc_ref(v_arg_1339_);
lean_dec_ref_known(v_x_1316_, 2);
v___x_1340_ = lean_array_set(v_x_1317_, v_x_1318_, v_arg_1339_);
v___x_1341_ = lean_unsigned_to_nat(1u);
v___x_1342_ = lean_nat_sub(v_x_1318_, v___x_1341_);
lean_dec(v_x_1318_);
v_x_1316_ = v_fn_1338_;
v_x_1317_ = v___x_1340_;
v_x_1318_ = v___x_1342_;
goto _start;
}
else
{
lean_dec(v_x_1318_);
if (lean_obj_tag(v_x_1316_) == 4)
{
lean_object* v_declName_1344_; 
v_declName_1344_ = lean_ctor_get(v_x_1316_, 0);
lean_inc(v_declName_1344_);
if (lean_obj_tag(v_declName_1344_) == 1)
{
lean_object* v_pre_1345_; 
v_pre_1345_ = lean_ctor_get(v_declName_1344_, 0);
lean_inc(v_pre_1345_);
if (lean_obj_tag(v_pre_1345_) == 0)
{
lean_object* v_us_1346_; lean_object* v_str_1347_; lean_object* v___x_1348_; uint8_t v___x_1349_; 
v_us_1346_ = lean_ctor_get(v_x_1316_, 1);
lean_inc(v_us_1346_);
lean_dec_ref_known(v_x_1316_, 2);
v_str_1347_ = lean_ctor_get(v_declName_1344_, 1);
lean_inc_ref(v_str_1347_);
lean_dec_ref_known(v_declName_1344_, 2);
v___x_1348_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__2));
v___x_1349_ = lean_string_dec_eq(v_str_1347_, v___x_1348_);
if (v___x_1349_ == 0)
{
lean_object* v___x_1350_; uint8_t v___x_1351_; 
lean_dec(v_us_1346_);
v___x_1350_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__3));
v___x_1351_ = lean_string_dec_eq(v_str_1347_, v___x_1350_);
lean_dec_ref(v_str_1347_);
if (v___x_1351_ == 0)
{
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
else
{
lean_object* v___x_1352_; lean_object* v___x_1353_; uint8_t v___x_1354_; 
v___x_1352_ = lean_array_get_size(v_x_1317_);
v___x_1353_ = lean_unsigned_to_nat(2u);
v___x_1354_ = lean_nat_dec_eq(v___x_1352_, v___x_1353_);
if (v___x_1354_ == 0)
{
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
else
{
lean_object* v___x_1355_; lean_object* v___x_1356_; 
v___x_1355_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__5));
v___x_1356_ = lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom(v_data_1311_, v___x_1355_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1356_) == 0)
{
lean_object* v_a_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; uint8_t v___x_1365_; lean_object* v___x_1366_; 
v_a_1357_ = lean_ctor_get(v___x_1356_, 0);
lean_inc(v_a_1357_);
lean_dec_ref_known(v___x_1356_, 1);
v___x_1358_ = lean_unsigned_to_nat(0u);
v___x_1359_ = lean_array_fget(v_x_1317_, v___x_1358_);
v___x_1360_ = lean_unsigned_to_nat(1u);
v___x_1361_ = lean_array_fget(v_x_1317_, v___x_1360_);
lean_dec_ref(v_x_1317_);
v___x_1362_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__8, &lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__8_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__8);
lean_inc_ref(v_a_1312_);
v___x_1363_ = l_Lean_mkAppN(v_a_1312_, v_ctx_1313_);
lean_inc_ref(v___x_1363_);
lean_inc(v___x_1361_);
lean_inc(v___x_1359_);
v___x_1364_ = l_Lean_mkApp3(v___x_1362_, v___x_1359_, v___x_1361_, v___x_1363_);
v___x_1365_ = 1;
v___x_1366_ = l_Lean_Meta_mkLambdaFVars(v_ctx_1313_, v___x_1364_, v___x_1349_, v___x_1354_, v___x_1349_, v___x_1354_, v___x_1365_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1366_) == 0)
{
lean_object* v_a_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; 
v_a_1367_ = lean_ctor_get(v___x_1366_, 0);
lean_inc(v_a_1367_);
lean_dec_ref_known(v___x_1366_, 1);
v___x_1368_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__11, &lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__11_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__11);
v___x_1369_ = l_Lean_mkApp3(v___x_1368_, v___x_1359_, v___x_1361_, v___x_1363_);
v___x_1370_ = l_Lean_Meta_mkLambdaFVars(v_ctx_1313_, v___x_1369_, v___x_1349_, v___x_1354_, v___x_1349_, v___x_1354_, v___x_1365_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
lean_dec_ref(v_ctx_1313_);
if (lean_obj_tag(v___x_1370_) == 0)
{
lean_object* v_a_1371_; lean_object* v___x_1372_; 
v_a_1371_ = lean_ctor_get(v___x_1370_, 0);
lean_inc(v_a_1371_);
lean_dec_ref_known(v___x_1370_, 1);
lean_inc(v___y_1322_);
lean_inc_ref(v___y_1321_);
lean_inc(v___y_1320_);
lean_inc_ref(v___y_1319_);
lean_inc(v_a_1367_);
v___x_1372_ = lean_infer_type(v_a_1367_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1372_) == 0)
{
lean_object* v_a_1373_; lean_object* v___x_1374_; 
v_a_1373_ = lean_ctor_get(v___x_1372_, 0);
lean_inc(v_a_1373_);
lean_dec_ref_known(v___x_1372_, 1);
lean_inc(v___y_1322_);
lean_inc_ref(v___y_1321_);
lean_inc(v___y_1320_);
lean_inc_ref(v___y_1319_);
lean_inc(v_a_1371_);
v___x_1374_ = lean_infer_type(v_a_1371_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1374_) == 0)
{
lean_object* v_a_1375_; lean_object* v___x_1376_; 
v_a_1375_ = lean_ctor_get(v___x_1374_, 0);
lean_inc(v_a_1375_);
lean_dec_ref_known(v___x_1374_, 1);
v___x_1376_ = l_Lean_MVarId_assert(v_fst_1314_, v_pre_1345_, v_a_1375_, v_a_1371_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1376_) == 0)
{
lean_object* v_a_1377_; lean_object* v___x_1378_; 
v_a_1377_ = lean_ctor_get(v___x_1376_, 0);
lean_inc(v_a_1377_);
lean_dec_ref_known(v___x_1376_, 1);
v___x_1378_ = l_Lean_MVarId_assert(v_a_1377_, v_a_1357_, v_a_1373_, v_a_1367_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1378_) == 0)
{
lean_object* v_a_1379_; lean_object* v___x_1380_; 
v_a_1379_ = lean_ctor_get(v___x_1378_, 0);
lean_inc(v_a_1379_);
lean_dec_ref_known(v___x_1378_, 1);
v___x_1380_ = l_Lean_Meta_intro1Core(v_a_1379_, v___x_1354_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1380_) == 0)
{
lean_object* v_a_1381_; lean_object* v___x_1383_; uint8_t v_isShared_1384_; uint8_t v_isSharedCheck_1413_; 
v_a_1381_ = lean_ctor_get(v___x_1380_, 0);
v_isSharedCheck_1413_ = !lean_is_exclusive(v___x_1380_);
if (v_isSharedCheck_1413_ == 0)
{
v___x_1383_ = v___x_1380_;
v_isShared_1384_ = v_isSharedCheck_1413_;
goto v_resetjp_1382_;
}
else
{
lean_inc(v_a_1381_);
lean_dec(v___x_1380_);
v___x_1383_ = lean_box(0);
v_isShared_1384_ = v_isSharedCheck_1413_;
goto v_resetjp_1382_;
}
v_resetjp_1382_:
{
lean_object* v_fst_1385_; lean_object* v_snd_1386_; lean_object* v___x_1388_; uint8_t v_isShared_1389_; uint8_t v_isSharedCheck_1412_; 
v_fst_1385_ = lean_ctor_get(v_a_1381_, 0);
v_snd_1386_ = lean_ctor_get(v_a_1381_, 1);
v_isSharedCheck_1412_ = !lean_is_exclusive(v_a_1381_);
if (v_isSharedCheck_1412_ == 0)
{
v___x_1388_ = v_a_1381_;
v_isShared_1389_ = v_isSharedCheck_1412_;
goto v_resetjp_1387_;
}
else
{
lean_inc(v_snd_1386_);
lean_inc(v_fst_1385_);
lean_dec(v_a_1381_);
v___x_1388_ = lean_box(0);
v_isShared_1389_ = v_isSharedCheck_1412_;
goto v_resetjp_1387_;
}
v_resetjp_1387_:
{
lean_object* v_g_1391_; 
if (lean_obj_tag(v_a_1312_) == 1)
{
lean_object* v_fvarId_1401_; lean_object* v___x_1402_; 
v_fvarId_1401_ = lean_ctor_get(v_a_1312_, 0);
lean_inc(v_fvarId_1401_);
lean_dec_ref_known(v_a_1312_, 1);
v___x_1402_ = l_Lean_MVarId_clear(v_snd_1386_, v_fvarId_1401_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1402_) == 0)
{
lean_object* v_a_1403_; 
v_a_1403_ = lean_ctor_get(v___x_1402_, 0);
lean_inc(v_a_1403_);
lean_dec_ref_known(v___x_1402_, 1);
v_g_1391_ = v_a_1403_;
goto v___jp_1390_;
}
else
{
lean_object* v_a_1404_; lean_object* v___x_1406_; uint8_t v_isShared_1407_; uint8_t v_isSharedCheck_1411_; 
lean_del_object(v___x_1388_);
lean_dec(v_fst_1385_);
lean_del_object(v___x_1383_);
v_a_1404_ = lean_ctor_get(v___x_1402_, 0);
v_isSharedCheck_1411_ = !lean_is_exclusive(v___x_1402_);
if (v_isSharedCheck_1411_ == 0)
{
v___x_1406_ = v___x_1402_;
v_isShared_1407_ = v_isSharedCheck_1411_;
goto v_resetjp_1405_;
}
else
{
lean_inc(v_a_1404_);
lean_dec(v___x_1402_);
v___x_1406_ = lean_box(0);
v_isShared_1407_ = v_isSharedCheck_1411_;
goto v_resetjp_1405_;
}
v_resetjp_1405_:
{
lean_object* v___x_1409_; 
if (v_isShared_1407_ == 0)
{
v___x_1409_ = v___x_1406_;
goto v_reusejp_1408_;
}
else
{
lean_object* v_reuseFailAlloc_1410_; 
v_reuseFailAlloc_1410_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1410_, 0, v_a_1404_);
v___x_1409_ = v_reuseFailAlloc_1410_;
goto v_reusejp_1408_;
}
v_reusejp_1408_:
{
return v___x_1409_;
}
}
}
}
else
{
lean_dec_ref(v_a_1312_);
v_g_1391_ = v_snd_1386_;
goto v___jp_1390_;
}
v___jp_1390_:
{
lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1395_; 
v___x_1392_ = lean_box(0);
v___x_1393_ = l_Lean_Expr_fvar___override(v_fst_1385_);
if (v_isShared_1389_ == 0)
{
lean_ctor_set(v___x_1388_, 1, v_g_1391_);
lean_ctor_set(v___x_1388_, 0, v___x_1393_);
v___x_1395_ = v___x_1388_;
goto v_reusejp_1394_;
}
else
{
lean_object* v_reuseFailAlloc_1400_; 
v_reuseFailAlloc_1400_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1400_, 0, v___x_1393_);
lean_ctor_set(v_reuseFailAlloc_1400_, 1, v_g_1391_);
v___x_1395_ = v_reuseFailAlloc_1400_;
goto v_reusejp_1394_;
}
v_reusejp_1394_:
{
lean_object* v___x_1396_; lean_object* v___x_1398_; 
v___x_1396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1396_, 0, v___x_1392_);
lean_ctor_set(v___x_1396_, 1, v___x_1395_);
if (v_isShared_1384_ == 0)
{
lean_ctor_set(v___x_1383_, 0, v___x_1396_);
v___x_1398_ = v___x_1383_;
goto v_reusejp_1397_;
}
else
{
lean_object* v_reuseFailAlloc_1399_; 
v_reuseFailAlloc_1399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1399_, 0, v___x_1396_);
v___x_1398_ = v_reuseFailAlloc_1399_;
goto v_reusejp_1397_;
}
v_reusejp_1397_:
{
return v___x_1398_;
}
}
}
}
}
}
else
{
lean_object* v_a_1414_; lean_object* v___x_1416_; uint8_t v_isShared_1417_; uint8_t v_isSharedCheck_1421_; 
lean_dec_ref(v_a_1312_);
v_a_1414_ = lean_ctor_get(v___x_1380_, 0);
v_isSharedCheck_1421_ = !lean_is_exclusive(v___x_1380_);
if (v_isSharedCheck_1421_ == 0)
{
v___x_1416_ = v___x_1380_;
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
else
{
lean_inc(v_a_1414_);
lean_dec(v___x_1380_);
v___x_1416_ = lean_box(0);
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
v_resetjp_1415_:
{
lean_object* v___x_1419_; 
if (v_isShared_1417_ == 0)
{
v___x_1419_ = v___x_1416_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1420_; 
v_reuseFailAlloc_1420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1420_, 0, v_a_1414_);
v___x_1419_ = v_reuseFailAlloc_1420_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
return v___x_1419_;
}
}
}
}
else
{
lean_object* v_a_1422_; lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1429_; 
lean_dec_ref(v_a_1312_);
v_a_1422_ = lean_ctor_get(v___x_1378_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1378_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___x_1378_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___x_1378_);
v___x_1424_ = lean_box(0);
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
v_resetjp_1423_:
{
lean_object* v___x_1427_; 
if (v_isShared_1425_ == 0)
{
v___x_1427_ = v___x_1424_;
goto v_reusejp_1426_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v_a_1422_);
v___x_1427_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1426_;
}
v_reusejp_1426_:
{
return v___x_1427_;
}
}
}
}
else
{
lean_object* v_a_1430_; lean_object* v___x_1432_; uint8_t v_isShared_1433_; uint8_t v_isSharedCheck_1437_; 
lean_dec(v_a_1373_);
lean_dec(v_a_1367_);
lean_dec(v_a_1357_);
lean_dec_ref(v_a_1312_);
v_a_1430_ = lean_ctor_get(v___x_1376_, 0);
v_isSharedCheck_1437_ = !lean_is_exclusive(v___x_1376_);
if (v_isSharedCheck_1437_ == 0)
{
v___x_1432_ = v___x_1376_;
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
else
{
lean_inc(v_a_1430_);
lean_dec(v___x_1376_);
v___x_1432_ = lean_box(0);
v_isShared_1433_ = v_isSharedCheck_1437_;
goto v_resetjp_1431_;
}
v_resetjp_1431_:
{
lean_object* v___x_1435_; 
if (v_isShared_1433_ == 0)
{
v___x_1435_ = v___x_1432_;
goto v_reusejp_1434_;
}
else
{
lean_object* v_reuseFailAlloc_1436_; 
v_reuseFailAlloc_1436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1436_, 0, v_a_1430_);
v___x_1435_ = v_reuseFailAlloc_1436_;
goto v_reusejp_1434_;
}
v_reusejp_1434_:
{
return v___x_1435_;
}
}
}
}
else
{
lean_object* v_a_1438_; lean_object* v___x_1440_; uint8_t v_isShared_1441_; uint8_t v_isSharedCheck_1445_; 
lean_dec(v_a_1373_);
lean_dec(v_a_1371_);
lean_dec(v_a_1367_);
lean_dec(v_a_1357_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_a_1312_);
v_a_1438_ = lean_ctor_get(v___x_1374_, 0);
v_isSharedCheck_1445_ = !lean_is_exclusive(v___x_1374_);
if (v_isSharedCheck_1445_ == 0)
{
v___x_1440_ = v___x_1374_;
v_isShared_1441_ = v_isSharedCheck_1445_;
goto v_resetjp_1439_;
}
else
{
lean_inc(v_a_1438_);
lean_dec(v___x_1374_);
v___x_1440_ = lean_box(0);
v_isShared_1441_ = v_isSharedCheck_1445_;
goto v_resetjp_1439_;
}
v_resetjp_1439_:
{
lean_object* v___x_1443_; 
if (v_isShared_1441_ == 0)
{
v___x_1443_ = v___x_1440_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1444_; 
v_reuseFailAlloc_1444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1444_, 0, v_a_1438_);
v___x_1443_ = v_reuseFailAlloc_1444_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
return v___x_1443_;
}
}
}
}
else
{
lean_object* v_a_1446_; lean_object* v___x_1448_; uint8_t v_isShared_1449_; uint8_t v_isSharedCheck_1453_; 
lean_dec(v_a_1371_);
lean_dec(v_a_1367_);
lean_dec(v_a_1357_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_a_1312_);
v_a_1446_ = lean_ctor_get(v___x_1372_, 0);
v_isSharedCheck_1453_ = !lean_is_exclusive(v___x_1372_);
if (v_isSharedCheck_1453_ == 0)
{
v___x_1448_ = v___x_1372_;
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
else
{
lean_inc(v_a_1446_);
lean_dec(v___x_1372_);
v___x_1448_ = lean_box(0);
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
v_resetjp_1447_:
{
lean_object* v___x_1451_; 
if (v_isShared_1449_ == 0)
{
v___x_1451_ = v___x_1448_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_a_1446_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
}
}
else
{
lean_object* v_a_1454_; lean_object* v___x_1456_; uint8_t v_isShared_1457_; uint8_t v_isSharedCheck_1461_; 
lean_dec(v_a_1367_);
lean_dec(v_a_1357_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_a_1312_);
v_a_1454_ = lean_ctor_get(v___x_1370_, 0);
v_isSharedCheck_1461_ = !lean_is_exclusive(v___x_1370_);
if (v_isSharedCheck_1461_ == 0)
{
v___x_1456_ = v___x_1370_;
v_isShared_1457_ = v_isSharedCheck_1461_;
goto v_resetjp_1455_;
}
else
{
lean_inc(v_a_1454_);
lean_dec(v___x_1370_);
v___x_1456_ = lean_box(0);
v_isShared_1457_ = v_isSharedCheck_1461_;
goto v_resetjp_1455_;
}
v_resetjp_1455_:
{
lean_object* v___x_1459_; 
if (v_isShared_1457_ == 0)
{
v___x_1459_ = v___x_1456_;
goto v_reusejp_1458_;
}
else
{
lean_object* v_reuseFailAlloc_1460_; 
v_reuseFailAlloc_1460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1460_, 0, v_a_1454_);
v___x_1459_ = v_reuseFailAlloc_1460_;
goto v_reusejp_1458_;
}
v_reusejp_1458_:
{
return v___x_1459_;
}
}
}
}
else
{
lean_object* v_a_1462_; lean_object* v___x_1464_; uint8_t v_isShared_1465_; uint8_t v_isSharedCheck_1469_; 
lean_dec_ref(v___x_1363_);
lean_dec(v___x_1361_);
lean_dec(v___x_1359_);
lean_dec(v_a_1357_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1462_ = lean_ctor_get(v___x_1366_, 0);
v_isSharedCheck_1469_ = !lean_is_exclusive(v___x_1366_);
if (v_isSharedCheck_1469_ == 0)
{
v___x_1464_ = v___x_1366_;
v_isShared_1465_ = v_isSharedCheck_1469_;
goto v_resetjp_1463_;
}
else
{
lean_inc(v_a_1462_);
lean_dec(v___x_1366_);
v___x_1464_ = lean_box(0);
v_isShared_1465_ = v_isSharedCheck_1469_;
goto v_resetjp_1463_;
}
v_resetjp_1463_:
{
lean_object* v___x_1467_; 
if (v_isShared_1465_ == 0)
{
v___x_1467_ = v___x_1464_;
goto v_reusejp_1466_;
}
else
{
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v_a_1462_);
v___x_1467_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1466_;
}
v_reusejp_1466_:
{
return v___x_1467_;
}
}
}
}
else
{
lean_object* v_a_1470_; lean_object* v___x_1472_; uint8_t v_isShared_1473_; uint8_t v_isSharedCheck_1477_; 
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1470_ = lean_ctor_get(v___x_1356_, 0);
v_isSharedCheck_1477_ = !lean_is_exclusive(v___x_1356_);
if (v_isSharedCheck_1477_ == 0)
{
v___x_1472_ = v___x_1356_;
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
else
{
lean_inc(v_a_1470_);
lean_dec(v___x_1356_);
v___x_1472_ = lean_box(0);
v_isShared_1473_ = v_isSharedCheck_1477_;
goto v_resetjp_1471_;
}
v_resetjp_1471_:
{
lean_object* v___x_1475_; 
if (v_isShared_1473_ == 0)
{
v___x_1475_ = v___x_1472_;
goto v_reusejp_1474_;
}
else
{
lean_object* v_reuseFailAlloc_1476_; 
v_reuseFailAlloc_1476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1476_, 0, v_a_1470_);
v___x_1475_ = v_reuseFailAlloc_1476_;
goto v_reusejp_1474_;
}
v_reusejp_1474_:
{
return v___x_1475_;
}
}
}
}
}
}
else
{
lean_dec_ref(v_str_1347_);
if (lean_obj_tag(v_us_1346_) == 1)
{
lean_object* v_tail_1478_; 
v_tail_1478_ = lean_ctor_get(v_us_1346_, 1);
if (lean_obj_tag(v_tail_1478_) == 0)
{
lean_object* v_head_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; uint8_t v___x_1482_; 
v_head_1479_ = lean_ctor_get(v_us_1346_, 0);
lean_inc(v_head_1479_);
v___x_1480_ = lean_array_get_size(v_x_1317_);
v___x_1481_ = lean_unsigned_to_nat(2u);
v___x_1482_ = lean_nat_dec_eq(v___x_1480_, v___x_1481_);
if (v___x_1482_ == 0)
{
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
else
{
lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___y_1486_; lean_object* v___y_1487_; uint8_t v___y_1488_; uint8_t v___y_1489_; lean_object* v___y_1490_; lean_object* v___y_1491_; uint8_t v___y_1492_; uint8_t v___y_1493_; lean_object* v___y_1494_; lean_object* v_dataVal_1495_; lean_object* v_specVal_1496_; lean_object* v___y_1497_; lean_object* v___y_1498_; lean_object* v___y_1499_; lean_object* v___y_1500_; lean_object* v___x_1550_; 
v___x_1483_ = lean_unsigned_to_nat(1u);
v___x_1484_ = lean_array_fget(v_x_1317_, v___x_1483_);
lean_inc(v___x_1484_);
v___x_1550_ = lp_mathlib_Lean_Expr_getBinderName(v___x_1484_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1550_) == 0)
{
lean_object* v_a_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___y_1555_; lean_object* v___y_1556_; lean_object* v___y_1557_; lean_object* v_ctx_x27_1558_; lean_object* v___y_1559_; lean_object* v___y_1560_; lean_object* v___y_1561_; lean_object* v___y_1562_; lean_object* v___y_1598_; lean_object* v___y_1599_; lean_object* v___y_1600_; lean_object* v___y_1601_; lean_object* v___y_1602_; lean_object* v___y_1603_; lean_object* v___y_1604_; lean_object* v___y_1605_; lean_object* v___y_1616_; 
v_a_1551_ = lean_ctor_get(v___x_1550_, 0);
lean_inc(v_a_1551_);
lean_dec_ref_known(v___x_1550_, 1);
v___x_1552_ = lean_unsigned_to_nat(0u);
v___x_1553_ = lean_array_fget(v_x_1317_, v___x_1552_);
lean_dec_ref(v_x_1317_);
if (lean_obj_tag(v_a_1551_) == 0)
{
lean_object* v___x_1691_; 
v___x_1691_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__5));
v___y_1616_ = v___x_1691_;
goto v___jp_1615_;
}
else
{
lean_object* v_val_1692_; 
v_val_1692_ = lean_ctor_get(v_a_1551_, 0);
lean_inc(v_val_1692_);
lean_dec_ref_known(v_a_1551_, 1);
v___y_1616_ = v_val_1692_;
goto v___jp_1615_;
}
v___jp_1554_:
{
uint8_t v___x_1563_; uint8_t v___x_1564_; lean_object* v___x_1565_; 
v___x_1563_ = 0;
v___x_1564_ = 1;
lean_inc(v___x_1553_);
v___x_1565_ = l_Lean_Meta_mkForallFVars(v_ctx_x27_1558_, v___x_1553_, v___x_1563_, v___x_1482_, v___x_1482_, v___x_1564_, v___y_1559_, v___y_1560_, v___y_1561_, v___y_1562_);
if (lean_obj_tag(v___x_1565_) == 0)
{
lean_object* v_a_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; 
v_a_1566_ = lean_ctor_get(v___x_1565_, 0);
lean_inc(v_a_1566_);
lean_dec_ref_known(v___x_1565_, 1);
v___x_1567_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__14));
lean_inc_ref(v_us_1346_);
v___x_1568_ = l_Lean_Expr_const___override(v___x_1567_, v_us_1346_);
lean_inc_ref(v_a_1312_);
v___x_1569_ = l_Lean_mkAppN(v_a_1312_, v_ctx_1313_);
lean_inc_ref(v___x_1569_);
lean_inc_n(v___x_1484_, 2);
lean_inc_n(v___x_1553_, 2);
v___x_1570_ = l_Lean_mkApp3(v___x_1568_, v___x_1553_, v___x_1484_, v___x_1569_);
v___x_1571_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__16));
v___x_1572_ = l_Lean_Expr_const___override(v___x_1571_, v_us_1346_);
v___x_1573_ = l_Lean_mkApp3(v___x_1572_, v___x_1553_, v___x_1484_, v___x_1569_);
if (lean_obj_tag(v___y_1557_) == 1)
{
lean_object* v_val_1574_; lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; 
v_val_1574_ = lean_ctor_get(v___y_1557_, 0);
lean_inc(v_val_1574_);
lean_dec_ref_known(v___y_1557_, 1);
lean_inc_ref(v_ctx_1313_);
v___x_1575_ = lean_array_to_list(v_ctx_1313_);
v___x_1576_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1576_, 0, v___x_1570_);
lean_ctor_set(v___x_1576_, 1, v___x_1573_);
lean_inc(v___x_1484_);
v___x_1577_ = lp_mathlib_Mathlib_Tactic_Choose_mkSometimes(v_head_1479_, v___x_1553_, v_val_1574_, v___x_1484_, v___x_1575_, v___x_1576_, v___y_1559_, v___y_1560_, v___y_1561_, v___y_1562_);
if (lean_obj_tag(v___x_1577_) == 0)
{
lean_object* v_a_1578_; lean_object* v_fst_1579_; lean_object* v_snd_1580_; 
v_a_1578_ = lean_ctor_get(v___x_1577_, 0);
lean_inc(v_a_1578_);
lean_dec_ref_known(v___x_1577_, 1);
v_fst_1579_ = lean_ctor_get(v_a_1578_, 0);
lean_inc(v_fst_1579_);
v_snd_1580_ = lean_ctor_get(v_a_1578_, 1);
lean_inc(v_snd_1580_);
lean_dec(v_a_1578_);
lean_inc_ref(v_ctx_x27_1558_);
v___y_1486_ = v___y_1555_;
v___y_1487_ = v_ctx_x27_1558_;
v___y_1488_ = v___x_1563_;
v___y_1489_ = v___x_1564_;
v___y_1490_ = v_a_1566_;
v___y_1491_ = v_ctx_x27_1558_;
v___y_1492_ = v___x_1563_;
v___y_1493_ = v___x_1564_;
v___y_1494_ = v___y_1556_;
v_dataVal_1495_ = v_fst_1579_;
v_specVal_1496_ = v_snd_1580_;
v___y_1497_ = v___y_1559_;
v___y_1498_ = v___y_1560_;
v___y_1499_ = v___y_1561_;
v___y_1500_ = v___y_1562_;
goto v___jp_1485_;
}
else
{
lean_object* v_a_1581_; lean_object* v___x_1583_; uint8_t v_isShared_1584_; uint8_t v_isSharedCheck_1588_; 
lean_dec(v_a_1566_);
lean_dec_ref(v_ctx_x27_1558_);
lean_dec(v___y_1556_);
lean_dec(v___y_1555_);
lean_dec(v___x_1484_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1581_ = lean_ctor_get(v___x_1577_, 0);
v_isSharedCheck_1588_ = !lean_is_exclusive(v___x_1577_);
if (v_isSharedCheck_1588_ == 0)
{
v___x_1583_ = v___x_1577_;
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
else
{
lean_inc(v_a_1581_);
lean_dec(v___x_1577_);
v___x_1583_ = lean_box(0);
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
v_resetjp_1582_:
{
lean_object* v___x_1586_; 
if (v_isShared_1584_ == 0)
{
v___x_1586_ = v___x_1583_;
goto v_reusejp_1585_;
}
else
{
lean_object* v_reuseFailAlloc_1587_; 
v_reuseFailAlloc_1587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1587_, 0, v_a_1581_);
v___x_1586_ = v_reuseFailAlloc_1587_;
goto v_reusejp_1585_;
}
v_reusejp_1585_:
{
return v___x_1586_;
}
}
}
}
else
{
lean_dec(v___y_1557_);
lean_dec(v___x_1553_);
lean_dec(v_head_1479_);
lean_inc_ref(v_ctx_x27_1558_);
v___y_1486_ = v___y_1555_;
v___y_1487_ = v_ctx_x27_1558_;
v___y_1488_ = v___x_1563_;
v___y_1489_ = v___x_1564_;
v___y_1490_ = v_a_1566_;
v___y_1491_ = v_ctx_x27_1558_;
v___y_1492_ = v___x_1563_;
v___y_1493_ = v___x_1564_;
v___y_1494_ = v___y_1556_;
v_dataVal_1495_ = v___x_1570_;
v_specVal_1496_ = v___x_1573_;
v___y_1497_ = v___y_1559_;
v___y_1498_ = v___y_1560_;
v___y_1499_ = v___y_1561_;
v___y_1500_ = v___y_1562_;
goto v___jp_1485_;
}
}
else
{
lean_object* v_a_1589_; lean_object* v___x_1591_; uint8_t v_isShared_1592_; uint8_t v_isSharedCheck_1596_; 
lean_dec_ref(v_ctx_x27_1558_);
lean_dec(v___y_1557_);
lean_dec(v___y_1556_);
lean_dec(v___y_1555_);
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1589_ = lean_ctor_get(v___x_1565_, 0);
v_isSharedCheck_1596_ = !lean_is_exclusive(v___x_1565_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1591_ = v___x_1565_;
v_isShared_1592_ = v_isSharedCheck_1596_;
goto v_resetjp_1590_;
}
else
{
lean_inc(v_a_1589_);
lean_dec(v___x_1565_);
v___x_1591_ = lean_box(0);
v_isShared_1592_ = v_isSharedCheck_1596_;
goto v_resetjp_1590_;
}
v_resetjp_1590_:
{
lean_object* v___x_1594_; 
if (v_isShared_1592_ == 0)
{
v___x_1594_ = v___x_1591_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_a_1589_);
v___x_1594_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1593_;
}
v_reusejp_1593_:
{
return v___x_1594_;
}
}
}
}
v___jp_1597_:
{
if (lean_obj_tag(v___y_1605_) == 0)
{
lean_object* v_a_1606_; 
v_a_1606_ = lean_ctor_get(v___y_1605_, 0);
lean_inc(v_a_1606_);
lean_dec_ref_known(v___y_1605_, 1);
v___y_1555_ = v___y_1598_;
v___y_1556_ = v___y_1601_;
v___y_1557_ = v___y_1604_;
v_ctx_x27_1558_ = v_a_1606_;
v___y_1559_ = v___y_1599_;
v___y_1560_ = v___y_1600_;
v___y_1561_ = v___y_1603_;
v___y_1562_ = v___y_1602_;
goto v___jp_1554_;
}
else
{
lean_object* v_a_1607_; lean_object* v___x_1609_; uint8_t v_isShared_1610_; uint8_t v_isSharedCheck_1614_; 
lean_dec(v___y_1604_);
lean_dec(v___y_1601_);
lean_dec(v___y_1598_);
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1607_ = lean_ctor_get(v___y_1605_, 0);
v_isSharedCheck_1614_ = !lean_is_exclusive(v___y_1605_);
if (v_isSharedCheck_1614_ == 0)
{
v___x_1609_ = v___y_1605_;
v_isShared_1610_ = v_isSharedCheck_1614_;
goto v_resetjp_1608_;
}
else
{
lean_inc(v_a_1607_);
lean_dec(v___y_1605_);
v___x_1609_ = lean_box(0);
v_isShared_1610_ = v_isSharedCheck_1614_;
goto v_resetjp_1608_;
}
v_resetjp_1608_:
{
lean_object* v___x_1612_; 
if (v_isShared_1610_ == 0)
{
v___x_1612_ = v___x_1609_;
goto v_reusejp_1611_;
}
else
{
lean_object* v_reuseFailAlloc_1613_; 
v_reuseFailAlloc_1613_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1613_, 0, v_a_1607_);
v___x_1612_ = v_reuseFailAlloc_1613_;
goto v_reusejp_1611_;
}
v_reusejp_1611_:
{
return v___x_1612_;
}
}
}
}
v___jp_1615_:
{
lean_object* v___x_1617_; 
v___x_1617_ = lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom(v_data_1311_, v___y_1616_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1617_) == 0)
{
if (v_nondep_1315_ == 0)
{
lean_object* v_a_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; 
v_a_1618_ = lean_ctor_get(v___x_1617_, 0);
lean_inc(v_a_1618_);
lean_dec_ref_known(v___x_1617_, 1);
v___x_1619_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__17));
v___x_1620_ = lean_box(0);
lean_inc_ref(v_ctx_1313_);
v___y_1555_ = v_a_1618_;
v___y_1556_ = v___x_1619_;
v___y_1557_ = v___x_1620_;
v_ctx_x27_1558_ = v_ctx_1313_;
v___y_1559_ = v___y_1319_;
v___y_1560_ = v___y_1320_;
v___y_1561_ = v___y_1321_;
v___y_1562_ = v___y_1322_;
goto v___jp_1554_;
}
else
{
lean_object* v_a_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; uint8_t v___x_1626_; lean_object* v___x_1627_; 
v_a_1621_ = lean_ctor_get(v___x_1617_, 0);
lean_inc(v_a_1621_);
lean_dec_ref_known(v___x_1617_, 1);
v___x_1622_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__18));
lean_inc_ref(v_us_1346_);
v___x_1623_ = l_Lean_Expr_const___override(v___x_1622_, v_us_1346_);
lean_inc(v___x_1553_);
lean_inc_ref(v___x_1623_);
v___x_1624_ = l_Lean_Expr_app___override(v___x_1623_, v___x_1553_);
lean_inc_ref(v___x_1624_);
v___x_1625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1625_, 0, v___x_1624_);
v___x_1626_ = 0;
v___x_1627_ = l_Lean_Meta_mkFreshExprMVar(v___x_1625_, v___x_1626_, v_pre_1345_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1627_) == 0)
{
lean_object* v_a_1628_; lean_object* v___x_1629_; size_t v_sz_1630_; size_t v___x_1631_; lean_object* v___x_1632_; 
v_a_1628_ = lean_ctor_get(v___x_1627_, 0);
lean_inc(v_a_1628_);
lean_dec_ref_known(v___x_1627_, 1);
v___x_1629_ = l_Lean_Expr_mvarId_x21(v_a_1628_);
v_sz_1630_ = lean_array_size(v_ctx_1313_);
v___x_1631_ = ((size_t)0ULL);
lean_inc_ref(v_us_1346_);
v___x_1632_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Choose_choose1_spec__5(v___x_1623_, v_us_1346_, v_pre_1345_, v_ctx_1313_, v_sz_1630_, v___x_1631_, v___x_1629_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1632_) == 0)
{
lean_object* v_a_1633_; lean_object* v___x_1634_; 
v_a_1633_ = lean_ctor_get(v___x_1632_, 0);
lean_inc(v_a_1633_);
lean_dec_ref_known(v___x_1632_, 1);
v___x_1634_ = l_Lean_MVarId_intros(v_a_1633_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1634_) == 0)
{
lean_object* v_a_1635_; lean_object* v_snd_1636_; lean_object* v___f_1637_; lean_object* v___x_1638_; 
v_a_1635_ = lean_ctor_get(v___x_1634_, 0);
lean_inc(v_a_1635_);
lean_dec_ref_known(v___x_1634_, 1);
v_snd_1636_ = lean_ctor_get(v_a_1635_, 1);
lean_inc_n(v_snd_1636_, 2);
lean_dec(v_a_1635_);
v___f_1637_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__2___boxed), 8, 3);
lean_closure_set(v___f_1637_, 0, v_snd_1636_);
lean_closure_set(v___f_1637_, 1, v___x_1624_);
lean_closure_set(v___f_1637_, 2, v_a_1628_);
v___x_1638_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(v_snd_1636_, v___f_1637_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
if (lean_obj_tag(v___x_1638_) == 0)
{
lean_object* v_a_1639_; lean_object* v_snd_1640_; 
v_a_1639_ = lean_ctor_get(v___x_1638_, 0);
lean_inc(v_a_1639_);
lean_dec_ref_known(v___x_1638_, 1);
v_snd_1640_ = lean_ctor_get(v_a_1639_, 1);
lean_inc(v_snd_1640_);
if (lean_obj_tag(v_snd_1640_) == 0)
{
lean_object* v_fst_1641_; 
v_fst_1641_ = lean_ctor_get(v_a_1639_, 0);
lean_inc(v_fst_1641_);
lean_dec(v_a_1639_);
lean_inc_ref(v_ctx_1313_);
v___y_1555_ = v_a_1621_;
v___y_1556_ = v_fst_1641_;
v___y_1557_ = v_snd_1640_;
v_ctx_x27_1558_ = v_ctx_1313_;
v___y_1559_ = v___y_1319_;
v___y_1560_ = v___y_1320_;
v___y_1561_ = v___y_1321_;
v___y_1562_ = v___y_1322_;
goto v___jp_1554_;
}
else
{
lean_object* v_fst_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; uint8_t v___x_1645_; 
v_fst_1642_ = lean_ctor_get(v_a_1639_, 0);
lean_inc(v_fst_1642_);
lean_dec(v_a_1639_);
v___x_1643_ = lean_array_get_size(v_ctx_1313_);
v___x_1644_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__19));
v___x_1645_ = lean_nat_dec_lt(v___x_1552_, v___x_1643_);
if (v___x_1645_ == 0)
{
v___y_1555_ = v_a_1621_;
v___y_1556_ = v_fst_1642_;
v___y_1557_ = v_snd_1640_;
v_ctx_x27_1558_ = v___x_1644_;
v___y_1559_ = v___y_1319_;
v___y_1560_ = v___y_1320_;
v___y_1561_ = v___y_1321_;
v___y_1562_ = v___y_1322_;
goto v___jp_1554_;
}
else
{
uint8_t v___x_1646_; 
v___x_1646_ = lean_nat_dec_le(v___x_1643_, v___x_1643_);
if (v___x_1646_ == 0)
{
if (v___x_1645_ == 0)
{
v___y_1555_ = v_a_1621_;
v___y_1556_ = v_fst_1642_;
v___y_1557_ = v_snd_1640_;
v_ctx_x27_1558_ = v___x_1644_;
v___y_1559_ = v___y_1319_;
v___y_1560_ = v___y_1320_;
v___y_1561_ = v___y_1321_;
v___y_1562_ = v___y_1322_;
goto v___jp_1554_;
}
else
{
size_t v___x_1647_; lean_object* v___x_1648_; 
v___x_1647_ = lean_usize_of_nat(v___x_1643_);
v___x_1648_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4(v_ctx_1313_, v___x_1631_, v___x_1647_, v___x_1644_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
v___y_1598_ = v_a_1621_;
v___y_1599_ = v___y_1319_;
v___y_1600_ = v___y_1320_;
v___y_1601_ = v_fst_1642_;
v___y_1602_ = v___y_1322_;
v___y_1603_ = v___y_1321_;
v___y_1604_ = v_snd_1640_;
v___y_1605_ = v___x_1648_;
goto v___jp_1597_;
}
}
else
{
size_t v___x_1649_; lean_object* v___x_1650_; 
v___x_1649_ = lean_usize_of_nat(v___x_1643_);
v___x_1650_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Choose_choose1_spec__4(v_ctx_1313_, v___x_1631_, v___x_1649_, v___x_1644_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
v___y_1598_ = v_a_1621_;
v___y_1599_ = v___y_1319_;
v___y_1600_ = v___y_1320_;
v___y_1601_ = v_fst_1642_;
v___y_1602_ = v___y_1322_;
v___y_1603_ = v___y_1321_;
v___y_1604_ = v_snd_1640_;
v___y_1605_ = v___x_1650_;
goto v___jp_1597_;
}
}
}
}
else
{
lean_object* v_a_1651_; lean_object* v___x_1653_; uint8_t v_isShared_1654_; uint8_t v_isSharedCheck_1658_; 
lean_dec(v_a_1621_);
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1651_ = lean_ctor_get(v___x_1638_, 0);
v_isSharedCheck_1658_ = !lean_is_exclusive(v___x_1638_);
if (v_isSharedCheck_1658_ == 0)
{
v___x_1653_ = v___x_1638_;
v_isShared_1654_ = v_isSharedCheck_1658_;
goto v_resetjp_1652_;
}
else
{
lean_inc(v_a_1651_);
lean_dec(v___x_1638_);
v___x_1653_ = lean_box(0);
v_isShared_1654_ = v_isSharedCheck_1658_;
goto v_resetjp_1652_;
}
v_resetjp_1652_:
{
lean_object* v___x_1656_; 
if (v_isShared_1654_ == 0)
{
v___x_1656_ = v___x_1653_;
goto v_reusejp_1655_;
}
else
{
lean_object* v_reuseFailAlloc_1657_; 
v_reuseFailAlloc_1657_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1657_, 0, v_a_1651_);
v___x_1656_ = v_reuseFailAlloc_1657_;
goto v_reusejp_1655_;
}
v_reusejp_1655_:
{
return v___x_1656_;
}
}
}
}
else
{
lean_object* v_a_1659_; lean_object* v___x_1661_; uint8_t v_isShared_1662_; uint8_t v_isSharedCheck_1666_; 
lean_dec(v_a_1628_);
lean_dec_ref(v___x_1624_);
lean_dec(v_a_1621_);
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1659_ = lean_ctor_get(v___x_1634_, 0);
v_isSharedCheck_1666_ = !lean_is_exclusive(v___x_1634_);
if (v_isSharedCheck_1666_ == 0)
{
v___x_1661_ = v___x_1634_;
v_isShared_1662_ = v_isSharedCheck_1666_;
goto v_resetjp_1660_;
}
else
{
lean_inc(v_a_1659_);
lean_dec(v___x_1634_);
v___x_1661_ = lean_box(0);
v_isShared_1662_ = v_isSharedCheck_1666_;
goto v_resetjp_1660_;
}
v_resetjp_1660_:
{
lean_object* v___x_1664_; 
if (v_isShared_1662_ == 0)
{
v___x_1664_ = v___x_1661_;
goto v_reusejp_1663_;
}
else
{
lean_object* v_reuseFailAlloc_1665_; 
v_reuseFailAlloc_1665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1665_, 0, v_a_1659_);
v___x_1664_ = v_reuseFailAlloc_1665_;
goto v_reusejp_1663_;
}
v_reusejp_1663_:
{
return v___x_1664_;
}
}
}
}
else
{
lean_object* v_a_1667_; lean_object* v___x_1669_; uint8_t v_isShared_1670_; uint8_t v_isSharedCheck_1674_; 
lean_dec(v_a_1628_);
lean_dec_ref(v___x_1624_);
lean_dec(v_a_1621_);
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1667_ = lean_ctor_get(v___x_1632_, 0);
v_isSharedCheck_1674_ = !lean_is_exclusive(v___x_1632_);
if (v_isSharedCheck_1674_ == 0)
{
v___x_1669_ = v___x_1632_;
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
else
{
lean_inc(v_a_1667_);
lean_dec(v___x_1632_);
v___x_1669_ = lean_box(0);
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
v_resetjp_1668_:
{
lean_object* v___x_1672_; 
if (v_isShared_1670_ == 0)
{
v___x_1672_ = v___x_1669_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_a_1667_);
v___x_1672_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
return v___x_1672_;
}
}
}
}
else
{
lean_object* v_a_1675_; lean_object* v___x_1677_; uint8_t v_isShared_1678_; uint8_t v_isSharedCheck_1682_; 
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1621_);
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1675_ = lean_ctor_get(v___x_1627_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v___x_1627_);
if (v_isSharedCheck_1682_ == 0)
{
v___x_1677_ = v___x_1627_;
v_isShared_1678_ = v_isSharedCheck_1682_;
goto v_resetjp_1676_;
}
else
{
lean_inc(v_a_1675_);
lean_dec(v___x_1627_);
v___x_1677_ = lean_box(0);
v_isShared_1678_ = v_isSharedCheck_1682_;
goto v_resetjp_1676_;
}
v_resetjp_1676_:
{
lean_object* v___x_1680_; 
if (v_isShared_1678_ == 0)
{
v___x_1680_ = v___x_1677_;
goto v_reusejp_1679_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v_a_1675_);
v___x_1680_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1679_;
}
v_reusejp_1679_:
{
return v___x_1680_;
}
}
}
}
}
else
{
lean_object* v_a_1683_; lean_object* v___x_1685_; uint8_t v_isShared_1686_; uint8_t v_isSharedCheck_1690_; 
lean_dec(v___x_1553_);
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1683_ = lean_ctor_get(v___x_1617_, 0);
v_isSharedCheck_1690_ = !lean_is_exclusive(v___x_1617_);
if (v_isSharedCheck_1690_ == 0)
{
v___x_1685_ = v___x_1617_;
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
else
{
lean_inc(v_a_1683_);
lean_dec(v___x_1617_);
v___x_1685_ = lean_box(0);
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
v_resetjp_1684_:
{
lean_object* v___x_1688_; 
if (v_isShared_1686_ == 0)
{
v___x_1688_ = v___x_1685_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v_a_1683_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
}
else
{
lean_object* v_a_1693_; lean_object* v___x_1695_; uint8_t v_isShared_1696_; uint8_t v_isSharedCheck_1700_; 
lean_dec(v___x_1484_);
lean_dec(v_head_1479_);
lean_dec_ref_known(v_us_1346_, 2);
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v_a_1693_ = lean_ctor_get(v___x_1550_, 0);
v_isSharedCheck_1700_ = !lean_is_exclusive(v___x_1550_);
if (v_isSharedCheck_1700_ == 0)
{
v___x_1695_ = v___x_1550_;
v_isShared_1696_ = v_isSharedCheck_1700_;
goto v_resetjp_1694_;
}
else
{
lean_inc(v_a_1693_);
lean_dec(v___x_1550_);
v___x_1695_ = lean_box(0);
v_isShared_1696_ = v_isSharedCheck_1700_;
goto v_resetjp_1694_;
}
v_resetjp_1694_:
{
lean_object* v___x_1698_; 
if (v_isShared_1696_ == 0)
{
v___x_1698_ = v___x_1695_;
goto v_reusejp_1697_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v_a_1693_);
v___x_1698_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1697_;
}
v_reusejp_1697_:
{
return v___x_1698_;
}
}
}
v___jp_1485_:
{
lean_object* v___x_1501_; 
v___x_1501_ = l_Lean_Meta_mkLambdaFVars(v___y_1491_, v_dataVal_1495_, v___y_1492_, v___x_1482_, v___y_1492_, v___x_1482_, v___y_1493_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_);
lean_dec_ref(v___y_1491_);
if (lean_obj_tag(v___x_1501_) == 0)
{
lean_object* v_a_1502_; lean_object* v___x_1503_; 
v_a_1502_ = lean_ctor_get(v___x_1501_, 0);
lean_inc(v_a_1502_);
lean_dec_ref_known(v___x_1501_, 1);
v___x_1503_ = l_Lean_Meta_mkLambdaFVars(v_ctx_1313_, v_specVal_1496_, v___y_1492_, v___x_1482_, v___y_1492_, v___x_1482_, v___y_1493_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_);
if (lean_obj_tag(v___x_1503_) == 0)
{
lean_object* v_a_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___f_1508_; lean_object* v___x_1509_; 
v_a_1504_ = lean_ctor_get(v___x_1503_, 0);
lean_inc(v_a_1504_);
lean_dec_ref_known(v___x_1503_, 1);
v___x_1505_ = lean_box(v___y_1488_);
v___x_1506_ = lean_box(v___x_1482_);
v___x_1507_ = lean_box(v___y_1489_);
lean_inc_ref(v___y_1490_);
v___f_1508_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___lam__1___boxed), 18, 12);
lean_closure_set(v___f_1508_, 0, v___y_1487_);
lean_closure_set(v___f_1508_, 1, v___x_1484_);
lean_closure_set(v___f_1508_, 2, v_ctx_1313_);
lean_closure_set(v___f_1508_, 3, v___x_1505_);
lean_closure_set(v___f_1508_, 4, v___x_1506_);
lean_closure_set(v___f_1508_, 5, v___x_1507_);
lean_closure_set(v___f_1508_, 6, v_fst_1314_);
lean_closure_set(v___f_1508_, 7, v___x_1483_);
lean_closure_set(v___f_1508_, 8, v_a_1502_);
lean_closure_set(v___f_1508_, 9, v_a_1504_);
lean_closure_set(v___f_1508_, 10, v___y_1486_);
lean_closure_set(v___f_1508_, 11, v___y_1490_);
v___x_1509_ = lp_mathlib_Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2___redArg(v_pre_1345_, v___y_1490_, v___f_1508_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_);
if (lean_obj_tag(v___x_1509_) == 0)
{
lean_object* v_a_1510_; 
v_a_1510_ = lean_ctor_get(v___x_1509_, 0);
lean_inc(v_a_1510_);
lean_dec_ref_known(v___x_1509_, 1);
if (lean_obj_tag(v_a_1312_) == 1)
{
lean_object* v_fst_1511_; lean_object* v_snd_1512_; lean_object* v_fvarId_1513_; lean_object* v___x_1514_; 
v_fst_1511_ = lean_ctor_get(v_a_1510_, 0);
lean_inc(v_fst_1511_);
v_snd_1512_ = lean_ctor_get(v_a_1510_, 1);
lean_inc(v_snd_1512_);
lean_dec(v_a_1510_);
v_fvarId_1513_ = lean_ctor_get(v_a_1312_, 0);
lean_inc(v_fvarId_1513_);
lean_dec_ref_known(v_a_1312_, 1);
v___x_1514_ = l_Lean_MVarId_clear(v_snd_1512_, v_fvarId_1513_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_);
if (lean_obj_tag(v___x_1514_) == 0)
{
lean_object* v_a_1515_; 
v_a_1515_ = lean_ctor_get(v___x_1514_, 0);
lean_inc(v_a_1515_);
lean_dec_ref_known(v___x_1514_, 1);
v___y_1325_ = v___y_1494_;
v___y_1326_ = v_fst_1511_;
v_g_1327_ = v_a_1515_;
goto v___jp_1324_;
}
else
{
lean_object* v_a_1516_; lean_object* v___x_1518_; uint8_t v_isShared_1519_; uint8_t v_isSharedCheck_1523_; 
lean_dec(v_fst_1511_);
lean_dec(v___y_1494_);
v_a_1516_ = lean_ctor_get(v___x_1514_, 0);
v_isSharedCheck_1523_ = !lean_is_exclusive(v___x_1514_);
if (v_isSharedCheck_1523_ == 0)
{
v___x_1518_ = v___x_1514_;
v_isShared_1519_ = v_isSharedCheck_1523_;
goto v_resetjp_1517_;
}
else
{
lean_inc(v_a_1516_);
lean_dec(v___x_1514_);
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
lean_object* v_fst_1524_; lean_object* v_snd_1525_; 
lean_dec_ref(v_a_1312_);
v_fst_1524_ = lean_ctor_get(v_a_1510_, 0);
lean_inc(v_fst_1524_);
v_snd_1525_ = lean_ctor_get(v_a_1510_, 1);
lean_inc(v_snd_1525_);
lean_dec(v_a_1510_);
v___y_1325_ = v___y_1494_;
v___y_1326_ = v_fst_1524_;
v_g_1327_ = v_snd_1525_;
goto v___jp_1324_;
}
}
else
{
lean_object* v_a_1526_; lean_object* v___x_1528_; uint8_t v_isShared_1529_; uint8_t v_isSharedCheck_1533_; 
lean_dec(v___y_1494_);
lean_dec_ref(v_a_1312_);
v_a_1526_ = lean_ctor_get(v___x_1509_, 0);
v_isSharedCheck_1533_ = !lean_is_exclusive(v___x_1509_);
if (v_isSharedCheck_1533_ == 0)
{
v___x_1528_ = v___x_1509_;
v_isShared_1529_ = v_isSharedCheck_1533_;
goto v_resetjp_1527_;
}
else
{
lean_inc(v_a_1526_);
lean_dec(v___x_1509_);
v___x_1528_ = lean_box(0);
v_isShared_1529_ = v_isSharedCheck_1533_;
goto v_resetjp_1527_;
}
v_resetjp_1527_:
{
lean_object* v___x_1531_; 
if (v_isShared_1529_ == 0)
{
v___x_1531_ = v___x_1528_;
goto v_reusejp_1530_;
}
else
{
lean_object* v_reuseFailAlloc_1532_; 
v_reuseFailAlloc_1532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1532_, 0, v_a_1526_);
v___x_1531_ = v_reuseFailAlloc_1532_;
goto v_reusejp_1530_;
}
v_reusejp_1530_:
{
return v___x_1531_;
}
}
}
}
else
{
lean_object* v_a_1534_; lean_object* v___x_1536_; uint8_t v_isShared_1537_; uint8_t v_isSharedCheck_1541_; 
lean_dec(v_a_1502_);
lean_dec(v___y_1494_);
lean_dec_ref(v___y_1490_);
lean_dec_ref(v___y_1487_);
lean_dec(v___y_1486_);
lean_dec(v___x_1484_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1534_ = lean_ctor_get(v___x_1503_, 0);
v_isSharedCheck_1541_ = !lean_is_exclusive(v___x_1503_);
if (v_isSharedCheck_1541_ == 0)
{
v___x_1536_ = v___x_1503_;
v_isShared_1537_ = v_isSharedCheck_1541_;
goto v_resetjp_1535_;
}
else
{
lean_inc(v_a_1534_);
lean_dec(v___x_1503_);
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
lean_object* v_a_1542_; lean_object* v___x_1544_; uint8_t v_isShared_1545_; uint8_t v_isSharedCheck_1549_; 
lean_dec_ref(v_specVal_1496_);
lean_dec(v___y_1494_);
lean_dec_ref(v___y_1490_);
lean_dec_ref(v___y_1487_);
lean_dec(v___y_1486_);
lean_dec(v___x_1484_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
v_a_1542_ = lean_ctor_get(v___x_1501_, 0);
v_isSharedCheck_1549_ = !lean_is_exclusive(v___x_1501_);
if (v_isSharedCheck_1549_ == 0)
{
v___x_1544_ = v___x_1501_;
v_isShared_1545_ = v_isSharedCheck_1549_;
goto v_resetjp_1543_;
}
else
{
lean_inc(v_a_1542_);
lean_dec(v___x_1501_);
v___x_1544_ = lean_box(0);
v_isShared_1545_ = v_isSharedCheck_1549_;
goto v_resetjp_1543_;
}
v_resetjp_1543_:
{
lean_object* v___x_1547_; 
if (v_isShared_1545_ == 0)
{
v___x_1547_ = v___x_1544_;
goto v_reusejp_1546_;
}
else
{
lean_object* v_reuseFailAlloc_1548_; 
v_reuseFailAlloc_1548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1548_, 0, v_a_1542_);
v___x_1547_ = v_reuseFailAlloc_1548_;
goto v_reusejp_1546_;
}
v_reusejp_1546_:
{
return v___x_1547_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_us_1346_, 2);
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
}
else
{
lean_dec(v_us_1346_);
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
}
}
else
{
lean_dec(v_pre_1345_);
lean_dec_ref_known(v_declName_1344_, 2);
lean_dec_ref_known(v_x_1316_, 2);
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
}
else
{
lean_dec(v_declName_1344_);
lean_dec_ref_known(v_x_1316_, 2);
lean_dec_ref(v_x_1317_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
}
else
{
lean_dec_ref(v_x_1317_);
lean_dec_ref(v_x_1316_);
lean_dec(v_fst_1314_);
lean_dec_ref(v_ctx_1313_);
lean_dec_ref(v_a_1312_);
lean_dec(v_data_1311_);
v___y_1332_ = v___y_1319_;
v___y_1333_ = v___y_1320_;
v___y_1334_ = v___y_1321_;
v___y_1335_ = v___y_1322_;
goto v___jp_1331_;
}
}
v___jp_1324_:
{
lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; 
v___x_1328_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1328_, 0, v___y_1326_);
lean_ctor_set(v___x_1328_, 1, v_g_1327_);
v___x_1329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1329_, 0, v___y_1325_);
lean_ctor_set(v___x_1329_, 1, v___x_1328_);
v___x_1330_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1329_);
return v___x_1330_;
}
v___jp_1331_:
{
lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1336_ = lean_obj_once(&lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__1, &lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__1_once, _init_lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__1);
v___x_1337_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0___redArg(v___x_1336_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
return v___x_1337_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___boxed(lean_object* v_data_1701_, lean_object* v_a_1702_, lean_object* v_ctx_1703_, lean_object* v_fst_1704_, lean_object* v_nondep_1705_, lean_object* v_x_1706_, lean_object* v_x_1707_, lean_object* v_x_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_){
_start:
{
uint8_t v_nondep_boxed_1714_; lean_object* v_res_1715_; 
v_nondep_boxed_1714_ = lean_unbox(v_nondep_1705_);
v_res_1715_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6(v_data_1701_, v_a_1702_, v_ctx_1703_, v_fst_1704_, v_nondep_boxed_1714_, v_x_1706_, v_x_1707_, v_x_1708_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_);
lean_dec(v___y_1712_);
lean_dec_ref(v___y_1711_);
lean_dec(v___y_1710_);
lean_dec_ref(v___y_1709_);
return v_res_1715_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1716_; lean_object* v_dummy_1717_; 
v___x_1716_ = lean_box(0);
v_dummy_1717_ = l_Lean_Expr_sort___override(v___x_1716_);
return v_dummy_1717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0(lean_object* v_data_1718_, lean_object* v_a_1719_, lean_object* v_fst_1720_, uint8_t v_nondep_1721_, lean_object* v_ctx_1722_, lean_object* v_t_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_){
_start:
{
lean_object* v_keyedConfig_1729_; uint8_t v_trackZetaDelta_1730_; lean_object* v_zetaDeltaSet_1731_; lean_object* v_lctx_1732_; lean_object* v_localInstances_1733_; lean_object* v_defEqCtx_x3f_1734_; lean_object* v_synthPendingDepth_1735_; lean_object* v_customCanUnfoldPredicate_x3f_1736_; uint8_t v_univApprox_1737_; uint8_t v_inTypeClassResolution_1738_; uint8_t v_cacheInferType_1739_; uint8_t v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; 
v_keyedConfig_1729_ = lean_ctor_get(v___y_1724_, 0);
v_trackZetaDelta_1730_ = lean_ctor_get_uint8(v___y_1724_, sizeof(void*)*7);
v_zetaDeltaSet_1731_ = lean_ctor_get(v___y_1724_, 1);
v_lctx_1732_ = lean_ctor_get(v___y_1724_, 2);
v_localInstances_1733_ = lean_ctor_get(v___y_1724_, 3);
v_defEqCtx_x3f_1734_ = lean_ctor_get(v___y_1724_, 4);
v_synthPendingDepth_1735_ = lean_ctor_get(v___y_1724_, 5);
v_customCanUnfoldPredicate_x3f_1736_ = lean_ctor_get(v___y_1724_, 6);
v_univApprox_1737_ = lean_ctor_get_uint8(v___y_1724_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1738_ = lean_ctor_get_uint8(v___y_1724_, sizeof(void*)*7 + 2);
v_cacheInferType_1739_ = lean_ctor_get_uint8(v___y_1724_, sizeof(void*)*7 + 3);
v___x_1740_ = 0;
lean_inc_ref(v_keyedConfig_1729_);
v___x_1741_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1740_, v_keyedConfig_1729_);
lean_inc(v_customCanUnfoldPredicate_x3f_1736_);
lean_inc(v_synthPendingDepth_1735_);
lean_inc(v_defEqCtx_x3f_1734_);
lean_inc_ref(v_localInstances_1733_);
lean_inc_ref(v_lctx_1732_);
lean_inc(v_zetaDeltaSet_1731_);
v___x_1742_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1742_, 0, v___x_1741_);
lean_ctor_set(v___x_1742_, 1, v_zetaDeltaSet_1731_);
lean_ctor_set(v___x_1742_, 2, v_lctx_1732_);
lean_ctor_set(v___x_1742_, 3, v_localInstances_1733_);
lean_ctor_set(v___x_1742_, 4, v_defEqCtx_x3f_1734_);
lean_ctor_set(v___x_1742_, 5, v_synthPendingDepth_1735_);
lean_ctor_set(v___x_1742_, 6, v_customCanUnfoldPredicate_x3f_1736_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*7, v_trackZetaDelta_1730_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*7 + 1, v_univApprox_1737_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1738_);
lean_ctor_set_uint8(v___x_1742_, sizeof(void*)*7 + 3, v_cacheInferType_1739_);
lean_inc(v___y_1727_);
lean_inc_ref(v___y_1726_);
lean_inc(v___y_1725_);
v___x_1743_ = lean_whnf(v_t_1723_, v___x_1742_, v___y_1725_, v___y_1726_, v___y_1727_);
if (lean_obj_tag(v___x_1743_) == 0)
{
lean_object* v_a_1744_; lean_object* v_dummy_1745_; lean_object* v_nargs_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; 
v_a_1744_ = lean_ctor_get(v___x_1743_, 0);
lean_inc(v_a_1744_);
lean_dec_ref_known(v___x_1743_, 1);
v_dummy_1745_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___closed__0);
v_nargs_1746_ = l_Lean_Expr_getAppNumArgs(v_a_1744_);
lean_inc(v_nargs_1746_);
v___x_1747_ = lean_mk_array(v_nargs_1746_, v_dummy_1745_);
v___x_1748_ = lean_unsigned_to_nat(1u);
v___x_1749_ = lean_nat_sub(v_nargs_1746_, v___x_1748_);
lean_dec(v_nargs_1746_);
v___x_1750_ = lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6(v_data_1718_, v_a_1719_, v_ctx_1722_, v_fst_1720_, v_nondep_1721_, v_a_1744_, v___x_1747_, v___x_1749_, v___y_1724_, v___y_1725_, v___y_1726_, v___y_1727_);
return v___x_1750_;
}
else
{
lean_object* v_a_1751_; lean_object* v___x_1753_; uint8_t v_isShared_1754_; uint8_t v_isSharedCheck_1758_; 
lean_dec_ref(v_ctx_1722_);
lean_dec(v_fst_1720_);
lean_dec_ref(v_a_1719_);
lean_dec(v_data_1718_);
v_a_1751_ = lean_ctor_get(v___x_1743_, 0);
v_isSharedCheck_1758_ = !lean_is_exclusive(v___x_1743_);
if (v_isSharedCheck_1758_ == 0)
{
v___x_1753_ = v___x_1743_;
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
else
{
lean_inc(v_a_1751_);
lean_dec(v___x_1743_);
v___x_1753_ = lean_box(0);
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
v_resetjp_1752_:
{
lean_object* v___x_1756_; 
if (v_isShared_1754_ == 0)
{
v___x_1756_ = v___x_1753_;
goto v_reusejp_1755_;
}
else
{
lean_object* v_reuseFailAlloc_1757_; 
v_reuseFailAlloc_1757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1757_, 0, v_a_1751_);
v___x_1756_ = v_reuseFailAlloc_1757_;
goto v_reusejp_1755_;
}
v_reusejp_1755_:
{
return v___x_1756_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___boxed(lean_object* v_data_1759_, lean_object* v_a_1760_, lean_object* v_fst_1761_, lean_object* v_nondep_1762_, lean_object* v_ctx_1763_, lean_object* v_t_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_){
_start:
{
uint8_t v_nondep_boxed_1770_; lean_object* v_res_1771_; 
v_nondep_boxed_1770_ = lean_unbox(v_nondep_1762_);
v_res_1771_ = lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0(v_data_1759_, v_a_1760_, v_fst_1761_, v_nondep_boxed_1770_, v_ctx_1763_, v_t_1764_, v___y_1765_, v___y_1766_, v___y_1767_, v___y_1768_);
lean_dec(v___y_1768_);
lean_dec_ref(v___y_1767_);
lean_dec(v___y_1766_);
lean_dec_ref(v___y_1765_);
return v_res_1771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__1(lean_object* v_snd_1772_, lean_object* v_data_1773_, lean_object* v_fst_1774_, uint8_t v_nondep_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_){
_start:
{
lean_object* v___x_1781_; lean_object* v_a_1782_; lean_object* v___x_1783_; 
v___x_1781_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Choose_choose1_spec__0___redArg(v_snd_1772_, v___y_1777_);
v_a_1782_ = lean_ctor_get(v___x_1781_, 0);
lean_inc_n(v_a_1782_, 2);
lean_dec_ref(v___x_1781_);
lean_inc(v___y_1779_);
lean_inc_ref(v___y_1778_);
lean_inc(v___y_1777_);
lean_inc_ref(v___y_1776_);
v___x_1783_ = lean_infer_type(v_a_1782_, v___y_1776_, v___y_1777_, v___y_1778_, v___y_1779_);
if (lean_obj_tag(v___x_1783_) == 0)
{
lean_object* v_a_1784_; lean_object* v___x_1785_; lean_object* v___f_1786_; uint8_t v___x_1787_; lean_object* v___x_1788_; 
v_a_1784_ = lean_ctor_get(v___x_1783_, 0);
lean_inc(v_a_1784_);
lean_dec_ref_known(v___x_1783_, 1);
v___x_1785_ = lean_box(v_nondep_1775_);
v___f_1786_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__0___boxed), 11, 4);
lean_closure_set(v___f_1786_, 0, v_data_1773_);
lean_closure_set(v___f_1786_, 1, v_a_1782_);
lean_closure_set(v___f_1786_, 2, v_fst_1774_);
lean_closure_set(v___f_1786_, 3, v___x_1785_);
v___x_1787_ = 0;
v___x_1788_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Tactic_Choose_choose1_spec__7___redArg(v_a_1784_, v___f_1786_, v___x_1787_, v___x_1787_, v___y_1776_, v___y_1777_, v___y_1778_, v___y_1779_);
lean_dec(v___y_1779_);
lean_dec_ref(v___y_1778_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
return v___x_1788_;
}
else
{
lean_object* v_a_1789_; lean_object* v___x_1791_; uint8_t v_isShared_1792_; uint8_t v_isSharedCheck_1796_; 
lean_dec(v_a_1782_);
lean_dec(v___y_1779_);
lean_dec_ref(v___y_1778_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
lean_dec(v_fst_1774_);
lean_dec(v_data_1773_);
v_a_1789_ = lean_ctor_get(v___x_1783_, 0);
v_isSharedCheck_1796_ = !lean_is_exclusive(v___x_1783_);
if (v_isSharedCheck_1796_ == 0)
{
v___x_1791_ = v___x_1783_;
v_isShared_1792_ = v_isSharedCheck_1796_;
goto v_resetjp_1790_;
}
else
{
lean_inc(v_a_1789_);
lean_dec(v___x_1783_);
v___x_1791_ = lean_box(0);
v_isShared_1792_ = v_isSharedCheck_1796_;
goto v_resetjp_1790_;
}
v_resetjp_1790_:
{
lean_object* v___x_1794_; 
if (v_isShared_1792_ == 0)
{
v___x_1794_ = v___x_1791_;
goto v_reusejp_1793_;
}
else
{
lean_object* v_reuseFailAlloc_1795_; 
v_reuseFailAlloc_1795_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1795_, 0, v_a_1789_);
v___x_1794_ = v_reuseFailAlloc_1795_;
goto v_reusejp_1793_;
}
v_reusejp_1793_:
{
return v___x_1794_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__1___boxed(lean_object* v_snd_1797_, lean_object* v_data_1798_, lean_object* v_fst_1799_, lean_object* v_nondep_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_){
_start:
{
uint8_t v_nondep_boxed_1806_; lean_object* v_res_1807_; 
v_nondep_boxed_1806_ = lean_unbox(v_nondep_1800_);
v_res_1807_ = lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__1(v_snd_1797_, v_data_1798_, v_fst_1799_, v_nondep_boxed_1806_, v___y_1801_, v___y_1802_, v___y_1803_, v___y_1804_);
return v_res_1807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1(lean_object* v_g_1808_, uint8_t v_nondep_1809_, lean_object* v_h_1810_, lean_object* v_data_1811_, lean_object* v_a_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_, lean_object* v_a_1815_){
_start:
{
lean_object* v_fst_1818_; lean_object* v_snd_1819_; lean_object* v___y_1820_; lean_object* v___y_1821_; lean_object* v___y_1822_; lean_object* v___y_1823_; 
if (lean_obj_tag(v_h_1810_) == 0)
{
uint8_t v___x_1827_; lean_object* v___x_1828_; 
v___x_1827_ = 1;
v___x_1828_ = l_Lean_Meta_intro1Core(v_g_1808_, v___x_1827_, v_a_1812_, v_a_1813_, v_a_1814_, v_a_1815_);
if (lean_obj_tag(v___x_1828_) == 0)
{
lean_object* v_a_1829_; lean_object* v_fst_1830_; lean_object* v_snd_1831_; lean_object* v___x_1832_; 
v_a_1829_ = lean_ctor_get(v___x_1828_, 0);
lean_inc(v_a_1829_);
lean_dec_ref_known(v___x_1828_, 1);
v_fst_1830_ = lean_ctor_get(v_a_1829_, 0);
lean_inc(v_fst_1830_);
v_snd_1831_ = lean_ctor_get(v_a_1829_, 1);
lean_inc(v_snd_1831_);
lean_dec(v_a_1829_);
v___x_1832_ = l_Lean_Expr_fvar___override(v_fst_1830_);
v_fst_1818_ = v_snd_1831_;
v_snd_1819_ = v___x_1832_;
v___y_1820_ = v_a_1812_;
v___y_1821_ = v_a_1813_;
v___y_1822_ = v_a_1814_;
v___y_1823_ = v_a_1815_;
goto v___jp_1817_;
}
else
{
lean_object* v_a_1833_; lean_object* v___x_1835_; uint8_t v_isShared_1836_; uint8_t v_isSharedCheck_1840_; 
lean_dec(v_data_1811_);
v_a_1833_ = lean_ctor_get(v___x_1828_, 0);
v_isSharedCheck_1840_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_1840_ == 0)
{
v___x_1835_ = v___x_1828_;
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
else
{
lean_inc(v_a_1833_);
lean_dec(v___x_1828_);
v___x_1835_ = lean_box(0);
v_isShared_1836_ = v_isSharedCheck_1840_;
goto v_resetjp_1834_;
}
v_resetjp_1834_:
{
lean_object* v___x_1838_; 
if (v_isShared_1836_ == 0)
{
v___x_1838_ = v___x_1835_;
goto v_reusejp_1837_;
}
else
{
lean_object* v_reuseFailAlloc_1839_; 
v_reuseFailAlloc_1839_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1839_, 0, v_a_1833_);
v___x_1838_ = v_reuseFailAlloc_1839_;
goto v_reusejp_1837_;
}
v_reusejp_1837_:
{
return v___x_1838_;
}
}
}
}
else
{
lean_object* v_val_1841_; 
v_val_1841_ = lean_ctor_get(v_h_1810_, 0);
lean_inc(v_val_1841_);
lean_dec_ref_known(v_h_1810_, 1);
v_fst_1818_ = v_g_1808_;
v_snd_1819_ = v_val_1841_;
v___y_1820_ = v_a_1812_;
v___y_1821_ = v_a_1813_;
v___y_1822_ = v_a_1814_;
v___y_1823_ = v_a_1815_;
goto v___jp_1817_;
}
v___jp_1817_:
{
lean_object* v___x_1824_; lean_object* v___f_1825_; lean_object* v___x_1826_; 
v___x_1824_ = lean_box(v_nondep_1809_);
lean_inc(v_fst_1818_);
v___f_1825_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Choose_choose1___lam__1___boxed), 9, 4);
lean_closure_set(v___f_1825_, 0, v_snd_1819_);
lean_closure_set(v___f_1825_, 1, v_data_1811_);
lean_closure_set(v___f_1825_, 2, v_fst_1818_);
lean_closure_set(v___f_1825_, 3, v___x_1824_);
v___x_1826_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1_spec__3___redArg(v_fst_1818_, v___f_1825_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_);
return v___x_1826_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1___boxed(lean_object* v_g_1842_, lean_object* v_nondep_1843_, lean_object* v_h_1844_, lean_object* v_data_1845_, lean_object* v_a_1846_, lean_object* v_a_1847_, lean_object* v_a_1848_, lean_object* v_a_1849_, lean_object* v_a_1850_){
_start:
{
uint8_t v_nondep_boxed_1851_; lean_object* v_res_1852_; 
v_nondep_boxed_1851_ = lean_unbox(v_nondep_1843_);
v_res_1852_ = lp_mathlib_Mathlib_Tactic_Choose_choose1(v_g_1842_, v_nondep_boxed_1851_, v_h_1844_, v_data_1845_, v_a_1846_, v_a_1847_, v_a_1848_, v_a_1849_);
lean_dec(v_a_1849_);
lean_dec_ref(v_a_1848_);
lean_dec(v_a_1847_);
lean_dec_ref(v_a_1846_);
return v_res_1852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1(lean_object* v_mvarId_1853_, lean_object* v_val_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_, lean_object* v___y_1858_){
_start:
{
lean_object* v___x_1860_; 
v___x_1860_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___redArg(v_mvarId_1853_, v_val_1854_, v___y_1856_);
return v___x_1860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1___boxed(lean_object* v_mvarId_1861_, lean_object* v_val_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_){
_start:
{
lean_object* v_res_1868_; 
v_res_1868_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1(v_mvarId_1861_, v_val_1862_, v___y_1863_, v___y_1864_, v___y_1865_, v___y_1866_);
lean_dec(v___y_1866_);
lean_dec_ref(v___y_1865_);
lean_dec(v___y_1864_);
lean_dec_ref(v___y_1863_);
return v_res_1868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3(lean_object* v_00_u03b1_1869_, lean_object* v_name_1870_, uint8_t v_bi_1871_, lean_object* v_type_1872_, lean_object* v_k_1873_, uint8_t v_kind_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_){
_start:
{
lean_object* v___x_1880_; 
v___x_1880_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___redArg(v_name_1870_, v_bi_1871_, v_type_1872_, v_k_1873_, v_kind_1874_, v___y_1875_, v___y_1876_, v___y_1877_, v___y_1878_);
return v___x_1880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3___boxed(lean_object* v_00_u03b1_1881_, lean_object* v_name_1882_, lean_object* v_bi_1883_, lean_object* v_type_1884_, lean_object* v_k_1885_, lean_object* v_kind_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_){
_start:
{
uint8_t v_bi_boxed_1892_; uint8_t v_kind_boxed_1893_; lean_object* v_res_1894_; 
v_bi_boxed_1892_ = lean_unbox(v_bi_1883_);
v_kind_boxed_1893_ = lean_unbox(v_kind_1886_);
v_res_1894_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_Meta_withLocalDeclD___at___00Mathlib_Tactic_Choose_choose1_spec__2_spec__3(v_00_u03b1_1881_, v_name_1882_, v_bi_boxed_1892_, v_type_1884_, v_k_1885_, v_kind_boxed_1893_, v___y_1887_, v___y_1888_, v___y_1889_, v___y_1890_);
lean_dec(v___y_1890_);
lean_dec_ref(v___y_1889_);
lean_dec(v___y_1888_);
lean_dec_ref(v___y_1887_);
return v_res_1894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1(lean_object* v_00_u03b2_1895_, lean_object* v_x_1896_, lean_object* v_x_1897_, lean_object* v_x_1898_){
_start:
{
lean_object* v___x_1899_; 
v___x_1899_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1___redArg(v_x_1896_, v_x_1897_, v_x_1898_);
return v___x_1899_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4(lean_object* v_00_u03b2_1900_, lean_object* v_x_1901_, size_t v_x_1902_, size_t v_x_1903_, lean_object* v_x_1904_, lean_object* v_x_1905_){
_start:
{
lean_object* v___x_1906_; 
v___x_1906_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___redArg(v_x_1901_, v_x_1902_, v_x_1903_, v_x_1904_, v_x_1905_);
return v___x_1906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4___boxed(lean_object* v_00_u03b2_1907_, lean_object* v_x_1908_, lean_object* v_x_1909_, lean_object* v_x_1910_, lean_object* v_x_1911_, lean_object* v_x_1912_){
_start:
{
size_t v_x_20802__boxed_1913_; size_t v_x_20803__boxed_1914_; lean_object* v_res_1915_; 
v_x_20802__boxed_1913_ = lean_unbox_usize(v_x_1909_);
lean_dec(v_x_1909_);
v_x_20803__boxed_1914_ = lean_unbox_usize(v_x_1910_);
lean_dec(v_x_1910_);
v_res_1915_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4(v_00_u03b2_1907_, v_x_1908_, v_x_20802__boxed_1913_, v_x_20803__boxed_1914_, v_x_1911_, v_x_1912_);
return v_res_1915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10(lean_object* v_00_u03b2_1916_, lean_object* v_n_1917_, lean_object* v_k_1918_, lean_object* v_v_1919_){
_start:
{
lean_object* v___x_1920_; 
v___x_1920_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10___redArg(v_n_1917_, v_k_1918_, v_v_1919_);
return v___x_1920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11(lean_object* v_00_u03b2_1921_, size_t v_depth_1922_, lean_object* v_keys_1923_, lean_object* v_vals_1924_, lean_object* v_heq_1925_, lean_object* v_i_1926_, lean_object* v_entries_1927_){
_start:
{
lean_object* v___x_1928_; 
v___x_1928_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___redArg(v_depth_1922_, v_keys_1923_, v_vals_1924_, v_i_1926_, v_entries_1927_);
return v___x_1928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11___boxed(lean_object* v_00_u03b2_1929_, lean_object* v_depth_1930_, lean_object* v_keys_1931_, lean_object* v_vals_1932_, lean_object* v_heq_1933_, lean_object* v_i_1934_, lean_object* v_entries_1935_){
_start:
{
size_t v_depth_boxed_1936_; lean_object* v_res_1937_; 
v_depth_boxed_1936_ = lean_unbox_usize(v_depth_1930_);
lean_dec(v_depth_1930_);
v_res_1937_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__11(v_00_u03b2_1929_, v_depth_boxed_1936_, v_keys_1931_, v_vals_1932_, v_heq_1933_, v_i_1934_, v_entries_1935_);
lean_dec_ref(v_vals_1932_);
lean_dec_ref(v_keys_1931_);
return v_res_1937_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10_spec__11(lean_object* v_00_u03b2_1938_, lean_object* v_x_1939_, lean_object* v_x_1940_, lean_object* v_x_1941_, lean_object* v_x_1942_){
_start:
{
lean_object* v___x_1943_; 
v___x_1943_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Choose_choose1_spec__1_spec__1_spec__4_spec__10_spec__11___redArg(v_x_1939_, v_x_1940_, v_x_1941_, v_x_1942_);
return v___x_1943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___lam__0(lean_object* v_x_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_){
_start:
{
lean_object* v___x_1952_; 
lean_inc(v___y_1946_);
lean_inc_ref(v___y_1945_);
v___x_1952_ = lean_apply_7(v_x_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_, v___y_1950_, lean_box(0));
return v___x_1952_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___lam__0___boxed(lean_object* v_x_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_){
_start:
{
lean_object* v_res_1961_; 
v_res_1961_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___lam__0(v_x_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_);
lean_dec(v___y_1955_);
lean_dec_ref(v___y_1954_);
return v_res_1961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg(lean_object* v_mvarId_1962_, lean_object* v_x_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_){
_start:
{
lean_object* v___f_1971_; lean_object* v___x_1972_; 
lean_inc(v___y_1965_);
lean_inc_ref(v___y_1964_);
v___f_1971_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_1971_, 0, v_x_1963_);
lean_closure_set(v___f_1971_, 1, v___y_1964_);
lean_closure_set(v___f_1971_, 2, v___y_1965_);
v___x_1972_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1962_, v___f_1971_, v___y_1966_, v___y_1967_, v___y_1968_, v___y_1969_);
if (lean_obj_tag(v___x_1972_) == 0)
{
return v___x_1972_;
}
else
{
lean_object* v_a_1973_; lean_object* v___x_1975_; uint8_t v_isShared_1976_; uint8_t v_isSharedCheck_1980_; 
v_a_1973_ = lean_ctor_get(v___x_1972_, 0);
v_isSharedCheck_1980_ = !lean_is_exclusive(v___x_1972_);
if (v_isSharedCheck_1980_ == 0)
{
v___x_1975_ = v___x_1972_;
v_isShared_1976_ = v_isSharedCheck_1980_;
goto v_resetjp_1974_;
}
else
{
lean_inc(v_a_1973_);
lean_dec(v___x_1972_);
v___x_1975_ = lean_box(0);
v_isShared_1976_ = v_isSharedCheck_1980_;
goto v_resetjp_1974_;
}
v_resetjp_1974_:
{
lean_object* v___x_1978_; 
if (v_isShared_1976_ == 0)
{
v___x_1978_ = v___x_1975_;
goto v_reusejp_1977_;
}
else
{
lean_object* v_reuseFailAlloc_1979_; 
v_reuseFailAlloc_1979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1979_, 0, v_a_1973_);
v___x_1978_ = v_reuseFailAlloc_1979_;
goto v_reusejp_1977_;
}
v_reusejp_1977_:
{
return v___x_1978_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg___boxed(lean_object* v_mvarId_1981_, lean_object* v_x_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_){
_start:
{
lean_object* v_res_1990_; 
v_res_1990_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg(v_mvarId_1981_, v_x_1982_, v___y_1983_, v___y_1984_, v___y_1985_, v___y_1986_, v___y_1987_, v___y_1988_);
lean_dec(v___y_1988_);
lean_dec_ref(v___y_1987_);
lean_dec(v___y_1986_);
lean_dec_ref(v___y_1985_);
lean_dec(v___y_1984_);
lean_dec_ref(v___y_1983_);
return v_res_1990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1(lean_object* v_00_u03b1_1991_, lean_object* v_mvarId_1992_, lean_object* v_x_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_){
_start:
{
lean_object* v___x_2001_; 
v___x_2001_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg(v_mvarId_1992_, v_x_1993_, v___y_1994_, v___y_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_);
return v___x_2001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___boxed(lean_object* v_00_u03b1_2002_, lean_object* v_mvarId_2003_, lean_object* v_x_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_, lean_object* v___y_2011_){
_start:
{
lean_object* v_res_2012_; 
v_res_2012_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1(v_00_u03b1_2002_, v_mvarId_2003_, v_x_2004_, v___y_2005_, v___y_2006_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_);
lean_dec(v___y_2010_);
lean_dec_ref(v___y_2009_);
lean_dec(v___y_2008_);
lean_dec_ref(v___y_2007_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
return v_res_2012_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0(void){
_start:
{
lean_object* v___x_2013_; lean_object* v___x_2014_; 
v___x_2013_ = lean_box(1);
v___x_2014_ = l_Lean_MessageData_ofFormat(v___x_2013_);
return v___x_2014_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__3(void){
_start:
{
lean_object* v___x_2018_; lean_object* v___x_2019_; 
v___x_2018_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__2));
v___x_2019_ = l_Lean_MessageData_ofFormat(v___x_2018_);
return v___x_2019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4(lean_object* v_x_2020_, lean_object* v_x_2021_){
_start:
{
if (lean_obj_tag(v_x_2021_) == 0)
{
return v_x_2020_;
}
else
{
lean_object* v_head_2022_; lean_object* v_tail_2023_; lean_object* v___x_2025_; uint8_t v_isShared_2026_; uint8_t v_isSharedCheck_2045_; 
v_head_2022_ = lean_ctor_get(v_x_2021_, 0);
v_tail_2023_ = lean_ctor_get(v_x_2021_, 1);
v_isSharedCheck_2045_ = !lean_is_exclusive(v_x_2021_);
if (v_isSharedCheck_2045_ == 0)
{
v___x_2025_ = v_x_2021_;
v_isShared_2026_ = v_isSharedCheck_2045_;
goto v_resetjp_2024_;
}
else
{
lean_inc(v_tail_2023_);
lean_inc(v_head_2022_);
lean_dec(v_x_2021_);
v___x_2025_ = lean_box(0);
v_isShared_2026_ = v_isSharedCheck_2045_;
goto v_resetjp_2024_;
}
v_resetjp_2024_:
{
lean_object* v_before_2027_; lean_object* v___x_2029_; uint8_t v_isShared_2030_; uint8_t v_isSharedCheck_2043_; 
v_before_2027_ = lean_ctor_get(v_head_2022_, 0);
v_isSharedCheck_2043_ = !lean_is_exclusive(v_head_2022_);
if (v_isSharedCheck_2043_ == 0)
{
lean_object* v_unused_2044_; 
v_unused_2044_ = lean_ctor_get(v_head_2022_, 1);
lean_dec(v_unused_2044_);
v___x_2029_ = v_head_2022_;
v_isShared_2030_ = v_isSharedCheck_2043_;
goto v_resetjp_2028_;
}
else
{
lean_inc(v_before_2027_);
lean_dec(v_head_2022_);
v___x_2029_ = lean_box(0);
v_isShared_2030_ = v_isSharedCheck_2043_;
goto v_resetjp_2028_;
}
v_resetjp_2028_:
{
lean_object* v___x_2031_; lean_object* v___x_2033_; 
v___x_2031_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0);
if (v_isShared_2030_ == 0)
{
lean_ctor_set_tag(v___x_2029_, 7);
lean_ctor_set(v___x_2029_, 1, v___x_2031_);
lean_ctor_set(v___x_2029_, 0, v_x_2020_);
v___x_2033_ = v___x_2029_;
goto v_reusejp_2032_;
}
else
{
lean_object* v_reuseFailAlloc_2042_; 
v_reuseFailAlloc_2042_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2042_, 0, v_x_2020_);
lean_ctor_set(v_reuseFailAlloc_2042_, 1, v___x_2031_);
v___x_2033_ = v_reuseFailAlloc_2042_;
goto v_reusejp_2032_;
}
v_reusejp_2032_:
{
lean_object* v___x_2034_; lean_object* v___x_2036_; 
v___x_2034_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__3);
if (v_isShared_2026_ == 0)
{
lean_ctor_set_tag(v___x_2025_, 7);
lean_ctor_set(v___x_2025_, 1, v___x_2034_);
lean_ctor_set(v___x_2025_, 0, v___x_2033_);
v___x_2036_ = v___x_2025_;
goto v_reusejp_2035_;
}
else
{
lean_object* v_reuseFailAlloc_2041_; 
v_reuseFailAlloc_2041_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2041_, 0, v___x_2033_);
lean_ctor_set(v_reuseFailAlloc_2041_, 1, v___x_2034_);
v___x_2036_ = v_reuseFailAlloc_2041_;
goto v_reusejp_2035_;
}
v_reusejp_2035_:
{
lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; 
v___x_2037_ = l_Lean_MessageData_ofSyntax(v_before_2027_);
v___x_2038_ = l_Lean_indentD(v___x_2037_);
v___x_2039_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2039_, 0, v___x_2036_);
lean_ctor_set(v___x_2039_, 1, v___x_2038_);
v_x_2020_ = v___x_2039_;
v_x_2021_ = v_tail_2023_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__3(lean_object* v_opts_2046_, lean_object* v_opt_2047_){
_start:
{
lean_object* v_name_2048_; lean_object* v_defValue_2049_; lean_object* v_map_2050_; lean_object* v___x_2051_; 
v_name_2048_ = lean_ctor_get(v_opt_2047_, 0);
v_defValue_2049_ = lean_ctor_get(v_opt_2047_, 1);
v_map_2050_ = lean_ctor_get(v_opts_2046_, 0);
v___x_2051_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2050_, v_name_2048_);
if (lean_obj_tag(v___x_2051_) == 0)
{
uint8_t v___x_2052_; 
v___x_2052_ = lean_unbox(v_defValue_2049_);
return v___x_2052_;
}
else
{
lean_object* v_val_2053_; 
v_val_2053_ = lean_ctor_get(v___x_2051_, 0);
lean_inc(v_val_2053_);
lean_dec_ref_known(v___x_2051_, 1);
if (lean_obj_tag(v_val_2053_) == 1)
{
uint8_t v_v_2054_; 
v_v_2054_ = lean_ctor_get_uint8(v_val_2053_, 0);
lean_dec_ref_known(v_val_2053_, 0);
return v_v_2054_;
}
else
{
uint8_t v___x_2055_; 
lean_dec(v_val_2053_);
v___x_2055_ = lean_unbox(v_defValue_2049_);
return v___x_2055_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_opts_2056_, lean_object* v_opt_2057_){
_start:
{
uint8_t v_res_2058_; lean_object* v_r_2059_; 
v_res_2058_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__3(v_opts_2056_, v_opt_2057_);
lean_dec_ref(v_opt_2057_);
lean_dec_ref(v_opts_2056_);
v_r_2059_ = lean_box(v_res_2058_);
return v_r_2059_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_2063_; lean_object* v___x_2064_; 
v___x_2063_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__1));
v___x_2064_ = l_Lean_MessageData_ofFormat(v___x_2063_);
return v___x_2064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg(lean_object* v_msgData_2065_, lean_object* v_macroStack_2066_, lean_object* v___y_2067_){
_start:
{
lean_object* v_options_2069_; lean_object* v___x_2070_; uint8_t v___x_2071_; 
v_options_2069_ = lean_ctor_get(v___y_2067_, 2);
v___x_2070_ = l_Lean_Elab_pp_macroStack;
v___x_2071_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__3(v_options_2069_, v___x_2070_);
if (v___x_2071_ == 0)
{
lean_object* v___x_2072_; 
lean_dec(v_macroStack_2066_);
v___x_2072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2072_, 0, v_msgData_2065_);
return v___x_2072_;
}
else
{
if (lean_obj_tag(v_macroStack_2066_) == 0)
{
lean_object* v___x_2073_; 
v___x_2073_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2073_, 0, v_msgData_2065_);
return v___x_2073_;
}
else
{
lean_object* v_head_2074_; lean_object* v_after_2075_; lean_object* v___x_2077_; uint8_t v_isShared_2078_; uint8_t v_isSharedCheck_2090_; 
v_head_2074_ = lean_ctor_get(v_macroStack_2066_, 0);
lean_inc(v_head_2074_);
v_after_2075_ = lean_ctor_get(v_head_2074_, 1);
v_isSharedCheck_2090_ = !lean_is_exclusive(v_head_2074_);
if (v_isSharedCheck_2090_ == 0)
{
lean_object* v_unused_2091_; 
v_unused_2091_ = lean_ctor_get(v_head_2074_, 0);
lean_dec(v_unused_2091_);
v___x_2077_ = v_head_2074_;
v_isShared_2078_ = v_isSharedCheck_2090_;
goto v_resetjp_2076_;
}
else
{
lean_inc(v_after_2075_);
lean_dec(v_head_2074_);
v___x_2077_ = lean_box(0);
v_isShared_2078_ = v_isSharedCheck_2090_;
goto v_resetjp_2076_;
}
v_resetjp_2076_:
{
lean_object* v___x_2079_; lean_object* v___x_2081_; 
v___x_2079_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4___closed__0);
if (v_isShared_2078_ == 0)
{
lean_ctor_set_tag(v___x_2077_, 7);
lean_ctor_set(v___x_2077_, 1, v___x_2079_);
lean_ctor_set(v___x_2077_, 0, v_msgData_2065_);
v___x_2081_ = v___x_2077_;
goto v_reusejp_2080_;
}
else
{
lean_object* v_reuseFailAlloc_2089_; 
v_reuseFailAlloc_2089_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2089_, 0, v_msgData_2065_);
lean_ctor_set(v_reuseFailAlloc_2089_, 1, v___x_2079_);
v___x_2081_ = v_reuseFailAlloc_2089_;
goto v_reusejp_2080_;
}
v_reusejp_2080_:
{
lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v_msgData_2086_; lean_object* v___x_2087_; lean_object* v___x_2088_; 
v___x_2082_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___closed__2);
v___x_2083_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2083_, 0, v___x_2081_);
lean_ctor_set(v___x_2083_, 1, v___x_2082_);
v___x_2084_ = l_Lean_MessageData_ofSyntax(v_after_2075_);
v___x_2085_ = l_Lean_indentD(v___x_2084_);
v_msgData_2086_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2086_, 0, v___x_2083_);
lean_ctor_set(v_msgData_2086_, 1, v___x_2085_);
v___x_2087_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2_spec__4(v_msgData_2086_, v_macroStack_2066_);
v___x_2088_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2088_, 0, v___x_2087_);
return v___x_2088_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_msgData_2092_, lean_object* v_macroStack_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_){
_start:
{
lean_object* v_res_2096_; 
v_res_2096_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg(v_msgData_2092_, v_macroStack_2093_, v___y_2094_);
lean_dec_ref(v___y_2094_);
return v_res_2096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(lean_object* v_msg_2097_, lean_object* v___y_2098_, lean_object* v___y_2099_, lean_object* v___y_2100_, lean_object* v___y_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_){
_start:
{
lean_object* v_ref_2105_; lean_object* v___x_2106_; lean_object* v_a_2107_; lean_object* v_macroStack_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v_a_2111_; lean_object* v___x_2113_; uint8_t v_isShared_2114_; uint8_t v_isSharedCheck_2119_; 
v_ref_2105_ = lean_ctor_get(v___y_2102_, 5);
v___x_2106_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_parseChooseArg_spec__0_spec__0_spec__1(v_msg_2097_, v___y_2100_, v___y_2101_, v___y_2102_, v___y_2103_);
v_a_2107_ = lean_ctor_get(v___x_2106_, 0);
lean_inc(v_a_2107_);
lean_dec_ref(v___x_2106_);
v_macroStack_2108_ = lean_ctor_get(v___y_2098_, 1);
v___x_2109_ = l_Lean_Elab_getBetterRef(v_ref_2105_, v_macroStack_2108_);
lean_inc(v_macroStack_2108_);
v___x_2110_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg(v_a_2107_, v_macroStack_2108_, v___y_2102_);
v_a_2111_ = lean_ctor_get(v___x_2110_, 0);
v_isSharedCheck_2119_ = !lean_is_exclusive(v___x_2110_);
if (v_isSharedCheck_2119_ == 0)
{
v___x_2113_ = v___x_2110_;
v_isShared_2114_ = v_isSharedCheck_2119_;
goto v_resetjp_2112_;
}
else
{
lean_inc(v_a_2111_);
lean_dec(v___x_2110_);
v___x_2113_ = lean_box(0);
v_isShared_2114_ = v_isSharedCheck_2119_;
goto v_resetjp_2112_;
}
v_resetjp_2112_:
{
lean_object* v___x_2115_; lean_object* v___x_2117_; 
v___x_2115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2115_, 0, v___x_2109_);
lean_ctor_set(v___x_2115_, 1, v_a_2111_);
if (v_isShared_2114_ == 0)
{
lean_ctor_set_tag(v___x_2113_, 1);
lean_ctor_set(v___x_2113_, 0, v___x_2115_);
v___x_2117_ = v___x_2113_;
goto v_reusejp_2116_;
}
else
{
lean_object* v_reuseFailAlloc_2118_; 
v_reuseFailAlloc_2118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2118_, 0, v___x_2115_);
v___x_2117_ = v_reuseFailAlloc_2118_;
goto v_reusejp_2116_;
}
v_reusejp_2116_:
{
return v___x_2117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg___boxed(lean_object* v_msg_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_){
_start:
{
lean_object* v_res_2128_; 
v_res_2128_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(v_msg_2120_, v___y_2121_, v___y_2122_, v___y_2123_, v___y_2124_, v___y_2125_, v___y_2126_);
lean_dec(v___y_2126_);
lean_dec_ref(v___y_2125_);
lean_dec(v___y_2124_);
lean_dec_ref(v___y_2123_);
lean_dec(v___y_2122_);
lean_dec_ref(v___y_2121_);
return v_res_2128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg(lean_object* v_ref_2129_, lean_object* v_msg_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_){
_start:
{
lean_object* v_fileName_2138_; lean_object* v_fileMap_2139_; lean_object* v_options_2140_; lean_object* v_currRecDepth_2141_; lean_object* v_maxRecDepth_2142_; lean_object* v_ref_2143_; lean_object* v_currNamespace_2144_; lean_object* v_openDecls_2145_; lean_object* v_initHeartbeats_2146_; lean_object* v_maxHeartbeats_2147_; lean_object* v_quotContext_2148_; lean_object* v_currMacroScope_2149_; uint8_t v_diag_2150_; lean_object* v_cancelTk_x3f_2151_; uint8_t v_suppressElabErrors_2152_; lean_object* v_inheritedTraceOptions_2153_; lean_object* v_ref_2154_; lean_object* v___x_2155_; lean_object* v___x_2156_; 
v_fileName_2138_ = lean_ctor_get(v___y_2135_, 0);
v_fileMap_2139_ = lean_ctor_get(v___y_2135_, 1);
v_options_2140_ = lean_ctor_get(v___y_2135_, 2);
v_currRecDepth_2141_ = lean_ctor_get(v___y_2135_, 3);
v_maxRecDepth_2142_ = lean_ctor_get(v___y_2135_, 4);
v_ref_2143_ = lean_ctor_get(v___y_2135_, 5);
v_currNamespace_2144_ = lean_ctor_get(v___y_2135_, 6);
v_openDecls_2145_ = lean_ctor_get(v___y_2135_, 7);
v_initHeartbeats_2146_ = lean_ctor_get(v___y_2135_, 8);
v_maxHeartbeats_2147_ = lean_ctor_get(v___y_2135_, 9);
v_quotContext_2148_ = lean_ctor_get(v___y_2135_, 10);
v_currMacroScope_2149_ = lean_ctor_get(v___y_2135_, 11);
v_diag_2150_ = lean_ctor_get_uint8(v___y_2135_, sizeof(void*)*14);
v_cancelTk_x3f_2151_ = lean_ctor_get(v___y_2135_, 12);
v_suppressElabErrors_2152_ = lean_ctor_get_uint8(v___y_2135_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2153_ = lean_ctor_get(v___y_2135_, 13);
v_ref_2154_ = l_Lean_replaceRef(v_ref_2129_, v_ref_2143_);
lean_inc_ref(v_inheritedTraceOptions_2153_);
lean_inc(v_cancelTk_x3f_2151_);
lean_inc(v_currMacroScope_2149_);
lean_inc(v_quotContext_2148_);
lean_inc(v_maxHeartbeats_2147_);
lean_inc(v_initHeartbeats_2146_);
lean_inc(v_openDecls_2145_);
lean_inc(v_currNamespace_2144_);
lean_inc(v_maxRecDepth_2142_);
lean_inc(v_currRecDepth_2141_);
lean_inc_ref(v_options_2140_);
lean_inc_ref(v_fileMap_2139_);
lean_inc_ref(v_fileName_2138_);
v___x_2155_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2155_, 0, v_fileName_2138_);
lean_ctor_set(v___x_2155_, 1, v_fileMap_2139_);
lean_ctor_set(v___x_2155_, 2, v_options_2140_);
lean_ctor_set(v___x_2155_, 3, v_currRecDepth_2141_);
lean_ctor_set(v___x_2155_, 4, v_maxRecDepth_2142_);
lean_ctor_set(v___x_2155_, 5, v_ref_2154_);
lean_ctor_set(v___x_2155_, 6, v_currNamespace_2144_);
lean_ctor_set(v___x_2155_, 7, v_openDecls_2145_);
lean_ctor_set(v___x_2155_, 8, v_initHeartbeats_2146_);
lean_ctor_set(v___x_2155_, 9, v_maxHeartbeats_2147_);
lean_ctor_set(v___x_2155_, 10, v_quotContext_2148_);
lean_ctor_set(v___x_2155_, 11, v_currMacroScope_2149_);
lean_ctor_set(v___x_2155_, 12, v_cancelTk_x3f_2151_);
lean_ctor_set(v___x_2155_, 13, v_inheritedTraceOptions_2153_);
lean_ctor_set_uint8(v___x_2155_, sizeof(void*)*14, v_diag_2150_);
lean_ctor_set_uint8(v___x_2155_, sizeof(void*)*14 + 1, v_suppressElabErrors_2152_);
v___x_2156_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(v_msg_2130_, v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___x_2155_, v___y_2136_);
lean_dec_ref_known(v___x_2155_, 14);
return v___x_2156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg___boxed(lean_object* v_ref_2157_, lean_object* v_msg_2158_, lean_object* v___y_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_){
_start:
{
lean_object* v_res_2166_; 
v_res_2166_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg(v_ref_2157_, v_msg_2158_, v___y_2159_, v___y_2160_, v___y_2161_, v___y_2162_, v___y_2163_, v___y_2164_);
lean_dec(v___y_2164_);
lean_dec_ref(v___y_2163_);
lean_dec(v___y_2162_);
lean_dec_ref(v___y_2161_);
lean_dec(v___y_2160_);
lean_dec_ref(v___y_2159_);
lean_dec(v_ref_2157_);
return v_res_2166_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2168_; lean_object* v___x_2169_; 
v___x_2168_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__0));
v___x_2169_ = l_Lean_stringToMessageData(v___x_2168_);
return v___x_2169_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3(void){
_start:
{
lean_object* v___x_2171_; lean_object* v___x_2172_; 
v___x_2171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__2));
v___x_2172_ = l_Lean_stringToMessageData(v___x_2171_);
return v___x_2172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0(lean_object* v_ref_2173_, lean_object* v_fst_2174_, lean_object* v_expectedType_x3f_2175_, lean_object* v_snd_2176_, lean_object* v_name_2177_, lean_object* v___y_2178_, lean_object* v___y_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_){
_start:
{
lean_object* v___x_2185_; 
lean_inc_ref(v_fst_2174_);
lean_inc(v_ref_2173_);
v___x_2185_ = l_Lean_Elab_Term_addLocalVarInfo(v_ref_2173_, v_fst_2174_, v___y_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2185_) == 0)
{
lean_object* v___x_2187_; uint8_t v_isShared_2188_; uint8_t v_isSharedCheck_2259_; 
v_isSharedCheck_2259_ = !lean_is_exclusive(v___x_2185_);
if (v_isSharedCheck_2259_ == 0)
{
lean_object* v_unused_2260_; 
v_unused_2260_ = lean_ctor_get(v___x_2185_, 0);
lean_dec(v_unused_2260_);
v___x_2187_ = v___x_2185_;
v_isShared_2188_ = v_isSharedCheck_2259_;
goto v_resetjp_2186_;
}
else
{
lean_dec(v___x_2185_);
v___x_2187_ = lean_box(0);
v_isShared_2188_ = v_isSharedCheck_2259_;
goto v_resetjp_2186_;
}
v_resetjp_2186_:
{
if (lean_obj_tag(v_expectedType_x3f_2175_) == 1)
{
lean_object* v_val_2189_; lean_object* v___x_2190_; 
lean_del_object(v___x_2187_);
v_val_2189_ = lean_ctor_get(v_expectedType_x3f_2175_, 0);
lean_inc(v_val_2189_);
lean_dec_ref_known(v_expectedType_x3f_2175_, 1);
lean_inc(v___y_2183_);
lean_inc_ref(v___y_2182_);
lean_inc(v___y_2181_);
lean_inc_ref(v___y_2180_);
lean_inc_ref(v_fst_2174_);
v___x_2190_ = lean_infer_type(v_fst_2174_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2190_) == 0)
{
lean_object* v_a_2191_; lean_object* v___x_2192_; 
v_a_2191_ = lean_ctor_get(v___x_2190_, 0);
lean_inc(v_a_2191_);
lean_dec_ref_known(v___x_2190_, 1);
v___x_2192_ = l_Lean_Elab_Term_elabType(v_val_2189_, v___y_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2192_) == 0)
{
lean_object* v_a_2193_; lean_object* v___y_2195_; lean_object* v___y_2196_; lean_object* v___y_2197_; lean_object* v___y_2198_; lean_object* v___x_2202_; 
v_a_2193_ = lean_ctor_get(v___x_2192_, 0);
lean_inc_n(v_a_2193_, 2);
lean_dec_ref_known(v___x_2192_, 1);
lean_inc(v_a_2191_);
v___x_2202_ = l_Lean_Meta_isExprDefEq(v_a_2191_, v_a_2193_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2202_) == 0)
{
lean_object* v_a_2203_; uint8_t v___x_2204_; 
v_a_2203_ = lean_ctor_get(v___x_2202_, 0);
lean_inc(v_a_2203_);
lean_dec_ref_known(v___x_2202_, 1);
v___x_2204_ = lean_unbox(v_a_2203_);
lean_dec(v_a_2203_);
if (v___x_2204_ == 0)
{
lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; 
lean_dec(v_snd_2176_);
lean_dec_ref(v_fst_2174_);
v___x_2205_ = lean_box(0);
v___x_2206_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__19));
v___x_2207_ = l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(v_a_2191_, v_a_2193_, v___x_2205_, v___x_2206_);
if (lean_obj_tag(v___x_2207_) == 0)
{
lean_object* v_a_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v_a_2216_; lean_object* v___x_2218_; uint8_t v_isShared_2219_; uint8_t v_isSharedCheck_2223_; 
v_a_2208_ = lean_ctor_get(v___x_2207_, 0);
lean_inc(v_a_2208_);
lean_dec_ref_known(v___x_2207_, 1);
v___x_2209_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1);
v___x_2210_ = l_Lean_MessageData_ofName(v_name_2177_);
v___x_2211_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2211_, 0, v___x_2209_);
lean_ctor_set(v___x_2211_, 1, v___x_2210_);
v___x_2212_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3);
v___x_2213_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2213_, 0, v___x_2211_);
lean_ctor_set(v___x_2213_, 1, v___x_2212_);
v___x_2214_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2214_, 0, v___x_2213_);
lean_ctor_set(v___x_2214_, 1, v_a_2208_);
v___x_2215_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg(v_ref_2173_, v___x_2214_, v___y_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_ref_2173_);
v_a_2216_ = lean_ctor_get(v___x_2215_, 0);
v_isSharedCheck_2223_ = !lean_is_exclusive(v___x_2215_);
if (v_isSharedCheck_2223_ == 0)
{
v___x_2218_ = v___x_2215_;
v_isShared_2219_ = v_isSharedCheck_2223_;
goto v_resetjp_2217_;
}
else
{
lean_inc(v_a_2216_);
lean_dec(v___x_2215_);
v___x_2218_ = lean_box(0);
v_isShared_2219_ = v_isSharedCheck_2223_;
goto v_resetjp_2217_;
}
v_resetjp_2217_:
{
lean_object* v___x_2221_; 
if (v_isShared_2219_ == 0)
{
v___x_2221_ = v___x_2218_;
goto v_reusejp_2220_;
}
else
{
lean_object* v_reuseFailAlloc_2222_; 
v_reuseFailAlloc_2222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2222_, 0, v_a_2216_);
v___x_2221_ = v_reuseFailAlloc_2222_;
goto v_reusejp_2220_;
}
v_reusejp_2220_:
{
return v___x_2221_;
}
}
}
else
{
lean_object* v_a_2224_; lean_object* v___x_2226_; uint8_t v_isShared_2227_; uint8_t v_isSharedCheck_2231_; 
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_name_2177_);
lean_dec(v_ref_2173_);
v_a_2224_ = lean_ctor_get(v___x_2207_, 0);
v_isSharedCheck_2231_ = !lean_is_exclusive(v___x_2207_);
if (v_isSharedCheck_2231_ == 0)
{
v___x_2226_ = v___x_2207_;
v_isShared_2227_ = v_isSharedCheck_2231_;
goto v_resetjp_2225_;
}
else
{
lean_inc(v_a_2224_);
lean_dec(v___x_2207_);
v___x_2226_ = lean_box(0);
v_isShared_2227_ = v_isSharedCheck_2231_;
goto v_resetjp_2225_;
}
v_resetjp_2225_:
{
lean_object* v___x_2229_; 
if (v_isShared_2227_ == 0)
{
v___x_2229_ = v___x_2226_;
goto v_reusejp_2228_;
}
else
{
lean_object* v_reuseFailAlloc_2230_; 
v_reuseFailAlloc_2230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2230_, 0, v_a_2224_);
v___x_2229_ = v_reuseFailAlloc_2230_;
goto v_reusejp_2228_;
}
v_reusejp_2228_:
{
return v___x_2229_;
}
}
}
}
else
{
lean_dec(v_a_2191_);
lean_dec(v_name_2177_);
lean_dec(v_ref_2173_);
v___y_2195_ = v___y_2180_;
v___y_2196_ = v___y_2181_;
v___y_2197_ = v___y_2182_;
v___y_2198_ = v___y_2183_;
goto v___jp_2194_;
}
}
else
{
lean_object* v_a_2232_; lean_object* v___x_2234_; uint8_t v_isShared_2235_; uint8_t v_isSharedCheck_2239_; 
lean_dec(v_a_2193_);
lean_dec(v_a_2191_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_name_2177_);
lean_dec(v_snd_2176_);
lean_dec_ref(v_fst_2174_);
lean_dec(v_ref_2173_);
v_a_2232_ = lean_ctor_get(v___x_2202_, 0);
v_isSharedCheck_2239_ = !lean_is_exclusive(v___x_2202_);
if (v_isSharedCheck_2239_ == 0)
{
v___x_2234_ = v___x_2202_;
v_isShared_2235_ = v_isSharedCheck_2239_;
goto v_resetjp_2233_;
}
else
{
lean_inc(v_a_2232_);
lean_dec(v___x_2202_);
v___x_2234_ = lean_box(0);
v_isShared_2235_ = v_isSharedCheck_2239_;
goto v_resetjp_2233_;
}
v_resetjp_2233_:
{
lean_object* v___x_2237_; 
if (v_isShared_2235_ == 0)
{
v___x_2237_ = v___x_2234_;
goto v_reusejp_2236_;
}
else
{
lean_object* v_reuseFailAlloc_2238_; 
v_reuseFailAlloc_2238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2238_, 0, v_a_2232_);
v___x_2237_ = v_reuseFailAlloc_2238_;
goto v_reusejp_2236_;
}
v_reusejp_2236_:
{
return v___x_2237_;
}
}
}
v___jp_2194_:
{
lean_object* v___x_2199_; uint8_t v___x_2200_; lean_object* v___x_2201_; 
v___x_2199_ = l_Lean_Expr_fvarId_x21(v_fst_2174_);
lean_dec_ref(v_fst_2174_);
v___x_2200_ = 1;
v___x_2201_ = l_Lean_MVarId_changeLocalDecl(v_snd_2176_, v___x_2199_, v_a_2193_, v___x_2200_, v___y_2195_, v___y_2196_, v___y_2197_, v___y_2198_);
lean_dec(v___y_2198_);
lean_dec_ref(v___y_2197_);
lean_dec(v___y_2196_);
lean_dec_ref(v___y_2195_);
return v___x_2201_;
}
}
else
{
lean_object* v_a_2240_; lean_object* v___x_2242_; uint8_t v_isShared_2243_; uint8_t v_isSharedCheck_2247_; 
lean_dec(v_a_2191_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_name_2177_);
lean_dec(v_snd_2176_);
lean_dec_ref(v_fst_2174_);
lean_dec(v_ref_2173_);
v_a_2240_ = lean_ctor_get(v___x_2192_, 0);
v_isSharedCheck_2247_ = !lean_is_exclusive(v___x_2192_);
if (v_isSharedCheck_2247_ == 0)
{
v___x_2242_ = v___x_2192_;
v_isShared_2243_ = v_isSharedCheck_2247_;
goto v_resetjp_2241_;
}
else
{
lean_inc(v_a_2240_);
lean_dec(v___x_2192_);
v___x_2242_ = lean_box(0);
v_isShared_2243_ = v_isSharedCheck_2247_;
goto v_resetjp_2241_;
}
v_resetjp_2241_:
{
lean_object* v___x_2245_; 
if (v_isShared_2243_ == 0)
{
v___x_2245_ = v___x_2242_;
goto v_reusejp_2244_;
}
else
{
lean_object* v_reuseFailAlloc_2246_; 
v_reuseFailAlloc_2246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2246_, 0, v_a_2240_);
v___x_2245_ = v_reuseFailAlloc_2246_;
goto v_reusejp_2244_;
}
v_reusejp_2244_:
{
return v___x_2245_;
}
}
}
}
else
{
lean_object* v_a_2248_; lean_object* v___x_2250_; uint8_t v_isShared_2251_; uint8_t v_isSharedCheck_2255_; 
lean_dec(v_val_2189_);
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_name_2177_);
lean_dec(v_snd_2176_);
lean_dec_ref(v_fst_2174_);
lean_dec(v_ref_2173_);
v_a_2248_ = lean_ctor_get(v___x_2190_, 0);
v_isSharedCheck_2255_ = !lean_is_exclusive(v___x_2190_);
if (v_isSharedCheck_2255_ == 0)
{
v___x_2250_ = v___x_2190_;
v_isShared_2251_ = v_isSharedCheck_2255_;
goto v_resetjp_2249_;
}
else
{
lean_inc(v_a_2248_);
lean_dec(v___x_2190_);
v___x_2250_ = lean_box(0);
v_isShared_2251_ = v_isSharedCheck_2255_;
goto v_resetjp_2249_;
}
v_resetjp_2249_:
{
lean_object* v___x_2253_; 
if (v_isShared_2251_ == 0)
{
v___x_2253_ = v___x_2250_;
goto v_reusejp_2252_;
}
else
{
lean_object* v_reuseFailAlloc_2254_; 
v_reuseFailAlloc_2254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2254_, 0, v_a_2248_);
v___x_2253_ = v_reuseFailAlloc_2254_;
goto v_reusejp_2252_;
}
v_reusejp_2252_:
{
return v___x_2253_;
}
}
}
}
else
{
lean_object* v___x_2257_; 
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_name_2177_);
lean_dec(v_expectedType_x3f_2175_);
lean_dec_ref(v_fst_2174_);
lean_dec(v_ref_2173_);
if (v_isShared_2188_ == 0)
{
lean_ctor_set(v___x_2187_, 0, v_snd_2176_);
v___x_2257_ = v___x_2187_;
goto v_reusejp_2256_;
}
else
{
lean_object* v_reuseFailAlloc_2258_; 
v_reuseFailAlloc_2258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2258_, 0, v_snd_2176_);
v___x_2257_ = v_reuseFailAlloc_2258_;
goto v_reusejp_2256_;
}
v_reusejp_2256_:
{
return v___x_2257_;
}
}
}
}
else
{
lean_object* v_a_2261_; lean_object* v___x_2263_; uint8_t v_isShared_2264_; uint8_t v_isSharedCheck_2268_; 
lean_dec(v___y_2183_);
lean_dec_ref(v___y_2182_);
lean_dec(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec(v_name_2177_);
lean_dec(v_snd_2176_);
lean_dec(v_expectedType_x3f_2175_);
lean_dec_ref(v_fst_2174_);
lean_dec(v_ref_2173_);
v_a_2261_ = lean_ctor_get(v___x_2185_, 0);
v_isSharedCheck_2268_ = !lean_is_exclusive(v___x_2185_);
if (v_isSharedCheck_2268_ == 0)
{
v___x_2263_ = v___x_2185_;
v_isShared_2264_ = v_isSharedCheck_2268_;
goto v_resetjp_2262_;
}
else
{
lean_inc(v_a_2261_);
lean_dec(v___x_2185_);
v___x_2263_ = lean_box(0);
v_isShared_2264_ = v_isSharedCheck_2268_;
goto v_resetjp_2262_;
}
v_resetjp_2262_:
{
lean_object* v___x_2266_; 
if (v_isShared_2264_ == 0)
{
v___x_2266_ = v___x_2263_;
goto v_reusejp_2265_;
}
else
{
lean_object* v_reuseFailAlloc_2267_; 
v_reuseFailAlloc_2267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2267_, 0, v_a_2261_);
v___x_2266_ = v_reuseFailAlloc_2267_;
goto v_reusejp_2265_;
}
v_reusejp_2265_:
{
return v___x_2266_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___boxed(lean_object* v_ref_2269_, lean_object* v_fst_2270_, lean_object* v_expectedType_x3f_2271_, lean_object* v_snd_2272_, lean_object* v_name_2273_, lean_object* v___y_2274_, lean_object* v___y_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_){
_start:
{
lean_object* v_res_2281_; 
v_res_2281_ = lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0(v_ref_2269_, v_fst_2270_, v_expectedType_x3f_2271_, v_snd_2272_, v_name_2273_, v___y_2274_, v___y_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
lean_dec(v___y_2275_);
lean_dec_ref(v___y_2274_);
return v_res_2281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo(lean_object* v_g_2282_, uint8_t v_nondep_2283_, lean_object* v_h_2284_, lean_object* v_arg_2285_, lean_object* v_a_2286_, lean_object* v_a_2287_, lean_object* v_a_2288_, lean_object* v_a_2289_, lean_object* v_a_2290_, lean_object* v_a_2291_){
_start:
{
lean_object* v_ref_2293_; lean_object* v_name_2294_; lean_object* v_expectedType_x3f_2295_; lean_object* v___x_2296_; 
v_ref_2293_ = lean_ctor_get(v_arg_2285_, 0);
lean_inc(v_ref_2293_);
v_name_2294_ = lean_ctor_get(v_arg_2285_, 1);
lean_inc_n(v_name_2294_, 2);
v_expectedType_x3f_2295_ = lean_ctor_get(v_arg_2285_, 2);
lean_inc(v_expectedType_x3f_2295_);
lean_dec_ref(v_arg_2285_);
v___x_2296_ = lp_mathlib_Mathlib_Tactic_Choose_choose1(v_g_2282_, v_nondep_2283_, v_h_2284_, v_name_2294_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_);
if (lean_obj_tag(v___x_2296_) == 0)
{
lean_object* v_a_2297_; lean_object* v_snd_2298_; lean_object* v_fst_2299_; lean_object* v_fst_2300_; lean_object* v_snd_2301_; lean_object* v___x_2303_; uint8_t v_isShared_2304_; uint8_t v_isSharedCheck_2326_; 
v_a_2297_ = lean_ctor_get(v___x_2296_, 0);
lean_inc(v_a_2297_);
lean_dec_ref_known(v___x_2296_, 1);
v_snd_2298_ = lean_ctor_get(v_a_2297_, 1);
lean_inc(v_snd_2298_);
v_fst_2299_ = lean_ctor_get(v_a_2297_, 0);
lean_inc(v_fst_2299_);
lean_dec(v_a_2297_);
v_fst_2300_ = lean_ctor_get(v_snd_2298_, 0);
v_snd_2301_ = lean_ctor_get(v_snd_2298_, 1);
v_isSharedCheck_2326_ = !lean_is_exclusive(v_snd_2298_);
if (v_isSharedCheck_2326_ == 0)
{
v___x_2303_ = v_snd_2298_;
v_isShared_2304_ = v_isSharedCheck_2326_;
goto v_resetjp_2302_;
}
else
{
lean_inc(v_snd_2301_);
lean_inc(v_fst_2300_);
lean_dec(v_snd_2298_);
v___x_2303_ = lean_box(0);
v_isShared_2304_ = v_isSharedCheck_2326_;
goto v_resetjp_2302_;
}
v_resetjp_2302_:
{
lean_object* v___f_2305_; lean_object* v___x_2306_; 
lean_inc(v_snd_2301_);
v___f_2305_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___boxed), 12, 5);
lean_closure_set(v___f_2305_, 0, v_ref_2293_);
lean_closure_set(v___f_2305_, 1, v_fst_2300_);
lean_closure_set(v___f_2305_, 2, v_expectedType_x3f_2295_);
lean_closure_set(v___f_2305_, 3, v_snd_2301_);
lean_closure_set(v___f_2305_, 4, v_name_2294_);
v___x_2306_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg(v_snd_2301_, v___f_2305_, v_a_2286_, v_a_2287_, v_a_2288_, v_a_2289_, v_a_2290_, v_a_2291_);
if (lean_obj_tag(v___x_2306_) == 0)
{
lean_object* v_a_2307_; lean_object* v___x_2309_; uint8_t v_isShared_2310_; uint8_t v_isSharedCheck_2317_; 
v_a_2307_ = lean_ctor_get(v___x_2306_, 0);
v_isSharedCheck_2317_ = !lean_is_exclusive(v___x_2306_);
if (v_isSharedCheck_2317_ == 0)
{
v___x_2309_ = v___x_2306_;
v_isShared_2310_ = v_isSharedCheck_2317_;
goto v_resetjp_2308_;
}
else
{
lean_inc(v_a_2307_);
lean_dec(v___x_2306_);
v___x_2309_ = lean_box(0);
v_isShared_2310_ = v_isSharedCheck_2317_;
goto v_resetjp_2308_;
}
v_resetjp_2308_:
{
lean_object* v___x_2312_; 
if (v_isShared_2304_ == 0)
{
lean_ctor_set(v___x_2303_, 1, v_a_2307_);
lean_ctor_set(v___x_2303_, 0, v_fst_2299_);
v___x_2312_ = v___x_2303_;
goto v_reusejp_2311_;
}
else
{
lean_object* v_reuseFailAlloc_2316_; 
v_reuseFailAlloc_2316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2316_, 0, v_fst_2299_);
lean_ctor_set(v_reuseFailAlloc_2316_, 1, v_a_2307_);
v___x_2312_ = v_reuseFailAlloc_2316_;
goto v_reusejp_2311_;
}
v_reusejp_2311_:
{
lean_object* v___x_2314_; 
if (v_isShared_2310_ == 0)
{
lean_ctor_set(v___x_2309_, 0, v___x_2312_);
v___x_2314_ = v___x_2309_;
goto v_reusejp_2313_;
}
else
{
lean_object* v_reuseFailAlloc_2315_; 
v_reuseFailAlloc_2315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2315_, 0, v___x_2312_);
v___x_2314_ = v_reuseFailAlloc_2315_;
goto v_reusejp_2313_;
}
v_reusejp_2313_:
{
return v___x_2314_;
}
}
}
}
else
{
lean_object* v_a_2318_; lean_object* v___x_2320_; uint8_t v_isShared_2321_; uint8_t v_isSharedCheck_2325_; 
lean_del_object(v___x_2303_);
lean_dec(v_fst_2299_);
v_a_2318_ = lean_ctor_get(v___x_2306_, 0);
v_isSharedCheck_2325_ = !lean_is_exclusive(v___x_2306_);
if (v_isSharedCheck_2325_ == 0)
{
v___x_2320_ = v___x_2306_;
v_isShared_2321_ = v_isSharedCheck_2325_;
goto v_resetjp_2319_;
}
else
{
lean_inc(v_a_2318_);
lean_dec(v___x_2306_);
v___x_2320_ = lean_box(0);
v_isShared_2321_ = v_isSharedCheck_2325_;
goto v_resetjp_2319_;
}
v_resetjp_2319_:
{
lean_object* v___x_2323_; 
if (v_isShared_2321_ == 0)
{
v___x_2323_ = v___x_2320_;
goto v_reusejp_2322_;
}
else
{
lean_object* v_reuseFailAlloc_2324_; 
v_reuseFailAlloc_2324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2324_, 0, v_a_2318_);
v___x_2323_ = v_reuseFailAlloc_2324_;
goto v_reusejp_2322_;
}
v_reusejp_2322_:
{
return v___x_2323_;
}
}
}
}
}
else
{
lean_object* v_a_2327_; lean_object* v___x_2329_; uint8_t v_isShared_2330_; uint8_t v_isSharedCheck_2334_; 
lean_dec(v_expectedType_x3f_2295_);
lean_dec(v_name_2294_);
lean_dec(v_ref_2293_);
v_a_2327_ = lean_ctor_get(v___x_2296_, 0);
v_isSharedCheck_2334_ = !lean_is_exclusive(v___x_2296_);
if (v_isSharedCheck_2334_ == 0)
{
v___x_2329_ = v___x_2296_;
v_isShared_2330_ = v_isSharedCheck_2334_;
goto v_resetjp_2328_;
}
else
{
lean_inc(v_a_2327_);
lean_dec(v___x_2296_);
v___x_2329_ = lean_box(0);
v_isShared_2330_ = v_isSharedCheck_2334_;
goto v_resetjp_2328_;
}
v_resetjp_2328_:
{
lean_object* v___x_2332_; 
if (v_isShared_2330_ == 0)
{
v___x_2332_ = v___x_2329_;
goto v_reusejp_2331_;
}
else
{
lean_object* v_reuseFailAlloc_2333_; 
v_reuseFailAlloc_2333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2333_, 0, v_a_2327_);
v___x_2332_ = v_reuseFailAlloc_2333_;
goto v_reusejp_2331_;
}
v_reusejp_2331_:
{
return v___x_2332_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___boxed(lean_object* v_g_2335_, lean_object* v_nondep_2336_, lean_object* v_h_2337_, lean_object* v_arg_2338_, lean_object* v_a_2339_, lean_object* v_a_2340_, lean_object* v_a_2341_, lean_object* v_a_2342_, lean_object* v_a_2343_, lean_object* v_a_2344_, lean_object* v_a_2345_){
_start:
{
uint8_t v_nondep_boxed_2346_; lean_object* v_res_2347_; 
v_nondep_boxed_2346_ = lean_unbox(v_nondep_2336_);
v_res_2347_ = lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo(v_g_2335_, v_nondep_boxed_2346_, v_h_2337_, v_arg_2338_, v_a_2339_, v_a_2340_, v_a_2341_, v_a_2342_, v_a_2343_, v_a_2344_);
lean_dec(v_a_2344_);
lean_dec_ref(v_a_2343_);
lean_dec(v_a_2342_);
lean_dec_ref(v_a_2341_);
lean_dec(v_a_2340_);
lean_dec_ref(v_a_2339_);
return v_res_2347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0(lean_object* v_00_u03b1_2348_, lean_object* v_ref_2349_, lean_object* v_msg_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
lean_object* v___x_2358_; 
v___x_2358_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg(v_ref_2349_, v_msg_2350_, v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_, v___y_2356_);
return v___x_2358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___boxed(lean_object* v_00_u03b1_2359_, lean_object* v_ref_2360_, lean_object* v_msg_2361_, lean_object* v___y_2362_, lean_object* v___y_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_){
_start:
{
lean_object* v_res_2369_; 
v_res_2369_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0(v_00_u03b1_2359_, v_ref_2360_, v_msg_2361_, v___y_2362_, v___y_2363_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_);
lean_dec(v___y_2367_);
lean_dec_ref(v___y_2366_);
lean_dec(v___y_2365_);
lean_dec_ref(v___y_2364_);
lean_dec(v___y_2363_);
lean_dec_ref(v___y_2362_);
lean_dec(v_ref_2360_);
return v_res_2369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0(lean_object* v_00_u03b1_2370_, lean_object* v_msg_2371_, lean_object* v___y_2372_, lean_object* v___y_2373_, lean_object* v___y_2374_, lean_object* v___y_2375_, lean_object* v___y_2376_, lean_object* v___y_2377_){
_start:
{
lean_object* v___x_2379_; 
v___x_2379_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(v_msg_2371_, v___y_2372_, v___y_2373_, v___y_2374_, v___y_2375_, v___y_2376_, v___y_2377_);
return v___x_2379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___boxed(lean_object* v_00_u03b1_2380_, lean_object* v_msg_2381_, lean_object* v___y_2382_, lean_object* v___y_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_){
_start:
{
lean_object* v_res_2389_; 
v_res_2389_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0(v_00_u03b1_2380_, v_msg_2381_, v___y_2382_, v___y_2383_, v___y_2384_, v___y_2385_, v___y_2386_, v___y_2387_);
lean_dec(v___y_2387_);
lean_dec_ref(v___y_2386_);
lean_dec(v___y_2385_);
lean_dec_ref(v___y_2384_);
lean_dec(v___y_2383_);
lean_dec_ref(v___y_2382_);
return v_res_2389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2(lean_object* v_msgData_2390_, lean_object* v_macroStack_2391_, lean_object* v___y_2392_, lean_object* v___y_2393_, lean_object* v___y_2394_, lean_object* v___y_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_){
_start:
{
lean_object* v___x_2399_; 
v___x_2399_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___redArg(v_msgData_2390_, v_macroStack_2391_, v___y_2396_);
return v___x_2399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2___boxed(lean_object* v_msgData_2400_, lean_object* v_macroStack_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_, lean_object* v___y_2405_, lean_object* v___y_2406_, lean_object* v___y_2407_, lean_object* v___y_2408_){
_start:
{
lean_object* v_res_2409_; 
v_res_2409_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0_spec__2(v_msgData_2400_, v_macroStack_2401_, v___y_2402_, v___y_2403_, v___y_2404_, v___y_2405_, v___y_2406_, v___y_2407_);
lean_dec(v___y_2407_);
lean_dec_ref(v___y_2406_);
lean_dec(v___y_2405_);
lean_dec_ref(v___y_2404_);
lean_dec(v___y_2403_);
lean_dec_ref(v___y_2402_);
return v_res_2409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___lam__0(lean_object* v_ref_2410_, lean_object* v___x_2411_, lean_object* v_expectedType_x3f_2412_, lean_object* v_snd_2413_, lean_object* v_fst_2414_, lean_object* v_name_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_, lean_object* v___y_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_){
_start:
{
lean_object* v___x_2423_; 
lean_inc_ref(v___x_2411_);
lean_inc(v_ref_2410_);
v___x_2423_ = l_Lean_Elab_Term_addLocalVarInfo(v_ref_2410_, v___x_2411_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_);
if (lean_obj_tag(v___x_2423_) == 0)
{
lean_object* v___x_2425_; uint8_t v_isShared_2426_; uint8_t v_isSharedCheck_2496_; 
v_isSharedCheck_2496_ = !lean_is_exclusive(v___x_2423_);
if (v_isSharedCheck_2496_ == 0)
{
lean_object* v_unused_2497_; 
v_unused_2497_ = lean_ctor_get(v___x_2423_, 0);
lean_dec(v_unused_2497_);
v___x_2425_ = v___x_2423_;
v_isShared_2426_ = v_isSharedCheck_2496_;
goto v_resetjp_2424_;
}
else
{
lean_dec(v___x_2423_);
v___x_2425_ = lean_box(0);
v_isShared_2426_ = v_isSharedCheck_2496_;
goto v_resetjp_2424_;
}
v_resetjp_2424_:
{
if (lean_obj_tag(v_expectedType_x3f_2412_) == 1)
{
lean_object* v_val_2427_; lean_object* v___x_2428_; 
lean_del_object(v___x_2425_);
v_val_2427_ = lean_ctor_get(v_expectedType_x3f_2412_, 0);
lean_inc(v_val_2427_);
lean_dec_ref_known(v_expectedType_x3f_2412_, 1);
lean_inc(v___y_2421_);
lean_inc_ref(v___y_2420_);
lean_inc(v___y_2419_);
lean_inc_ref(v___y_2418_);
v___x_2428_ = lean_infer_type(v___x_2411_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_);
if (lean_obj_tag(v___x_2428_) == 0)
{
lean_object* v_a_2429_; lean_object* v___x_2430_; 
v_a_2429_ = lean_ctor_get(v___x_2428_, 0);
lean_inc(v_a_2429_);
lean_dec_ref_known(v___x_2428_, 1);
v___x_2430_ = l_Lean_Elab_Term_elabType(v_val_2427_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_);
if (lean_obj_tag(v___x_2430_) == 0)
{
lean_object* v_a_2431_; lean_object* v___y_2433_; lean_object* v___y_2434_; lean_object* v___y_2435_; lean_object* v___y_2436_; lean_object* v___x_2439_; 
v_a_2431_ = lean_ctor_get(v___x_2430_, 0);
lean_inc_n(v_a_2431_, 2);
lean_dec_ref_known(v___x_2430_, 1);
lean_inc(v_a_2429_);
v___x_2439_ = l_Lean_Meta_isExprDefEq(v_a_2429_, v_a_2431_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_);
if (lean_obj_tag(v___x_2439_) == 0)
{
lean_object* v_a_2440_; uint8_t v___x_2441_; 
v_a_2440_ = lean_ctor_get(v___x_2439_, 0);
lean_inc(v_a_2440_);
lean_dec_ref_known(v___x_2439_, 1);
v___x_2441_ = lean_unbox(v_a_2440_);
lean_dec(v_a_2440_);
if (v___x_2441_ == 0)
{
lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; 
lean_dec(v_fst_2414_);
lean_dec(v_snd_2413_);
v___x_2442_ = lean_box(0);
v___x_2443_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__19));
v___x_2444_ = l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(v_a_2429_, v_a_2431_, v___x_2442_, v___x_2443_);
if (lean_obj_tag(v___x_2444_) == 0)
{
lean_object* v_a_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v_a_2453_; lean_object* v___x_2455_; uint8_t v_isShared_2456_; uint8_t v_isSharedCheck_2460_; 
v_a_2445_ = lean_ctor_get(v___x_2444_, 0);
lean_inc(v_a_2445_);
lean_dec_ref_known(v___x_2444_, 1);
v___x_2446_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__1);
v___x_2447_ = l_Lean_MessageData_ofName(v_name_2415_);
v___x_2448_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2448_, 0, v___x_2446_);
lean_ctor_set(v___x_2448_, 1, v___x_2447_);
v___x_2449_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo___lam__0___closed__3);
v___x_2450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2450_, 0, v___x_2448_);
lean_ctor_set(v___x_2450_, 1, v___x_2449_);
v___x_2451_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2451_, 0, v___x_2450_);
lean_ctor_set(v___x_2451_, 1, v_a_2445_);
v___x_2452_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0___redArg(v_ref_2410_, v___x_2451_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_);
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_ref_2410_);
v_a_2453_ = lean_ctor_get(v___x_2452_, 0);
v_isSharedCheck_2460_ = !lean_is_exclusive(v___x_2452_);
if (v_isSharedCheck_2460_ == 0)
{
v___x_2455_ = v___x_2452_;
v_isShared_2456_ = v_isSharedCheck_2460_;
goto v_resetjp_2454_;
}
else
{
lean_inc(v_a_2453_);
lean_dec(v___x_2452_);
v___x_2455_ = lean_box(0);
v_isShared_2456_ = v_isSharedCheck_2460_;
goto v_resetjp_2454_;
}
v_resetjp_2454_:
{
lean_object* v___x_2458_; 
if (v_isShared_2456_ == 0)
{
v___x_2458_ = v___x_2455_;
goto v_reusejp_2457_;
}
else
{
lean_object* v_reuseFailAlloc_2459_; 
v_reuseFailAlloc_2459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2459_, 0, v_a_2453_);
v___x_2458_ = v_reuseFailAlloc_2459_;
goto v_reusejp_2457_;
}
v_reusejp_2457_:
{
return v___x_2458_;
}
}
}
else
{
lean_object* v_a_2461_; lean_object* v___x_2463_; uint8_t v_isShared_2464_; uint8_t v_isSharedCheck_2468_; 
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_name_2415_);
lean_dec(v_ref_2410_);
v_a_2461_ = lean_ctor_get(v___x_2444_, 0);
v_isSharedCheck_2468_ = !lean_is_exclusive(v___x_2444_);
if (v_isSharedCheck_2468_ == 0)
{
v___x_2463_ = v___x_2444_;
v_isShared_2464_ = v_isSharedCheck_2468_;
goto v_resetjp_2462_;
}
else
{
lean_inc(v_a_2461_);
lean_dec(v___x_2444_);
v___x_2463_ = lean_box(0);
v_isShared_2464_ = v_isSharedCheck_2468_;
goto v_resetjp_2462_;
}
v_resetjp_2462_:
{
lean_object* v___x_2466_; 
if (v_isShared_2464_ == 0)
{
v___x_2466_ = v___x_2463_;
goto v_reusejp_2465_;
}
else
{
lean_object* v_reuseFailAlloc_2467_; 
v_reuseFailAlloc_2467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2467_, 0, v_a_2461_);
v___x_2466_ = v_reuseFailAlloc_2467_;
goto v_reusejp_2465_;
}
v_reusejp_2465_:
{
return v___x_2466_;
}
}
}
}
else
{
lean_dec(v_a_2429_);
lean_dec(v_name_2415_);
lean_dec(v_ref_2410_);
v___y_2433_ = v___y_2418_;
v___y_2434_ = v___y_2419_;
v___y_2435_ = v___y_2420_;
v___y_2436_ = v___y_2421_;
goto v___jp_2432_;
}
}
else
{
lean_object* v_a_2469_; lean_object* v___x_2471_; uint8_t v_isShared_2472_; uint8_t v_isSharedCheck_2476_; 
lean_dec(v_a_2431_);
lean_dec(v_a_2429_);
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_name_2415_);
lean_dec(v_fst_2414_);
lean_dec(v_snd_2413_);
lean_dec(v_ref_2410_);
v_a_2469_ = lean_ctor_get(v___x_2439_, 0);
v_isSharedCheck_2476_ = !lean_is_exclusive(v___x_2439_);
if (v_isSharedCheck_2476_ == 0)
{
v___x_2471_ = v___x_2439_;
v_isShared_2472_ = v_isSharedCheck_2476_;
goto v_resetjp_2470_;
}
else
{
lean_inc(v_a_2469_);
lean_dec(v___x_2439_);
v___x_2471_ = lean_box(0);
v_isShared_2472_ = v_isSharedCheck_2476_;
goto v_resetjp_2470_;
}
v_resetjp_2470_:
{
lean_object* v___x_2474_; 
if (v_isShared_2472_ == 0)
{
v___x_2474_ = v___x_2471_;
goto v_reusejp_2473_;
}
else
{
lean_object* v_reuseFailAlloc_2475_; 
v_reuseFailAlloc_2475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2475_, 0, v_a_2469_);
v___x_2474_ = v_reuseFailAlloc_2475_;
goto v_reusejp_2473_;
}
v_reusejp_2473_:
{
return v___x_2474_;
}
}
}
v___jp_2432_:
{
uint8_t v___x_2437_; lean_object* v___x_2438_; 
v___x_2437_ = 1;
v___x_2438_ = l_Lean_MVarId_changeLocalDecl(v_snd_2413_, v_fst_2414_, v_a_2431_, v___x_2437_, v___y_2433_, v___y_2434_, v___y_2435_, v___y_2436_);
lean_dec(v___y_2436_);
lean_dec_ref(v___y_2435_);
lean_dec(v___y_2434_);
lean_dec_ref(v___y_2433_);
return v___x_2438_;
}
}
else
{
lean_object* v_a_2477_; lean_object* v___x_2479_; uint8_t v_isShared_2480_; uint8_t v_isSharedCheck_2484_; 
lean_dec(v_a_2429_);
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_name_2415_);
lean_dec(v_fst_2414_);
lean_dec(v_snd_2413_);
lean_dec(v_ref_2410_);
v_a_2477_ = lean_ctor_get(v___x_2430_, 0);
v_isSharedCheck_2484_ = !lean_is_exclusive(v___x_2430_);
if (v_isSharedCheck_2484_ == 0)
{
v___x_2479_ = v___x_2430_;
v_isShared_2480_ = v_isSharedCheck_2484_;
goto v_resetjp_2478_;
}
else
{
lean_inc(v_a_2477_);
lean_dec(v___x_2430_);
v___x_2479_ = lean_box(0);
v_isShared_2480_ = v_isSharedCheck_2484_;
goto v_resetjp_2478_;
}
v_resetjp_2478_:
{
lean_object* v___x_2482_; 
if (v_isShared_2480_ == 0)
{
v___x_2482_ = v___x_2479_;
goto v_reusejp_2481_;
}
else
{
lean_object* v_reuseFailAlloc_2483_; 
v_reuseFailAlloc_2483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2483_, 0, v_a_2477_);
v___x_2482_ = v_reuseFailAlloc_2483_;
goto v_reusejp_2481_;
}
v_reusejp_2481_:
{
return v___x_2482_;
}
}
}
}
else
{
lean_object* v_a_2485_; lean_object* v___x_2487_; uint8_t v_isShared_2488_; uint8_t v_isSharedCheck_2492_; 
lean_dec(v_val_2427_);
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_name_2415_);
lean_dec(v_fst_2414_);
lean_dec(v_snd_2413_);
lean_dec(v_ref_2410_);
v_a_2485_ = lean_ctor_get(v___x_2428_, 0);
v_isSharedCheck_2492_ = !lean_is_exclusive(v___x_2428_);
if (v_isSharedCheck_2492_ == 0)
{
v___x_2487_ = v___x_2428_;
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
else
{
lean_inc(v_a_2485_);
lean_dec(v___x_2428_);
v___x_2487_ = lean_box(0);
v_isShared_2488_ = v_isSharedCheck_2492_;
goto v_resetjp_2486_;
}
v_resetjp_2486_:
{
lean_object* v___x_2490_; 
if (v_isShared_2488_ == 0)
{
v___x_2490_ = v___x_2487_;
goto v_reusejp_2489_;
}
else
{
lean_object* v_reuseFailAlloc_2491_; 
v_reuseFailAlloc_2491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2491_, 0, v_a_2485_);
v___x_2490_ = v_reuseFailAlloc_2491_;
goto v_reusejp_2489_;
}
v_reusejp_2489_:
{
return v___x_2490_;
}
}
}
}
else
{
lean_object* v___x_2494_; 
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_name_2415_);
lean_dec(v_fst_2414_);
lean_dec(v_expectedType_x3f_2412_);
lean_dec_ref(v___x_2411_);
lean_dec(v_ref_2410_);
if (v_isShared_2426_ == 0)
{
lean_ctor_set(v___x_2425_, 0, v_snd_2413_);
v___x_2494_ = v___x_2425_;
goto v_reusejp_2493_;
}
else
{
lean_object* v_reuseFailAlloc_2495_; 
v_reuseFailAlloc_2495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2495_, 0, v_snd_2413_);
v___x_2494_ = v_reuseFailAlloc_2495_;
goto v_reusejp_2493_;
}
v_reusejp_2493_:
{
return v___x_2494_;
}
}
}
}
else
{
lean_object* v_a_2498_; lean_object* v___x_2500_; uint8_t v_isShared_2501_; uint8_t v_isSharedCheck_2505_; 
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v_name_2415_);
lean_dec(v_fst_2414_);
lean_dec(v_snd_2413_);
lean_dec(v_expectedType_x3f_2412_);
lean_dec_ref(v___x_2411_);
lean_dec(v_ref_2410_);
v_a_2498_ = lean_ctor_get(v___x_2423_, 0);
v_isSharedCheck_2505_ = !lean_is_exclusive(v___x_2423_);
if (v_isSharedCheck_2505_ == 0)
{
v___x_2500_ = v___x_2423_;
v_isShared_2501_ = v_isSharedCheck_2505_;
goto v_resetjp_2499_;
}
else
{
lean_inc(v_a_2498_);
lean_dec(v___x_2423_);
v___x_2500_ = lean_box(0);
v_isShared_2501_ = v_isSharedCheck_2505_;
goto v_resetjp_2499_;
}
v_resetjp_2499_:
{
lean_object* v___x_2503_; 
if (v_isShared_2501_ == 0)
{
v___x_2503_ = v___x_2500_;
goto v_reusejp_2502_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v_a_2498_);
v___x_2503_ = v_reuseFailAlloc_2504_;
goto v_reusejp_2502_;
}
v_reusejp_2502_:
{
return v___x_2503_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___lam__0___boxed(lean_object* v_ref_2506_, lean_object* v___x_2507_, lean_object* v_expectedType_x3f_2508_, lean_object* v_snd_2509_, lean_object* v_fst_2510_, lean_object* v_name_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_){
_start:
{
lean_object* v_res_2519_; 
v_res_2519_ = lp_mathlib_Mathlib_Tactic_Choose_elabChoose___lam__0(v_ref_2506_, v___x_2507_, v_expectedType_x3f_2508_, v_snd_2509_, v_fst_2510_, v_name_2511_, v___y_2512_, v___y_2513_, v___y_2514_, v___y_2515_, v___y_2516_, v___y_2517_);
lean_dec(v___y_2513_);
lean_dec_ref(v___y_2512_);
return v_res_2519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg(lean_object* v_as_x27_2520_, lean_object* v_b_2521_, lean_object* v___y_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_){
_start:
{
if (lean_obj_tag(v_as_x27_2520_) == 0)
{
lean_object* v___x_2527_; 
v___x_2527_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2527_, 0, v_b_2521_);
return v___x_2527_;
}
else
{
lean_object* v_head_2528_; lean_object* v_tail_2529_; lean_object* v___x_2530_; uint8_t v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; 
v_head_2528_ = lean_ctor_get(v_as_x27_2520_, 0);
v_tail_2529_ = lean_ctor_get(v_as_x27_2520_, 1);
lean_inc(v_head_2528_);
v___x_2530_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2530_, 0, v_head_2528_);
v___x_2531_ = 0;
v___x_2532_ = lean_box(0);
v___x_2533_ = l_Lean_Meta_mkFreshExprMVar(v___x_2530_, v___x_2531_, v___x_2532_, v___y_2522_, v___y_2523_, v___y_2524_, v___y_2525_);
if (lean_obj_tag(v___x_2533_) == 0)
{
lean_object* v_a_2534_; lean_object* v___x_2535_; lean_object* v___x_2536_; lean_object* v___x_2537_; 
v_a_2534_ = lean_ctor_get(v___x_2533_, 0);
lean_inc(v_a_2534_);
lean_dec_ref_known(v___x_2533_, 1);
v___x_2535_ = l_Lean_Expr_mvarId_x21(v_a_2534_);
lean_dec(v_a_2534_);
v___x_2536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2536_, 0, v___x_2535_);
v___x_2537_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2537_, 0, v_b_2521_);
lean_ctor_set(v___x_2537_, 1, v___x_2536_);
v_as_x27_2520_ = v_tail_2529_;
v_b_2521_ = v___x_2537_;
goto _start;
}
else
{
lean_object* v_a_2539_; lean_object* v___x_2541_; uint8_t v_isShared_2542_; uint8_t v_isSharedCheck_2546_; 
lean_dec_ref(v_b_2521_);
v_a_2539_ = lean_ctor_get(v___x_2533_, 0);
v_isSharedCheck_2546_ = !lean_is_exclusive(v___x_2533_);
if (v_isSharedCheck_2546_ == 0)
{
v___x_2541_ = v___x_2533_;
v_isShared_2542_ = v_isSharedCheck_2546_;
goto v_resetjp_2540_;
}
else
{
lean_inc(v_a_2539_);
lean_dec(v___x_2533_);
v___x_2541_ = lean_box(0);
v_isShared_2542_ = v_isSharedCheck_2546_;
goto v_resetjp_2540_;
}
v_resetjp_2540_:
{
lean_object* v___x_2544_; 
if (v_isShared_2542_ == 0)
{
v___x_2544_ = v___x_2541_;
goto v_reusejp_2543_;
}
else
{
lean_object* v_reuseFailAlloc_2545_; 
v_reuseFailAlloc_2545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2545_, 0, v_a_2539_);
v___x_2544_ = v_reuseFailAlloc_2545_;
goto v_reusejp_2543_;
}
v_reusejp_2543_:
{
return v___x_2544_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg___boxed(lean_object* v_as_x27_2547_, lean_object* v_b_2548_, lean_object* v___y_2549_, lean_object* v___y_2550_, lean_object* v___y_2551_, lean_object* v___y_2552_, lean_object* v___y_2553_){
_start:
{
lean_object* v_res_2554_; 
v_res_2554_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg(v_as_x27_2547_, v_b_2548_, v___y_2549_, v___y_2550_, v___y_2551_, v___y_2552_);
lean_dec(v___y_2552_);
lean_dec_ref(v___y_2551_);
lean_dec(v___y_2550_);
lean_dec_ref(v___y_2549_);
lean_dec(v_as_x27_2547_);
return v_res_2554_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__1(void){
_start:
{
lean_object* v___x_2556_; lean_object* v___x_2557_; 
v___x_2556_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__0));
v___x_2557_ = l_Lean_stringToMessageData(v___x_2556_);
return v___x_2557_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__3(void){
_start:
{
lean_object* v___x_2559_; lean_object* v_msg_2560_; 
v___x_2559_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__2));
v_msg_2560_ = l_Lean_stringToMessageData(v___x_2559_);
return v_msg_2560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose(uint8_t v_nondep_2561_, lean_object* v_h_2562_, lean_object* v_x_2563_, lean_object* v_x_2564_, lean_object* v_x_2565_, lean_object* v_a_2566_, lean_object* v_a_2567_, lean_object* v_a_2568_, lean_object* v_a_2569_, lean_object* v_a_2570_, lean_object* v_a_2571_){
_start:
{
if (lean_obj_tag(v_x_2563_) == 0)
{
lean_object* v___x_2573_; lean_object* v___x_2574_; 
lean_dec(v_x_2565_);
lean_dec(v_x_2564_);
lean_dec(v_h_2562_);
v___x_2573_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__1, &lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__1);
v___x_2574_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(v___x_2573_, v_a_2566_, v_a_2567_, v_a_2568_, v_a_2569_, v_a_2570_, v_a_2571_);
return v___x_2574_;
}
else
{
lean_object* v_head_2575_; lean_object* v_tail_2576_; lean_object* v_____x_2578_; lean_object* v___y_2579_; lean_object* v___y_2580_; lean_object* v___y_2581_; lean_object* v___y_2582_; lean_object* v___y_2583_; lean_object* v___y_2584_; lean_object* v___y_2594_; lean_object* v___y_2595_; lean_object* v___y_2596_; lean_object* v___y_2597_; lean_object* v___y_2598_; lean_object* v___y_2599_; 
v_head_2575_ = lean_ctor_get(v_x_2563_, 0);
lean_inc(v_head_2575_);
v_tail_2576_ = lean_ctor_get(v_x_2563_, 1);
lean_inc(v_tail_2576_);
lean_dec_ref_known(v_x_2563_, 2);
if (lean_obj_tag(v_tail_2576_) == 0)
{
lean_dec(v_h_2562_);
if (v_nondep_2561_ == 1)
{
if (lean_obj_tag(v_x_2564_) == 1)
{
lean_object* v_ts_2624_; lean_object* v_msg_2625_; lean_object* v___x_2626_; 
lean_dec(v_head_2575_);
lean_dec(v_x_2565_);
v_ts_2624_ = lean_ctor_get(v_x_2564_, 0);
lean_inc(v_ts_2624_);
lean_dec_ref_known(v_x_2564_, 1);
v_msg_2625_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__3, &lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Choose_elabChoose___closed__3);
v___x_2626_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg(v_ts_2624_, v_msg_2625_, v_a_2568_, v_a_2569_, v_a_2570_, v_a_2571_);
lean_dec(v_ts_2624_);
if (lean_obj_tag(v___x_2626_) == 0)
{
lean_object* v_a_2627_; lean_object* v___x_2628_; 
v_a_2627_ = lean_ctor_get(v___x_2626_, 0);
lean_inc(v_a_2627_);
lean_dec_ref_known(v___x_2626_, 1);
v___x_2628_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__0_spec__0___redArg(v_a_2627_, v_a_2566_, v_a_2567_, v_a_2568_, v_a_2569_, v_a_2570_, v_a_2571_);
return v___x_2628_;
}
else
{
lean_object* v_a_2629_; lean_object* v___x_2631_; uint8_t v_isShared_2632_; uint8_t v_isSharedCheck_2636_; 
v_a_2629_ = lean_ctor_get(v___x_2626_, 0);
v_isSharedCheck_2636_ = !lean_is_exclusive(v___x_2626_);
if (v_isSharedCheck_2636_ == 0)
{
v___x_2631_ = v___x_2626_;
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
else
{
lean_inc(v_a_2629_);
lean_dec(v___x_2626_);
v___x_2631_ = lean_box(0);
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
v_resetjp_2630_:
{
lean_object* v___x_2634_; 
if (v_isShared_2632_ == 0)
{
v___x_2634_ = v___x_2631_;
goto v_reusejp_2633_;
}
else
{
lean_object* v_reuseFailAlloc_2635_; 
v_reuseFailAlloc_2635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2635_, 0, v_a_2629_);
v___x_2634_ = v_reuseFailAlloc_2635_;
goto v_reusejp_2633_;
}
v_reusejp_2633_:
{
return v___x_2634_;
}
}
}
}
else
{
lean_dec(v_x_2564_);
v___y_2594_ = v_a_2566_;
v___y_2595_ = v_a_2567_;
v___y_2596_ = v_a_2568_;
v___y_2597_ = v_a_2569_;
v___y_2598_ = v_a_2570_;
v___y_2599_ = v_a_2571_;
goto v___jp_2593_;
}
}
else
{
lean_dec(v_x_2564_);
v___y_2594_ = v_a_2566_;
v___y_2595_ = v_a_2567_;
v___y_2596_ = v_a_2568_;
v___y_2597_ = v_a_2569_;
v___y_2598_ = v_a_2570_;
v___y_2599_ = v_a_2571_;
goto v___jp_2593_;
}
}
else
{
lean_object* v___x_2637_; 
v___x_2637_ = lp_mathlib_Mathlib_Tactic_Choose_choose1WithInfo(v_x_2565_, v_nondep_2561_, v_h_2562_, v_head_2575_, v_a_2566_, v_a_2567_, v_a_2568_, v_a_2569_, v_a_2570_, v_a_2571_);
if (lean_obj_tag(v___x_2637_) == 0)
{
lean_object* v_a_2638_; lean_object* v_fst_2639_; lean_object* v_snd_2640_; lean_object* v___x_2641_; lean_object* v___x_2642_; 
v_a_2638_ = lean_ctor_get(v___x_2637_, 0);
lean_inc(v_a_2638_);
lean_dec_ref_known(v___x_2637_, 1);
v_fst_2639_ = lean_ctor_get(v_a_2638_, 0);
lean_inc(v_fst_2639_);
v_snd_2640_ = lean_ctor_get(v_a_2638_, 1);
lean_inc(v_snd_2640_);
lean_dec(v_a_2638_);
v___x_2641_ = lean_box(0);
v___x_2642_ = lp_mathlib_Mathlib_Tactic_Choose_ElimStatus_merge(v_x_2564_, v_fst_2639_);
v_h_2562_ = v___x_2641_;
v_x_2563_ = v_tail_2576_;
v_x_2564_ = v___x_2642_;
v_x_2565_ = v_snd_2640_;
goto _start;
}
else
{
lean_object* v_a_2644_; lean_object* v___x_2646_; uint8_t v_isShared_2647_; uint8_t v_isSharedCheck_2651_; 
lean_dec(v_tail_2576_);
lean_dec(v_x_2564_);
v_a_2644_ = lean_ctor_get(v___x_2637_, 0);
v_isSharedCheck_2651_ = !lean_is_exclusive(v___x_2637_);
if (v_isSharedCheck_2651_ == 0)
{
v___x_2646_ = v___x_2637_;
v_isShared_2647_ = v_isSharedCheck_2651_;
goto v_resetjp_2645_;
}
else
{
lean_inc(v_a_2644_);
lean_dec(v___x_2637_);
v___x_2646_ = lean_box(0);
v_isShared_2647_ = v_isSharedCheck_2651_;
goto v_resetjp_2645_;
}
v_resetjp_2645_:
{
lean_object* v___x_2649_; 
if (v_isShared_2647_ == 0)
{
v___x_2649_ = v___x_2646_;
goto v_reusejp_2648_;
}
else
{
lean_object* v_reuseFailAlloc_2650_; 
v_reuseFailAlloc_2650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2650_, 0, v_a_2644_);
v___x_2649_ = v_reuseFailAlloc_2650_;
goto v_reusejp_2648_;
}
v_reusejp_2648_:
{
return v___x_2649_;
}
}
}
}
v___jp_2577_:
{
lean_object* v_fst_2585_; lean_object* v_snd_2586_; lean_object* v_ref_2587_; lean_object* v_name_2588_; lean_object* v_expectedType_x3f_2589_; lean_object* v___x_2590_; lean_object* v___f_2591_; lean_object* v___x_2592_; 
v_fst_2585_ = lean_ctor_get(v_____x_2578_, 0);
lean_inc_n(v_fst_2585_, 2);
v_snd_2586_ = lean_ctor_get(v_____x_2578_, 1);
lean_inc_n(v_snd_2586_, 2);
lean_dec_ref(v_____x_2578_);
v_ref_2587_ = lean_ctor_get(v_head_2575_, 0);
lean_inc(v_ref_2587_);
v_name_2588_ = lean_ctor_get(v_head_2575_, 1);
lean_inc(v_name_2588_);
v_expectedType_x3f_2589_ = lean_ctor_get(v_head_2575_, 2);
lean_inc(v_expectedType_x3f_2589_);
lean_dec(v_head_2575_);
v___x_2590_ = l_Lean_Expr_fvar___override(v_fst_2585_);
v___f_2591_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Choose_elabChoose___lam__0___boxed), 13, 6);
lean_closure_set(v___f_2591_, 0, v_ref_2587_);
lean_closure_set(v___f_2591_, 1, v___x_2590_);
lean_closure_set(v___f_2591_, 2, v_expectedType_x3f_2589_);
lean_closure_set(v___f_2591_, 3, v_snd_2586_);
lean_closure_set(v___f_2591_, 4, v_fst_2585_);
lean_closure_set(v___f_2591_, 5, v_name_2588_);
v___x_2592_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Choose_choose1WithInfo_spec__1___redArg(v_snd_2586_, v___f_2591_, v___y_2579_, v___y_2580_, v___y_2581_, v___y_2582_, v___y_2583_, v___y_2584_);
return v___x_2592_;
}
v___jp_2593_:
{
lean_object* v_name_2600_; lean_object* v___x_2601_; uint8_t v___x_2602_; 
v_name_2600_ = lean_ctor_get(v_head_2575_, 1);
v___x_2601_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_mkFreshNameFrom___closed__1));
v___x_2602_ = lean_name_eq(v_name_2600_, v___x_2601_);
if (v___x_2602_ == 0)
{
lean_object* v___x_2603_; 
lean_inc(v_name_2600_);
v___x_2603_ = l_Lean_MVarId_intro(v_x_2565_, v_name_2600_, v___y_2596_, v___y_2597_, v___y_2598_, v___y_2599_);
if (lean_obj_tag(v___x_2603_) == 0)
{
lean_object* v_a_2604_; 
v_a_2604_ = lean_ctor_get(v___x_2603_, 0);
lean_inc(v_a_2604_);
lean_dec_ref_known(v___x_2603_, 1);
v_____x_2578_ = v_a_2604_;
v___y_2579_ = v___y_2594_;
v___y_2580_ = v___y_2595_;
v___y_2581_ = v___y_2596_;
v___y_2582_ = v___y_2597_;
v___y_2583_ = v___y_2598_;
v___y_2584_ = v___y_2599_;
goto v___jp_2577_;
}
else
{
lean_object* v_a_2605_; lean_object* v___x_2607_; uint8_t v_isShared_2608_; uint8_t v_isSharedCheck_2612_; 
lean_dec(v_head_2575_);
v_a_2605_ = lean_ctor_get(v___x_2603_, 0);
v_isSharedCheck_2612_ = !lean_is_exclusive(v___x_2603_);
if (v_isSharedCheck_2612_ == 0)
{
v___x_2607_ = v___x_2603_;
v_isShared_2608_ = v_isSharedCheck_2612_;
goto v_resetjp_2606_;
}
else
{
lean_inc(v_a_2605_);
lean_dec(v___x_2603_);
v___x_2607_ = lean_box(0);
v_isShared_2608_ = v_isSharedCheck_2612_;
goto v_resetjp_2606_;
}
v_resetjp_2606_:
{
lean_object* v___x_2610_; 
if (v_isShared_2608_ == 0)
{
v___x_2610_ = v___x_2607_;
goto v_reusejp_2609_;
}
else
{
lean_object* v_reuseFailAlloc_2611_; 
v_reuseFailAlloc_2611_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2611_, 0, v_a_2605_);
v___x_2610_ = v_reuseFailAlloc_2611_;
goto v_reusejp_2609_;
}
v_reusejp_2609_:
{
return v___x_2610_;
}
}
}
}
else
{
uint8_t v___x_2613_; lean_object* v___x_2614_; 
v___x_2613_ = 0;
v___x_2614_ = l_Lean_Meta_intro1Core(v_x_2565_, v___x_2613_, v___y_2596_, v___y_2597_, v___y_2598_, v___y_2599_);
if (lean_obj_tag(v___x_2614_) == 0)
{
lean_object* v_a_2615_; 
v_a_2615_ = lean_ctor_get(v___x_2614_, 0);
lean_inc(v_a_2615_);
lean_dec_ref_known(v___x_2614_, 1);
v_____x_2578_ = v_a_2615_;
v___y_2579_ = v___y_2594_;
v___y_2580_ = v___y_2595_;
v___y_2581_ = v___y_2596_;
v___y_2582_ = v___y_2597_;
v___y_2583_ = v___y_2598_;
v___y_2584_ = v___y_2599_;
goto v___jp_2577_;
}
else
{
lean_object* v_a_2616_; lean_object* v___x_2618_; uint8_t v_isShared_2619_; uint8_t v_isSharedCheck_2623_; 
lean_dec(v_head_2575_);
v_a_2616_ = lean_ctor_get(v___x_2614_, 0);
v_isSharedCheck_2623_ = !lean_is_exclusive(v___x_2614_);
if (v_isSharedCheck_2623_ == 0)
{
v___x_2618_ = v___x_2614_;
v_isShared_2619_ = v_isSharedCheck_2623_;
goto v_resetjp_2617_;
}
else
{
lean_inc(v_a_2616_);
lean_dec(v___x_2614_);
v___x_2618_ = lean_box(0);
v_isShared_2619_ = v_isSharedCheck_2623_;
goto v_resetjp_2617_;
}
v_resetjp_2617_:
{
lean_object* v___x_2621_; 
if (v_isShared_2619_ == 0)
{
v___x_2621_ = v___x_2618_;
goto v_reusejp_2620_;
}
else
{
lean_object* v_reuseFailAlloc_2622_; 
v_reuseFailAlloc_2622_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2622_, 0, v_a_2616_);
v___x_2621_ = v_reuseFailAlloc_2622_;
goto v_reusejp_2620_;
}
v_reusejp_2620_:
{
return v___x_2621_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose_elabChoose___boxed(lean_object* v_nondep_2652_, lean_object* v_h_2653_, lean_object* v_x_2654_, lean_object* v_x_2655_, lean_object* v_x_2656_, lean_object* v_a_2657_, lean_object* v_a_2658_, lean_object* v_a_2659_, lean_object* v_a_2660_, lean_object* v_a_2661_, lean_object* v_a_2662_, lean_object* v_a_2663_){
_start:
{
uint8_t v_nondep_boxed_2664_; lean_object* v_res_2665_; 
v_nondep_boxed_2664_ = lean_unbox(v_nondep_2652_);
v_res_2665_ = lp_mathlib_Mathlib_Tactic_Choose_elabChoose(v_nondep_boxed_2664_, v_h_2653_, v_x_2654_, v_x_2655_, v_x_2656_, v_a_2657_, v_a_2658_, v_a_2659_, v_a_2660_, v_a_2661_, v_a_2662_);
lean_dec(v_a_2662_);
lean_dec_ref(v_a_2661_);
lean_dec(v_a_2660_);
lean_dec_ref(v_a_2659_);
lean_dec(v_a_2658_);
lean_dec_ref(v_a_2657_);
return v_res_2665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0(lean_object* v_as_2666_, lean_object* v_as_x27_2667_, lean_object* v_b_2668_, lean_object* v_a_2669_, lean_object* v___y_2670_, lean_object* v___y_2671_, lean_object* v___y_2672_, lean_object* v___y_2673_, lean_object* v___y_2674_, lean_object* v___y_2675_){
_start:
{
lean_object* v___x_2677_; 
v___x_2677_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___redArg(v_as_x27_2667_, v_b_2668_, v___y_2672_, v___y_2673_, v___y_2674_, v___y_2675_);
return v___x_2677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0___boxed(lean_object* v_as_2678_, lean_object* v_as_x27_2679_, lean_object* v_b_2680_, lean_object* v_a_2681_, lean_object* v___y_2682_, lean_object* v___y_2683_, lean_object* v___y_2684_, lean_object* v___y_2685_, lean_object* v___y_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_){
_start:
{
lean_object* v_res_2689_; 
v_res_2689_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Choose_elabChoose_spec__0(v_as_2678_, v_as_x27_2679_, v_b_2680_, v_a_2681_, v___y_2682_, v___y_2683_, v___y_2684_, v___y_2685_, v___y_2686_, v___y_2687_);
lean_dec(v___y_2687_);
lean_dec_ref(v___y_2686_);
lean_dec(v___y_2685_);
lean_dec_ref(v___y_2684_);
lean_dec(v___y_2683_);
lean_dec_ref(v___y_2682_);
lean_dec(v_as_x27_2679_);
lean_dec(v_as_2678_);
return v_res_2689_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__19(void){
_start:
{
lean_object* v___x_2731_; lean_object* v___x_2732_; lean_object* v___x_2733_; lean_object* v___x_2734_; 
v___x_2731_ = lp_mathlib_Mathlib_Tactic_Choose_chooseBinder;
v___x_2732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__18));
v___x_2733_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2));
v___x_2734_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2734_, 0, v___x_2733_);
lean_ctor_set(v___x_2734_, 1, v___x_2732_);
lean_ctor_set(v___x_2734_, 2, v___x_2731_);
return v___x_2734_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20(void){
_start:
{
lean_object* v___x_2735_; lean_object* v___x_2736_; lean_object* v___x_2737_; 
v___x_2735_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__19, &lp_mathlib_Mathlib_Tactic_Choose_choose___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__19);
v___x_2736_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__11));
v___x_2737_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2737_, 0, v___x_2736_);
lean_ctor_set(v___x_2737_, 1, v___x_2735_);
return v___x_2737_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__21(void){
_start:
{
lean_object* v___x_2738_; lean_object* v___x_2739_; lean_object* v___x_2740_; lean_object* v___x_2741_; 
v___x_2738_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20, &lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20);
v___x_2739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__9));
v___x_2740_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2));
v___x_2741_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2741_, 0, v___x_2740_);
lean_ctor_set(v___x_2741_, 1, v___x_2739_);
lean_ctor_set(v___x_2741_, 2, v___x_2738_);
return v___x_2741_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__29(void){
_start:
{
lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2761_; 
v___x_2758_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__28));
v___x_2759_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__21, &lp_mathlib_Mathlib_Tactic_Choose_choose___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__21);
v___x_2760_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2));
v___x_2761_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_2761_, 0, v___x_2760_);
lean_ctor_set(v___x_2761_, 1, v___x_2759_);
lean_ctor_set(v___x_2761_, 2, v___x_2758_);
return v___x_2761_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__30(void){
_start:
{
lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v___x_2764_; lean_object* v___x_2765_; 
v___x_2762_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__29, &lp_mathlib_Mathlib_Tactic_Choose_choose___closed__29_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__29);
v___x_2763_ = lean_unsigned_to_nat(1022u);
v___x_2764_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0));
v___x_2765_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_2765_, 0, v___x_2764_);
lean_ctor_set(v___x_2765_, 1, v___x_2763_);
lean_ctor_set(v___x_2765_, 2, v___x_2762_);
return v___x_2765_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_choose(void){
_start:
{
lean_object* v___x_2766_; 
v___x_2766_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__30, &lp_mathlib_Mathlib_Tactic_Choose_choose___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__30);
return v___x_2766_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2767_; lean_object* v___x_2768_; lean_object* v___x_2769_; 
v___x_2767_ = lean_box(0);
v___x_2768_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2769_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2769_, 0, v___x_2768_);
lean_ctor_set(v___x_2769_, 1, v___x_2767_);
return v___x_2769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2771_; lean_object* v___x_2772_; 
v___x_2771_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___closed__0);
v___x_2772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2772_, 0, v___x_2771_);
return v___x_2772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg___boxed(lean_object* v___y_2773_){
_start:
{
lean_object* v_res_2774_; 
v_res_2774_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
return v_res_2774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0(lean_object* v_00_u03b1_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_){
_start:
{
lean_object* v___x_2785_; 
v___x_2785_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
return v___x_2785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___boxed(lean_object* v_00_u03b1_2786_, lean_object* v___y_2787_, lean_object* v___y_2788_, lean_object* v___y_2789_, lean_object* v___y_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_){
_start:
{
lean_object* v_res_2796_; 
v_res_2796_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0(v_00_u03b1_2786_, v___y_2787_, v___y_2788_, v___y_2789_, v___y_2790_, v___y_2791_, v___y_2792_, v___y_2793_, v___y_2794_);
lean_dec(v___y_2794_);
lean_dec_ref(v___y_2793_);
lean_dec(v___y_2792_);
lean_dec_ref(v___y_2791_);
lean_dec(v___y_2790_);
lean_dec_ref(v___y_2789_);
lean_dec(v___y_2788_);
lean_dec_ref(v___y_2787_);
return v_res_2796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg(lean_object* v_a_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_, lean_object* v___y_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_, lean_object* v___y_2805_){
_start:
{
lean_object* v___x_2807_; lean_object* v___x_2808_; 
lean_inc(v___y_2799_);
lean_inc_ref(v___y_2798_);
v___x_2807_ = lean_apply_2(v_a_2797_, v___y_2798_, v___y_2799_);
v___x_2808_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___x_2807_, v___y_2800_, v___y_2801_, v___y_2802_, v___y_2803_, v___y_2804_, v___y_2805_);
return v___x_2808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg___boxed(lean_object* v_a_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_, lean_object* v___y_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_){
_start:
{
lean_object* v_res_2819_; 
v_res_2819_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg(v_a_2809_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_);
lean_dec(v___y_2817_);
lean_dec_ref(v___y_2816_);
lean_dec(v___y_2815_);
lean_dec_ref(v___y_2814_);
lean_dec(v___y_2813_);
lean_dec_ref(v___y_2812_);
lean_dec(v___y_2811_);
lean_dec_ref(v___y_2810_);
return v_res_2819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3(lean_object* v_00_u03b1_2820_, lean_object* v_a_2821_, lean_object* v___y_2822_, lean_object* v___y_2823_, lean_object* v___y_2824_, lean_object* v___y_2825_, lean_object* v___y_2826_, lean_object* v___y_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_){
_start:
{
lean_object* v___x_2831_; 
v___x_2831_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg(v_a_2821_, v___y_2822_, v___y_2823_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_, v___y_2829_);
return v___x_2831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___boxed(lean_object* v_00_u03b1_2832_, lean_object* v_a_2833_, lean_object* v___y_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_, lean_object* v___y_2841_, lean_object* v___y_2842_){
_start:
{
lean_object* v_res_2843_; 
v_res_2843_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3(v_00_u03b1_2832_, v_a_2833_, v___y_2834_, v___y_2835_, v___y_2836_, v___y_2837_, v___y_2838_, v___y_2839_, v___y_2840_, v___y_2841_);
lean_dec(v___y_2841_);
lean_dec_ref(v___y_2840_);
lean_dec(v___y_2839_);
lean_dec_ref(v___y_2838_);
lean_dec(v___y_2837_);
lean_dec_ref(v___y_2836_);
lean_dec(v___y_2835_);
lean_dec_ref(v___y_2834_);
return v_res_2843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__0(lean_object* v_a_2844_, lean_object* v_a_2845_, lean_object* v_b_2846_, uint8_t v___x_2847_, lean_object* v___y_2848_, lean_object* v___y_2849_, lean_object* v___y_2850_, lean_object* v___y_2851_, lean_object* v___y_2852_, lean_object* v___y_2853_, lean_object* v___y_2854_, lean_object* v___y_2855_){
_start:
{
lean_object* v___x_2857_; 
v___x_2857_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2849_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
if (lean_obj_tag(v___x_2857_) == 0)
{
lean_object* v_a_2858_; uint8_t v___y_2860_; 
v_a_2858_ = lean_ctor_get(v___x_2857_, 0);
lean_inc(v_a_2858_);
lean_dec_ref_known(v___x_2857_, 1);
if (lean_obj_tag(v_b_2846_) == 0)
{
uint8_t v___x_2875_; 
v___x_2875_ = 0;
v___y_2860_ = v___x_2875_;
goto v___jp_2859_;
}
else
{
v___y_2860_ = v___x_2847_;
goto v___jp_2859_;
}
v___jp_2859_:
{
lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; 
v___x_2861_ = lean_box(0);
v___x_2862_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__17));
v___x_2863_ = lp_mathlib_Mathlib_Tactic_Choose_elabChoose(v___y_2860_, v_a_2844_, v_a_2845_, v___x_2862_, v_a_2858_, v___y_2850_, v___y_2851_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
if (lean_obj_tag(v___x_2863_) == 0)
{
lean_object* v_a_2864_; lean_object* v___x_2865_; lean_object* v___x_2866_; 
v_a_2864_ = lean_ctor_get(v___x_2863_, 0);
lean_inc(v_a_2864_);
lean_dec_ref_known(v___x_2863_, 1);
v___x_2865_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2865_, 0, v_a_2864_);
lean_ctor_set(v___x_2865_, 1, v___x_2861_);
v___x_2866_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_2865_, v___y_2849_, v___y_2852_, v___y_2853_, v___y_2854_, v___y_2855_);
return v___x_2866_;
}
else
{
lean_object* v_a_2867_; lean_object* v___x_2869_; uint8_t v_isShared_2870_; uint8_t v_isSharedCheck_2874_; 
v_a_2867_ = lean_ctor_get(v___x_2863_, 0);
v_isSharedCheck_2874_ = !lean_is_exclusive(v___x_2863_);
if (v_isSharedCheck_2874_ == 0)
{
v___x_2869_ = v___x_2863_;
v_isShared_2870_ = v_isSharedCheck_2874_;
goto v_resetjp_2868_;
}
else
{
lean_inc(v_a_2867_);
lean_dec(v___x_2863_);
v___x_2869_ = lean_box(0);
v_isShared_2870_ = v_isSharedCheck_2874_;
goto v_resetjp_2868_;
}
v_resetjp_2868_:
{
lean_object* v___x_2872_; 
if (v_isShared_2870_ == 0)
{
v___x_2872_ = v___x_2869_;
goto v_reusejp_2871_;
}
else
{
lean_object* v_reuseFailAlloc_2873_; 
v_reuseFailAlloc_2873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2873_, 0, v_a_2867_);
v___x_2872_ = v_reuseFailAlloc_2873_;
goto v_reusejp_2871_;
}
v_reusejp_2871_:
{
return v___x_2872_;
}
}
}
}
}
else
{
lean_object* v_a_2876_; lean_object* v___x_2878_; uint8_t v_isShared_2879_; uint8_t v_isSharedCheck_2883_; 
lean_dec(v_a_2845_);
lean_dec(v_a_2844_);
v_a_2876_ = lean_ctor_get(v___x_2857_, 0);
v_isSharedCheck_2883_ = !lean_is_exclusive(v___x_2857_);
if (v_isSharedCheck_2883_ == 0)
{
v___x_2878_ = v___x_2857_;
v_isShared_2879_ = v_isSharedCheck_2883_;
goto v_resetjp_2877_;
}
else
{
lean_inc(v_a_2876_);
lean_dec(v___x_2857_);
v___x_2878_ = lean_box(0);
v_isShared_2879_ = v_isSharedCheck_2883_;
goto v_resetjp_2877_;
}
v_resetjp_2877_:
{
lean_object* v___x_2881_; 
if (v_isShared_2879_ == 0)
{
v___x_2881_ = v___x_2878_;
goto v_reusejp_2880_;
}
else
{
lean_object* v_reuseFailAlloc_2882_; 
v_reuseFailAlloc_2882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2882_, 0, v_a_2876_);
v___x_2881_ = v_reuseFailAlloc_2882_;
goto v_reusejp_2880_;
}
v_reusejp_2880_:
{
return v___x_2881_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__0___boxed(lean_object* v_a_2884_, lean_object* v_a_2885_, lean_object* v_b_2886_, lean_object* v___x_2887_, lean_object* v___y_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_){
_start:
{
uint8_t v___x_3788__boxed_2897_; lean_object* v_res_2898_; 
v___x_3788__boxed_2897_ = lean_unbox(v___x_2887_);
v_res_2898_ = lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__0(v_a_2884_, v_a_2885_, v_b_2886_, v___x_3788__boxed_2897_, v___y_2888_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_, v___y_2893_, v___y_2894_, v___y_2895_);
lean_dec(v___y_2895_);
lean_dec_ref(v___y_2894_);
lean_dec(v___y_2893_);
lean_dec_ref(v___y_2892_);
lean_dec(v___y_2891_);
lean_dec_ref(v___y_2890_);
lean_dec(v___y_2889_);
lean_dec_ref(v___y_2888_);
lean_dec(v_b_2886_);
return v_res_2898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg(lean_object* v_x_2899_, lean_object* v_x_2900_, lean_object* v___y_2901_, lean_object* v___y_2902_, lean_object* v___y_2903_, lean_object* v___y_2904_){
_start:
{
if (lean_obj_tag(v_x_2899_) == 0)
{
lean_object* v___x_2906_; lean_object* v___x_2907_; 
v___x_2906_ = l_List_reverse___redArg(v_x_2900_);
v___x_2907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2907_, 0, v___x_2906_);
return v___x_2907_;
}
else
{
lean_object* v_head_2908_; lean_object* v_tail_2909_; lean_object* v___x_2911_; uint8_t v_isShared_2912_; uint8_t v_isSharedCheck_2927_; 
v_head_2908_ = lean_ctor_get(v_x_2899_, 0);
v_tail_2909_ = lean_ctor_get(v_x_2899_, 1);
v_isSharedCheck_2927_ = !lean_is_exclusive(v_x_2899_);
if (v_isSharedCheck_2927_ == 0)
{
v___x_2911_ = v_x_2899_;
v_isShared_2912_ = v_isSharedCheck_2927_;
goto v_resetjp_2910_;
}
else
{
lean_inc(v_tail_2909_);
lean_inc(v_head_2908_);
lean_dec(v_x_2899_);
v___x_2911_ = lean_box(0);
v_isShared_2912_ = v_isSharedCheck_2927_;
goto v_resetjp_2910_;
}
v_resetjp_2910_:
{
lean_object* v___x_2913_; 
v___x_2913_ = lp_mathlib_Mathlib_Tactic_Choose_parseChooseArg(v_head_2908_, v___y_2901_, v___y_2902_, v___y_2903_, v___y_2904_);
if (lean_obj_tag(v___x_2913_) == 0)
{
lean_object* v_a_2914_; lean_object* v___x_2916_; 
v_a_2914_ = lean_ctor_get(v___x_2913_, 0);
lean_inc(v_a_2914_);
lean_dec_ref_known(v___x_2913_, 1);
if (v_isShared_2912_ == 0)
{
lean_ctor_set(v___x_2911_, 1, v_x_2900_);
lean_ctor_set(v___x_2911_, 0, v_a_2914_);
v___x_2916_ = v___x_2911_;
goto v_reusejp_2915_;
}
else
{
lean_object* v_reuseFailAlloc_2918_; 
v_reuseFailAlloc_2918_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2918_, 0, v_a_2914_);
lean_ctor_set(v_reuseFailAlloc_2918_, 1, v_x_2900_);
v___x_2916_ = v_reuseFailAlloc_2918_;
goto v_reusejp_2915_;
}
v_reusejp_2915_:
{
v_x_2899_ = v_tail_2909_;
v_x_2900_ = v___x_2916_;
goto _start;
}
}
else
{
lean_object* v_a_2919_; lean_object* v___x_2921_; uint8_t v_isShared_2922_; uint8_t v_isSharedCheck_2926_; 
lean_del_object(v___x_2911_);
lean_dec(v_tail_2909_);
lean_dec(v_x_2900_);
v_a_2919_ = lean_ctor_get(v___x_2913_, 0);
v_isSharedCheck_2926_ = !lean_is_exclusive(v___x_2913_);
if (v_isSharedCheck_2926_ == 0)
{
v___x_2921_ = v___x_2913_;
v_isShared_2922_ = v_isSharedCheck_2926_;
goto v_resetjp_2920_;
}
else
{
lean_inc(v_a_2919_);
lean_dec(v___x_2913_);
v___x_2921_ = lean_box(0);
v_isShared_2922_ = v_isSharedCheck_2926_;
goto v_resetjp_2920_;
}
v_resetjp_2920_:
{
lean_object* v___x_2924_; 
if (v_isShared_2922_ == 0)
{
v___x_2924_ = v___x_2921_;
goto v_reusejp_2923_;
}
else
{
lean_object* v_reuseFailAlloc_2925_; 
v_reuseFailAlloc_2925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2925_, 0, v_a_2919_);
v___x_2924_ = v_reuseFailAlloc_2925_;
goto v_reusejp_2923_;
}
v_reusejp_2923_:
{
return v___x_2924_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg___boxed(lean_object* v_x_2928_, lean_object* v_x_2929_, lean_object* v___y_2930_, lean_object* v___y_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_, lean_object* v___y_2934_){
_start:
{
lean_object* v_res_2935_; 
v_res_2935_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg(v_x_2928_, v_x_2929_, v___y_2930_, v___y_2931_, v___y_2932_, v___y_2933_);
lean_dec(v___y_2933_);
lean_dec_ref(v___y_2932_);
lean_dec(v___y_2931_);
lean_dec_ref(v___y_2930_);
return v_res_2935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__1(lean_object* v_val_2936_, lean_object* v_b_2937_, uint8_t v___x_2938_, lean_object* v___y_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_, lean_object* v___y_2943_, lean_object* v___y_2944_, lean_object* v___y_2945_, lean_object* v___y_2946_, lean_object* v___y_2947_){
_start:
{
lean_object* v_a_2950_; 
if (lean_obj_tag(v___y_2939_) == 0)
{
lean_object* v___x_2966_; 
v___x_2966_ = lean_box(0);
v_a_2950_ = v___x_2966_;
goto v___jp_2949_;
}
else
{
lean_object* v_val_2967_; lean_object* v___x_2969_; uint8_t v_isShared_2970_; uint8_t v_isSharedCheck_2986_; 
v_val_2967_ = lean_ctor_get(v___y_2939_, 0);
v_isSharedCheck_2986_ = !lean_is_exclusive(v___y_2939_);
if (v_isSharedCheck_2986_ == 0)
{
v___x_2969_ = v___y_2939_;
v_isShared_2970_ = v_isSharedCheck_2986_;
goto v_resetjp_2968_;
}
else
{
lean_inc(v_val_2967_);
lean_dec(v___y_2939_);
v___x_2969_ = lean_box(0);
v_isShared_2970_ = v_isSharedCheck_2986_;
goto v_resetjp_2968_;
}
v_resetjp_2968_:
{
lean_object* v___x_2971_; uint8_t v___x_2972_; lean_object* v___x_2973_; 
v___x_2971_ = lean_box(0);
v___x_2972_ = 0;
v___x_2973_ = l_Lean_Elab_Tactic_elabTerm(v_val_2967_, v___x_2971_, v___x_2972_, v___y_2940_, v___y_2941_, v___y_2942_, v___y_2943_, v___y_2944_, v___y_2945_, v___y_2946_, v___y_2947_);
if (lean_obj_tag(v___x_2973_) == 0)
{
lean_object* v_a_2974_; lean_object* v___x_2976_; 
v_a_2974_ = lean_ctor_get(v___x_2973_, 0);
lean_inc(v_a_2974_);
lean_dec_ref_known(v___x_2973_, 1);
if (v_isShared_2970_ == 0)
{
lean_ctor_set(v___x_2969_, 0, v_a_2974_);
v___x_2976_ = v___x_2969_;
goto v_reusejp_2975_;
}
else
{
lean_object* v_reuseFailAlloc_2977_; 
v_reuseFailAlloc_2977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2977_, 0, v_a_2974_);
v___x_2976_ = v_reuseFailAlloc_2977_;
goto v_reusejp_2975_;
}
v_reusejp_2975_:
{
v_a_2950_ = v___x_2976_;
goto v___jp_2949_;
}
}
else
{
lean_object* v_a_2978_; lean_object* v___x_2980_; uint8_t v_isShared_2981_; uint8_t v_isSharedCheck_2985_; 
lean_del_object(v___x_2969_);
lean_dec(v_b_2937_);
lean_dec_ref(v_val_2936_);
v_a_2978_ = lean_ctor_get(v___x_2973_, 0);
v_isSharedCheck_2985_ = !lean_is_exclusive(v___x_2973_);
if (v_isSharedCheck_2985_ == 0)
{
v___x_2980_ = v___x_2973_;
v_isShared_2981_ = v_isSharedCheck_2985_;
goto v_resetjp_2979_;
}
else
{
lean_inc(v_a_2978_);
lean_dec(v___x_2973_);
v___x_2980_ = lean_box(0);
v_isShared_2981_ = v_isSharedCheck_2985_;
goto v_resetjp_2979_;
}
v_resetjp_2979_:
{
lean_object* v___x_2983_; 
if (v_isShared_2981_ == 0)
{
v___x_2983_ = v___x_2980_;
goto v_reusejp_2982_;
}
else
{
lean_object* v_reuseFailAlloc_2984_; 
v_reuseFailAlloc_2984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2984_, 0, v_a_2978_);
v___x_2983_ = v_reuseFailAlloc_2984_;
goto v_reusejp_2982_;
}
v_reusejp_2982_:
{
return v___x_2983_;
}
}
}
}
}
v___jp_2949_:
{
lean_object* v___x_2951_; lean_object* v___x_2952_; lean_object* v___x_2953_; 
v___x_2951_ = lean_array_to_list(v_val_2936_);
v___x_2952_ = lean_box(0);
v___x_2953_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg(v___x_2951_, v___x_2952_, v___y_2944_, v___y_2945_, v___y_2946_, v___y_2947_);
if (lean_obj_tag(v___x_2953_) == 0)
{
lean_object* v_a_2954_; lean_object* v___x_2955_; lean_object* v___f_2956_; lean_object* v___x_2957_; 
v_a_2954_ = lean_ctor_get(v___x_2953_, 0);
lean_inc(v_a_2954_);
lean_dec_ref_known(v___x_2953_, 1);
v___x_2955_ = lean_box(v___x_2938_);
v___f_2956_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__0___boxed), 13, 4);
lean_closure_set(v___f_2956_, 0, v_a_2950_);
lean_closure_set(v___f_2956_, 1, v_a_2954_);
lean_closure_set(v___f_2956_, 2, v_b_2937_);
lean_closure_set(v___f_2956_, 3, v___x_2955_);
v___x_2957_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__3___redArg(v___f_2956_, v___y_2940_, v___y_2941_, v___y_2942_, v___y_2943_, v___y_2944_, v___y_2945_, v___y_2946_, v___y_2947_);
return v___x_2957_;
}
else
{
lean_object* v_a_2958_; lean_object* v___x_2960_; uint8_t v_isShared_2961_; uint8_t v_isSharedCheck_2965_; 
lean_dec(v_a_2950_);
lean_dec(v_b_2937_);
v_a_2958_ = lean_ctor_get(v___x_2953_, 0);
v_isSharedCheck_2965_ = !lean_is_exclusive(v___x_2953_);
if (v_isSharedCheck_2965_ == 0)
{
v___x_2960_ = v___x_2953_;
v_isShared_2961_ = v_isSharedCheck_2965_;
goto v_resetjp_2959_;
}
else
{
lean_inc(v_a_2958_);
lean_dec(v___x_2953_);
v___x_2960_ = lean_box(0);
v_isShared_2961_ = v_isSharedCheck_2965_;
goto v_resetjp_2959_;
}
v_resetjp_2959_:
{
lean_object* v___x_2963_; 
if (v_isShared_2961_ == 0)
{
v___x_2963_ = v___x_2960_;
goto v_reusejp_2962_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v_a_2958_);
v___x_2963_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2962_;
}
v_reusejp_2962_:
{
return v___x_2963_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__1___boxed(lean_object* v_val_2987_, lean_object* v_b_2988_, lean_object* v___x_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_){
_start:
{
uint8_t v___x_3943__boxed_3000_; lean_object* v_res_3001_; 
v___x_3943__boxed_3000_ = lean_unbox(v___x_2989_);
v_res_3001_ = lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__1(v_val_2987_, v_b_2988_, v___x_3943__boxed_3000_, v___y_2990_, v___y_2991_, v___y_2992_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_, v___y_2997_, v___y_2998_);
lean_dec(v___y_2998_);
lean_dec_ref(v___y_2997_);
lean_dec(v___y_2996_);
lean_dec_ref(v___y_2995_);
lean_dec(v___y_2994_);
lean_dec_ref(v___y_2993_);
lean_dec(v___y_2992_);
lean_dec_ref(v___y_2991_);
return v_res_3001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1(size_t v_sz_3002_, size_t v_i_3003_, lean_object* v_bs_3004_){
_start:
{
uint8_t v___x_3005_; 
v___x_3005_ = lean_usize_dec_lt(v_i_3003_, v_sz_3002_);
if (v___x_3005_ == 0)
{
lean_object* v___x_3006_; 
v___x_3006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3006_, 0, v_bs_3004_);
return v___x_3006_;
}
else
{
lean_object* v_v_3007_; lean_object* v___x_3008_; uint8_t v___x_3009_; 
v_v_3007_ = lean_array_uget(v_bs_3004_, v_i_3003_);
v___x_3008_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_chooseBinder___closed__4));
lean_inc(v_v_3007_);
v___x_3009_ = l_Lean_Syntax_isOfKind(v_v_3007_, v___x_3008_);
if (v___x_3009_ == 0)
{
lean_object* v___x_3010_; 
lean_dec(v_v_3007_);
lean_dec_ref(v_bs_3004_);
v___x_3010_ = lean_box(0);
return v___x_3010_;
}
else
{
lean_object* v___x_3011_; lean_object* v_bs_x27_3012_; size_t v___x_3013_; size_t v___x_3014_; lean_object* v___x_3015_; 
v___x_3011_ = lean_unsigned_to_nat(0u);
v_bs_x27_3012_ = lean_array_uset(v_bs_3004_, v_i_3003_, v___x_3011_);
v___x_3013_ = ((size_t)1ULL);
v___x_3014_ = lean_usize_add(v_i_3003_, v___x_3013_);
v___x_3015_ = lean_array_uset(v_bs_x27_3012_, v_i_3003_, v_v_3007_);
v_i_3003_ = v___x_3014_;
v_bs_3004_ = v___x_3015_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1___boxed(lean_object* v_sz_3017_, lean_object* v_i_3018_, lean_object* v_bs_3019_){
_start:
{
size_t v_sz_boxed_3020_; size_t v_i_boxed_3021_; lean_object* v_res_3022_; 
v_sz_boxed_3020_ = lean_unbox_usize(v_sz_3017_);
lean_dec(v_sz_3017_);
v_i_boxed_3021_ = lean_unbox_usize(v_i_3018_);
lean_dec(v_i_3018_);
v_res_3022_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1(v_sz_boxed_3020_, v_i_boxed_3021_, v_bs_3019_);
return v_res_3022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1(lean_object* v_x_3023_, lean_object* v_a_3024_, lean_object* v_a_3025_, lean_object* v_a_3026_, lean_object* v_a_3027_, lean_object* v_a_3028_, lean_object* v_a_3029_, lean_object* v_a_3030_, lean_object* v_a_3031_){
_start:
{
lean_object* v___x_3033_; uint8_t v___x_3034_; lean_object* v___y_3036_; lean_object* v___y_3037_; lean_object* v___y_3038_; lean_object* v___y_3039_; lean_object* v___y_3040_; lean_object* v___y_3041_; lean_object* v___y_3042_; lean_object* v___y_3043_; lean_object* v___y_3044_; lean_object* v___y_3045_; lean_object* v___y_3046_; 
v___x_3033_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0));
lean_inc(v_x_3023_);
v___x_3034_ = l_Lean_Syntax_isOfKind(v_x_3023_, v___x_3033_);
if (v___x_3034_ == 0)
{
lean_object* v___x_3050_; 
lean_dec(v_x_3023_);
v___x_3050_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
return v___x_3050_;
}
else
{
lean_object* v___x_3051_; lean_object* v_b_3053_; lean_object* v___y_3054_; lean_object* v___y_3055_; lean_object* v___y_3056_; lean_object* v___y_3057_; lean_object* v___y_3058_; lean_object* v___y_3059_; lean_object* v___y_3060_; lean_object* v___y_3061_; lean_object* v___x_3084_; uint8_t v___x_3085_; 
v___x_3051_ = lean_unsigned_to_nat(1u);
v___x_3084_ = l_Lean_Syntax_getArg(v_x_3023_, v___x_3051_);
v___x_3085_ = l_Lean_Syntax_isNone(v___x_3084_);
if (v___x_3085_ == 0)
{
uint8_t v___x_3086_; 
lean_inc(v___x_3084_);
v___x_3086_ = l_Lean_Syntax_matchesNull(v___x_3084_, v___x_3051_);
if (v___x_3086_ == 0)
{
lean_object* v___x_3087_; 
lean_dec(v___x_3084_);
lean_dec(v_x_3023_);
v___x_3087_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
return v___x_3087_;
}
else
{
lean_object* v___x_3088_; lean_object* v_b_3089_; lean_object* v___x_3090_; 
v___x_3088_ = lean_unsigned_to_nat(0u);
v_b_3089_ = l_Lean_Syntax_getArg(v___x_3084_, v___x_3088_);
lean_dec(v___x_3084_);
v___x_3090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3090_, 0, v_b_3089_);
v_b_3053_ = v___x_3090_;
v___y_3054_ = v_a_3024_;
v___y_3055_ = v_a_3025_;
v___y_3056_ = v_a_3026_;
v___y_3057_ = v_a_3027_;
v___y_3058_ = v_a_3028_;
v___y_3059_ = v_a_3029_;
v___y_3060_ = v_a_3030_;
v___y_3061_ = v_a_3031_;
goto v___jp_3052_;
}
}
else
{
lean_object* v___x_3091_; 
lean_dec(v___x_3084_);
v___x_3091_ = lean_box(0);
v_b_3053_ = v___x_3091_;
v___y_3054_ = v_a_3024_;
v___y_3055_ = v_a_3025_;
v___y_3056_ = v_a_3026_;
v___y_3057_ = v_a_3027_;
v___y_3058_ = v_a_3028_;
v___y_3059_ = v_a_3029_;
v___y_3060_ = v_a_3030_;
v___y_3061_ = v_a_3031_;
goto v___jp_3052_;
}
v___jp_3052_:
{
lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; size_t v_sz_3065_; size_t v___x_3066_; lean_object* v___x_3067_; 
v___x_3062_ = lean_unsigned_to_nat(2u);
v___x_3063_ = l_Lean_Syntax_getArg(v_x_3023_, v___x_3062_);
v___x_3064_ = l_Lean_Syntax_getArgs(v___x_3063_);
lean_dec(v___x_3063_);
v_sz_3065_ = lean_array_size(v___x_3064_);
v___x_3066_ = ((size_t)0ULL);
v___x_3067_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1(v_sz_3065_, v___x_3066_, v___x_3064_);
if (lean_obj_tag(v___x_3067_) == 0)
{
lean_object* v___x_3068_; 
lean_dec(v_b_3053_);
lean_dec(v_x_3023_);
v___x_3068_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
return v___x_3068_;
}
else
{
lean_object* v_val_3069_; lean_object* v___x_3071_; uint8_t v_isShared_3072_; uint8_t v_isSharedCheck_3083_; 
v_val_3069_ = lean_ctor_get(v___x_3067_, 0);
v_isSharedCheck_3083_ = !lean_is_exclusive(v___x_3067_);
if (v_isSharedCheck_3083_ == 0)
{
v___x_3071_ = v___x_3067_;
v_isShared_3072_ = v_isSharedCheck_3083_;
goto v_resetjp_3070_;
}
else
{
lean_inc(v_val_3069_);
lean_dec(v___x_3067_);
v___x_3071_ = lean_box(0);
v_isShared_3072_ = v_isSharedCheck_3083_;
goto v_resetjp_3070_;
}
v_resetjp_3070_:
{
lean_object* v___x_3073_; lean_object* v___x_3074_; uint8_t v___x_3075_; 
v___x_3073_ = lean_unsigned_to_nat(3u);
v___x_3074_ = l_Lean_Syntax_getArg(v_x_3023_, v___x_3073_);
lean_dec(v_x_3023_);
v___x_3075_ = l_Lean_Syntax_isNone(v___x_3074_);
if (v___x_3075_ == 0)
{
uint8_t v___x_3076_; 
lean_inc(v___x_3074_);
v___x_3076_ = l_Lean_Syntax_matchesNull(v___x_3074_, v___x_3062_);
if (v___x_3076_ == 0)
{
lean_object* v___x_3077_; 
lean_dec(v___x_3074_);
lean_del_object(v___x_3071_);
lean_dec(v_val_3069_);
lean_dec(v_b_3053_);
v___x_3077_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__0___redArg();
return v___x_3077_;
}
else
{
lean_object* v_h_3078_; lean_object* v___x_3080_; 
v_h_3078_ = l_Lean_Syntax_getArg(v___x_3074_, v___x_3051_);
lean_dec(v___x_3074_);
if (v_isShared_3072_ == 0)
{
lean_ctor_set(v___x_3071_, 0, v_h_3078_);
v___x_3080_ = v___x_3071_;
goto v_reusejp_3079_;
}
else
{
lean_object* v_reuseFailAlloc_3081_; 
v_reuseFailAlloc_3081_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3081_, 0, v_h_3078_);
v___x_3080_ = v_reuseFailAlloc_3081_;
goto v_reusejp_3079_;
}
v_reusejp_3079_:
{
v___y_3036_ = v_b_3053_;
v___y_3037_ = v_val_3069_;
v___y_3038_ = v___y_3059_;
v___y_3039_ = v___y_3061_;
v___y_3040_ = v___y_3060_;
v___y_3041_ = v___y_3056_;
v___y_3042_ = v___y_3058_;
v___y_3043_ = v___y_3057_;
v___y_3044_ = v___y_3055_;
v___y_3045_ = v___y_3054_;
v___y_3046_ = v___x_3080_;
goto v___jp_3035_;
}
}
}
else
{
lean_object* v___x_3082_; 
lean_dec(v___x_3074_);
lean_del_object(v___x_3071_);
v___x_3082_ = lean_box(0);
v___y_3036_ = v_b_3053_;
v___y_3037_ = v_val_3069_;
v___y_3038_ = v___y_3059_;
v___y_3039_ = v___y_3061_;
v___y_3040_ = v___y_3060_;
v___y_3041_ = v___y_3056_;
v___y_3042_ = v___y_3058_;
v___y_3043_ = v___y_3057_;
v___y_3044_ = v___y_3055_;
v___y_3045_ = v___y_3054_;
v___y_3046_ = v___x_3082_;
goto v___jp_3035_;
}
}
}
}
}
v___jp_3035_:
{
lean_object* v___x_3047_; lean_object* v___f_3048_; lean_object* v___x_3049_; 
v___x_3047_ = lean_box(v___x_3034_);
v___f_3048_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___lam__1___boxed), 13, 4);
lean_closure_set(v___f_3048_, 0, v___y_3037_);
lean_closure_set(v___f_3048_, 1, v___y_3036_);
lean_closure_set(v___f_3048_, 2, v___x_3047_);
lean_closure_set(v___f_3048_, 3, v___y_3046_);
v___x_3049_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3048_, v___y_3045_, v___y_3044_, v___y_3041_, v___y_3043_, v___y_3042_, v___y_3038_, v___y_3040_, v___y_3039_);
return v___x_3049_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1___boxed(lean_object* v_x_3092_, lean_object* v_a_3093_, lean_object* v_a_3094_, lean_object* v_a_3095_, lean_object* v_a_3096_, lean_object* v_a_3097_, lean_object* v_a_3098_, lean_object* v_a_3099_, lean_object* v_a_3100_, lean_object* v_a_3101_){
_start:
{
lean_object* v_res_3102_; 
v_res_3102_ = lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1(v_x_3092_, v_a_3093_, v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_, v_a_3099_, v_a_3100_);
lean_dec(v_a_3100_);
lean_dec_ref(v_a_3099_);
lean_dec(v_a_3098_);
lean_dec_ref(v_a_3097_);
lean_dec(v_a_3096_);
lean_dec_ref(v_a_3095_);
lean_dec(v_a_3094_);
lean_dec_ref(v_a_3093_);
return v_res_3102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2(lean_object* v_x_3103_, lean_object* v_x_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_, lean_object* v___y_3107_, lean_object* v___y_3108_, lean_object* v___y_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_){
_start:
{
lean_object* v___x_3114_; 
v___x_3114_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___redArg(v_x_3103_, v_x_3104_, v___y_3109_, v___y_3110_, v___y_3111_, v___y_3112_);
return v___x_3114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2___boxed(lean_object* v_x_3115_, lean_object* v_x_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_, lean_object* v___y_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_, lean_object* v___y_3124_, lean_object* v___y_3125_){
_start:
{
lean_object* v_res_3126_; 
v_res_3126_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__2(v_x_3115_, v_x_3116_, v___y_3117_, v___y_3118_, v___y_3119_, v___y_3120_, v___y_3121_, v___y_3122_, v___y_3123_, v___y_3124_);
lean_dec(v___y_3124_);
lean_dec_ref(v___y_3123_);
lean_dec(v___y_3122_);
lean_dec_ref(v___y_3121_);
lean_dec(v___y_3120_);
lean_dec_ref(v___y_3119_);
lean_dec(v___y_3118_);
lean_dec_ref(v___y_3117_);
return v_res_3126_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__4(void){
_start:
{
lean_object* v___x_3137_; lean_object* v___x_3138_; lean_object* v___x_3139_; lean_object* v___x_3140_; 
v___x_3137_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20, &lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_Choose_choose___closed__20);
v___x_3138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__3));
v___x_3139_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2));
v___x_3140_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3140_, 0, v___x_3139_);
lean_ctor_set(v___x_3140_, 1, v___x_3138_);
lean_ctor_set(v___x_3140_, 2, v___x_3137_);
return v___x_3140_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__5(void){
_start:
{
lean_object* v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; 
v___x_3141_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__28));
v___x_3142_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__4, &lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__4);
v___x_3143_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__2));
v___x_3144_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_3144_, 0, v___x_3143_);
lean_ctor_set(v___x_3144_, 1, v___x_3142_);
lean_ctor_set(v___x_3144_, 2, v___x_3141_);
return v___x_3144_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__6(void){
_start:
{
lean_object* v___x_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; 
v___x_3145_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__5, &lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__5);
v___x_3146_ = lean_unsigned_to_nat(1022u);
v___x_3147_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1));
v___x_3148_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_3148_, 0, v___x_3147_);
lean_ctor_set(v___x_3148_, 1, v___x_3146_);
lean_ctor_set(v___x_3148_, 2, v___x_3145_);
return v___x_3148_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using__(void){
_start:
{
lean_object* v___x_3149_; 
v___x_3149_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__6, &lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__6);
return v___x_3149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1_spec__0(size_t v_sz_3150_, size_t v_i_3151_, lean_object* v_bs_3152_){
_start:
{
uint8_t v___x_3153_; 
v___x_3153_ = lean_usize_dec_lt(v_i_3151_, v_sz_3150_);
if (v___x_3153_ == 0)
{
return v_bs_3152_;
}
else
{
lean_object* v_v_3154_; lean_object* v___x_3155_; lean_object* v_bs_x27_3156_; size_t v___x_3157_; size_t v___x_3158_; lean_object* v___x_3159_; 
v_v_3154_ = lean_array_uget(v_bs_3152_, v_i_3151_);
v___x_3155_ = lean_unsigned_to_nat(0u);
v_bs_x27_3156_ = lean_array_uset(v_bs_3152_, v_i_3151_, v___x_3155_);
v___x_3157_ = ((size_t)1ULL);
v___x_3158_ = lean_usize_add(v_i_3151_, v___x_3157_);
v___x_3159_ = lean_array_uset(v_bs_x27_3156_, v_i_3151_, v_v_3154_);
v_i_3151_ = v___x_3158_;
v_bs_3152_ = v___x_3159_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1_spec__0___boxed(lean_object* v_sz_3161_, lean_object* v_i_3162_, lean_object* v_bs_3163_){
_start:
{
size_t v_sz_boxed_3164_; size_t v_i_boxed_3165_; lean_object* v_res_3166_; 
v_sz_boxed_3164_ = lean_unbox_usize(v_sz_3161_);
lean_dec(v_sz_3161_);
v_i_boxed_3165_ = lean_unbox_usize(v_i_3162_);
lean_dec(v_i_3162_);
v_res_3166_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1_spec__0(v_sz_boxed_3164_, v_i_boxed_3165_, v_bs_3163_);
return v_res_3166_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__2(void){
_start:
{
lean_object* v___x_3170_; 
v___x_3170_ = l_Array_mkArray0(lean_box(0));
return v___x_3170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1(lean_object* v_x_3174_, lean_object* v_a_3175_, lean_object* v_a_3176_){
_start:
{
lean_object* v___y_3178_; lean_object* v___y_3179_; lean_object* v___y_3180_; lean_object* v___y_3181_; lean_object* v___y_3182_; lean_object* v___y_3183_; lean_object* v___y_3184_; lean_object* v___y_3185_; lean_object* v___y_3186_; lean_object* v___x_3191_; uint8_t v___x_3192_; 
v___x_3191_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using___00__closed__1));
lean_inc(v_x_3174_);
v___x_3192_ = l_Lean_Syntax_isOfKind(v_x_3174_, v___x_3191_);
if (v___x_3192_ == 0)
{
lean_object* v___x_3193_; lean_object* v___x_3194_; 
lean_dec(v_x_3174_);
v___x_3193_ = lean_box(1);
v___x_3194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3194_, 0, v___x_3193_);
lean_ctor_set(v___x_3194_, 1, v_a_3176_);
return v___x_3194_;
}
else
{
lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; size_t v_sz_3198_; size_t v___x_3199_; lean_object* v___x_3200_; 
v___x_3195_ = lean_unsigned_to_nat(1u);
v___x_3196_ = l_Lean_Syntax_getArg(v_x_3174_, v___x_3195_);
v___x_3197_ = l_Lean_Syntax_getArgs(v___x_3196_);
lean_dec(v___x_3196_);
v_sz_3198_ = lean_array_size(v___x_3197_);
v___x_3199_ = ((size_t)0ULL);
v___x_3200_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______elabRules__Mathlib__Tactic__Choose__choose__1_spec__1(v_sz_3198_, v___x_3199_, v___x_3197_);
if (lean_obj_tag(v___x_3200_) == 0)
{
lean_object* v___x_3201_; lean_object* v___x_3202_; 
lean_dec(v_x_3174_);
v___x_3201_ = lean_box(1);
v___x_3202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3202_, 0, v___x_3201_);
lean_ctor_set(v___x_3202_, 1, v_a_3176_);
return v___x_3202_;
}
else
{
lean_object* v_val_3203_; lean_object* v___x_3205_; uint8_t v_isShared_3206_; uint8_t v_isSharedCheck_3242_; 
v_val_3203_ = lean_ctor_get(v___x_3200_, 0);
v_isSharedCheck_3242_ = !lean_is_exclusive(v___x_3200_);
if (v_isSharedCheck_3242_ == 0)
{
v___x_3205_ = v___x_3200_;
v_isShared_3206_ = v_isSharedCheck_3242_;
goto v_resetjp_3204_;
}
else
{
lean_inc(v_val_3203_);
lean_dec(v___x_3200_);
v___x_3205_ = lean_box(0);
v_isShared_3206_ = v_isSharedCheck_3242_;
goto v_resetjp_3204_;
}
v_resetjp_3204_:
{
lean_object* v_h_3208_; lean_object* v___y_3209_; lean_object* v___y_3210_; lean_object* v___x_3231_; lean_object* v___x_3232_; uint8_t v___x_3233_; 
v___x_3231_ = lean_unsigned_to_nat(2u);
v___x_3232_ = l_Lean_Syntax_getArg(v_x_3174_, v___x_3231_);
lean_dec(v_x_3174_);
v___x_3233_ = l_Lean_Syntax_isNone(v___x_3232_);
if (v___x_3233_ == 0)
{
uint8_t v___x_3234_; 
lean_inc(v___x_3232_);
v___x_3234_ = l_Lean_Syntax_matchesNull(v___x_3232_, v___x_3231_);
if (v___x_3234_ == 0)
{
lean_object* v___x_3235_; lean_object* v___x_3236_; 
lean_dec(v___x_3232_);
lean_del_object(v___x_3205_);
lean_dec(v_val_3203_);
v___x_3235_ = lean_box(1);
v___x_3236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3236_, 0, v___x_3235_);
lean_ctor_set(v___x_3236_, 1, v_a_3176_);
return v___x_3236_;
}
else
{
lean_object* v_h_3237_; lean_object* v___x_3239_; 
v_h_3237_ = l_Lean_Syntax_getArg(v___x_3232_, v___x_3195_);
lean_dec(v___x_3232_);
if (v_isShared_3206_ == 0)
{
lean_ctor_set(v___x_3205_, 0, v_h_3237_);
v___x_3239_ = v___x_3205_;
goto v_reusejp_3238_;
}
else
{
lean_object* v_reuseFailAlloc_3240_; 
v_reuseFailAlloc_3240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3240_, 0, v_h_3237_);
v___x_3239_ = v_reuseFailAlloc_3240_;
goto v_reusejp_3238_;
}
v_reusejp_3238_:
{
v_h_3208_ = v___x_3239_;
v___y_3209_ = v_a_3175_;
v___y_3210_ = v_a_3176_;
goto v___jp_3207_;
}
}
}
else
{
lean_object* v___x_3241_; 
lean_dec(v___x_3232_);
lean_del_object(v___x_3205_);
v___x_3241_ = lean_box(0);
v_h_3208_ = v___x_3241_;
v___y_3209_ = v_a_3175_;
v___y_3210_ = v_a_3176_;
goto v___jp_3207_;
}
v___jp_3207_:
{
lean_object* v_ref_3211_; uint8_t v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; lean_object* v___x_3215_; lean_object* v___x_3216_; lean_object* v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v___x_3220_; lean_object* v___x_3221_; size_t v_sz_3222_; lean_object* v___x_3223_; lean_object* v___x_3224_; lean_object* v___x_3225_; 
v_ref_3211_ = lean_ctor_get(v___y_3209_, 5);
v___x_3212_ = 0;
v___x_3213_ = l_Lean_SourceInfo_fromRef(v_ref_3211_, v___x_3212_);
v___x_3214_ = ((lean_object*)(lp_mathlib_Lean_Expr_withAppAux___at___00Mathlib_Tactic_Choose_choose1_spec__6___closed__13));
v___x_3215_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__0));
lean_inc_n(v___x_3213_, 4);
v___x_3216_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3216_, 0, v___x_3213_);
lean_ctor_set(v___x_3216_, 1, v___x_3214_);
v___x_3217_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__1));
v___x_3218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose_choose___closed__6));
v___x_3219_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3219_, 0, v___x_3213_);
lean_ctor_set(v___x_3219_, 1, v___x_3218_);
v___x_3220_ = l_Lean_Syntax_node1(v___x_3213_, v___x_3217_, v___x_3219_);
v___x_3221_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__2, &lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__2);
v_sz_3222_ = lean_array_size(v_val_3203_);
v___x_3223_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1_spec__0(v_sz_3222_, v___x_3199_, v_val_3203_);
v___x_3224_ = l_Array_append___redArg(v___x_3221_, v___x_3223_);
lean_dec_ref(v___x_3223_);
v___x_3225_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3225_, 0, v___x_3213_);
lean_ctor_set(v___x_3225_, 1, v___x_3217_);
lean_ctor_set(v___x_3225_, 2, v___x_3224_);
if (lean_obj_tag(v_h_3208_) == 1)
{
lean_object* v_val_3226_; lean_object* v___x_3227_; lean_object* v___x_3228_; lean_object* v___x_3229_; 
v_val_3226_ = lean_ctor_get(v_h_3208_, 0);
lean_inc(v_val_3226_);
lean_dec_ref_known(v_h_3208_, 1);
v___x_3227_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__3));
lean_inc(v___x_3213_);
v___x_3228_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3228_, 0, v___x_3213_);
lean_ctor_set(v___x_3228_, 1, v___x_3227_);
v___x_3229_ = l_Array_mkArray2___redArg(v___x_3228_, v_val_3226_);
v___y_3178_ = v___x_3213_;
v___y_3179_ = v___x_3221_;
v___y_3180_ = v___y_3210_;
v___y_3181_ = v___x_3217_;
v___y_3182_ = v___x_3220_;
v___y_3183_ = v___x_3216_;
v___y_3184_ = v___x_3215_;
v___y_3185_ = v___x_3225_;
v___y_3186_ = v___x_3229_;
goto v___jp_3177_;
}
else
{
lean_object* v___x_3230_; 
lean_dec(v_h_3208_);
v___x_3230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___closed__4));
v___y_3178_ = v___x_3213_;
v___y_3179_ = v___x_3221_;
v___y_3180_ = v___y_3210_;
v___y_3181_ = v___x_3217_;
v___y_3182_ = v___x_3220_;
v___y_3183_ = v___x_3216_;
v___y_3184_ = v___x_3215_;
v___y_3185_ = v___x_3225_;
v___y_3186_ = v___x_3230_;
goto v___jp_3177_;
}
}
}
}
}
v___jp_3177_:
{
lean_object* v___x_3187_; lean_object* v___x_3188_; lean_object* v___x_3189_; lean_object* v___x_3190_; 
v___x_3187_ = l_Array_append___redArg(v___y_3179_, v___y_3186_);
lean_dec_ref(v___y_3186_);
lean_inc(v___y_3178_);
v___x_3188_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3188_, 0, v___y_3178_);
lean_ctor_set(v___x_3188_, 1, v___y_3181_);
lean_ctor_set(v___x_3188_, 2, v___x_3187_);
lean_inc(v___y_3184_);
v___x_3189_ = l_Lean_Syntax_node4(v___y_3178_, v___y_3184_, v___y_3183_, v___y_3182_, v___y_3185_, v___x_3188_);
v___x_3190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3190_, 0, v___x_3189_);
lean_ctor_set(v___x_3190_, 1, v___y_3180_);
return v___x_3190_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1___boxed(lean_object* v_x_3243_, lean_object* v_a_3244_, lean_object* v_a_3245_){
_start:
{
lean_object* v_res_3246_; 
v_res_3246_ = lp_mathlib_Mathlib_Tactic_Choose___aux__Mathlib__Tactic__Choose______macroRules__Mathlib__Tactic__Choose__tacticChoose_x21______Using____1(v_x_3243_, v_a_3244_, v_a_3245_);
lean_dec_ref(v_a_3244_);
return v_res_3246_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Choose(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Choose(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Choose_chooseBinder = _init_lp_mathlib_Mathlib_Tactic_Choose_chooseBinder();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Choose_chooseBinder);
lp_mathlib_Mathlib_Tactic_Choose_choose = _init_lp_mathlib_Mathlib_Tactic_Choose_choose();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Choose_choose);
lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using__ = _init_lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Choose_tacticChoose_x21______Using__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Choose(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Choose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Choose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Choose(builtin);
}
#ifdef __cplusplus
}
#endif
